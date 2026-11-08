"""Vision through a separate model, so a text-only executor (e.g. nesi3) can still 'see'."""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..llm import LLMError, get_provider
from .workspace import ToolError, Workspace, _q, _sandbox_paths, interruptible

IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
# By extension, not via mimetypes: Python's own table has no .webp (it only worked where the OS supplied one).
EXT_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".jpe": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp"}
# Providers limit an image to 5 MB of base64 (Anthropic), i.e. about 3.7 MB of raw bytes.
MAX_IMAGE_BYTES = 3_700_000
MAX_SOURCE_BYTES = 60_000_000  # bigger files are refused before reading them


def image_mime(path: str) -> str | None:
    return EXT_MIME.get(os.path.splitext(str(path))[1].lower())


def sniff_mime(data: bytes) -> str | None:
    """The real type from the first bytes (a .png that is really a JPEG must be sent as image/jpeg)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _sips_profile(src: Path, out_dir: Path) -> str:
    """Seatbelt profile for one `sips` run on an image the agent controls: it may read that image and the OS,
    write only its output dir (and the UUID-named temp file sips makes first), and has no network."""
    first, _, after = _sandbox_paths()
    deny = " ".join(f"(subpath {_q(p)})" for p in first + after)
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        f"(deny file-read* file-write* {deny})",
        '(deny file-write* (subpath "/"))',
        "(allow file-read-metadata)",
        f"(allow file-read* (literal {_q(src)}))",
        f"(allow file-read* file-write* (subpath {_q(out_dir)}))",
        # sips saves through T/<UUID> and T/.<UUID> in the per-user temp dir, whatever TMPDIR says
        '(allow file-read* file-write* (regex #"^/private/var/folders/[^/]+/[^/]+/T/'
        '\\.?[0-9A-Fa-f]+-[0-9A-Fa-f]+-[0-9A-Fa-f]+-[0-9A-Fa-f]+-[0-9A-Fa-f]+"))',
        '(allow file-read* file-write* (literal "/dev/null"))',
        "(deny lsopen)",
        "(deny appleevent-send)",
    ])


def _shrink(src: Path, size: int) -> tuple[bytes, str]:
    """A JPEG copy small enough for the vision model (a Retina screenshot is often 5-15 MB), made by macOS
    `sips` in its own sandbox: the untrusted image is never parsed by the server process itself."""
    if sys.platform != "darwin" or not shutil.which("sips") or not shutil.which("sandbox-exec"):
        raise ToolError(
            f"image is {size:,} bytes; the vision limit is {MAX_IMAGE_BYTES:,}. Make a smaller copy first "
            "(e.g. `python3 -c \"from PIL import Image; im=Image.open('in.png'); im.thumbnail((1600,1600)); "
            "im.convert('RGB').save('small.jpg', quality=85)\"`) and view that."
        )
    out_dir = Path(tempfile.mkdtemp(prefix="nesicode-view-")).resolve()  # the server's own temp: the agent can't touch it
    try:
        for edge, quality in ((2000, 80), (1400, 70), (1000, 60)):
            out = out_dir / f"{edge}.jpg"
            argv = ["sandbox-exec", "-p", _sips_profile(src, out_dir), "/usr/bin/sips", "-Z", str(edge),
                    "-s", "format", "jpeg", "-s", "formatOptions", str(quality), str(src), "--out", str(out)]
            try:
                r = subprocess.run(argv, capture_output=True, timeout=60, stdin=subprocess.DEVNULL,
                                   env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(out_dir), "TMPDIR": str(out_dir)})
            except (OSError, subprocess.SubprocessError) as exc:
                raise ToolError(f"could not make a smaller copy of the image: {exc}") from None
            if r.returncode != 0 or not out.is_file():
                msg = (r.stdout + r.stderr).decode("utf-8", "replace").strip()[-300:]
                raise ToolError(f"could not make a smaller copy of the image: {msg}")
            data = out.read_bytes()
            if len(data) <= MAX_IMAGE_BYTES:
                return data, "image/jpeg"
        raise ToolError(f"image is {size:,} bytes and still too large after shrinking it")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _call(cancel, method, *args):
    """A vision / image-model request that Stop ends at once (a stalled endpoint would otherwise hold the run — and a
    shared project — for its read timeouts, ~20 min). A provider whose method takes `cancel` gets it and aborts the
    request itself; the wait is abandoned on Stop in any case (Stopped, a ToolError)."""
    try:
        takes = cancel is not None and "cancel" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        takes = False
    return interruptible(cancel, lambda: method(*args, cancel=cancel) if takes else method(*args), "nesi-vision-call")


def describe_image(ws: Workspace, path: str, vision_model: str, user_keys: dict | None, question: str,
                   cancel=None) -> str:
    """The vision model's answer about an image of the workspace. `cancel`: the run's Stop (default: the workspace's,
    which the agent sets)."""
    cancel = cancel if cancel is not None else getattr(ws, "cancel", None)
    if not vision_model:
        raise ToolError("no vision model is configured for this project")
    p = ws.resolve(path)
    if not p.is_file():
        raise ToolError(f"image not found: {path}")
    mime = image_mime(p.name)
    if not mime:
        raise ToolError("unsupported image type; use png, jpeg, gif or webp")
    size = p.stat().st_size  # before reading: a huge file is refused without loading it
    if size > MAX_SOURCE_BYTES:
        raise ToolError(f"image is {size:,} bytes; too large to view")
    if size > MAX_IMAGE_BYTES:
        data, mime = _shrink(p, size)
    else:
        data = ws.read_bytes(p, MAX_IMAGE_BYTES + 1)  # bounded: the file may grow (or be swapped) after the stat
        if len(data) > MAX_IMAGE_BYTES:
            raise ToolError("the image changed while it was being read; try again")
        mime = sniff_mime(data) or mime
    try:
        provider, model = get_provider(vision_model, user_keys)
        text, _ = _call(cancel, provider.vision, model, question or "Describe this image in detail.", data, mime)
    except LLMError as exc:
        raise ToolError(f"vision model failed: {exc}") from exc
    return text.strip() or "(the vision model returned nothing)"


EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}


def generate_image(ws: Workspace, prompt: str, path: str, image_model: str, user_keys: dict | None,
                   cancel=None) -> str:
    """Generate an image with the image model and save it in the workspace. Returns the relative path. `cancel`: the
    run's Stop (default: the workspace's, which the agent sets)."""
    import re

    cancel = cancel if cancel is not None else getattr(ws, "cancel", None)
    if not image_model:
        raise ToolError("no image-generation model is configured for this project")
    if not prompt.strip():
        raise ToolError("`prompt` is required")
    try:
        provider, model = get_provider(image_model, user_keys)
        data, mime = _call(cancel, provider.generate_image, model, prompt.strip())
    except LLMError as exc:
        raise ToolError(f"image model failed: {exc}") from exc
    ext = EXT.get(mime, "png")
    if path.strip():
        rel = re.sub(r"\.(png|jpe?g|webp|gif)$", "", path.strip(), flags=re.I) + f".{ext}"
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:40].strip("-") or "image"
        rel = f"images/{slug}.{ext}"
    target = ws.resolve(rel)
    n = 1
    while target.exists() and not path.strip():
        n += 1
        target = ws.resolve(rel.rsplit(".", 1)[0] + f"-{n}.{ext}")
    ws.write_bytes(target, data)
    return ws.rel(target)

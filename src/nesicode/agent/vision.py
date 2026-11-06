"""Vision through a separate model, so a text-only executor (e.g. nesi3) can still 'see'."""

from __future__ import annotations

import mimetypes

from ..llm import LLMError, get_provider
from .workspace import ToolError, Workspace

IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def image_mime(path: str) -> str | None:
    mime = mimetypes.guess_type(path)[0]
    return mime if mime in IMAGE_TYPES else None


def describe_image(ws: Workspace, path: str, vision_model: str, user_keys: dict | None, question: str) -> str:
    if not vision_model:
        raise ToolError("no vision model is configured for this project")
    p = ws.resolve(path)
    if not p.is_file():
        raise ToolError(f"image not found: {path}")
    mime = image_mime(p.name)
    if not mime:
        raise ToolError("unsupported image type; use png, jpeg, gif or webp")
    data = p.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ToolError(f"image is {len(data):,} bytes; the limit is {MAX_IMAGE_BYTES:,}")
    try:
        provider, model = get_provider(vision_model, user_keys)
        text, _ = provider.vision(model, question or "Describe this image in detail.", data, mime)
    except LLMError as exc:
        raise ToolError(f"vision model failed: {exc}") from exc
    return text.strip() or "(the vision model returned nothing)"


EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}


def generate_image(ws: Workspace, prompt: str, path: str, image_model: str, user_keys: dict | None) -> str:
    """Generate an image with the image model and save it in the workspace. Returns the relative path."""
    import re

    if not image_model:
        raise ToolError("no image-generation model is configured for this project")
    if not prompt.strip():
        raise ToolError("`prompt` is required")
    try:
        provider, model = get_provider(image_model, user_keys)
        data, mime = provider.generate_image(model, prompt.strip())
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
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return ws.rel(target)

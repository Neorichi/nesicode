"""Packs the PyInstaller folder (dist/nesicode) into a .deb for Ubuntu and Debian.

    python packaging/linux/make_deb.py --version 0.4.9 [--arch amd64] [--bundle dist/nesicode] [--out dist]

Layout: the app in /opt/nesicode, the `nesicode` command in /usr/bin, the menu entry and its icons under /usr/share.
The window is the system's WebKitGTK (see bundle.py), hence the Depends. dpkg's triggers refresh the menu and the icon
cache, so there are no maintainer scripts. Run on Linux: dpkg-deb builds the package (the tree alone elsewhere).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
MAINTAINER = "Nesi Labs <3147619+Neorichi@users.noreply.github.com>"
HOMEPAGE = "https://github.com/Neorichi/nesicode"
# gir1.2-* bring GTK 3, WebKitGTK (4.1 API) and their typelibs; the rest is what the bundled PyGObject links to.
DEPENDS = "gir1.2-gtk-3.0, gir1.2-webkit2-4.1, libgirepository-1.0-1, libcairo-gobject2, libffi8"
# xdg-utils opens links in the browser; the GStreamer ones give WebKitGTK the microphone (the mic button)
RECOMMENDS = "xdg-utils, git, gstreamer1.0-plugins-good, gstreamer1.0-pipewire | gstreamer1.0-pulseaudio"
DESCRIPTION = """Coding agent with a chat, projects, connectors and bots
 Nesicode is a desktop app for working with AI coding agents: a chat that edits
 and runs code in your projects, connectors (MCP servers, Gmail, GitHub…),
 scheduled tasks, Telegram and Slack bots, and a harness that improves itself.
 It starts its own server on this computer and shows it in a native window.
 Models come from the providers you configure (Anthropic, OpenAI-compatible
 endpoints, local models)."""


def deb_arch() -> str:
    try:
        return subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(os.uname().machine, os.uname().machine)


def control(version: str, arch: str, installed_kib: int) -> str:
    return (f"Package: nesicode\nVersion: {version}\nSection: devel\nPriority: optional\nArchitecture: {arch}\n"
            f"Maintainer: {MAINTAINER}\nInstalled-Size: {installed_kib}\nDepends: {DEPENDS}\n"
            f"Recommends: {RECOMMENDS}\nHomepage: {HOMEPAGE}\nDescription: {DESCRIPTION}\n")


def copyright_text() -> str:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8").strip()
    return (f"Nesicode\n{HOMEPAGE}\n\n{notice}\n\n"
            "The full text of the Apache License 2.0 is in /usr/share/doc/nesicode/LICENSE-harness2.\n\n"
            "/opt/nesicode bundles a Python interpreter (Python Software Foundation License) and the Python packages "
            "Nesicode uses, each under its own license (their *.dist-info folders in /opt/nesicode/_internal).\n")


def _normalize_modes(tree: Path) -> None:
    """Folders 755, executables 755, everything else 644 (whoever built it, whatever their umask)."""
    for dirpath, dirnames, filenames in os.walk(tree):
        os.chmod(dirpath, 0o755)
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            mode = p.stat().st_mode
            os.chmod(p, 0o755 if mode & stat.S_IXUSR else 0o644)


def _installed_kib(tree: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(tree):
        for name in filenames:
            p = Path(dirpath) / name
            if not p.is_symlink():
                total += (p.stat().st_size + 1023) // 1024
    return total


def _icons(dest: Path) -> None:
    from PIL import Image

    with Image.open(ROOT / "packaging" / "icons" / "nesicode.png") as src:
        src = src.convert("RGBA")
        for size in ICON_SIZES:
            folder = dest / "usr" / "share" / "icons" / "hicolor" / f"{size}x{size}" / "apps"
            folder.mkdir(parents=True, exist_ok=True)
            src.resize((size, size), Image.LANCZOS).save(folder / "nesicode.png")


def build_tree(bundle: Path, version: str, arch: str, work: Path) -> Path:
    """The package's file tree (with DEBIAN/control and md5sums) under `work`; returns its root."""
    if not (bundle / "nesicode").is_file() or not (bundle / "_internal").is_dir():
        raise SystemExit(f"{bundle} is not the PyInstaller folder (build it first: packaging/linux/build_deb.sh)")
    tree = work / f"nesicode_{version}_{arch}"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(bundle, tree / "opt" / "nesicode", symlinks=True)
    (tree / "usr" / "bin").mkdir(parents=True)
    os.symlink("/opt/nesicode/nesicode", tree / "usr" / "bin" / "nesicode")
    apps = tree / "usr" / "share" / "applications"
    apps.mkdir(parents=True)
    shutil.copyfile(HERE / "nesicode.desktop", apps / "nesicode.desktop")
    _icons(tree)
    doc = tree / "usr" / "share" / "doc" / "nesicode"
    doc.mkdir(parents=True)
    (doc / "copyright").write_text(copyright_text(), encoding="utf-8")
    shutil.copyfile(ROOT / "LICENSE-harness2", doc / "LICENSE-harness2")
    _normalize_modes(tree)
    sums = []
    for dirpath, dirnames, filenames in os.walk(tree):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if not p.is_symlink():
                sums.append(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p.relative_to(tree).as_posix()}\n")
    meta = tree / "DEBIAN"
    meta.mkdir()
    (meta / "control").write_text(control(version, arch, _installed_kib(tree)), encoding="utf-8")
    (meta / "md5sums").write_text("".join(sums), encoding="utf-8")
    os.chmod(meta, 0o755)
    for f in meta.iterdir():
        os.chmod(f, 0o644)
    return tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--arch", default=None, help="Debian architecture (default: this machine's)")
    ap.add_argument("--bundle", default=str(ROOT / "dist" / "nesicode"))
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--work", default=str(ROOT / "build" / "deb"))
    args = ap.parse_args()
    arch = args.arch or deb_arch()
    tree = build_tree(Path(args.bundle), args.version, arch, Path(args.work))
    out = Path(args.out) / f"Nesicode-{args.version}-linux-{arch}.deb"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("dpkg-deb"):
        sys.exit(f"tree ready in {tree}; dpkg-deb (Debian/Ubuntu) builds the package")
    # xz: every dpkg reads it (Ubuntu's default, zstd, needs dpkg 1.21.18+)
    subprocess.run(["dpkg-deb", "--root-owner-group", "-Zxz", "--build", str(tree), str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()

"""Folders of the user's that the agent may READ (never change): a project setting, off (empty) by default.

The OS sandbox keeps the user's home, other disks and every other folder out of a caged command's reach. Here the user
names a few folders it may read anyway (the APK their browser downloaded to ~/Downloads…). What may be named is decided
here, on the server, on the path as the OS spells it (links followed, and on macOS the real case of every name, without
the /System/Volumes/Data alias), which is also the spelling the sandbox matches: never the disk root, the home or a
folder that holds it, another user's home, Nesicode's own folders (its data, the desktop app's folder, a source
checkout with its .env) or a folder that holds them, a place with keys or credentials (agent/workspace.py,
SECRET_PLACES) or a folder inside one, a folder that only holds apps' private data (~/Library, ~/.config…: name a
folder inside it), the temporary folders every program shares (/tmp…: other programs' files and sockets), or the
system. Only in local mode: on a shared server the folders would be the server's.

Each run gets the folders again through usable_read_dirs, checked again: one that went away, changed into a link or
stopped passing is left out of that run. Taking folders off the list never checks the ones kept (check_read_dirs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..agent.workspace import REPO_DIR, SECRET_PLACES, TOOLCHAIN_SECRETS, path_key, spellings, vault_key_file
from ..config import app_data_dir, settings
from .db import loads

MAX_READ_DIRS = 10
MAX_PATH = 1000
# Folders that only hold other apps' private data: one inside them can be named, not they themselves.
APP_DATA_ROOTS = ("Library", "Library/Application Support", "Library/Containers", "Library/Group Containers",
                  ".config", ".local", ".local/share", "AppData", "AppData/Local", "AppData/Roaming")
# The OS itself, and places that show the whole disk again under another name (/System/Volumes/Data/Users/…,
# /proc/<pid>/root/…).
SYSTEM_ROOTS = ("/System", "/dev", "/proc", "/sys") if os.name != "nt" else ()
ROOT_HOMES = ("/var/root", "/private/var/root", "/root") if os.name != "nt" else ()
SHARED_HOMES = {"shared", "public"}  # /Users/Shared (macOS), C:\Users\Public (Windows): nobody's home
# Places every program shares (temporary files, sockets, the local databases Homebrew/MacPorts run): never one of them,
# a folder inside one or a folder that holds one. Other programs' files are there (the sandbox keeps them out for that:
# workspace.SHARED_DIRS), and in docker mode a folder is mounted, and a read-only mount still lets a socket in it be
# connected to (tmux's, an ssh-agent's, a rootless Docker's: commands run as the user outside the container).
SHARED_ROOTS = ("/tmp", "/var/tmp", "/run", "/var/run", "/private/tmp", "/private/var/tmp", "/private/var/run",
                "/private/var/folders", "/var/folders", "/opt/homebrew/var", "/usr/local/var", "/opt/local/var",
                ) if os.name != "nt" else ()
REMOVABLE_ROOTS = ("/run/media",)  # (where some Linux desktops mount USB disks: not a shared place)


class ReadDirError(ValueError):
    """Why a folder cannot be one the agent reads (Spanish: it is shown to the user as the API's detail)."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _getpath(p: Path) -> Path:
    """macOS: the folder's path as the kernel names it (F_GETPATH): the real case of every name and no firmlink alias
    (/System/Volumes/Data/private/tmp is /private/tmp), as the sandbox sees it. Elsewhere, or when the folder cannot be
    opened (macOS privacy settings may keep the server out of ~/Downloads until the user allows it), `p` itself."""
    if sys.platform != "darwin":
        return p
    try:
        import fcntl

        fd = os.open(p, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            got = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0]
        finally:
            os.close(fd)
        return Path(os.fsdecode(got)) if got.startswith(b"/") else p
    except (OSError, AttributeError, ValueError):
        return p


def canonical(p: Path | str) -> Path:
    """`p` (absolute) with its links resolved and spelled as the OS spells it (see _getpath)."""
    real = Path(os.path.realpath(p))
    return _getpath(real) if real.is_dir() else real


# Paths are compared as the disk compares names: on macOS and Windows blind to case (and on macOS to Unicode
# normalisation), so a folder spelled two ways is one folder.
_key = path_key


def _inside(child: Path | str, parent: Path | str) -> bool:
    """child is parent or a path under it."""
    c, p = _key(child), _key(parent)
    return len(c) >= len(p) and c[: len(p)] == p


# A place as written, where it really is (a link to a dotfiles folder…) and without macOS's data volume alias.
_spellings = spellings


def _nesicode_paths() -> list[Path]:
    """Nesicode's own folders: its data (every user's projects and conversations), the desktop app's folder (its data
    and .env) and, run from source, the checkout (its .env holds keys)."""
    paths = [settings.data_dir, app_data_dir()] + ([REPO_DIR] if (REPO_DIR / "pyproject.toml").is_file() else [])
    return [q for p in paths for q in _spellings(p)]


def _secret_paths(home: Path) -> list[Path]:
    places = [home / s for s in (*SECRET_PLACES, *TOOLCHAIN_SECRETS)] + [vault_key_file()]
    return [q for p in places for q in _spellings(p)]


def _vetoed_paths() -> list[Path]:
    return [q for p in os.environ.get("NESICODE_SANDBOX_DENY", "").split(":") if p for q in _spellings(Path(p))]


def check_read_dir(raw: object) -> str:
    """The folder `raw` names, canonical, if the agent may be let read it; ReadDirError (Spanish) if not."""
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_PATH or any(ord(c) < 32 for c in raw):
        shown = "".join(c if c.isprintable() else "?" for c in str(raw)[:200])
        raise ReadDirError(f"Ruta no válida: {shown}")
    text = raw.strip()
    p = Path(os.path.expanduser(text))
    if not p.is_absolute():
        raise ReadDirError(f"Escribe la ruta completa de la carpeta (p. ej. {Path.home() / 'Downloads'}): {text}")
    try:
        real = canonical(p)
        is_dir = real.is_dir()
    except (OSError, ValueError):
        is_dir = False
    if not is_dir:
        raise ReadDirError(f"La carpeta no existe: {text}")
    home = canonical(Path(os.path.expanduser("~")))
    if _key(real) == _key(real.anchor):
        raise ReadDirError(f"Elige una carpeta concreta, no la raíz del disco: {real}")
    if _inside(home, real):
        raise ReadDirError(f"Elige una carpeta concreta (p. ej. Descargas), no tu carpeta personal ni una que la contenga: {real}")
    if any(_key(real) == _key(home / a) for a in APP_DATA_ROOTS):  # (~/Library also holds the desktop app's folder)
        raise ReadDirError(f"Esa carpeta solo guarda datos privados de tus apps; elige una carpeta dentro de ella: {real}")
    if any(_inside(real, n) or _inside(n, real) for n in _nesicode_paths()):
        raise ReadDirError(f"Esa carpeta es de Nesicode (sus datos, su código o su configuración) o los contiene: {real}")
    homes = home.parent
    if _key(homes) != _key(homes.anchor) and _inside(real, homes) and not _inside(real, home):
        if _key(real)[len(_key(homes))] not in SHARED_HOMES:
            raise ReadDirError(f"Esa carpeta es de otro usuario: {real}")
    if any(_inside(real, r) for r in ROOT_HOMES):
        raise ReadDirError(f"Esa carpeta es de otro usuario: {real}")
    if any(_inside(real, s) for s in _secret_paths(home)):
        raise ReadDirError(f"Esa carpeta guarda claves o credenciales; el agente nunca puede leerla: {real}")
    if any(_inside(real, s) for s in _vetoed_paths()):
        raise ReadDirError(f"Esa carpeta está vetada para el agente (NESICODE_SANDBOX_DENY): {real}")
    if any(_inside(real, s) for s in SYSTEM_ROOTS):
        raise ReadDirError(f"Esa carpeta es del sistema: {real}")
    if any(_inside(s, real) or (_inside(real, s) and not any(_inside(real, m) for m in REMOVABLE_ROOTS))
           for s in SHARED_ROOTS):
        raise ReadDirError(f"Esa carpeta la comparten todos los programas (temporales, sockets); elige una tuya "
                           f"(p. ej. Descargas): {real}")
    try:  # what the server cannot read, no command of the agent can (the same user, fewer rights). On macOS this is
        # also when the system asks the user to let Nesicode into ~/Downloads, ~/Documents… (privacy settings).
        with os.scandir(real) as entries:
            next(entries, None)
    except PermissionError:
        raise ReadDirError("Nesicode no tiene permiso para leer esa carpeta (en el Mac: Ajustes del Sistema → "
                           f"Privacidad y seguridad → Archivos y carpetas): {real}") from None
    except OSError:
        pass
    return str(real)


def check_read_dirs(items: object, kept=()) -> list[str]:
    """The list as stored: every folder added checked and canonical, each once, at most MAX_READ_DIRS. A folder already
    on the list (`kept`: the stored one, sent back as it is) stays as it is, unchecked: taking folders off the list must
    never fail because another one does not pass now (an unplugged disk, a folder macOS no longer lets Nesicode into),
    and each run checks them all again anyway (usable_read_dirs). Adding is for local mode only."""
    if not isinstance(items, list):
        raise ReadDirError("Lista de carpetas no válida")
    kept = [d for d in kept if isinstance(d, str)]
    if settings.auth_enabled and any(raw not in kept for raw in items):
        raise ReadDirError("Las carpetas que el agente puede leer solo se pueden elegir en modo local (sin login)", 403)
    out: list[str] = []
    for raw in items:
        d = raw if raw in kept else check_read_dir(raw)
        if not any(_key(d) == _key(x) for x in out):
            out.append(d)
    if len(out) > MAX_READ_DIRS:
        raise ReadDirError(f"Como mucho {MAX_READ_DIRS} carpetas")
    return out


def stored_read_dirs(project: dict) -> list[str]:
    value = loads(project.get("read_dirs"), [])
    return [d for d in value if isinstance(d, str)] if isinstance(value, list) else []


def usable_read_dirs(project: dict) -> list[str]:
    """The folders a chat or bot run of `project` may read: the stored ones that still pass, as stored. One that went
    away (an unplugged disk), now resolves elsewhere (replaced by a link) or no longer passes is left out of the run;
    on a shared server (login on) none."""
    if settings.auth_enabled:
        return []
    out: list[str] = []
    for d in stored_read_dirs(project)[:MAX_READ_DIRS]:
        try:
            now = check_read_dir(d)
        except ReadDirError as exc:
            print(f"nesicode: project {project.get('id')}: the agent may not read {d} in this run: {exc}")
            continue
        if _key(now) != _key(d):  # (the same folder may come back better spelled: its real case once macOS lets us see it)
            print(f"nesicode: project {project.get('id')}: the agent may not read {d} in this run: it leads to {now} now")
            continue
        if now not in out:
            out.append(now)
    return out

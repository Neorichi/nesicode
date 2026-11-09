"""A confined project workspace: path resolution, command execution, copies and diffs.

The server process is not sandboxed, but the agent can create symbolic links in its cage that point
anywhere. So everything the server does inside a workspace (reads, writes, walks, diffs, applies)
goes through helpers that never follow a link: on POSIX every path component is opened relative to
its parent with O_NOFOLLOW, so even a link swapped in by a background process is refused."""

from __future__ import annotations

import contextvars
import difflib
import errno
import hashlib
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path

from ..config import settings
from ..harness import Harness

# Not listed, not diffed, not copied back.
IGNORE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".nesicode", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".next", ".turbo", ".cache",
    ".attachments",  # files the user attached to a conversation: readable, never diffed or applied
}
ATTACHMENTS_DIR = ".attachments"
MAX_DIFF_FILE = 1_000_000
MAX_HASH = 50_000_000  # bigger files are tracked by size and mtime instead of a hash
HARNESS_DIR = ".nesicode"
# Manifests record links and big files since v2; older ones did not (see changes_since). "" is never a path.
MANIFEST_KEY, MANIFEST_VERSION = "", "2"
OUTPUT_KEEP = 64_000  # bytes of a command's output kept from its start and from its end
OUTPUT_CAP = 50_000_000  # a command printing more than this is stopped (a runaway loop would exhaust memory)
EOF_GRACE = 2.0  # seconds to collect output after the command exits while children it left still hold the pipe


class ToolError(Exception):
    """A tool failed in a way the model should see and recover from."""


class Stopped(ToolError):
    """The user pressed Stop while a call that cannot be interrupted was in flight (it is abandoned, not waited for)."""


def interruptible(cancel: threading.Event | None, fn, name: str = "nesi-call"):
    """Run a blocking call (an MCP request, a Chrome command, a vision model…) in a helper thread and wait for it in
    short slices, so Stop returns at once instead of when the call ends (up to minutes): Stopped. The abandoned call
    finishes, or times out, in the background and its result is dropped. No `cancel`: a plain call. The call sees the
    caller's context variables (a new thread starts without them): the model layer's request scope — the Stop and
    the retry notices the agent gives a tool's vision or image request — must reach it as if called directly."""
    if cancel is None:
        return fn()
    if cancel.is_set():
        raise Stopped("stopped by the user")
    box: dict = {}
    done = threading.Event()
    context = contextvars.copy_context()

    def target() -> None:
        try:
            box["value"] = context.run(fn)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True, name=name).start()
    while not done.wait(0.2):
        if cancel.is_set():
            raise Stopped("stopped by the user")
    if "error" in box:
        if cancel.is_set():  # a call that honours Stop itself ended because of it: the same answer either way
            raise Stopped("stopped by the user") from box["error"]
        raise box["error"]
    return box["value"]


REPO_DIR = Path(__file__).resolve().parents[3]
# Toolchains under $HOME that caged commands may READ (never write): only the parts that hold programs,
# so other apps' data under ~/.local/share stays out of reach.
TOOLCHAIN_DIRS = (
    ".local/bin", ".local/lib", ".local/include", ".local/opt", ".local/pipx", ".local/share/uv", ".local/share/mise",
    ".local/state/mise", ".local/share/pipx", ".local/share/pnpm", ".local/share/fnm", ".local/share/virtualenvs",
    ".local/pyvenv.cfg",  # Python started as ~/.local/bin/python3 looks for it next to bin/ (unreadable: crash)
    ".nvm", ".cargo", ".rustup", ".bun", ".pyenv", ".deno", "go", ".sdkman", ".volta",
)
TOOLCHAIN_SECRETS = (".cargo/credentials", ".cargo/credentials.toml")  # registry tokens inside a toolchain dir
# Shared places a caged command may neither read nor write: other disks (other projects), other users' homes
# and /Users/Shared, the shared temp dirs (other programs' files and sockets, other conversations' mktemp
# files), the per-user temp and cache dirs, and the data of Homebrew/MacPorts services (the user's local
# MySQL/PostgreSQL/MongoDB files and their logs; no toolchain reads them).
SHARED_DIRS = ("/Volumes", "/Users", "/private/tmp", "/private/var/tmp", "/private/var/folders",
               "/opt/homebrew/var", "/usr/local/var", "/opt/local/var")


# Variables Windows programs need to start at all (PATH alone is not enough there).
WINDOWS_ENV = {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "PROGRAMDATA", "PROGRAMFILES",
               "PROGRAMFILES(X86)", "LOCALAPPDATA", "APPDATA", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"}


def windows_bash() -> str | None:
    """Git for Windows' bash, if installed (gives the agent the same shell as on macOS/Linux)."""
    for c in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
        if c and Path(c).exists() and "system32" not in c.lower():  # skip the WSL launcher
            return c
    return None


def shell_name() -> str:
    if os.name == "nt" and not windows_bash():
        return "powershell"
    return "bash"


# A login shell runs /etc/profile: macOS path_helper moves /usr/bin & co. in front of the user's PATH (so
# `python3` became Apple's 3.9 instead of the user's toolchain) and Debian's resets PATH. Put it back first.
RESTORE_PATH = 'if [ -n "${NESICODE_PATH:-}" ]; then PATH="$NESICODE_PATH${PATH:+:$PATH}"; export PATH; fi; unset NESICODE_PATH; '


def powershell_argv(command: str) -> list[str]:
    # Redirected PowerShell 5.1 output uses the OEM code page: make it UTF-8 like everything else we decode.
    utf8 = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $OutputEncoding=[Console]::OutputEncoding; "
    return ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", utf8 + command]


def shell_argv(command: str) -> list[str]:
    if os.name == "nt":
        bash = windows_bash()
        if bash:
            return [bash, "-lc", command]
        return powershell_argv(command)
    return ["bash", "-lc", RESTORE_PATH + command]


def sandbox_mode() -> str:
    """docker | macos | none. `auto`/`local` pick the OS sandbox when one is available."""
    mode = settings.sandbox
    if mode in ("docker", "none"):
        return mode
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "macos"
    return "none"


def _q(p: Path | str) -> str:
    return '"' + str(p).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sandbox_paths() -> tuple[list[Path], list[Path], list[Path]]:
    """(denied, then readable toolchains, then denied again) — the order the profile applies them in."""
    real_home = Path(os.path.expanduser("~")).resolve()
    first = [*(Path(d) for d in SHARED_DIRS), real_home, settings.data_dir.resolve()]
    tools = [real_home / d for d in TOOLCHAIN_DIRS]
    # A source checkout of nesicode (its .env holds keys); an installed package's parents[3] is just a lib dir.
    repo = [REPO_DIR] if (REPO_DIR / "pyproject.toml").is_file() else []
    after = [*repo, *(real_home / s for s in TOOLCHAIN_SECRETS)] + [
        Path(p).resolve() for p in os.environ.get("NESICODE_SANDBOX_DENY", "").split(":") if p
    ]
    return first, tools, after


def macos_profile(cage: Path, home: Path, deny_ports=()) -> str:
    """Seatbelt profile: the command may read the OS and the toolchains, and read or write only its own
    cage and private home. Off limits: the user's home, the data dir (other users, projects and
    conversations), the nesicode repo (.env), other disks and users, the shared and per-user temp dirs,
    other programs' unix sockets (tmux, ssh-agent, docker), the clipboard, launching apps or sending
    Apple Events (both run code outside the sandbox), signals to other processes and nesicode's own API
    port (`deny_ports`). Seatbelt cannot stop sysctl(KERN_PROCARGS2) from reading another same-user
    process's launch environment, so secrets must never be in one (caged commands get a minimal env).
    In SBPL the LAST matching rule wins, so denies come first and the cage allow last."""
    cage, home = cage.resolve(), home.resolve()
    first, tools, after = _sandbox_paths()
    sub = lambda paths: " ".join(f"(subpath {_q(p)})" for p in paths)  # noqa: E731
    return "\n".join([
        "(version 1)",
        "(allow default)",
        f"(deny file-read* file-write* {sub(first)})",
        '(deny file-write* (subpath "/"))',
        f"(allow file-read* {sub(tools)})",
        f"(deny file-read* file-write* {sub(after)})",
        "(allow file-read-metadata)",
        # Apple's xcrun shims (/usr/bin/python3, git…) read their lookup cache; writing it is denied (the user's
        # own shell trusts it). mktemp, which ignores TMPDIR, goes through a shim (see _cage_bin).
        '(allow file-read* (regex #"^/private/var/folders/[^/]+/[^/]+/T/xcrun_db$"))',
        f"(allow file-read* file-write* (subpath {_q(cage)}) (subpath {_q(home)}))",
        '(allow file-read* file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") '
        '(literal "/dev/random") (literal "/dev/urandom") (literal "/dev/dtracehelper") (regex #"^/dev/fd/"))',
        "(deny signal)",
        "(allow signal (target same-sandbox))",
        "(deny network-outbound (remote unix-socket))",
        f"(allow network-outbound (remote unix-socket (subpath {_q(cage)}) (subpath {_q(home)}) "
        '(literal "/private/var/run/mDNSResponder") (literal "/private/var/run/syslog")))',
        '(deny mach-lookup (global-name "com.apple.pasteboard.1"))',
        "(deny lsopen)",
        "(deny appleevent-send)",
        *[f'(deny network-outbound (remote ip "*:{int(p)}"))' for p in sorted(set(deny_ports))],
    ])


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _cage_path(path: str, cage: Path, home: Path) -> str:
    """The server's PATH minus directories the sandbox denies (a denied interpreter, e.g. the repo's .venv
    under `uv run`, would crash instead of being 'not found'), with the cage's shims first."""
    first, tools, after = _sandbox_paths()
    allowed = [str(cage.resolve()), str(home.resolve())]
    keep = [str(_cage_bin(home))]
    for entry in path.split(os.pathsep):
        if not entry:
            continue
        real = os.path.realpath(entry)
        if any(_within(real, a) for a in allowed):
            keep.append(entry)
        elif any(_within(real, str(d)) for d in after):
            continue
        elif any(_within(real, str(t)) for t in tools) or not any(_within(real, str(d)) for d in first):
            keep.append(entry)
    return os.pathsep.join(dict.fromkeys(keep))


# macOS mktemp ignores $TMPDIR and uses the shared per-user temp dir, which the sandbox denies (it let one
# conversation list and rewrite another's temp scripts). Without an explicit template, use our $TMPDIR.
MKTEMP_SHIM = r"""#!/bin/bash
# nesicode: mktemp ignores $TMPDIR on macOS; keep temporary files in this conversation's private $TMPDIR.
args=("$@"); opts=(); prefix=tmp
while [ $# -gt 0 ]; do
  case "$1" in
    -t) prefix=${2:-tmp}; shift; [ $# -gt 0 ] && shift ;;
    -t?*) prefix=${1#-t}; shift ;;
    -*t) opts+=("${1%t}"); prefix=${2:-tmp}; shift; [ $# -gt 0 ] && shift ;;
    -*) opts+=("$1"); shift ;;
    *) exec /usr/bin/mktemp "${args[@]}" ;;
  esac
done
exec /usr/bin/mktemp "${opts[@]}" "${TMPDIR:-/tmp}/$prefix.XXXXXXXXXX"
"""


def _cage_bin(home: Path) -> Path:
    return home / ".nesicode-bin"


def _install_shims(home: Path) -> None:
    shim = MKTEMP_SHIM.encode()
    rel = f"{_cage_bin(home).name}/mktemp"
    try:
        try:
            if _read_beneath(home, rel, len(shim) + 1) == shim:
                return
        except OSError:
            pass
        if _FD_WALK:  # the private home is writable from the cage: never write through a link planted there
            dfd = _parent_fd(home, _parts(home, rel), mkdirs=True)
            try:
                _clear_at(dfd, "mktemp")
            finally:
                os.close(dfd)
        with os.fdopen(_open_beneath(home, rel, write=True, mkdirs=True), "wb") as f:
            f.truncate(0)
            f.write(shim)
            os.chmod(f.fileno() if os.chmod in os.supports_fd else Path(home, rel), 0o755)
    except OSError:
        pass  # mktemp without a template then fails inside the sandbox; everything else works


def _listening_ports() -> set[int]:
    """TCP ports this process listens on (nesicode's own API): caged commands must not reach them, or they
    could drive the app (approve their own commands, read other projects) from inside the sandbox."""
    ports: set[int] = set()
    try:
        fds = [int(x) for x in os.listdir("/dev/fd" if sys.platform == "darwin" else "/proc/self/fd") if x.isdigit()]
    except OSError:
        return ports
    for fd in fds:
        try:
            if not stat.S_ISSOCK(os.fstat(fd).st_mode):
                continue
            dup = os.dup(fd)
        except OSError:
            continue
        try:
            s = socket.socket(fileno=dup)
        except OSError:
            os.close(dup)
            continue
        try:
            if s.family in (socket.AF_INET, socket.AF_INET6) and s.type == socket.SOCK_STREAM and _listening(s):
                ports.add(int(s.getsockname()[1]))
        except OSError:
            pass
        finally:
            s.close()
    return ports


def _listening(s: socket.socket) -> bool:
    try:
        return bool(s.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN))
    except (OSError, AttributeError):  # macOS has no SO_ACCEPTCONN: a bound TCP socket without a peer
        pass
    try:
        s.getpeername()
        return False
    except OSError:
        return s.getsockname()[1] != 0


# ---------------------------------------------------------------- files, never through a symlink
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # opening a FIFO must never block the server
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_BINARY = getattr(os, "O_BINARY", 0)
_FD_WALK = bool(_NOFOLLOW and _DIRECTORY) and all(
    f in os.supports_dir_fd for f in (os.open, os.mkdir, os.unlink, os.rmdir, os.symlink, os.stat)
)


def _parts(root: Path, p: Path | str) -> tuple[str, ...]:
    rel = Path(p)
    if rel.is_absolute():
        rel = rel.relative_to(root)
    parts = rel.parts
    if not parts or any(x in ("", ".", "..") for x in parts):
        raise OSError(errno.EINVAL, "not a path inside the workspace", str(p))
    return parts


def _parent_fd(root: Path, parts: tuple[str, ...], mkdirs: bool = False) -> int:
    """fd of root/<parts[:-1]>, every component opened with O_NOFOLLOW (POSIX only)."""
    dfd = os.open(root, os.O_RDONLY | _DIRECTORY | _CLOEXEC)
    try:
        for name in parts[:-1]:
            if mkdirs:
                try:
                    os.mkdir(name, 0o777, dir_fd=dfd)
                except FileExistsError:
                    pass
            nfd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)
            os.close(dfd)
            dfd = nfd
        return dfd
    except BaseException:
        os.close(dfd)
        raise


def _no_links(root: Path, parts: tuple[str, ...]) -> Path:
    """Windows fallback (no dir fds): refuse a path whose components include a link."""
    cur = Path(root)
    for name in parts:
        cur = cur / name
        if os.path.islink(cur):
            raise OSError(errno.ELOOP, "symbolic link", str(cur))
    return cur


def _open_beneath(root: Path, p: Path | str, write: bool = False, mkdirs: bool = False) -> int:
    """fd of the regular file root/p, opened without following any symlink; never blocks on a FIFO.
    Writing creates the file (and its directories) but does not truncate it."""
    parts = _parts(root, p)
    flags = (os.O_WRONLY | os.O_CREAT) if write else os.O_RDONLY
    if _FD_WALK:
        dfd = _parent_fd(root, parts, mkdirs)
        try:
            fd = os.open(parts[-1], flags | _NOFOLLOW | _NONBLOCK | _CLOEXEC, 0o666, dir_fd=dfd)
        finally:
            os.close(dfd)
    else:
        target = _no_links(root, parts)
        if mkdirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, flags | _BINARY, 0o666)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise OSError(errno.EINVAL, "not a regular file", str(p))
    return fd


def _read_beneath(root: Path, p: Path | str, limit: int | None = None) -> bytes:
    with os.fdopen(_open_beneath(root, p), "rb") as f:
        return f.read() if limit is None else f.read(limit)


def _why(exc: OSError) -> str:
    if exc.errno in (errno.ELOOP, errno.EMLINK):
        return "it is a symbolic link (links are not followed; use the path it points to)"
    if exc.errno == errno.ENOTDIR:
        return "a parent folder is a symbolic link or a file"
    if exc.errno == errno.ENOENT:
        return "not found"
    if exc.errno == errno.EINVAL:
        return "not a regular file"
    return exc.strerror or str(exc)


def _remove_at(dfd: int, name: str) -> None:
    """Delete the file, link or directory tree `name` inside dir fd `dfd` without following links."""
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(name, ignore_errors=True, dir_fd=dfd)
    else:
        os.unlink(name, dir_fd=dfd)


def _real_dir_at(dfd: int, name: str) -> int:
    """fd of the real directory `name` inside `dfd`, replacing a link or file that sits there."""
    for attempt in (0, 1):
        try:
            return os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if attempt or exc.errno not in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
                raise
            os.unlink(name, dir_fd=dfd)  # a link (or a file) where our directory belongs
        try:
            os.mkdir(name, 0o777, dir_fd=dfd)
        except FileExistsError:
            pass
    return os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)


def store_new_file(root: Path, folder: str, name: str, data: bytes) -> str:
    """Save `data` as a NEW file in root/<folder> (made if missing) named `name`, or name-2, name-3… when that is taken;
    returns the name used. Nothing is followed: a link at the folder (the agent can plant one, `ln -s ~/x .attachments`)
    is replaced by a real folder, never written through, and every name is created with O_EXCL, which refuses an
    existing entry — a dangling link too (writing through it would create a file wherever it points). A file where the
    folder should be is not touched: OSError (ToolError for a path out of root, or a `name` that is not a plain file
    name: 'sub/x' would be opened through `sub` — a link maybe)."""
    if name in ("", ".", "..") or any(s and s in name for s in ("/", os.sep, os.altsep)):
        raise ToolError(f"{name!r} is not a file name")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    names = (name if n == 1 else f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}" for n in range(1, 100_000))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _CLOEXEC | _BINARY
    try:
        parts = _parts(root, folder)
    except (OSError, ValueError):
        raise ToolError(f"{folder!r} is not a folder inside the workspace") from None

    def _real_folder(dfd: int, part: str) -> int:
        try:  # a link there goes (only the link); a file there stays and makes the open below fail
            if stat.S_ISLNK(os.stat(part, dir_fd=dfd, follow_symlinks=False).st_mode):
                os.unlink(part, dir_fd=dfd)
        except FileNotFoundError:
            pass
        try:
            os.mkdir(part, 0o777, dir_fd=dfd)
        except FileExistsError:
            pass
        return os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)

    if _FD_WALK:
        dfd = os.open(root, os.O_RDONLY | _DIRECTORY | _CLOEXEC)
        try:
            for part in parts:
                nfd = _real_folder(dfd, part)
                os.close(dfd)
                dfd = nfd
            for candidate in names:
                try:
                    fd = os.open(candidate, flags | _NOFOLLOW, 0o666, dir_fd=dfd)
                    break
                except FileExistsError:
                    continue
            else:
                raise OSError(errno.EEXIST, "no free name", name)
        finally:
            os.close(dfd)
    else:  # Windows (links need admin rights there): check each folder instead
        d = Path(root)
        for part in parts:
            d = d / part
            if d.is_symlink():
                d.unlink()
            d.mkdir(exist_ok=True)
            if d.is_symlink() or not d.is_dir():
                raise OSError(errno.ENOTDIR, "not a folder", str(d))
        for candidate in names:
            try:
                fd = os.open(d / candidate, flags)
                break
            except FileExistsError:
                continue
        else:
            raise OSError(errno.EEXIST, "no free name", name)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return candidate


# ---------------------------------------------------------------- text files
_UTF8_BOM = b"\xef\xbb\xbf"
_SURROGATES = re.compile("[\udc80-\udcff]")


def text_codec(data: bytes) -> tuple[bytes, str, str]:
    """(bom, encoding, errors) that decode `data` and encode it back byte for byte. UTF-8 first; a mostly
    UTF-8 file keeps its stray bytes (surrogateescape); otherwise a legacy single-byte file (cp1252, the
    usual Windows/Excel encoding here, or latin-1). Raises ValueError for binary data."""
    if data.startswith(_UTF8_BOM):
        bom, body, enc = _UTF8_BOM, data[3:], "utf-8"
    elif data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16-le" if data[:2] == b"\xff\xfe" else "utf-16-be"
        try:
            data[2:].decode(enc)
        except UnicodeDecodeError:
            raise ValueError("binary file") from None
        return data[:2], enc, "strict"
    else:
        bom, body, enc = b"", data, ""
    if b"\0" in body[:8192]:
        raise ValueError("binary file")
    try:
        body.decode("utf-8")
        return bom, "utf-8", "strict"
    except UnicodeDecodeError:
        pass
    loose = body.decode("utf-8", "surrogateescape")
    bad = len(_SURROGATES.findall(loose))
    good = sum(1 for ch in loose if ord(ch) > 0x7F) - bad
    if enc or good > bad:
        return bom, "utf-8", "surrogateescape"
    try:
        body.decode("cp1252")
        return b"", "cp1252", "strict"
    except UnicodeDecodeError:
        return b"", "latin-1", "strict"


def decode_text(data: bytes) -> tuple[str, tuple[bytes, str, str]]:
    """(text, codec) of file bytes; raises ValueError for binary data. Line endings are kept as they are."""
    codec = text_codec(data)
    bom, enc, errors = codec
    return data[len(bom):].decode(enc, errors), codec


def encode_text(text: str, codec: tuple[bytes, str, str]) -> bytes:
    """Inverse of decode_text; raises UnicodeEncodeError when `text` has characters the file's encoding lacks."""
    bom, enc, errors = codec
    return bom + text.encode(enc, errors)


def shown(text: str) -> str:
    """Text safe to show (stray bytes of a mostly-UTF-8 file appear as U+FFFD)."""
    return _SURROGATES.sub("\ufffd", text)


def codec_note(codec: tuple[bytes, str, str]) -> str:
    bom, enc, errors = codec
    if errors == "surrogateescape":
        return "some bytes are not valid UTF-8 (shown as \ufffd); edits keep them unchanged"
    if enc != "utf-8":
        return f"encoding: {enc}" + (" with BOM" if bom and not enc.startswith("utf-16") else "")
    return "UTF-8 with BOM" if bom else ""


# ---------------------------------------------------------------- walking
def _scan(root: Path):
    """(path, DirEntry) of every non-directory entry under root, sorted per directory, skipping IGNORE_DIRS.
    Symbolic links (to files or directories) are yielded as entries, never followed."""
    stack = [Path(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        subdirs = []
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in IGNORE_DIRS:
                        subdirs.append(Path(e.path))
                    continue
                if e.name in IGNORE_DIRS and e.is_symlink():
                    continue
            except OSError:
                continue
            yield Path(e.path), e
        stack.extend(reversed(subdirs))


def _walk(root: Path):
    """Regular files under root: never symlinks (they may point out of the cage), FIFOs, sockets or devices."""
    for p, e in _scan(root):
        try:
            if e.is_file(follow_symlinks=False):
                yield p
        except OSError:
            continue


def _lexists_exact(root: Path, rel: str, cache: dict | None = None) -> bool:
    """root/rel exists spelled exactly so (on macOS/Windows a case-only rename leaves the old spelling resolvable),
    reached without a symlink: a folder the agent replaced by a link (maybe to a dir outside) is never listed."""
    cache = {} if cache is None else cache
    root = Path(root)
    cur = root
    for name in Path(rel).parts:
        if cur not in cache:
            try:
                cache[cur] = set() if cur != root and os.path.islink(cur) else set(os.listdir(cur))
            except OSError:
                cache[cur] = set()
        if name not in cache[cur]:
            return False
        cur = cur / name
    return True


# ---------------------------------------------------------------- manifests and diffs
def _file_sig(root: Path, rel: str, st: os.stat_result) -> str:
    if st.st_size > MAX_HASH:
        return f"large:{st.st_size}:{st.st_mtime_ns}"
    h = hashlib.sha256()
    with os.fdopen(_open_beneath(root, rel), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path) -> dict[str, str]:
    """{relative path: sha256 | 'large:<size>:<mtime>' | 'symlink:<target>'} of every diffable entry, plus a
    version key; used to detect a cage's own changes. Links and big files are recorded too, so they are
    never mistaken for deletions."""
    root = Path(root)
    out = {MANIFEST_KEY: MANIFEST_VERSION}
    for p, e in _scan(root):
        rel = str(p.relative_to(root))
        try:
            if e.is_symlink():
                out[rel] = "symlink:" + os.readlink(p)
            elif e.is_file(follow_symlinks=False):
                out[rel] = _file_sig(root, rel, e.stat(follow_symlinks=False))
        except OSError:
            continue
    return out


def _is_hash(v: str) -> bool:
    return len(v) == 64 and ":" not in v


def _changes(base: dict[str, str], root: Path) -> tuple[list[str], list[str], list[str]]:
    legacy = base.get(MANIFEST_KEY) != MANIFEST_VERSION
    base = {k: v for k, v in base.items() if k != MANIFEST_KEY}
    now = manifest(root)
    now.pop(MANIFEST_KEY, None)
    if legacy:  # manifests from before v2 never recorded links or big files: unknown, so not 'added'
        now = {k: v for k, v in now.items() if k in base or _is_hash(v)}
    cache: dict = {}
    added = sorted(p for p in now if p not in base)
    modified = sorted(p for p in now if p in base and now[p] != base[p])
    # Gone from the manifest is not enough (unreadable, special file...): it must be gone from the cage.
    removed = sorted(p for p in base if p not in now and not _lexists_exact(root, p, cache))
    return added, modified, removed


def _side(root: Path, rel: str) -> tuple[str, str]:
    """('missing' | 'link' | 'binary' | 'text', content) of root/rel for a diff; never read through a link."""
    try:
        st = os.lstat(Path(root, rel))
    except OSError:
        return "missing", ""
    if stat.S_ISLNK(st.st_mode):
        try:
            return "link", os.readlink(Path(root, rel))
        except OSError:
            return "binary", ""
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_DIFF_FILE:
        return "binary", ""
    try:
        data = _read_beneath(root, rel, MAX_DIFF_FILE + 1)  # refuses a path reached through a linked folder
        return "text", shown(decode_text(data)[0])
    except (OSError, ValueError):
        return "binary", ""


def _diff_chunks(old_root: Path, new_root: Path, rel: str) -> list[str]:
    (ka, ta), (kb, tb) = _side(old_root, rel), _side(new_root, rel)
    if "binary" in (ka, kb):
        return [f"Binary or large file changed: {rel}\n"]
    if "link" in (ka, kb):
        return [f"Symlink changed: {rel}: {ta if ka == 'link' else ka} -> {tb if kb == 'link' else kb}\n"]
    return list(difflib.unified_diff(
        ta.splitlines(keepends=True), tb.splitlines(keepends=True),
        fromfile=f"a/{rel}" if ka != "missing" else "/dev/null", tofile=f"b/{rel}" if kb != "missing" else "/dev/null",
    ))


def _diff_text(chunks: list[str], max_chars: int) -> str:
    text = "".join(c if c.endswith("\n") else c + "\n" for c in chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (diff truncated)\n"
    return text


def changes_since(base: dict[str, str], root: Path, compare_to: Path, max_chars: int = 80_000) -> dict:
    """What `root` changed relative to the manifest it started from, diffed against `compare_to`."""
    added, modified, removed = _changes(base, Path(root))
    chunks: list[str] = []
    for rel in added + modified + removed:
        chunks.extend(_diff_chunks(Path(compare_to), Path(root), rel))
    return {"added": added, "modified": modified, "removed": removed, "diff": _diff_text(chunks, max_chars)}


def _real(p: Path | str) -> str:
    # (realpath leaves the rest of the path as written past a link loop — maybe '..' — so it is normalised too: a path
    # through a loop is judged by its letters, which never makes it look inside when it is not)
    return os.path.normpath(os.path.realpath(p))


def _link_stays_inside(target_root: Path, rel: str, link: str) -> bool:
    root = os.path.realpath(target_root)
    return _within(_real(os.path.join(root, os.path.dirname(rel), link)), root)


def _up_after_down(link: str) -> bool:
    """A link target that goes into a folder and back up ('y/../x', 'y/z/../../x', '/abs/y/../x'). '..' after a name is
    applied where that name LEADS (a link's target), so where such a link ends depends on links made later — by
    another «Aplicar» of the same conversation, too: `a -> y/../x` is inside while y is a folder and leaves the project
    once a later apply replaces y with `y -> .`. No check of the folder as it is now can vouch for it, so it is never
    created. Leading '..' are fine (applied to the real folders the link is written in: `.bin/x -> ../pkg/bin/x`)."""
    down = False
    for part in re.split(r"[\\/]" if os.name == "nt" else "/", link):
        if part == "..":
            if down:
                return True
        elif part not in ("", "."):
            down = True
    return False


def _escaping_links(target: Path, rels: list[str]) -> dict[str, str]:
    """Of the entries `rels` an apply just wrote into `target`, the links that point out of it now that all of them
    exist — each was checked when written, against the folder as it was then, but a link of the user's can still
    complete a chain: one made meanwhile, or one with a '..' after a folder name (`w -> q/..`, which the apply's own
    `q -> .` takes out of the project, and `a -> w/x` with it). They are removed again (until none is left: removing
    one changes where others lead). {rel: link text}."""
    links = {}
    for rel in rels:
        try:
            if stat.S_ISLNK(os.lstat(Path(target, rel)).st_mode):
                links[rel] = os.readlink(Path(target, rel))
        except OSError:
            continue
    root, out = os.path.realpath(target), {}
    while bad := [rel for rel in links if rel not in out and not _within(_real(Path(target, rel)), root)]:
        for rel in bad:
            try:
                _unlink_entry(Path(target), rel)  # the link itself, never what it points to
            except OSError:
                pass
            out[rel] = links[rel]
    return out


def _clear_at(dfd: int, name: str) -> None:
    """Make room for a new entry `name`: unlink a file or link, remove an empty directory."""
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        os.rmdir(name, dir_fd=dfd)  # OSError if not empty: the caller reports it as skipped
    else:
        os.unlink(name, dir_fd=dfd)


def _copy_entry(src_root: Path, dst_root: Path, rel: str, dst_rel: str | None = None) -> None:
    """Copy src_root/rel (a regular file or a link) to dst_root/<dst_rel or rel> without following any link on either
    side; a link that would point out of dst_root is refused (EPERM)."""
    src_rel, rel = rel, dst_rel or rel
    src = Path(src_root, src_rel)
    st = os.lstat(src)
    parts = _parts(dst_root, rel)
    if stat.S_ISLNK(st.st_mode):
        link = os.readlink(src)
        # Judged in dst_root as it is now — which holds for good only for a target without a '..' after a folder name
        # (_up_after_down): its leading '..' climb the real folders it is written in, the rest only goes down, through
        # links that are inside too. One that goes down and back up (`a -> y/z/../../x`) follows those folders where
        # they lead, so links made after it (`y -> .`, `z -> .`: later in this apply, in a later «Aplicar» of the same
        # conversation, or a folder replaced by one) take it out of the project: refused whatever the folders are now.
        # (Not judged in src_root: an absolute link into the project, and links through one, point out of the cage.)
        # One message for both (the web UI translates it: "path: symlink to X would point outside the project").
        if not _link_stays_inside(dst_root, rel, link) or _up_after_down(link):
            raise OSError(errno.EPERM, f"symlink to {link} would point outside the project")
        if _FD_WALK:
            dfd = _parent_fd(dst_root, parts, mkdirs=True)
            try:
                _clear_at(dfd, parts[-1])
                os.symlink(link, parts[-1], dir_fd=dfd)
            finally:
                os.close(dfd)
        else:
            dst = _no_links(dst_root, parts[:-1]) / parts[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            os.symlink(link, dst)
        return
    if not stat.S_ISREG(st.st_mode):
        raise OSError(errno.EINVAL, "not a regular file")
    sfd = _open_beneath(src_root, src_rel)
    try:
        if _FD_WALK:
            dfd = _parent_fd(dst_root, parts, mkdirs=True)
            try:
                try:
                    cur = os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
                    if not stat.S_ISREG(cur.st_mode):
                        _clear_at(dfd, parts[-1])  # a link or empty dir in the way: replace it, never write through it
                except FileNotFoundError:
                    pass
                dfd_out = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | _NOFOLLOW | _NONBLOCK | _CLOEXEC, 0o666, dir_fd=dfd)
            finally:
                os.close(dfd)
        else:
            dst = _no_links(dst_root, parts[:-1]) / parts[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            dfd_out = os.open(dst, os.O_WRONLY | os.O_CREAT | _BINARY, 0o666)
        with os.fdopen(dfd_out, "wb") as out, os.fdopen(sfd, "rb") as inp:
            sfd = -1
            if not stat.S_ISREG(os.fstat(out.fileno()).st_mode):
                raise OSError(errno.EINVAL, "destination is not a regular file")
            out.truncate(0)
            shutil.copyfileobj(inp, out, 1 << 20)
            out.flush()
            try:
                if os.chmod in os.supports_fd:
                    os.chmod(out.fileno(), stat.S_IMODE(st.st_mode))
                if os.utime in os.supports_fd:
                    os.utime(out.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
            except OSError:
                pass
    finally:
        if sfd >= 0:
            os.close(sfd)


def _unlink_entry(root: Path, rel: str) -> None:
    parts = _parts(root, rel)
    if _FD_WALK:
        try:
            dfd = _parent_fd(root, parts)
        except FileNotFoundError:
            return
        try:
            st = os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
            if stat.S_ISDIR(st.st_mode):
                raise OSError(errno.EISDIR, "it is a directory now")
            os.unlink(parts[-1], dir_fd=dfd)  # removes a link itself, never what it points to
        except FileNotFoundError:
            pass
        finally:
            os.close(dfd)
    else:
        p = _no_links(root, parts[:-1]) / parts[-1]
        if p.is_dir() and not p.is_symlink():
            raise OSError(errno.EISDIR, "it is a directory now")
        p.unlink(missing_ok=True)


def _apply(src_root: Path, target: Path, added: list[str], modified: list[str], removed: list[str]) -> dict:
    """Copy changes onto `target`. Removals go first: after a case-only rename (button.tsx -> Button.tsx) on a
    case-insensitive disk, removing the old name afterwards would delete the renamed file."""
    out: dict = {"added": [], "modified": [], "removed": [], "skipped": []}
    for rel in removed:
        try:
            _unlink_entry(Path(target), rel)
            out["removed"].append(rel)
        except OSError as exc:
            out["skipped"].append(f"{rel}: {_why(exc)}")
    for key, rels in (("added", added), ("modified", modified)):
        for rel in rels:
            try:
                _copy_entry(Path(src_root), Path(target), rel)
                out[key].append(rel)
            except OSError as exc:
                out["skipped"].append(f"{rel}: {_why(exc)}")
    for rel, link in _escaping_links(Path(target), out["added"] + out["modified"]).items():
        out["added" if rel in out["added"] else "modified"].remove(rel)
        out["skipped"].append(f"{rel}: symlink to {link} would point outside the project")
    return out


def apply_cage(base: dict[str, str], root: Path, target: Path) -> dict:
    """Copy a cage's own changes (relative to its starting manifest) onto `target`. Never writes through a
    link in `target` and never creates a link pointing out of it; what can't be applied is in 'skipped'."""
    added, modified, removed = _changes(base, Path(root))
    return _apply(Path(root), Path(target), added, modified, removed)


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._bg = _Background()
        self._watcher: threading.Thread | None = None
        weakref.finalize(self, self._bg.kill)  # the run that used this workspace is over

    def resolve(self, rel: str) -> Path:
        rel = (rel or ".").strip()
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path {rel!r} is outside the workspace")
        return p

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    def contains(self, p: Path | str) -> bool:
        return _within(os.path.realpath(p), str(self.root))

    # Server-side file access for the tools: regular files only, never through a link (even one swapped in
    # after resolve() by a background process), never a FIFO or device.
    def open_file(self, p: Path):
        try:
            return os.fdopen(_open_beneath(self.root, p), "rb")
        except OSError as exc:
            raise ToolError(f"cannot read {self.rel(p)}: {_why(exc)}") from None
        except ValueError:
            raise ToolError(f"path {str(p)!r} is outside the workspace") from None

    def open_dir(self, p: Path) -> int | None:
        """fd of the directory p, reached without following any link (POSIX; None where dir fds are missing).
        Walking by fds means a folder swapped for a link mid-walk is refused, never listed through."""
        if not _FD_WALK:
            return None
        try:
            if Path(p) == self.root:
                return os.open(self.root, os.O_RDONLY | _DIRECTORY | _CLOEXEC)
            parts = _parts(self.root, p)
            dfd = _parent_fd(self.root, parts)
            try:
                return os.open(parts[-1], os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)
            finally:
                os.close(dfd)
        except OSError as exc:
            raise ToolError(f"cannot list {self.rel(p)}: {_why(exc)}") from None
        except ValueError:
            raise ToolError(f"path {str(p)!r} is outside the workspace") from None

    def read_bytes(self, p: Path, limit: int | None = None) -> bytes:
        with self.open_file(p) as f:
            return f.read() if limit is None else f.read(limit)

    def write_bytes(self, p: Path, data: bytes, mode: int | None = None) -> None:
        """Write a file in place (keeps its permissions), creating missing directories."""
        try:
            fd = _open_beneath(self.root, p, write=True, mkdirs=True)
        except OSError as exc:
            raise ToolError(f"cannot write {self.rel(p)}: {_why(exc)}") from None
        except ValueError:
            raise ToolError(f"path {str(p)!r} is outside the workspace") from None
        with os.fdopen(fd, "wb") as f:
            f.truncate(0)
            f.write(data)
            if mode is not None:
                try:
                    os.chmod(f.fileno() if os.chmod in os.supports_fd else p, mode)
                except OSError:
                    pass

    def install_harness(self, h: Harness) -> None:
        """Materialise harness scripts so bash can run them at .nesicode/<path>. The agent may have turned
        .nesicode, scripts/ or skills/ into links (to the user's folders): those are replaced by real
        directories and nothing is ever deleted or written through a link."""
        if _FD_WALK:
            root_fd = os.open(self.root, os.O_RDONLY | _DIRECTORY | _CLOEXEC)
            try:
                nesi_fd = _real_dir_at(root_fd, HARNESS_DIR)
                try:
                    for sub in ("scripts", "skills"):  # keep .nesicode/context, outputs and chats (the archive)
                        _remove_at(nesi_fd, sub)
                finally:
                    os.close(nesi_fd)
            finally:
                os.close(root_fd)
        else:
            target = self.root / HARNESS_DIR
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                target.unlink()
            target.mkdir(exist_ok=True)
            for sub in ("scripts", "skills"):
                p = target / sub
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
        for path, text in h.items():
            if path.startswith("scripts/") or "/scripts/" in path:
                rel = Path(HARNESS_DIR, path)
                if ".." in rel.parts or Path(path).is_absolute():
                    continue
                self.write_bytes(self.root / rel, text.encode("utf-8"), mode=0o755)  # bytes: no CRLF on Windows

    @property
    def home(self) -> Path:
        """Private HOME, outside the cage so tool caches never pollute diffs: next to a cage nesicode made
        (it goes away with the conversation), under the data dir for a folder of the user's (never litter
        next to their project, nor fail when its parent is not writable, e.g. a volume root)."""
        data = settings.data_dir.resolve()
        if _within(str(self.root), str(data)) and self.root != data:
            return self.root.parent / f".{self.root.name}.home"
        key = hashlib.sha256(str(self.root).encode("utf-8", "surrogateescape")).hexdigest()[:16]
        return data / "homes" / f"{re.sub(r'[^A-Za-z0-9._-]', '_', self.root.name)[:40]}-{key}"

    def _local_env(self, mode: str) -> dict:
        home = self.home
        tmp = home / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        path = os.environ.get("PATH", "/usr/bin:/bin")
        env = {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",  # Windows: Python children otherwise write the ANSI code page to the pipe
            "PYTHONIOENCODING": "utf-8",
            "LANG": "C.UTF-8",
            "TERM": "dumb",
        }
        if mode == "macos":
            _install_shims(home)
            path = _cage_path(path, self.root, home)
            env["CLANG_MODULE_CACHE_PATH"] = str(home / ".cache" / "clang")  # the default per-user cache is denied
        env["PATH"] = path
        if os.name == "nt":
            env.update({k: v for k, v in os.environ.items() if k.upper() in WINDOWS_ENV})
            env["USERPROFILE"] = str(home)
        else:
            env["NESICODE_PATH"] = path  # see RESTORE_PATH
        return env

    def run(self, command: str, timeout: int = 120) -> tuple[int, str]:
        timeout = max(1, min(int(timeout or 120), 900))
        mode = sandbox_mode()
        container = None
        if mode == "docker":
            container = f"nesicode-{uuid.uuid4().hex[:16]}"
            argv, env = docker_argv(self.root, command, container), None
        else:
            env = self._local_env(mode)
            argv = shell_argv(command)
            if mode == "macos":
                argv = ["sandbox-exec", "-p", macos_profile(self.root, self.home, _listening_ports())] + argv
        # Windows: started suspended, put in its own job, then let go (see _WinJob): not even its first child can start
        # outside the job
        suspend = os.name == "nt" and _WinJob.ready()
        try:
            group = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                 | (_CREATE_SUSPENDED if suspend else 0)}
                if os.name == "nt" else {"start_new_session": True}
            )
            proc = subprocess.Popen(
                argv, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, **group,
            )
        except OSError as exc:
            raise ToolError(f"could not start command: {exc}") from exc
        job = None
        if suspend:
            try:
                job = _WinJob.adopt(proc)
            except OSError as exc:  # it could not be let go: it never ran
                proc.kill()
                proc.wait()
                proc.stdout.close()
                raise ToolError(f"could not start command: {exc}") from exc
        try:
            if container:
                _CONTAINERS.add(container)
            out = _Capture(proc.stdout)
            # Wait in short slices so the user's Stop kills the command at once (not when it ends).
            cancel = getattr(self, "cancel", None)
            deadline = time.monotonic() + timeout
            why = ""
            while True:
                try:
                    proc.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancel is not None and cancel.is_set():
                    why = "stopped"
                elif out.over.is_set():
                    why = "overflow"
                elif time.monotonic() >= deadline:
                    why = "timeout"
                if why:
                    _kill_tree(proc, container, job)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    break
            # The command is over, but processes it left (`server &`, a setsid'd helper) may still hold the output
            # pipe: wait a little for the last bytes, never for them to exit.
            out.eof.wait(min(0.5, EOF_GRACE) if why else EOF_GRACE)
        except BaseException:
            # A raw KeyboardInterrupt (a second Ctrl+C, library use without the CLI's handler) or anything else that
            # aborts the wait: the command and whatever it started must not outlive us as orphans.
            _kill_tree(proc, container, job)
            if container:
                _CONTAINERS.discard(container)
            raise
        if container:
            _CONTAINERS.discard(container)
        if not why:
            self._track_background(proc, job)
        text = out.text()
        if mode == "macos":
            text = _XCRUN_NOISE.sub("", text)
        if why == "stopped":
            return 130, text + "\n[stopped by the user]"
        if why == "timeout":
            return 124, text + f"\n[timed out after {timeout}s]"
        if why == "overflow":
            return 137, text + (f"\n[stopped: the command printed more than {OUTPUT_CAP // 1_000_000} MB; "
                                "redirect big outputs to a file]")
        return proc.returncode if proc.returncode is not None else -1, text

    # ------------------------------------------------------------ background processes (`cmd &`)
    def _track_background(self, proc: subprocess.Popen, job: _WinJob | None = None) -> None:
        group: int | _WinJob
        if os.name == "nt":
            if job is None:
                return  # (no job: Windows refused one) nothing can find them once their parent is gone
            if not job.active():  # processes left in the command's job?
                job.close()
                return
            group = job
        else:
            try:
                os.killpg(proc.pid, 0)  # members left in the command's process group?
            except OSError:
                return
            group = proc.pid
        if self._bg.add(group):  # decided under the lock, so a watcher that is just exiting can't miss this group
            self._watcher = threading.Thread(target=_watch_background, args=(weakref.ref(self), self._bg), daemon=True)
            self._watcher.start()

    def kill_background(self) -> None:
        """Kill what this workspace's commands left running (call when the run ends; Stop and exit do it too)."""
        self._bg.kill()


_XCRUN_NOISE = re.compile(r"^[^\n:]{1,80}: error: couldn't create cache file '[^'\n]*xcrun_db[^'\n]*' \(errno=[^)\n]*\)\n?", re.M)
_CONTAINERS: set[str] = set()


def docker_argv(root: Path, command: str, name: str) -> list[str]:
    """A throwaway container with a name (so Stop/timeout can `docker kill` it: killing the CLI does not stop
    it) and, on Linux, the server's uid (files it creates stay the server's, not root's)."""
    user: list[str] = []
    if sys.platform.startswith("linux") and hasattr(os, "getuid"):
        user = ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    return [
        "docker", "run", "--rm", "--init", "--name", name, *user, "--network", settings.docker_network,
        "--memory", "2g", "--cpus", "2", "--pids-limit", "512", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUTF8=1",
        "-v", f"{root}:/workspace", "-w", "/workspace", settings.docker_image, "bash", "-lc", command,
    ]


def _kill_tree(proc: subprocess.Popen, container: str | None = None, job: _WinJob | None = None) -> None:
    if container:
        try:
            subprocess.run(["docker", "kill", container], capture_output=True, timeout=20, **_NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        if os.name == "nt":
            if job is not None and job.kill():
                return  # every process the command started, also those whose parent is gone
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=20, **_NO_WINDOW)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass


# Helpers started from the desktop app (a window, no console) must not flash a console window of their own.
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
_CREATE_SUSPENDED = 0x00000004


class _WinJob:
    """Windows has no process groups: a command runs in a job object instead (started suspended and let go once it is
    in its job, so not even its first child starts outside). Stop, a timeout and the run's end kill everything in the
    job, also what outlived the command (`server &`, Start-Process): taskkill /T only finds the children of a live
    process, and a background process's parent — the shell — is gone by then. The job kills what is left when its
    last handle closes (KILL_ON_JOB_CLOSE), so nothing a command started outlives Nesicode, even when Nesicode itself
    is killed. No breakaway: Python's launchers (a venv's python.exe, py.exe) put their interpreter in a job of their
    own that lets children go silently, and they would leave ours too if it allowed it (they stop at the first job
    up the chain that does not)."""

    _api: tuple | None = None
    _api_failed = False
    _api_lock = threading.Lock()

    def __init__(self, handle: int):
        self._handle: int | None = handle
        self._lock = threading.Lock()

    @classmethod
    def _load(cls) -> tuple | None:
        with cls._api_lock:
            if cls._api is None and not cls._api_failed:
                try:
                    cls._api = _win_job_api()
                except Exception as exc:  # noqa: BLE001 - commands still run (their tree killed with taskkill)
                    cls._api_failed = True
                    print("nesicode: Windows job objects are not available:", exc)
            return cls._api

    @classmethod
    def ready(cls) -> bool:
        return os.name == "nt" and cls._load() is not None

    @classmethod
    def adopt(cls, proc: subprocess.Popen) -> _WinJob | None:
        """Put a command started suspended in a new job, then let it run. None: Windows refused the job (the command
        runs all the same; its tree is then killed with taskkill /T). OSError: it could not be let go (it never ran)."""
        ctypes, k32, ntdll, extended, _ = cls._load()
        process = k32.OpenProcess(0x0800 | 0x0100 | 0x0001, False, proc.pid)  # SUSPEND_RESUME | SET_QUOTA | TERMINATE
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        job = None
        try:
            job = k32.CreateJobObjectW(None, None)
            info = extended()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if job and not (k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # extended limits
                            and k32.AssignProcessToJobObject(job, process)):
                k32.CloseHandle(job)  # (it holds no process)
                job = None
        except Exception as exc:  # noqa: BLE001 - no job then (the command still runs, killed with taskkill /T)
            print("nesicode: could not put a command in a job object:", exc)
            if job:
                k32.CloseHandle(job)
            job = None
        finally:
            status = ntdll.NtResumeProcess(process)
            k32.CloseHandle(process)
        if status != 0:
            if job:
                k32.CloseHandle(job)  # KILL_ON_JOB_CLOSE: the suspended command goes with it
            raise OSError(f"could not resume the command (NTSTATUS {status & 0xFFFFFFFF:#010x})")
        return cls(job) if job else None

    def active(self) -> int:
        """How many processes of the command are still running (1 when it cannot be told: kept until killed)."""
        ctypes, k32, _, _, accounting = self._api
        with self._lock:
            if not self._handle:
                return 0
            acct = accounting()
            if not k32.QueryInformationJobObject(self._handle, 1, ctypes.byref(acct), ctypes.sizeof(acct), None):
                return 1
            return int(acct.ActiveProcesses)

    def kill(self) -> bool:
        """Kill every process in the job and let the job go. False if Windows refused (or it was gone already)."""
        _, k32, _, _, _ = self._api
        with self._lock:
            handle, self._handle = self._handle, None
        if not handle:
            return False
        ok = bool(k32.TerminateJobObject(handle, 1))
        k32.CloseHandle(handle)  # (KILL_ON_JOB_CLOSE: what TerminateJobObject missed goes now)
        return ok

    close = kill  # a job with nothing left: closing it is all there is to do


def _win_job_api() -> tuple:
    """kernel32's job object functions and structures (Windows only), set up once."""
    import ctypes
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

    class Extended(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class Accounting(ctypes.Structure):  # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", w.DWORD), ("TotalProcesses", w.DWORD),
                    ("ActiveProcesses", w.DWORD), ("TotalTerminatedProcesses", w.DWORD)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # (own instances: argtypes never clash with other users')
    ntdll = ctypes.WinDLL("ntdll")
    for name, res, args in (
        ("OpenProcess", w.HANDLE, (w.DWORD, w.BOOL, w.DWORD)),
        ("CloseHandle", w.BOOL, (w.HANDLE,)),
        ("CreateJobObjectW", w.HANDLE, (w.LPVOID, w.LPCWSTR)),
        ("SetInformationJobObject", w.BOOL, (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD)),
        ("QueryInformationJobObject", w.BOOL, (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, w.LPDWORD)),
        ("AssignProcessToJobObject", w.BOOL, (w.HANDLE, w.HANDLE)),
        ("TerminateJobObject", w.BOOL, (w.HANDLE, w.UINT)),
    ):
        fn = getattr(k32, name)
        fn.restype, fn.argtypes = res, args
    # (NtResumeProcess: the one call that lets a whole suspended process go — subprocess keeps no thread handle)
    ntdll.NtResumeProcess.restype, ntdll.NtResumeProcess.argtypes = ctypes.c_long, (w.HANDLE,)
    return ctypes, k32, ntdll, Extended, Accounting


class _Capture:
    """Drains a command's output in a thread, keeping only its first and last OUTPUT_KEEP bytes."""

    def __init__(self, pipe):
        self.head, self.tail, self.total = bytearray(), bytearray(), 0
        self.over, self.eof = threading.Event(), threading.Event()
        self._discard = False
        self._lock = threading.Lock()
        threading.Thread(target=self._drain, args=(pipe,), daemon=True).start()

    def _drain(self, pipe) -> None:
        fd = pipe.fileno()
        try:
            while True:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                with self._lock:
                    self.total += len(chunk)
                    if self.total > OUTPUT_CAP:
                        self.over.set()
                    if self._discard:  # the call returned: later output of background processes is dropped
                        continue
                    room = OUTPUT_KEEP - len(self.head)
                    if room > 0:
                        self.head += chunk[:room]
                        chunk = chunk[room:]
                    if chunk:
                        self.tail += chunk
                        if len(self.tail) > OUTPUT_KEEP:
                            del self.tail[: len(self.tail) - OUTPUT_KEEP]
        except (OSError, ValueError):
            pass
        finally:
            self.eof.set()
            try:
                pipe.close()
            except OSError:
                pass

    def text(self) -> str:
        with self._lock:
            head, tail, total = bytes(self.head), bytes(self.tail), self.total
            self._discard = True
        omitted = total - len(head) - len(tail)
        if omitted <= 0:  # nothing dropped: decode in one piece (a character may straddle head and tail)
            return (head + tail).decode("utf-8", "replace")
        return (head.decode("utf-8", "replace") + f"\n... ({omitted:,} bytes of output omitted) ...\n"
                + tail.decode("utf-8", "replace"))


class _Background:
    """Process groups (Windows: jobs, see _WinJob) a workspace's commands left running."""

    def __init__(self):
        self.groups: set[int | _WinJob] = set()
        self.lock = threading.Lock()
        self.watched = False  # a watcher thread is running (or about to) for these groups

    def add(self, pgid: int | _WinJob) -> bool:
        """Track a group; True when the caller must start a watcher (none is running)."""
        with self.lock:
            self.groups.add(pgid)
            start, self.watched = not self.watched, True
            return start

    def prune(self) -> bool:
        """Forget groups that are gone (so their ids can never hit a reused one); True while any is alive. When none
        is, the watcher stops, and the next add() starts a new one (both under the lock: no group is left unwatched)."""
        with self.lock:
            for g in list(self.groups):
                if isinstance(g, _WinJob):
                    if not g.active():  # every process in it has ended
                        g.close()
                        self.groups.discard(g)
                    continue
                try:
                    os.killpg(g, 0)
                except (ProcessLookupError, PermissionError):  # gone, or the id now belongs to another user's process
                    self.groups.discard(g)
                except OSError:
                    pass
            self.watched = bool(self.groups)
            return self.watched

    def kill(self) -> None:
        with self.lock:
            groups, self.groups = self.groups, set()
        for g in groups:
            if isinstance(g, _WinJob):
                g.kill()
                continue
            try:
                os.killpg(g, signal.SIGKILL)
            except OSError:
                pass


def _watch_background(ws_ref, bg: _Background) -> None:
    """Kill the workspace's background processes as soon as the user presses Stop."""
    while True:
        ws = ws_ref()
        cancel = getattr(ws, "cancel", None) if ws is not None else None
        released = ws is None  # the run is over (its finalizer killed them already)
        del ws
        if released or (cancel is not None and cancel.is_set()):
            bg.kill()
        if not bg.prune():
            return
        time.sleep(0.5)


def _kill_containers() -> None:
    for name in list(_CONTAINERS):
        try:
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=20, **_NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            pass


__import__("atexit").register(_kill_containers)


class Cancelled(Exception):
    """The user pressed Stop."""


def _copy_data(fsrc, fdst) -> None:
    """A file's contents, the fast way where the OS has one (macOS fcopyfile, Linux sendfile: what shutil.copyfile
    does), else in chunks."""
    try:
        if getattr(shutil, "_HAS_FCOPYFILE", False) and hasattr(shutil, "_fastcopy_fcopyfile"):
            import posix

            shutil._fastcopy_fcopyfile(fsrc, fdst, posix._COPYFILE_DATA)
            return
        if getattr(shutil, "_USE_CP_SENDFILE", False) and hasattr(shutil, "_fastcopy_sendfile"):
            shutil._fastcopy_sendfile(fsrc, fdst)
            return
    except getattr(shutil, "_GiveupOnFastCopy", ()):  # raised before anything was copied: the plain way below
        pass
    shutil.copyfileobj(fsrc, fdst, 1 << 20)


def _copy_regular(src: str, dst: str) -> str:
    """copytree's copy_function: a regular file with its mode and times. Whatever else is found there by the time it is
    copied — a FIFO swapped in after the listing (opening one for reading waits for a writer: the copy would hang), a
    socket, a device, a link, a file that vanished — is skipped, never opened as a file."""
    try:
        fd = os.open(src, os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _CLOEXEC | _BINARY)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENXIO, errno.EOPNOTSUPP, errno.ENOENT):
            return dst
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return dst
        with os.fdopen(fd, "rb") as fsrc:
            fd = -1
            out = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW | _CLOEXEC | _BINARY, 0o666)
            with os.fdopen(out, "wb") as fdst:
                _copy_data(fsrc, fdst)
                fdst.flush()
                try:
                    if os.chmod in os.supports_fd:
                        os.chmod(fdst.fileno(), stat.S_IMODE(st.st_mode))
                    if os.utime in os.supports_fd:
                        os.utime(fdst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
                except OSError:
                    pass
        if os.utime not in os.supports_fd:  # (Windows)
            shutil.copystat(src, dst)
    finally:
        if fd >= 0:
            os.close(fd)
    return dst


def copy_workspace(src: Path, dst: Path) -> None:
    """Copy a project into a fresh cage. Sockets, FIFOs and devices (git's fsmonitor socket, a dev database's
    socket...) can't be copied and made copytree fail after copying everything else: they are skipped — also one
    that appears while the copy runs (nothing but a regular file is ever opened, see _copy_regular)."""
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(d: str, names: list[str]) -> set[str]:
        skip = {HARNESS_DIR} & set(names)
        for n in names:
            try:
                m = os.lstat(os.path.join(d, n)).st_mode
            except OSError:
                skip.add(n)
                continue
            if not (stat.S_ISREG(m) or stat.S_ISDIR(m) or stat.S_ISLNK(m)):
                skip.add(n)
        return skip

    shutil.copytree(src, dst, symlinks=True, ignore=ignore, copy_function=_copy_regular)


def copy_files_into(src: Path, dst_root: Path, prefix: str) -> list[str]:
    """Merge the files and links under the folder `src` into dst_root/<prefix> (a conversation's attachments into a
    harnessing snapshot…) without following a link on either side: a `src` that is itself a link, a link that would
    point out of dst_root and anything that is not a regular file are skipped. Returns what was skipped, and why."""
    src = Path(src)
    if src.is_symlink() or not src.is_dir():
        return [f"{prefix}: not a folder"]
    skipped = []
    for p, _e in _scan(src):
        rel = p.relative_to(src).as_posix()
        try:
            _copy_entry(src, Path(dst_root), rel, dst_rel=f"{prefix}/{rel}")
        except OSError as exc:
            skipped.append(f"{prefix}/{rel}: {_why(exc)}")
        except ValueError as exc:
            skipped.append(f"{prefix}/{rel}: {exc}")
    return skipped


def _tree(root: Path) -> dict[str, os.stat_result]:
    out = {}
    if not Path(root).exists():
        return out
    for p, e in _scan(Path(root)):
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode):
            out[str(p.relative_to(root))] = st
    return out


def _same(a_root: Path, b_root: Path, rel: str, sa: os.stat_result, sb: os.stat_result) -> bool:
    if stat.S_ISLNK(sa.st_mode) or stat.S_ISLNK(sb.st_mode):
        return stat.S_ISLNK(sa.st_mode) and stat.S_ISLNK(sb.st_mode) and \
            os.readlink(Path(a_root, rel)) == os.readlink(Path(b_root, rel))
    if sa.st_size != sb.st_size:
        return False
    with os.fdopen(_open_beneath(a_root, rel), "rb") as fa, os.fdopen(_open_beneath(b_root, rel), "rb") as fb:
        while True:
            ca, cb = fa.read(1 << 20), fb.read(1 << 20)
            if ca != cb:
                return False
            if not ca:
                return True


def diff_workspaces(before: Path, after: Path, max_chars: int = 60_000) -> dict:
    """Changed files and links between two workspace trees (ignored dirs skipped)."""
    before, after = Path(before), Path(after)
    a, b = _tree(before), _tree(after)
    cache: dict = {}
    added = sorted(set(b) - set(a))
    removed = sorted(p for p in set(a) - set(b) if not (after.exists() and _lexists_exact(after, p, cache)))
    modified = []
    for rel in sorted(set(a) & set(b)):
        try:
            if not _same(before, after, rel, a[rel], b[rel]):
                modified.append(rel)
        except OSError:
            continue
    chunks: list[str] = []
    for rel in sorted(added + removed + modified) if max_chars > 0 else ():
        chunks.extend(_diff_chunks(before, after, rel))
    return {"added": added, "removed": removed, "modified": modified, "diff": _diff_text(chunks, max_chars)}


def apply_changes(before: Path, after: Path, target: Path) -> dict:
    """Copy what changed between `before` and `after` onto `target` (see _apply: removals first, no links
    followed or created out of `target`; what can't be applied is in 'skipped')."""
    d = diff_workspaces(before, after, max_chars=0)
    return _apply(Path(after), Path(target), d["added"], d["modified"], d["removed"])

"""A confined project workspace: path resolution, command execution, copies and diffs."""

from __future__ import annotations

import difflib
import hashlib
import os
import sys
import shutil
import signal
import subprocess
import time
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
HARNESS_DIR = ".nesicode"


class ToolError(Exception):
    """A tool failed in a way the model should see and recover from."""


REPO_DIR = Path(__file__).resolve().parents[3]
# Toolchains under $HOME that caged commands may READ (never write).
TOOLCHAIN_DIRS = (".local", ".nvm", ".cargo", ".rustup", ".bun", ".pyenv", ".deno", "go", ".sdkman", ".volta")


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


def shell_argv(command: str) -> list[str]:
    if os.name == "nt":
        bash = windows_bash()
        if bash:
            return [bash, "-lc", command]
        return ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
    return ["bash", "-lc", command]


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


def macos_profile(cage: Path, home: Path) -> str:
    """Seatbelt profile: the command may read the OS and toolchains, but may only read or
    write inside its own cage; the user's home, the data dir (other users, projects and
    conversations), the nesicode repo (.env) and other processes are off limits.
    In SBPL the LAST matching rule wins, so denies come first and the cage allow last."""
    cage, home = cage.resolve(), home.resolve()
    real_home = Path(os.path.expanduser("~")).resolve()
    denied = [real_home, settings.data_dir.resolve(), REPO_DIR] + [
        Path(p).resolve() for p in os.environ.get("NESICODE_SANDBOX_DENY", "").split(":") if p
    ]
    tools = " ".join(f"(subpath {_q(real_home / d)})" for d in TOOLCHAIN_DIRS)
    return "\n".join([
        "(version 1)",
        "(allow default)",
        *[f"(deny file-read* file-write* (subpath {_q(d)}))" for d in denied],
        '(deny file-write* (subpath "/"))',
        f"(allow file-read* {tools})",
        "(allow file-read-metadata)",
        f"(allow file-read* file-write* (subpath {_q(cage)}) (subpath {_q(home)}))",
        '(allow file-read* file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") '
        '(literal "/dev/random") (literal "/dev/urandom") (literal "/dev/dtracehelper") (regex #"^/dev/fd/"))',
        # mktemp ignores TMPDIR on macOS: allow only random-named entries, never listing the dir.
        '(allow file-read* file-write* (regex #"^/private/var/folders/[^/]+/[^/]+/T/tmp\\.[A-Za-z0-9]+"))',
        # Apple's xcrun shims (/usr/bin/python3, git…) cache their lookup database here.
        '(allow file-read* file-write* (regex #"^/private/var/folders/[^/]+/[^/]+/T/xcrun_db"))',
        "(deny signal)",
        "(allow signal (target same-sandbox))",
    ])


def manifest(root: Path) -> dict[str, str]:
    """{relative path: sha256} of every diffable file; used to detect a cage's own changes."""
    out = {}
    for p in _walk(root):
        try:
            if p.is_symlink() or p.stat().st_size > 50_000_000:
                continue
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def changes_since(base: dict[str, str], root: Path, compare_to: Path, max_chars: int = 80_000) -> dict:
    """What `root` changed relative to the manifest it started from, diffed against `compare_to`."""
    now = manifest(root)
    added = sorted(p for p in now if p not in base)
    modified = sorted(p for p in now if p in base and now[p] != base[p])
    removed = sorted(p for p in base if p not in now)
    chunks: list[str] = []
    for rel in added + modified + removed:
        old = compare_to / rel
        new = root / rel
        ta = _read_text(old) if old.exists() else ""
        tb = _read_text(new) if new.exists() else ""
        if ta is None or tb is None:
            chunks.append(f"Binary or large file changed: {rel}\n")
            continue
        chunks.extend(difflib.unified_diff(
            ta.splitlines(keepends=True), tb.splitlines(keepends=True),
            fromfile=f"a/{rel}" if old.exists() else "/dev/null", tofile=f"b/{rel}" if new.exists() else "/dev/null",
        ))
    text = "".join(c if c.endswith("\n") else c + "\n" for c in chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (diff truncated)\n"
    return {"added": added, "modified": modified, "removed": removed, "diff": text}


def apply_cage(base: dict[str, str], root: Path, target: Path) -> dict:
    """Copy a cage's own changes (relative to its starting manifest) onto `target`."""
    ch = changes_since(base, root, target, max_chars=0)
    for rel in ch["added"] + ch["modified"]:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dst, follow_symlinks=False)
    for rel in ch["removed"]:
        (target / rel).unlink(missing_ok=True)
    return {k: ch[k] for k in ("added", "modified", "removed")}


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> Path:
        rel = (rel or ".").strip()
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path {rel!r} is outside the workspace")
        return p

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    def install_harness(self, h: Harness) -> None:
        """Materialise harness scripts so bash can run them at .nesicode/<path>."""
        target = self.root / HARNESS_DIR
        for sub in ("scripts", "skills"):  # keep .nesicode/context and .nesicode/outputs (the archive)
            shutil.rmtree(target / sub, ignore_errors=True)
        for path, text in h.items():
            if path.startswith("scripts/") or "/scripts/" in path:
                f = target / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(text, encoding="utf-8")
                f.chmod(0o755)

    @property
    def home(self) -> Path:
        """Private HOME next to (not inside) the cage, so tool caches never pollute diffs."""
        return self.root.parent / f".{self.root.name}.home"

    def run(self, command: str, timeout: int = 120) -> tuple[int, str]:
        timeout = max(1, min(int(timeout or 120), 900))
        mode = sandbox_mode()
        if mode == "docker":
            argv = [
                "docker", "run", "--rm", "--network", settings.docker_network,
                "--memory", "2g", "--cpus", "2", "--pids-limit", "512",
                "-v", f"{self.root}:/workspace", "-w", "/workspace",
                settings.docker_image, "bash", "-lc", command,
            ]
            env = None
        else:
            tmp = self.home / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(self.home),
                "TMPDIR": str(tmp),
                "PYTHONDONTWRITEBYTECODE": "1",
                "LANG": "C.UTF-8",
                "TERM": "dumb",
            }
            argv = shell_argv(command)
            if os.name == "nt":
                env.update({k: v for k, v in os.environ.items() if k.upper() in WINDOWS_ENV})
                env["USERPROFILE"] = str(self.home)
            if mode == "macos":
                argv = ["sandbox-exec", "-p", macos_profile(self.root, self.home)] + argv
        try:
            group = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)}
                if os.name == "nt" else {"start_new_session": True}
            )
            proc = subprocess.Popen(
                argv, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, **group,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"could not start command: {exc}") from exc
        # Wait in short slices so the user's Stop kills the command at once (not when it ends).
        cancel = getattr(self, "cancel", None)
        deadline = time.monotonic() + timeout
        while True:
            try:
                out, _ = proc.communicate(timeout=0.25)
                return proc.returncode, (out or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired:
                stopped = cancel is not None and cancel.is_set()
                if not stopped and time.monotonic() < deadline:
                    continue
                _kill_tree(proc)
                out, _ = proc.communicate()
                text = (out or b"").decode("utf-8", "replace")
                if stopped:
                    return 130, text + "\n[stopped by the user]"
                return 124, text + f"\n[timed out after {timeout}s]"


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


class Cancelled(Exception):
    """The user pressed Stop."""


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def copy_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(HARNESS_DIR))


def _read_text(p: Path) -> str | None:
    try:
        if p.is_symlink() or p.stat().st_size > MAX_DIFF_FILE:
            return None
        data = p.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", "replace")


def diff_workspaces(before: Path, after: Path, max_chars: int = 60_000) -> dict:
    """Changed files between two workspace trees (ignored dirs skipped)."""
    a_files = {str(p.relative_to(before)): p for p in _walk(before)} if before.exists() else {}
    b_files = {str(p.relative_to(after)): p for p in _walk(after)} if after.exists() else {}
    added = sorted(set(b_files) - set(a_files))
    removed = sorted(set(a_files) - set(b_files))
    modified = []
    chunks: list[str] = []
    for rel in sorted(set(a_files) | set(b_files)):
        pa, pb = a_files.get(rel), b_files.get(rel)
        if pa and pb:
            try:
                if pa.stat().st_size == pb.stat().st_size and pa.read_bytes() == pb.read_bytes():
                    continue
            except OSError:
                continue
            modified.append(rel)
        ta = _read_text(pa) if pa else ""
        tb = _read_text(pb) if pb else ""
        if ta is None or tb is None:
            chunks.append(f"Binary or large file changed: {rel}\n")
            continue
        chunks.extend(
            difflib.unified_diff(
                ta.splitlines(keepends=True), tb.splitlines(keepends=True),
                fromfile=f"a/{rel}" if pa else "/dev/null", tofile=f"b/{rel}" if pb else "/dev/null",
            )
        )
    text = "".join(c if c.endswith("\n") else c + "\n" for c in chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (diff truncated)\n"
    return {"added": added, "removed": removed, "modified": modified, "diff": text}


def apply_changes(before: Path, after: Path, target: Path) -> dict:
    """Copy what changed between `before` and `after` onto `target`."""
    d = diff_workspaces(before, after, max_chars=0)
    for rel in d["added"] + d["modified"]:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(after / rel, dst, follow_symlinks=False)
    for rel in d["removed"]:
        (target / rel).unlink(missing_ok=True)
    return {k: d[k] for k in ("added", "removed", "modified")}

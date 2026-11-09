"""Runtime settings, read once from the environment (and an optional .env file)."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _read_dotenv(path: Path) -> str | None:
    """The text of a .env as Windows tools write it too: UTF-8 with or without a BOM (PowerShell 5.1 `-Encoding UTF8`,
    older Notepad), UTF-16 with a BOM (PowerShell 5.1 `echo … > .env`, Out-File) or the ANSI code page (Set-Content).
    None, with a warning naming the file, when it cannot be read: the app still starts (this runs at import, before
    the desktop app even opens its log)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"nesicode: could not read {path} ({exc}); its settings were not loaded", file=sys.stderr)
        return None
    if data[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        encodings = ["utf-32"]
    elif data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings = ["utf-16"]
    else:
        import locale

        encodings = ["utf-8-sig", locale.getpreferredencoding(False), "cp1252"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    print(f"nesicode: {path} is not text in a known encoding (save it as UTF-8); its settings were not loaded",
          file=sys.stderr)
    return None


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win. A line the environment cannot hold (no name, a NUL
    character) is skipped: a .env must never keep the app from starting."""
    if os.environ.get("NESICODE_NO_DOTENV") == "1" or not path.is_file():
        return
    text = _read_dotenv(path)
    if text is None:
        return
    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")  # (a BOM inside: files joined with `type a b > .env`)
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] in ('"', "'") and value[1:].find(value[0]) != -1:
            value = value[1 : 1 + value[1:].find(value[0])]
        else:
            value = "" if value.startswith("#") else re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if not key or "\0" in key + value:
            continue
        try:
            os.environ.setdefault(key, value)
        except (ValueError, OSError):  # a name the system refuses
            continue


def app_data_dir() -> Path:
    """Per-user data folder of the desktop app (also used when NESICODE_DESKTOP=1)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "nesicode"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "nesicode"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "nesicode"


# Under pytest, never the real data dir, vault key or .env: tests run through tests/conftest.py, which isolates all of
# them and sets NESICODE_TEST_ISOLATED. A test file run from elsewhere skips that conftest, and once opened ./data (the
# dev server's database) as the local account — its fixtures deleted that account's bots. Refuse before touching anything.
if "pytest" in sys.modules and os.environ.get("NESICODE_TEST_ISOLATED") != "1":
    raise RuntimeError(
        "nesicode imported under pytest without tests/conftest.py: refusing to open the real data dir, keys and .env. "
        "Run the test from the repo's tests folder, or put a copy of tests/conftest.py next to it."
    )

IS_DESKTOP = getattr(sys, "frozen", False) or os.environ.get("NESICODE_DESKTOP") == "1"
if IS_DESKTOP:
    os.environ.setdefault("NESICODE_DATA_DIR", str(app_data_dir() / "data"))
    _load_dotenv(app_data_dir() / ".env")
_load_dotenv(Path.cwd() / ".env")
# The .env files settings (API keys…) come from: no caged command may read them, not even inside a folder the user lets
# the agent read (agent/workspace.py, secret_paths).
DOTENV_PATHS = ([app_data_dir() / ".env"] if IS_DESKTOP else []) + [Path.cwd() / ".env"]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("NESICODE_DATA_DIR", "./data")).resolve())
    # "auto": OS sandbox (macOS sandbox-exec) caging each workspace; "none": no sandbox;
    # "docker" runs every command in a throwaway container with no network.
    sandbox: str = field(default_factory=lambda: _env("NESICODE_SANDBOX", "auto"))
    docker_image: str = field(default_factory=lambda: _env("NESICODE_DOCKER_IMAGE", "python:3.12-slim"))
    docker_network: str = field(default_factory=lambda: _env("NESICODE_DOCKER_NETWORK", "none"))
    default_executor_model: str = field(
        default_factory=lambda: _env("NESICODE_EXECUTOR_MODEL", "anthropic/claude-opus-5-5")
    )
    default_improver_model: str = field(
        default_factory=lambda: _env("NESICODE_IMPROVER_MODEL", "anthropic/claude-opus-5-5")
    )
    # Model used to read images when the executor has no vision (empty = no vision).
    vision_model: str = field(default_factory=lambda: _env("NESICODE_VISION_MODEL", ""))
    # Model used to CREATE images (empty = image generation off).
    image_model: str = field(default_factory=lambda: _env("NESICODE_IMAGE_MODEL", ""))
    secret_key: str = field(default_factory=lambda: _env("NESICODE_SECRET_KEY", ""))
    max_jobs: int = field(default_factory=lambda: int(_env("NESICODE_MAX_JOBS", "4")))
    rollout_width: int = field(default_factory=lambda: int(_env("NESICODE_ROLLOUT_WIDTH", "4")))
    # Login screen and multi-user accounts. Off by default: nesicode runs as a single local user.
    auth_enabled: bool = field(default_factory=lambda: _env("NESICODE_AUTH", "false").lower() in ("1", "true", "yes", "on"))
    # Which account is used when auth is off (email). Empty = the first account, created if none exists.
    local_user: str = field(default_factory=lambda: _env("NESICODE_LOCAL_USER", ""))
    allow_signup: bool = field(default_factory=lambda: _env("NESICODE_ALLOW_SIGNUP", "1") == "1")
    cookie_secure: bool = field(default_factory=lambda: _env("NESICODE_COOKIE_SECURE", "0") == "1")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "nesicode.db"

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


settings = Settings()

"""A test run without tests/conftest.py must never reach the real data dir, keys or .env: a review's throwaway test,
run from a scratch folder, once opened ./data (the dev server's database) as the real local account."""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def load_config(tmp_path, under_pytest: bool, **env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("NESICODE_")}
    clean.update(PYTHONPATH=str(SRC), NESICODE_NO_DOTENV="1", NESICODE_DATA_DIR=str(tmp_path / "data"), **env)
    code = ("import pytest\n" if under_pytest else "") + "import nesicode.config\nprint('loaded')"
    return subprocess.run([sys.executable, "-c", code], env=clean, cwd=tmp_path, capture_output=True, text=True, timeout=120)


def test_nesicode_refuses_to_load_under_pytest_without_the_tests_isolation(tmp_path):
    p = load_config(tmp_path, True)
    assert p.returncode != 0 and "refusing to open the real data dir" in p.stderr, p.stderr
    assert load_config(tmp_path, True, NESICODE_TEST_ISOLATED="1").stdout.strip() == "loaded"
    assert load_config(tmp_path, False).stdout.strip() == "loaded"  # (the app and its servers never import pytest)

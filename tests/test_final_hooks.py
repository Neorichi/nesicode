"""Hook regexes that can freeze the server are refused when a harness is validated (security#3, hook half).

`search_history` and `search` already refuse nested repetitions such as `(a+)+$`; a harness hook (written by the
improver or a user) ran them unguarded on every tool call, and Python's re holds the GIL while it backtracks."""

import json

from nesicode.harness import base_harness
from nesicode.harness.base import LEGACY_RM_MATCHES, RM_MATCH
from nesicode.harness.tree import HOOKS_FILE, _validate_hooks


def _hooks(match):
    return json.dumps({"pre_tool": [{"tool": "bash", "match": match, "action": "block", "message": "no"}]})


def test_a_nested_repetition_is_refused():
    for bad in (r"(\w+\s?)+:", r"(a+)+$", r"(\s*\w+)*x"):
        errs = _validate_hooks(_hooks(bad))
        assert errs and "freeze the server" in errs[0], (bad, errs)


def test_ordinary_regexes_and_the_shipped_rm_guard_are_accepted():
    assert _validate_hooks(_hooks(r"rm\s+-rf\s+/")) == []
    assert _validate_hooks(_hooks(r"(\w+\.)+py$")) == []  # a mandatory '.' each round: linear
    assert _validate_hooks(_hooks(RM_MATCH)) == []
    for old in LEGACY_RM_MATCHES:  # stored in projects created with earlier versions
        assert _validate_hooks(_hooks(old)) == []
    base = base_harness()
    files = base.files if hasattr(base, "files") else base
    assert _validate_hooks(files[HOOKS_FILE]) == []

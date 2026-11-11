"""The child process that runs the agent's regex searches that could backtrack for a long time.

Python's `re` holds the GIL while it matches, and agents run as threads of the server: a pattern such as `(a|aa)+$` or
`foo.*bar.*baz` on an unlucky line would freeze every conversation, the HTTP loop and the Stop button with it. The
`search` tool (nesicode.agent.tools) runs such patterns here and kills this process at its deadline or on Stop.

It is only ever sent text the server read itself (the server keeps reading files without following links), and it
imports nothing but the standard library: it lives outside nesicode.agent because that package loads the whole agent,
seconds per search."""

from __future__ import annotations

import faulthandler
import os
import re

_devnull = None


def serve(conn, pattern: str, flags: int, lifetime: float) -> None:
    """Say "ready", then answer (kind, data, limit) requests until the pipe closes: "lines" -> [(index, start, end)] of
    the first match on each line that has one; "text" -> [(start, end)] of the matches in the whole text. At most
    `limit` of either."""
    global _devnull
    # Never outlive the search that started it, even when the server dies in the middle of a match: faulthandler's
    # watchdog is a C thread, it needs no GIL.
    _devnull = open(os.devnull, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(lifetime, exit=True, file=_devnull)
    rx = re.compile(pattern, flags)
    conn.send("ready")
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        kind, data, limit = msg
        out: list | str = []
        try:
            if kind == "lines":
                for i, line in enumerate(data):
                    m = rx.search(line)
                    if m:
                        out.append((i, m.start(), m.end()))
                        if len(out) >= limit:
                            break
            else:
                for m in rx.finditer(data):
                    out.append((m.start(), m.end()))
                    if len(out) >= limit:
                        break
        except Exception as exc:  # (RecursionError, MemoryError…): the caller reports it instead of waiting
            out = f"{type(exc).__name__}: {exc}"
        try:
            conn.send(out)
        except (EOFError, OSError):
            return

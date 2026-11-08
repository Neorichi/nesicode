"""Context window and output limits per model, cached per user (provider → table → default).

A limit `learned` from a provider error ("maximum context length is N") is the truth for that endpoint: it never
expires and no table/default/probe value replaces it — only the provider reporting a SMALLER window does, or the user
forgetting it (forget_learned: e.g. a local server restarted with a bigger window)."""

from __future__ import annotations

import ipaddress
import sys
import threading
import time
from contextlib import contextmanager, nullcontext

from ..llm import DEFAULT_LIMITS, fetch_limits, known_limits, server_provider_config
from ..llm.transport import address_problem, private_providers_allowed, proxied
from .auth import user_keys
from .db import DB

TTL = 24 * 3600


def model_limits(db: DB, user_id: int, spec: str, refresh: bool = False) -> dict:
    """{"context", "output", "source"} for `provider/model`. source: learned | provider | table | default."""
    if "/" not in spec:
        return {"context": DEFAULT_LIMITS[0], "output": DEFAULT_LIMITS[1], "source": "default"}
    name, model = spec.split("/", 1)
    row = db.one("SELECT * FROM model_limits WHERE user_id=? AND spec=?", (user_id, spec))
    learned = row if row and row["source"] == "learned" else None
    if learned and not refresh:
        return {"context": learned["context"], "output": learned["output"] or (known_limits(model) or DEFAULT_LIMITS)[1],
                "source": "learned"}
    if row and not learned and not refresh and time.time() - row["updated_at"] < TTL:
        return {"context": row["context"], "output": row["output"], "source": row["source"]}
    entry = user_keys(db, user_id).get(name)
    mine = isinstance(entry, dict)  # the user's own endpoint (vetted where it connects); else the operator's built-in one
    cfg = entry if mine else server_provider_config(name)
    got = None
    if cfg and name != "mock":
        with vetted(cfg["base_url"]) if mine else nullcontext():
            got = fetch_limits(cfg["kind"], cfg["base_url"], cfg.get("api_key", ""), model)
    if got and got.get("context"):
        out = {"context": got["context"], "output": got.get("output") or (known_limits(model) or DEFAULT_LIMITS)[1],
               "source": "provider"}
    elif (k := known_limits(model)):
        out = {"context": k[0], "output": k[1], "source": "table"}
    else:
        out = {"context": DEFAULT_LIMITS[0], "output": DEFAULT_LIMITS[1], "source": "default"}
    if learned and not (out["source"] == "provider" and out["context"] < learned["context"]):
        # what the provider proved stays (only a smaller window reported by the provider replaces it)
        out = {"context": learned["context"], "output": learned["output"] or out["output"], "source": "learned"}
    db.execute(
        "INSERT INTO model_limits (user_id, spec, context, output, source, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(user_id, spec) DO UPDATE SET context=excluded.context, output=excluded.output, "
        "source=excluded.source, updated_at=excluded.updated_at",
        (user_id, spec, out["context"], out["output"], out["source"], time.time()),
    )
    return out


def learn_limit(db: DB, user_id: int, spec: str, context: int) -> None:
    """A provider error revealed the real context window: store it (overrides tables/estimates, never expires)."""
    cur = db.one("SELECT output FROM model_limits WHERE user_id=? AND spec=?", (user_id, spec)) or {}
    db.execute(
        "INSERT INTO model_limits (user_id, spec, context, output, source, updated_at) VALUES (?, ?, ?, ?, 'learned', ?) "
        "ON CONFLICT(user_id, spec) DO UPDATE SET context=excluded.context, source='learned', updated_at=excluded.updated_at",
        (user_id, spec, context, cur.get("output"), time.time()),
    )


def forget_learned(db: DB, user_id: int, spec: str) -> bool:
    """Drop a learned limit (the user's «Olvidar»): the next model_limits() asks the provider again, or falls back to
    the table / default. True if there was one."""
    return db.execute("DELETE FROM model_limits WHERE user_id=? AND spec=? AND source='learned'",
                      (user_id, spec)).rowcount > 0


def refresh_provider_limits(db: DB, user_id: int, provider: str, models: list[str]) -> None:
    """Called when a provider is added or its models are refreshed (runs in the background). Learned limits stay."""
    db.execute("DELETE FROM model_limits WHERE user_id=? AND spec LIKE ? AND source<>'learned'", (user_id, provider + "/%"))
    for model in models[:60]:
        try:
            model_limits(db, user_id, f"{provider}/{model}", refresh=True)
        except Exception:
            pass


# ------------------------------------------------- user endpoints: vetted where each connection really goes
def endpoint_problem(ip: str) -> str:
    """Why a user-defined endpoint may not be reached at `ip` on a server that forbids private networks ('' = fine):
    the LLM layer's rule (link-local, multicast, unspecified, private, loopback, reserved) plus anything else that is
    not public unicast (CGNAT 100.64/10, benchmarking and documentation ranges…). The same rule when an endpoint is
    saved (app.check_base_url) and when it is connected to (vetted)."""
    problem = address_problem(ip, strict=True)
    if problem:
        return problem
    addr = ipaddress.ip_address(ip.split("%", 1)[0])
    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return "" if addr.is_global and not addr.is_multicast else f"{addr} is not a public address"


class RefusedAddress(ConnectionRefusedError):
    """Raised in place of a connection to a refused address (an OSError: HTTP clients report a plain connect error)."""


_vetting = threading.local()  # .policy: ip -> problem, while this thread is inside vetted()
_hook_lock = threading.Lock()
_hooked = False


def _audit(event: str, args: tuple) -> None:
    # Runs for every audited event of the process: return at once unless this thread is inside vetted().
    if event != "socket.connect":
        return
    policy = getattr(_vetting, "policy", None)
    if policy is None:
        return
    address = args[1] if len(args) > 1 else None
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        # a host name instead of an address (a connect that resolves by itself) is refused too: nothing to vet
        problem = policy(address[0])
        if problem:
            raise RefusedAddress(f"refused to connect: {problem} (private networks are not allowed on this server)")


def _install_audit_hook() -> None:
    global _hooked
    with _hook_lock:
        if not _hooked:  # (audit hooks cannot be removed: installed once, the first time it is needed)
            sys.addaudithook(_audit)
            _hooked = True


@contextmanager
def vetted(url: str):
    """Requests made in this block go to a USER-defined endpoint (model lists, limits and limit probes, transcription).
    With NESICODE_ALLOW_PRIVATE_PROVIDERS=0, every connection this thread opens meanwhile is checked against the
    address it really goes to and refused BEFORE it is made — no request, no key, no port probe: the endpoint was
    checked when it was saved, but its name can resolve somewhere else now (DNS rebinding). Chat requests do the
    same through the LLM layer (build_provider's guard_private). Through a proxy the socket talks to the proxy, which
    picks the target itself, so nothing is vetted (as in chat). The operator's built-in providers never come here."""
    target = url if "://" in (url or "") else "https://" + (url or "")
    if private_providers_allowed() or proxied(target):
        yield
        return
    _install_audit_hook()
    previous = getattr(_vetting, "policy", None)
    _vetting.policy = endpoint_problem
    try:
        yield
    finally:
        _vetting.policy = previous

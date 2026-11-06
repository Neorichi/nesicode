"""Context window and output limits per model, cached per user (provider → table → default)."""

from __future__ import annotations

import time

from ..llm import DEFAULT_LIMITS, fetch_limits, known_limits, server_provider_config
from .auth import user_keys
from .db import DB

TTL = 24 * 3600


def model_limits(db: DB, user_id: int, spec: str, refresh: bool = False) -> dict:
    """{"context", "output", "source"} for `provider/model`. source: provider | table | default."""
    if "/" not in spec:
        return {"context": DEFAULT_LIMITS[0], "output": DEFAULT_LIMITS[1], "source": "default"}
    row = db.one("SELECT * FROM model_limits WHERE user_id=? AND spec=?", (user_id, spec))
    if row and not refresh and time.time() - row["updated_at"] < TTL:
        return {"context": row["context"], "output": row["output"], "source": row["source"]}
    name, model = spec.split("/", 1)
    entry = user_keys(db, user_id).get(name)
    cfg = entry if isinstance(entry, dict) else server_provider_config(name)
    got = None
    if cfg and name != "mock":
        got = fetch_limits(cfg["kind"], cfg["base_url"], cfg.get("api_key", ""), model)
    if got and got.get("context"):
        out = {"context": got["context"], "output": got.get("output") or (known_limits(model) or DEFAULT_LIMITS)[1],
               "source": "provider"}
    elif (k := known_limits(model)):
        out = {"context": k[0], "output": k[1], "source": "table"}
    else:
        out = {"context": DEFAULT_LIMITS[0], "output": DEFAULT_LIMITS[1], "source": "default"}
    db.execute(
        "INSERT INTO model_limits (user_id, spec, context, output, source, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(user_id, spec) DO UPDATE SET context=excluded.context, output=excluded.output, "
        "source=excluded.source, updated_at=excluded.updated_at",
        (user_id, spec, out["context"], out["output"], out["source"], time.time()),
    )
    return out


def learn_limit(db: DB, user_id: int, spec: str, context: int) -> None:
    """A provider error revealed the real context window: store it (overrides tables/estimates)."""
    cur = db.one("SELECT output FROM model_limits WHERE user_id=? AND spec=?", (user_id, spec)) or {}
    db.execute(
        "INSERT INTO model_limits (user_id, spec, context, output, source, updated_at) VALUES (?, ?, ?, ?, 'learned', ?) "
        "ON CONFLICT(user_id, spec) DO UPDATE SET context=excluded.context, source='learned', updated_at=excluded.updated_at",
        (user_id, spec, context, cur.get("output"), time.time()),
    )


def refresh_provider_limits(db: DB, user_id: int, provider: str, models: list[str]) -> None:
    """Called when a provider is added or its models are refreshed (runs in the background)."""
    db.execute("DELETE FROM model_limits WHERE user_id=? AND spec LIKE ?", (user_id, provider + "/%"))
    for model in models[:60]:
        try:
            model_limits(db, user_id, f"{provider}/{model}", refresh=True)
        except Exception:
            pass

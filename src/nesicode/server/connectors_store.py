"""Reading the user's MCP connectors from the DB, with secrets resolved from the vault."""

from __future__ import annotations

from ..connectors import ConnectorConfig, catalog_entry
from .db import DB, loads
from .vault import Vault


def _env_names(row: dict) -> set[str] | None:
    """The environment variables a connector's fields may set: any (None) for a built-in or remote one — they reach no
    process; a built-in reads its own by name —, and for a local (stdio) one only those its catalog entry declares, for
    exactly the catalog's command. Anything else would make its command run other code than the line it shows
    (NODE_OPTIONS, PYTHONPATH, PATH…): the API never gives a local connector such fields, and an imported one comes
    without them; only an import made before that could have left some."""
    if row["kind"] != "stdio":
        return None
    entry = catalog_entry(row.get("template") or "")
    if (not entry or entry.get("kind") != "stdio"
            or (row["command"], loads(row["args"], [])) != (entry.get("command"), entry.get("args", []))):
        return set()
    return {f["env"] for f in entry.get("fields", []) if f.get("env")}


def connector_configs(db: DB, user_id: int, enabled_only: bool = False) -> list[tuple[dict, ConnectorConfig]]:
    v = Vault(db)
    sql = "SELECT * FROM user_connectors WHERE user_id=?" + (" AND enabled=1" if enabled_only else "") + " ORDER BY id"
    out = []
    for row in db.all(sql, (user_id,)):
        headers, env, allowed = {}, {}, _env_names(row)
        for f in loads(row["fields"], []):
            value = v.get(user_id, f"connector:{row['name']}:{f['key']}") if f.get("secret") else f.get("value", "")
            if not value:
                continue
            if f.get("header"):
                headers[f["header"]] = f.get("prefix", "") + value
            if f.get("env") and (allowed is None or f["env"] in allowed):
                env[f["env"]] = value
        cfg = ConnectorConfig(
            name=row["name"], kind=row["kind"], command=row["command"], args=loads(row["args"], []),
            env=env, url=row["url"], headers=headers,
        )
        out.append((row, cfg))
    return out

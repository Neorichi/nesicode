"""Reading the user's MCP connectors from the DB, with secrets resolved from the vault."""

from __future__ import annotations

from ..connectors import ConnectorConfig
from .db import DB, loads
from .vault import Vault


def connector_configs(db: DB, user_id: int, enabled_only: bool = False) -> list[tuple[dict, ConnectorConfig]]:
    v = Vault(db)
    sql = "SELECT * FROM user_connectors WHERE user_id=?" + (" AND enabled=1" if enabled_only else "") + " ORDER BY id"
    out = []
    for row in db.all(sql, (user_id,)):
        headers, env = {}, {}
        for f in loads(row["fields"], []):
            value = v.get(user_id, f"connector:{row['name']}:{f['key']}") if f.get("secret") else f.get("value", "")
            if not value:
                continue
            if f.get("header"):
                headers[f["header"]] = f.get("prefix", "") + value
            if f.get("env"):
                env[f["env"]] = value
        cfg = ConnectorConfig(
            name=row["name"], kind=row["kind"], command=row["command"], args=loads(row["args"], []),
            env=env, url=row["url"], headers=headers,
        )
        out.append((row, cfg))
    return out

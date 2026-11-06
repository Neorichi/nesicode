"""Burp Suite Professional through its built-in REST API (v0.1) — works on old versions (2.x, 2020–2022)
that cannot run the official MCP Server extension.

Enable it in Burp: User options / Settings → Misc → REST API → "Service running" (default 127.0.0.1:1337)
and create an API key. The key goes in the URL path: http://127.0.0.1:1337/<key>/v0.1/...

The v0.1 API only covers scans: start a crawl-and-audit, poll its status and issues, and read the issue
catalog. There is no proxy history, Repeater or scan cancel in it (those need the MCP Server extension).

Each tool is `fn(args, ws, cfg) -> (text_for_the_model, written_relpath | None)`.
"""

from __future__ import annotations

import json
import re

import httpx

from .agent.workspace import ToolError
from .llm import ToolSpec

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3, "information": 3}


def _base(cfg) -> str:
    key = (cfg.env or {}).get("BURP_API_KEY", "").strip()
    return cfg.url.rstrip("/") + (f"/{key}" if key else "") + "/v0.1"


def _request(cfg, method: str, path: str, **kw) -> httpx.Response:
    try:
        r = httpx.request(method, _base(cfg) + path, timeout=30, **kw)
    except httpx.HTTPError as exc:
        raise ToolError(f"Burp REST API not reachable at {cfg.url} ({type(exc).__name__}). Is Burp open with "
                        "Settings → Suite → REST API → 'Service running' checked?") from exc
    if r.status_code in (401, 403):
        raise ToolError("Burp rejected the API key (create one in Burp → REST API → API keys)")
    if r.status_code >= 400:
        raise ToolError(f"Burp REST API error {r.status_code}: {r.text[:500]}")
    return r


def ping(cfg) -> None:
    _request(cfg, "GET", "/")


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


# ------------------------------------------------------------------ tools
def burp_scan(args: dict, ws, cfg):
    urls = [u.strip() for u in (args.get("urls") or []) if isinstance(u, str) and u.strip()]
    if not urls or not all(re.match(r"^https?://", u) for u in urls):
        raise ToolError("`urls` must be a list of http(s) URLs")
    body: dict = {"urls": urls}
    if args.get("scan_configurations"):
        body["scan_configurations"] = [{"name": n, "type": "NamedConfiguration"} for n in args["scan_configurations"]]
    include, exclude = args.get("scope_include") or [], args.get("scope_exclude") or []
    if include or exclude:
        body["scope"] = {"include": [{"rule": r, "type": "SimpleScopeDef"} for r in include] or
                         [{"rule": u, "type": "SimpleScopeDef"} for u in urls],
                         "exclude": [{"rule": r, "type": "SimpleScopeDef"} for r in exclude],
                         "type": "SimpleScope"}
    if args.get("resource_pool"):
        body["resource_pool"] = args["resource_pool"]
    r = _request(cfg, "POST", "/scan", json=body)
    task = (r.headers.get("Location") or "").rstrip("/").split("/")[-1]
    if not task:
        raise ToolError(f"Burp accepted the request but returned no task id (HTTP {r.status_code})")
    return (f"Scan started in Burp: task_id={task} for {', '.join(urls)}. It runs inside Burp (visible in its "
            f"Dashboard); poll it with burp_scan_status. Stop it from Burp's Dashboard if needed."), None


def burp_scan_status(args: dict, ws, cfg):
    task = str(args.get("task_id", "")).strip()
    if not task.isdigit():
        raise ToolError("`task_id` (a number returned by burp_scan) is required")
    data = _request(cfg, "GET", f"/scan/{task}").json()
    m = data.get("scan_metrics") or {}
    issues = [e.get("issue") or {} for e in data.get("issue_events") or [] if e.get("type") in ("issue_found", None)]
    issues.sort(key=lambda i: (SEVERITY_ORDER.get(str(i.get("severity", "")).lower(), 9), i.get("name", "")))
    lines = [f"task {task}: {data.get('scan_status', '?')} — progress {m.get('crawl_and_audit_progress', '?')}%, "
             f"requests {m.get('total_elapsed_time', '?')}s elapsed, {m.get('crawl_requests_made', 0)} crawl / "
             f"{m.get('audit_requests_made', 0)} audit requests, {len(issues)} issues"]
    counts: dict[str, int] = {}
    for i in issues:
        counts[i.get("severity", "?")] = counts.get(i.get("severity", "?"), 0) + 1
    if counts:
        lines.append("by severity: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    detail = bool(args.get("details"))
    for i in issues[: int(args.get("limit", 60))]:
        lines.append(f"- [{i.get('severity')}/{i.get('confidence')}] {i.get('name')} — {i.get('origin', '')}{i.get('path', '')} "
                     f"(serial {i.get('serial_number')})")
        if detail:
            lines.append("  " + _strip_html(i.get("description") or i.get("issue_background") or "")[:1200])
            if i.get("remediation"):
                lines.append("  Remediation: " + _strip_html(i["remediation"])[:600])
    text = "\n".join(lines)
    if issues and args.get("save_json"):
        rel = f"burp/scan-{task}.json"
        p = ws.resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return text + f"\nFull JSON saved to {rel}.", rel
    return text[:30000], None


def burp_issue_definitions(args: dict, ws, cfg):
    q = str(args.get("query", "")).lower().strip()
    defs = _request(cfg, "GET", "/knowledge_base/issue_definitions").json()
    hits = [d for d in defs if not q or q in (d.get("name", "") + " " + d.get("description", "")).lower()]
    out = [f"{len(hits)} of {len(defs)} issue definitions" + (f" matching {q!r}" if q else "")]
    for d in hits[:40]:
        out.append(f"- {d.get('name')} (type {d.get('issue_type_id')}, typical {d.get('typical_severity')})")
        if q:
            out.append("  " + _strip_html(d.get("description", ""))[:500])
    return "\n".join(out), None


_URLS = {"type": "array", "items": {"type": "string"}}
TOOLS: dict[str, tuple[ToolSpec, object]] = {
    "burp_scan": (ToolSpec(
        "burp_scan", "Start an active Burp Suite crawl-and-audit scan (sends many real requests, including attack "
        "payloads) against the given URLs. ONLY for targets the user owns or is explicitly authorized to test. "
        "Optional named scan configurations from Burp's library (e.g. 'Crawl strategy - fastest', "
        "'Audit checks - light active', 'Audit checks - passive') and scope rules.",
        {"type": "object", "properties": {"urls": _URLS, "scan_configurations": _URLS, "scope_include": _URLS,
                                          "scope_exclude": _URLS, "resource_pool": {"type": "string"}},
         "required": ["urls"]}), burp_scan),
    "burp_scan_status": (ToolSpec(
        "burp_scan_status", "Status, progress and issues found so far for a Burp scan task (sorted by severity). "
        "details:true adds descriptions and remediation; save_json:true saves the full result to burp/scan-<id>.json.",
        {"type": "object", "properties": {"task_id": {"type": "string"}, "details": {"type": "boolean"},
                                          "limit": {"type": "integer"}, "save_json": {"type": "boolean"}},
         "required": ["task_id"]}), burp_scan_status),
    "burp_issue_definitions": (ToolSpec(
        "burp_issue_definitions", "Search Burp's catalog of issue types (name, typical severity, description).",
        {"type": "object", "properties": {"query": {"type": "string"}}}), burp_issue_definitions),
}

"""VirusTotal, built into Nesicode (official REST API v3, your API key; nothing to install).

Lookups (hash, URL, domain, IP) only read VirusTotal's existing reports. Submitting a URL or uploading a
file shares it with VirusTotal (and its community): those two always ask the user first.
Each tool is `fn(args, ws, cfg) -> (text_for_the_model, written_relpath | None)`.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone

import httpx

from .agent.workspace import ToolError
from .llm import ToolSpec

API = "https://www.virustotal.com/api/v3"
GUI = "https://www.virustotal.com/gui"
EICAR = "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"
MAX_UPLOAD = 32 * 1024 * 1024
transport: httpx.BaseTransport | None = None  # tests plug a mock in here

# The public (free) API allows 4 requests a minute and 500 a day. Analyses are polled every POLL_EVERY seconds
# (VirusTotal's own client waits 20 s) and never faster than the per-minute quota, counting every request made
# with that key (the lookup and the upload before the polls too).
PER_MINUTE = 4
POLL_EVERY = 20.0
FIRST_POLL = 15.0  # an analysis is never ready right after the submission
_recent: dict[str, collections.deque] = {}  # key fingerprint -> times of the last PER_MINUTE requests
_recent_lock = threading.Lock()


class RateLimited(ToolError):
    """HTTP 429: the API key's quota is used up for now."""


def _clock() -> float:
    return time.monotonic()


def _pause(ws, seconds: float) -> bool:
    """Wait; True if the user pressed Stop meanwhile (the run's cancel event lives on its workspace)."""
    cancel = getattr(ws, "cancel", None)
    if cancel is not None:
        return cancel.wait(max(0.0, seconds))
    time.sleep(max(0.0, seconds))
    return False


def _key(cfg) -> str:
    key = (cfg.env or {}).get("VT_API_KEY", "").strip()
    if not key:
        raise ToolError("falta la API key de VirusTotal (Integraciones → VirusTotal)")
    return key


def _fingerprint(cfg) -> str:
    return hashlib.sha256(_key(cfg).encode()).hexdigest()[:16]


def _slot_in(cfg) -> float:
    """Seconds until one more request fits in the per-minute quota of this key."""
    with _recent_lock:
        q = _recent.get(_fingerprint(cfg))
        if not q or len(q) < PER_MINUTE:
            return 0.0
        return max(0.0, q[0] + 61.0 - _clock())


def _req(cfg, method: str, path: str, **kw) -> dict | None:
    key = _key(cfg)
    with _recent_lock:
        _recent.setdefault(_fingerprint(cfg), collections.deque(maxlen=PER_MINUTE)).append(_clock())
    try:
        with httpx.Client(timeout=60, transport=transport) as c:
            r = c.request(method, API + path, headers={"x-apikey": key, "accept": "application/json"}, **kw)
    except httpx.HTTPError as exc:
        raise ToolError(f"no se pudo conectar con VirusTotal: {exc}") from exc
    if r.status_code == 404:
        return None
    if r.status_code == 401:
        raise ToolError("VirusTotal rechazó la API key")
    if r.status_code == 429:
        raise RateLimited("VirusTotal: límite de consultas alcanzado (la API gratuita permite 4 por minuto y 500 al día). Espera un poco.")
    if r.status_code >= 400:
        raise ToolError(f"VirusTotal HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def ping(cfg) -> None:
    _req(cfg, "GET", f"/files/{EICAR}")


def _date(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") if ts else "?"


def _verdict(attrs: dict) -> list[str]:
    st = attrs.get("last_analysis_stats") or attrs.get("stats") or {}
    mal, sus = st.get("malicious", 0), st.get("suspicious", 0)
    total = sum(v for v in st.values() if isinstance(v, int))
    verdict = "MALICIOUS" if mal >= 3 else "suspicious" if mal or sus else "clean (no engine flags it)"
    out = [f"Verdict: {verdict} — {mal} malicious, {sus} suspicious, of {total} engines"]
    results = attrs.get("last_analysis_results") or attrs.get("results") or {}
    flagged = [f"{eng}: {r.get('result')}" for eng, r in results.items() if r.get("category") in ("malicious", "suspicious")]
    if flagged:
        out.append("Flagged by: " + "; ".join(flagged[:12]) + (f" (+{len(flagged) - 12} more)" if len(flagged) > 12 else ""))
    if "reputation" in attrs:
        out.append(f"Community reputation: {attrs['reputation']}")
    if attrs.get("tags"):
        out.append("Tags: " + ", ".join(attrs["tags"][:15]))
    return out


def _clean(text: str, field: str) -> str:
    text = text.strip()
    if not text:
        raise ToolError(f"`{field}` es obligatorio")
    return text


# ------------------------------------------------------------------ lookups
def vt_file_report(args, ws, cfg):
    h = _clean(str(args.get("hash", "")), "hash").lower()
    if not re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64}", h):
        raise ToolError("`hash` debe ser MD5, SHA-1 o SHA-256")
    data = _req(cfg, "GET", f"/files/{h}")
    if not data:
        return f"VirusTotal has no report for {h} (never seen). You can upload the file with vt_scan_file.", None
    a = data["data"]["attributes"]
    lines = [f"File {a.get('meaningful_name') or (a.get('names') or ['?'])[0]} · {a.get('type_description', '?')} · "
             f"{a.get('size', '?')} bytes", f"SHA-256 {a.get('sha256')}",
             f"First seen {_date(a.get('first_submission_date'))}, last analysed {_date(a.get('last_analysis_date'))}"]
    lines += _verdict(a)
    if a.get("popular_threat_classification", {}).get("suggested_threat_label"):
        lines.append("Threat label: " + a["popular_threat_classification"]["suggested_threat_label"])
    lines.append(f"Report: {GUI}/file/{a.get('sha256', h)}")
    return "\n".join(lines), None


def vt_url_report(args, ws, cfg):
    url = _clean(str(args.get("url", "")), "url")
    uid = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    data = _req(cfg, "GET", f"/urls/{uid}")
    if not data:
        return f"VirusTotal has no report for {url}. Submitting it (vt_scan_url) shares it with VirusTotal — ask the user first.", None
    a = data["data"]["attributes"]
    lines = [f"URL {a.get('url', url)}", f"Last analysed {_date(a.get('last_analysis_date'))}"]
    lines += _verdict(a)
    if a.get("categories"):
        lines.append("Categories: " + ", ".join(sorted(set(a["categories"].values())))[:300])
    if a.get("last_final_url") and a["last_final_url"] != url:
        lines.append("Redirects to: " + a["last_final_url"])
    lines.append(f"Report: {GUI}/url/{uid}")
    return "\n".join(lines), None


def vt_domain_report(args, ws, cfg):
    d = _clean(str(args.get("domain", "")), "domain").lower()
    d = re.sub(r"^https?://", "", d).split("/")[0]
    data = _req(cfg, "GET", f"/domains/{d}")
    if not data:
        return f"VirusTotal has no report for {d}.", None
    a = data["data"]["attributes"]
    lines = [f"Domain {d}", f"Registrar: {a.get('registrar', '?')} · created {_date(a.get('creation_date'))}"]
    lines += _verdict(a)
    if a.get("categories"):
        lines.append("Categories: " + ", ".join(sorted(set(a["categories"].values())))[:300])
    recs = [f"{r.get('type')} {r.get('value')}" for r in (a.get("last_dns_records") or [])[:8]]
    if recs:
        lines.append("DNS: " + "; ".join(recs))
    lines.append(f"Report: {GUI}/domain/{d}")
    return "\n".join(lines), None


def vt_ip_report(args, ws, cfg):
    ip = _clean(str(args.get("ip", "")), "ip")
    data = _req(cfg, "GET", f"/ip_addresses/{ip}")
    if not data:
        return f"VirusTotal has no report for {ip}.", None
    a = data["data"]["attributes"]
    lines = [f"IP {ip} · {a.get('as_owner', '?')} (AS{a.get('asn', '?')}) · {a.get('country', '?')}"]
    lines += _verdict(a)
    lines.append(f"Report: {GUI}/ip-address/{ip}")
    return "\n".join(lines), None


RELATIONS = {
    "domain": ("domains", ["subdomains", "siblings", "resolutions", "urls", "communicating_files", "referrer_files"]),
    "ip": ("ip_addresses", ["resolutions", "urls", "communicating_files", "referrer_files"]),
}


def vt_relations(args, ws, cfg):
    kind = str(args.get("type") or "domain").lower()
    if kind not in RELATIONS:
        raise ToolError("`type` es domain o ip")
    target = _clean(str(args.get("target", "")), "target").lower()
    target = re.sub(r"^https?://", "", target).split("/")[0]
    coll, allowed = RELATIONS[kind]
    rel = str(args.get("relation") or "subdomains")
    if rel not in allowed:
        raise ToolError(f"`relation` para {kind}: {', '.join(allowed)}")
    limit = max(1, min(int(args.get("limit") or 40), 40))
    cursor = str(args.get("cursor") or "")
    params = {"limit": limit} | ({"cursor": cursor} if cursor else {})
    data = _req(cfg, "GET", f"/{coll}/{target}/{rel}", params=params) or {}
    items = data.get("data") or []
    lines = [f"{rel} of {target}: {len(items)} shown" + (" (more available: pass cursor)" if data.get("meta", {}).get("cursor") else "")]
    for it in items:
        a = it.get("attributes") or {}
        st = a.get("last_analysis_stats") or {}
        flag = f" · {st.get('malicious', 0)} malicious" if st.get("malicious") else ""
        if rel == "resolutions":
            lines.append(f"- {a.get('host_name', '?')} → {a.get('ip_address', '?')} (seen {_date(a.get('date'))})")
        elif rel in ("urls",):
            lines.append(f"- {a.get('url', it.get('id'))}{flag}")
        elif rel.endswith("files"):
            lines.append(f"- {a.get('meaningful_name') or it.get('id')} ({a.get('type_description', '?')}){flag}")
        else:
            lines.append(f"- {it.get('id')}{flag}")
    if data.get("meta", {}).get("cursor"):
        lines.append(f"cursor: {data['meta']['cursor']}")
    return "\n".join(lines), None


# ------------------------------------------------------------------ submissions (shared with VirusTotal: always asked)
def _wait_analysis(cfg, ws, analysis_id: str, timeout: float = 90) -> tuple[dict, str]:
    """Poll an analysis within the quota. Returns (attributes, "") once it is completed, or (the last attributes
    seen, why it is not) when it is still running at the deadline, the user pressed Stop or polling failed. The
    submission already happened, so a busy quota (429) means waiting longer, never an error that loses it."""
    deadline = _clock() + timeout
    a: dict = {}
    wait = FIRST_POLL
    while True:
        wait = max(wait, _slot_in(cfg))
        if _clock() + wait > deadline:
            return a, f"still running after {timeout:.0f} s"
        if _pause(ws, wait):
            return a, "the user pressed Stop before it finished"
        try:
            data = _req(cfg, "GET", f"/analyses/{analysis_id}") or {}
        except RateLimited:
            wait = 60.0  # quota used up (other requests with this key too): back off, then try again
            continue
        except ToolError as exc:
            return a, f"could not check it ({exc})"
        a = (data.get("data") or {}).get("attributes") or {}
        if a.get("status") == "completed":
            return a, ""
        wait = POLL_EVERY


def _not_finished(a: dict, why: str, analysis_id: str, how: str, report: str) -> list[str]:
    """What to say about an analysis without a final result: never a verdict (0 engines is not 'clean')."""
    return [f"Analysis not finished — status: {a.get('status') or 'queued'}, {why}. NO VERDICT YET: do not report "
            "it as clean or malicious.", f"Analysis id: {analysis_id}",
            f"Check the result later with {how} (a free lookup, no new submission), or open {report}"]


def _submitted_id(data: dict | None) -> str:
    aid = ((data or {}).get("data") or {}).get("id")
    if not aid:
        raise ToolError(f"VirusTotal no devolvió el id del análisis: {json.dumps(data)[:200]}")
    return aid


def vt_scan_url(args, ws, cfg):
    url = _clean(str(args.get("url", "")), "url")
    aid = _submitted_id(_req(cfg, "POST", "/urls", data={"url": url}))
    a, why = _wait_analysis(cfg, ws, aid)
    uid = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    report = f"{GUI}/url/{uid}"
    if why:
        lines = [f"Submitted {url} to VirusTotal."] + _not_finished(a, why, aid, f"vt_url_report (url {url})", report)
    else:
        lines = [f"Submitted {url} to VirusTotal (status: completed)"] + _verdict(a) + [f"Report: {report}"]
    return "\n".join(lines), None


def vt_scan_file(args, ws, cfg):
    rel = _clean(str(args.get("path", "")), "path")
    p = ws.resolve(rel)
    if not p.is_file():
        raise ToolError(f"no existe el archivo {rel}")
    if p.stat().st_size > MAX_UPLOAD:
        raise ToolError("archivo demasiado grande (máx. 32 MB con la API estándar)")
    # read once, through the workspace: never a link swapped in meanwhile, never a FIFO (its read would never end)
    data = ws.read_bytes(p, MAX_UPLOAD + 1)
    size = len(data)
    if size > MAX_UPLOAD:
        raise ToolError("archivo demasiado grande (máx. 32 MB con la API estándar)")
    sha = hashlib.sha256(data).hexdigest()
    known = _req(cfg, "GET", f"/files/{sha}")
    if known:  # already known: no need to upload anything
        text, _ = vt_file_report({"hash": sha}, ws, cfg)
        return "Already known to VirusTotal (not uploaded again).\n" + text, None
    aid = _submitted_id(_req(cfg, "POST", "/files", files={"file": (p.name, data)}))
    a, why = _wait_analysis(cfg, ws, aid, timeout=150)
    report = f"{GUI}/file/{sha}"
    lines = [f"Uploaded {rel} ({size} bytes) to VirusTotal", f"SHA-256 {sha}"]
    if why:
        lines += _not_finished(a, why, aid, f"vt_file_report (hash {sha})", report)
    else:
        lines[0] += " — status: completed"
        lines += _verdict(a) + [f"Report: {report}"]
    return "\n".join(lines), None


S = lambda props, req: {"type": "object", "properties": props, "required": req}  # noqa: E731
STR = {"type": "string"}
TOOLS: dict[str, tuple[ToolSpec, object]] = {
    "vt_file_report": (ToolSpec("vt_file_report", "VirusTotal report for a file hash (MD5/SHA-1/SHA-256): verdict by "
                                "antivirus engines, names, type, first seen, threat label.", S({"hash": STR}, ["hash"])), vt_file_report),
    "vt_url_report": (ToolSpec("vt_url_report", "VirusTotal's existing report for a URL (does not submit it).",
                               S({"url": STR}, ["url"])), vt_url_report),
    "vt_domain_report": (ToolSpec("vt_domain_report", "VirusTotal report for a domain: verdict, categories, registrar, DNS.",
                                  S({"domain": STR}, ["domain"])), vt_domain_report),
    "vt_ip_report": (ToolSpec("vt_ip_report", "VirusTotal report for an IP address: verdict, owner/ASN, country.",
                              S({"ip": STR}, ["ip"])), vt_ip_report),
    "vt_relations": (ToolSpec("vt_relations", "VirusTotal relationships of a domain or IP: subdomains, sibling domains, "
                              "DNS resolutions (passive DNS), URLs and related files. Paginate with `cursor`.",
                              S({"type": {"type": "string", "enum": ["domain", "ip"]}, "target": STR,
                                 "relation": {"type": "string", "enum": ["subdomains", "siblings", "resolutions", "urls",
                                                                         "communicating_files", "referrer_files"]},
                                 "limit": {"type": "integer"}, "cursor": STR}, ["target", "relation"])), vt_relations),
    "vt_scan_url": (ToolSpec("vt_scan_url", "Submit a URL to VirusTotal for a fresh scan and wait (up to ~1.5 min) for "
                             "the verdict; if the analysis is still running it says so: check it later with vt_url_report "
                             "instead of submitting again. It is shared with VirusTotal: the user approves every submission.",
                             S({"url": STR}, ["url"])), vt_scan_url),
    "vt_scan_file": (ToolSpec("vt_scan_file", "Upload a workspace file (max 32 MB) to VirusTotal and wait (up to ~2.5 min) "
                              "for the verdict; if the analysis is still running it says so: check it later with "
                              "vt_file_report (its SHA-256) instead of uploading again. The file is shared with VirusTotal and "
                              "its community: the user approves every upload. Known files are only looked up by hash, never "
                              "re-uploaded.", S({"path": STR}, ["path"])), vt_scan_file),
}

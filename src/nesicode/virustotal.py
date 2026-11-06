"""VirusTotal, built into Nesicode (official REST API v3, your API key; nothing to install).

Lookups (hash, URL, domain, IP) only read VirusTotal's existing reports. Submitting a URL or uploading a
file shares it with VirusTotal (and its community): those two always ask the user first.
Each tool is `fn(args, ws, cfg) -> (text_for_the_model, written_relpath | None)`.
"""

from __future__ import annotations

import base64
import json
import re
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


def _key(cfg) -> str:
    key = (cfg.env or {}).get("VT_API_KEY", "").strip()
    if not key:
        raise ToolError("falta la API key de VirusTotal (Integraciones → VirusTotal)")
    return key


def _req(cfg, method: str, path: str, **kw) -> dict | None:
    try:
        with httpx.Client(timeout=60, transport=transport) as c:
            r = c.request(method, API + path, headers={"x-apikey": _key(cfg), "accept": "application/json"}, **kw)
    except httpx.HTTPError as exc:
        raise ToolError(f"no se pudo conectar con VirusTotal: {exc}") from exc
    if r.status_code == 404:
        return None
    if r.status_code == 401:
        raise ToolError("VirusTotal rechazó la API key")
    if r.status_code == 429:
        raise ToolError("VirusTotal: límite de consultas alcanzado (la API gratuita permite 4 por minuto y 500 al día). Espera un poco.")
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


# ------------------------------------------------------------------ submissions (shared with VirusTotal: always asked)
def _wait_analysis(cfg, analysis_id: str, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while True:
        data = _req(cfg, "GET", f"/analyses/{analysis_id}") or {}
        a = (data.get("data") or {}).get("attributes") or {}
        if a.get("status") == "completed" or time.time() > deadline:
            return a
        time.sleep(8)


def vt_scan_url(args, ws, cfg):
    url = _clean(str(args.get("url", "")), "url")
    data = _req(cfg, "POST", "/urls", data={"url": url}) or {}
    a = _wait_analysis(cfg, data["data"]["id"])
    lines = [f"Submitted {url} to VirusTotal (status: {a.get('status', 'queued')})"] + _verdict(a)
    uid = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    lines.append(f"Report: {GUI}/url/{uid}")
    return "\n".join(lines), None


def vt_scan_file(args, ws, cfg):
    rel = _clean(str(args.get("path", "")), "path")
    p = ws.resolve(rel)
    if not p.is_file():
        raise ToolError(f"no existe el archivo {rel}")
    size = p.stat().st_size
    if size > MAX_UPLOAD:
        raise ToolError("archivo demasiado grande (máx. 32 MB con la API estándar)")
    import hashlib

    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    known = _req(cfg, "GET", f"/files/{sha}")
    if known:  # already known: no need to upload anything
        text, _ = vt_file_report({"hash": sha}, ws, cfg)
        return "Already known to VirusTotal (not uploaded again).\n" + text, None
    with p.open("rb") as fh:
        data = _req(cfg, "POST", "/files", files={"file": (p.name, fh)}) or {}
    a = _wait_analysis(cfg, data["data"]["id"], timeout=150)
    lines = [f"Uploaded {rel} ({size} bytes) to VirusTotal — status: {a.get('status', 'queued')}", f"SHA-256 {sha}"] + _verdict(a)
    lines.append(f"Report: {GUI}/file/{sha}")
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
    "vt_scan_url": (ToolSpec("vt_scan_url", "Submit a URL to VirusTotal for a fresh scan. It is shared with VirusTotal: "
                             "the user approves every submission.", S({"url": STR}, ["url"])), vt_scan_url),
    "vt_scan_file": (ToolSpec("vt_scan_file", "Upload a workspace file (max 32 MB) to VirusTotal and wait for the verdict. "
                              "The file is shared with VirusTotal and its community: the user approves every upload. "
                              "Known files are only looked up by hash, never re-uploaded.", S({"path": STR}, ["path"])), vt_scan_file),
}

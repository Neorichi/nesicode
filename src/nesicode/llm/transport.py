"""HTTP plumbing shared by the model providers.

- `Call`: the sockets of one request, so Stop or an idle timeout can abort it from another thread at once.
  shutdown(SHUT_RDWR) wakes a blocked recv on every platform (close() alone does not on Linux) and the server sees
  the disconnect, so it stops generating (and billing).
- `pump`: runs the blocking request on a helper thread while the caller's thread forwards its deltas to `on_event`
  (which may raise the agent's Cancelled), watches Stop and enforces the first-byte / idle timeouts. There is no
  total-duration limit while data flows.
- `with_retries`: exponential backoff with jitter, honours Retry-After, notifies before each retry, never sleeps
  after the last attempt, and every wait can be cancelled.
- The private-network guard for user-defined endpoints and for image URLs returned by providers.
"""

from __future__ import annotations

import email.utils
import ipaddress
import os
import queue
import random
import socket
import threading
import time
import urllib.request
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

from .base import LLMError

CANCELLED = "cancelled by the user"
CONNECT_TIMEOUT = 15.0
DEFAULT_ATTEMPTS = 5
BACKOFF_BASE = 2.0  # seconds before the 2nd attempt; doubles each time (±25% jitter), capped
BACKOFF_CAP = 30.0
RETRY_AFTER_CAP = 60.0  # never wait longer than this before one retry, whatever Retry-After says
RETRY_AFTER_GIVE_UP = 300.0  # a provider asking for a longer wait (quota exhausted) is not retried
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
IMAGE_MAX_BYTES = 20 * 1024 * 1024


def _env_seconds(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def first_byte_timeout() -> float:
    """Longest wait for a streamed reply to start (prompt processing, reasoning before the first token)."""
    return _env_seconds("NESICODE_LLM_FIRST_BYTE_TIMEOUT", 300.0)


def idle_timeout() -> float:
    """Longest silence between two chunks once a streamed reply has started."""
    return _env_seconds("NESICODE_LLM_IDLE_TIMEOUT", 180.0)


class Retryable(Exception):
    """A failure worth another attempt: connection error, timeout, 429/5xx, a reply stream cut short."""

    def __init__(self, reason: str, retry_after: float | None = None, partial: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after
        self.partial = partial  # part of the reply had already been streamed to the caller


def retry_after_seconds(headers) -> float | None:
    """retry-after-ms / Retry-After (seconds or an HTTP date); None when absent or unreadable."""
    if headers is None:
        return None
    try:
        ms = headers.get("retry-after-ms")
        if ms:
            return max(0.0, float(ms) / 1000)
        value = (headers.get("retry-after") or "").strip()
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
    except Exception:  # noqa: BLE001 — a malformed header just means "no hint"
        return None


def backoff(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait after failed attempt number `attempt` (1-based)."""
    if retry_after is not None:
        return min(RETRY_AFTER_CAP, retry_after) * random.uniform(1.0, 1.1)
    return min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (attempt - 1)) * random.uniform(0.75, 1.25)


def wait_or_cancel(cancel, seconds: float) -> None:
    if cancel is None:
        time.sleep(max(0.0, seconds))
    elif cancel.wait(max(0.0, seconds)):
        raise LLMError(CANCELLED)


def notify_safely(notify, info: dict) -> None:
    if notify is None:
        return
    try:
        notify(info)
    except Exception:  # noqa: BLE001 — a broken listener must never break the request
        pass


def with_retries(attempt: Callable[[int], Any], *, label: str, cancel=None, notify=None,
                 attempts: int | None = None) -> Any:
    """Run attempt(n) until it succeeds. `Retryable` failures are retried (after notify), anything else propagates."""
    total = max(1, int(attempts or DEFAULT_ATTEMPTS))
    last = Retryable("no attempt was made")
    n = 0
    for n in range(1, total + 1):
        if cancel is not None and cancel.is_set():
            raise LLMError(CANCELLED)
        try:
            return attempt(n)
        except Retryable as exc:
            last = exc
        if n == total:
            break
        if last.retry_after is not None and last.retry_after > RETRY_AFTER_GIVE_UP:
            raise LLMError(f"{label} request failed: {last.reason} (the provider asks to wait {last.retry_after:.0f} s)")
        wait = backoff(n, last.retry_after)
        notify_safely(notify, {"type": "llm_retry", "attempt": n + 1, "max": total, "wait": round(wait, 1),
                               "reason": last.reason[:300], "partial": last.partial})
        wait_or_cancel(cancel, wait)
    raise LLMError(f"{label} request failed: {last.reason}" + (f" (after {n} attempts)" if n > 1 else ""))


# ------------------------------------------------------------------ abortable requests
def _shutdown(sock) -> None:
    try:
        # the plain socket method also for TLS sockets: wakes a read blocked in another thread and sends FIN
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except (OSError, ValueError, TypeError):
        pass


class Call:
    """One request's sockets, so another thread can abort it at once. httpx/httpcore report each new connection
    through the `trace` request extension; pass `call.trace` there. `policy(ip) -> problem` (optional) vets the
    address actually connected to, before anything (key included) is sent: no DNS-rebinding window."""

    def __init__(self, policy: Callable[[str], str] | None = None):
        self._lock = threading.Lock()
        self._socks: list = []
        self.policy = policy
        self.reason = ""  # why it was aborted: cancel | idle | caller | blocked
        self.blocked = ""  # the policy refused the address it connected to
        self.last = time.monotonic()  # last bytes received
        self.data = False  # the reply has started: the idle limit replaces the first-byte limit
        self.emitted = 0  # deltas handed to the caller

    def touch(self) -> None:
        self.last = time.monotonic()

    def trace(self, name: str, info: dict) -> None:
        if not name.endswith((".connect_tcp.complete", ".start_tls.complete")):
            return
        stream = (info or {}).get("return_value")
        try:
            sock = stream.get_extra_info("socket") if stream is not None else None
        except Exception:  # noqa: BLE001
            sock = None
        if sock is None:
            return
        with self._lock:
            self._socks.append(sock)
            aborted = bool(self.reason)
        if aborted:
            _shutdown(sock)
            return
        if self.policy is not None and name.endswith(".connect_tcp.complete"):
            try:
                peer = sock.getpeername()[0]
            except (OSError, IndexError, TypeError):
                peer = ""
            problem = self.policy(peer) if peer else ""
            if problem:
                self.blocked = f"refused to connect: {problem} (private networks are not allowed on this server)"
                self.abort("blocked")

    def abort(self, reason: str) -> None:
        with self._lock:
            if not self.reason:
                self.reason = reason
            socks = list(self._socks)
        for sock in socks:
            _shutdown(sock)


def pump(produce: Callable[[Call, Callable[[str, str], None]], Any], call: Call, *, cancel=None,
         on_event: Callable[[str, str], None] | None = None, first_byte: float | None = None,
         idle: float | None = None) -> Any:
    """Run produce(call, emit) on a helper thread and wait for it here. emit(kind, text) deltas reach `on_event` in
    THIS thread, so the agent's Cancelled propagates unchanged. Stop aborts the request within ~0.1 s, also while it
    connects; with first_byte/idle, a reply that stays silent too long is aborted and raised as Retryable."""
    q: queue.SimpleQueue = queue.SimpleQueue()

    def work() -> None:
        try:
            q.put(("done", produce(call, lambda kind, text: q.put(("event", kind, text)))))
        except BaseException as exc:  # noqa: BLE001 — handed to the caller's thread
            q.put(("error", exc))

    threading.Thread(target=work, daemon=True, name="nesi-llm-request").start()
    try:
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                item = None
            if cancel is not None and cancel.is_set():
                call.abort("cancel")
                raise LLMError(CANCELLED)
            if item is None:
                limit = idle if call.data else first_byte
                if limit and time.monotonic() - call.last > limit:
                    call.abort("idle")
                    what = "the reply stalled" if call.data else "the reply did not start"
                    raise Retryable(f"{what}: no data for {limit:.0f} s", partial=call.emitted > 0)
                continue
            if item[0] == "event":
                if on_event is not None:
                    call.emitted += 1
                    on_event(item[1], item[2])
            elif item[0] == "done":
                return item[1]
            else:
                raise item[1]
    except BaseException:
        call.abort("caller")  # whatever happened here, the abandoned request must not keep running
        raise


# ------------------------------------------------------------------ private-network guard
def private_providers_allowed() -> bool:
    """NESICODE_ALLOW_PRIVATE_PROVIDERS=0 (multi-user servers): user endpoints may not reach private networks."""
    return os.environ.get("NESICODE_ALLOW_PRIVATE_PROVIDERS", "1") != "0"


def address_problem(ip: str, strict: bool) -> str:
    """Why `ip` must not be contacted ("" = fine). Link-local (cloud metadata), multicast and unspecified addresses
    are always refused; strict also refuses private, loopback and reserved ones."""
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return f"invalid address {ip!r}"
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    if addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return f"{addr} is a link-local, multicast or unspecified address"
    if strict and (addr.is_private or addr.is_loopback or addr.is_reserved):
        return f"{addr} is a private or local address"
    return ""


def _routing(lib: str):
    """(proxy mounts from the environment, URLPattern, URL) exactly as the library that makes the request builds and
    matches them: httpx for the providers, model lists, probes and image downloads; the Anthropic SDK reads the
    environment its own way and its client (httpx2) matches the mounts."""
    if lib == "anthropic":
        import httpx2
        from anthropic._utils._httpx import get_environment_proxies

        return get_environment_proxies(), httpx2._utils.URLPattern, httpx2.URL
    import httpx

    return httpx._utils.get_environment_proxies(), httpx._utils.URLPattern, httpx.URL


def proxied(url: str, lib: str = "httpx") -> bool:
    """True when the HTTP library that makes the request (`lib`: "httpx", or "anthropic" for the SDK) sends this URL
    through a proxy: the socket then talks to the proxy, so its address says nothing about the target. Decided with
    that library's own rules (HTTP(S)_PROXY / ALL_PROXY, else the system's proxy settings, minus NO_PROXY), never
    urllib's: they differ (`localhost,*` means no proxy at all, `host:port` and `scheme://host` entries, a leading dot,
    CIDR ranges), and a disagreement either skipped the private-network check of a direct connection or refused a
    legitimate endpoint behind the proxy."""
    try:
        mounts, pattern, URL = _routing(lib)
        target = URL(url)
        rules = []
        for key, proxy in mounts.items():
            try:
                rules.append((pattern(key), proxy))
            except Exception:  # noqa: BLE001 — an entry the library cannot parse (httpx2 drops it too)
                continue
        # the client tries its mounts in this order (port, host, then scheme first) and takes the first that matches
        for rule, proxy in sorted(rules, key=lambda r: r[0]):
            if rule.matches(target):
                return proxy is not None
        return False
    except Exception:  # noqa: BLE001 — (library internals moved) the environment as urllib reads it
        try:
            proxies = urllib.request.getproxies()
            u = urlparse(url)
            if not (proxies.get(u.scheme) or proxies.get("all")):
                return False
            return not urllib.request.proxy_bypass_environment(u.hostname or "", proxies)
        except Exception:  # noqa: BLE001
            return False


def check_host(host: str, strict: bool) -> None:
    """Resolve `host` and refuse it if any of its addresses is not allowed (LLMError)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise LLMError(f"cannot resolve {host}: {exc}") from exc
    for info in infos:
        problem = address_problem(str(info[4][0]), strict)
        if problem:
            raise LLMError(f"refused to fetch from {host}: {problem}")


def image_mime(data: bytes) -> str | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def fetch_image(url: str, *, strict: bool, timeout: float = 120.0, max_bytes: int = IMAGE_MAX_BYTES,
                transport=None, cancel=None) -> tuple[bytes, str]:
    """Download an image a provider returned by URL: http(s) only, every hop (no automatic redirects) checked
    against the address policy before and at connect time, at most `max_bytes`, and it must really be an image.
    `cancel` (the user's Stop) aborts it at once."""
    import httpx

    for _hop in range(4):
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise LLMError("the provider returned an image URL that is not http(s)")
        if transport is None:
            check_host(u.hostname, strict)  # also through a proxy: the target itself is vetted before any request
        # at connect time too (DNS can change), unless a proxy makes the connection
        guarded = transport is None and not proxied(url)
        call = Call(policy=(lambda ip: address_problem(ip, strict)) if guarded else None)

        def produce(call: Call, emit, url=url) -> tuple[bytes, str]:  # (image bytes, or where it redirects)
            buf = bytearray()
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT), follow_redirects=False,
                              transport=transport) as client:
                with client.stream("GET", url, headers={"Accept-Encoding": "identity"},
                                   extensions={"trace": call.trace}) as r:
                    if r.is_redirect and r.headers.get("location"):
                        return b"", urljoin(url, r.headers["location"])
                    if r.status_code >= 400:
                        raise LLMError(f"could not download the generated image: HTTP {r.status_code}")
                    size = r.headers.get("content-length", "")
                    if size.isdigit() and int(size) > max_bytes:
                        raise LLMError(f"the generated image is larger than {max_bytes // (1024 * 1024)} MB")
                    for chunk in r.iter_bytes():
                        buf += chunk
                        if len(buf) > max_bytes:
                            raise LLMError(f"the generated image is larger than {max_bytes // (1024 * 1024)} MB")
            return bytes(buf), ""

        try:
            data, location = pump(produce, call, cancel=cancel)  # (on a helper thread: Stop is watched meanwhile)
        except httpx.HTTPError as exc:
            if call.blocked:
                raise LLMError(call.blocked) from exc
            raise LLMError(f"could not download the generated image: {exc}") from exc
        if call.blocked:
            raise LLMError(call.blocked)
        if location:
            url = location
            continue
        mime = image_mime(data)
        if not mime:
            raise LLMError("the provider's image URL did not return a PNG, JPEG, WebP or GIF image")
        return data, mime
    raise LLMError("too many redirects while downloading the generated image")

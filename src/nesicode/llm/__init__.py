"""Model registry: `provider/model` strings -> a Provider instance.

    anthropic/claude-opus-5-5     Claude (official SDK)
    openai/gpt-5                  OpenAI
    openrouter/z-ai/glm-5         OpenRouter (anything after the first slash is the model)
    gemini/gemini-3.5-flash       Google Gemini (OpenAI-compatible endpoint)
    zai/glm-5                     Z.ai (GLM)
    ollama/qwen3-coder            local Ollama
    compat/<model>                any OpenAI-compatible server (NESICODE_COMPAT_BASE_URL)
    mock/demo                     deterministic offline provider for demos and tests
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .. import config as _config  # noqa: F401  (loads .env before keys are read)
from .anthropic_provider import AnthropicProvider
from .base import (
    Conversation,
    LLMError,
    ModelOptions,
    Provider,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
)
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider
from .transport import private_providers_allowed


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    label: str
    key_env: str  # "" = no key needed
    base_url: str = ""
    base_url_env: str = ""
    examples: tuple[str, ...] = ()


PROVIDERS: dict[str, ProviderInfo] = {
    "nesilabs": ProviderInfo(
        "nesilabs", "Nesi Labs", "NESILABS_API_KEY", "https://api.nesilabs.com/v1", "NESILABS_BASE_URL",
        examples=("nesilabs/nesi3",),
    ),
    "anthropic": ProviderInfo(
        "anthropic", "Anthropic (Claude)", "ANTHROPIC_API_KEY",
        examples=("anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5"),
    ),
    "openai": ProviderInfo("openai", "OpenAI", "OPENAI_API_KEY", "https://api.openai.com/v1", "OPENAI_BASE_URL"),
    "openrouter": ProviderInfo(
        "openrouter", "OpenRouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
        examples=("openrouter/z-ai/glm-5",),
    ),
    "gemini": ProviderInfo(
        "gemini", "Google Gemini", "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai", examples=("gemini/gemini-3.5-flash",),
    ),
    "zai": ProviderInfo("zai", "Z.ai (GLM)", "ZAI_API_KEY", "https://api.z.ai/api/paas/v4", examples=("zai/glm-5",)),
    "xai": ProviderInfo("xai", "xAI (Grok)", "XAI_API_KEY", "https://api.x.ai/v1"),
    "deepseek": ProviderInfo("deepseek", "DeepSeek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "ollama": ProviderInfo("ollama", "Ollama (local)", "", "http://localhost:11434/v1", "OLLAMA_BASE_URL"),
    "compat": ProviderInfo("compat", "OpenAI-compatible", "NESICODE_COMPAT_API_KEY", "", "NESICODE_COMPAT_BASE_URL"),
    "mock": ProviderInfo("mock", "Mock (offline demo)", "", examples=("mock/demo",)),
}


def split_model(spec: str) -> tuple[str, str]:
    if "/" not in spec:
        raise LLMError(f"model must look like 'provider/model', got {spec!r}")
    provider, model = spec.split("/", 1)
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")
    if not model:
        raise LLMError(f"missing model name in {spec!r}")
    return provider, model


# A "provider context" maps a provider name to either:
#   str  -> the user's own API key for a built-in provider, or
#   dict -> a user-defined provider {"kind": "openai"|"anthropic", "base_url": ..., "api_key": ...}
# User-defined providers win over built-ins with the same name.
ProviderContext = dict[str, "str | dict"]

KINDS = ("openai", "anthropic")


def provider_available(name: str, ctx: ProviderContext | None = None) -> bool:
    entry = (ctx or {}).get(name)
    if isinstance(entry, dict):
        return bool(entry.get("base_url"))
    info = PROVIDERS.get(name)
    if info is None:
        return False
    if name == "compat" and not os.environ.get(info.base_url_env):
        return False
    if not info.key_env:
        return True
    return bool(entry or os.environ.get(info.key_env))


def split_any(spec: str, ctx: ProviderContext | None = None) -> tuple[str, str]:
    if "/" not in spec:
        raise LLMError(f"model must look like 'provider/model', got {spec!r}")
    name, model = spec.split("/", 1)
    if not model:
        raise LLMError(f"missing model name in {spec!r}")
    if name not in PROVIDERS and not isinstance((ctx or {}).get(name), dict):
        raise LLMError(f"unknown provider {name!r}: add it in Ajustes del proyecto → Proveedores")
    return name, model


def build_provider(kind: str, name: str, base_url: str, api_key: str) -> Provider:
    """A user-defined provider. It only ever uses its own endpoint and key: never the server's (no fallback to
    ANTHROPIC_API_KEY & co.). With NESICODE_ALLOW_PRIVATE_PROVIDERS=0 the address it connects to is checked on
    every request, not only when the endpoint was saved (DNS can change)."""
    guard = not private_providers_allowed()
    if kind == "anthropic":
        default = base_url.rstrip("/") in ("", "https://api.anthropic.com")
        return AnthropicProvider(api_key=api_key or "", base_url=None if default else base_url, isolated=True,
                                 guard_private=guard)
    if kind == "openai":
        return OpenAICompatProvider(name, base_url, api_key, guard_private=guard)
    raise LLMError(f"unknown provider kind {kind!r}")


def get_provider(spec: str, user_keys: ProviderContext | None = None) -> tuple[Provider, str]:
    """Resolve `provider/model` to (Provider, bare model id)."""
    name, model = split_any(spec, user_keys)
    entry = (user_keys or {}).get(name)
    if isinstance(entry, dict):
        return build_provider(entry["kind"], name, entry["base_url"], entry.get("api_key", "")), model
    info = PROVIDERS[name]
    key = entry or (os.environ.get(info.key_env, "") if info.key_env else "")
    if name == "mock":
        return MockProvider(), model
    if name == "anthropic":
        return AnthropicProvider(api_key=key or None, base_url=os.environ.get("ANTHROPIC_BASE_URL") or None), model
    base = (os.environ.get(info.base_url_env) if info.base_url_env else "") or info.base_url
    if not base:
        raise LLMError(f"provider {name!r} needs {info.base_url_env} to be set")
    if info.key_env and not key:
        raise LLMError(f"no API key for {info.label}: add it in Ajustes del proyecto → Proveedores")
    return OpenAICompatProvider(name, base, key), model


def server_provider_config(name: str) -> dict | None:
    """Endpoint/key of a built-in provider as configured on the server (for model discovery)."""
    info = PROVIDERS.get(name)
    if not info or name == "mock":
        return None
    kind = "anthropic" if name == "anthropic" else "openai"
    base = (os.environ.get(info.base_url_env) if info.base_url_env else "") or info.base_url
    if name == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    key = os.environ.get(info.key_env, "") if info.key_env else ""
    return {"kind": kind, "base_url": base, "api_key": key}


def list_models(kind: str, base_url: str, api_key: str, timeout: float = 20.0) -> list[str]:
    """Ask a provider which models it serves (GET /models)."""
    import httpx

    base = base_url.rstrip("/")
    if kind == "anthropic":
        url = (base or "https://api.anthropic.com") + "/v1/models?limit=100"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    else:
        url = base + "/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise LLMError(f"no se pudo conectar: {exc}") from exc
    if r.status_code >= 400:
        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
    hint = "¿es una API compatible con OpenAI? Revisa la URL (suele terminar en /v1)"
    try:
        data = r.json()
    except ValueError as exc:  # an HTML page (web site, SPA, captive portal) instead of the API
        raise LLMError(f"la respuesta de {url} no es JSON ({r.headers.get('content-type', '?')}): {hint}") from exc
    items = (data.get("data") if "data" in data else data.get("models")) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise LLMError(f"la respuesta de {url} no trae una lista de modelos: {hint}")
    ids = sorted({str(m.get("id") or m.get("name")) for m in items if isinstance(m, dict) and (m.get("id") or m.get("name"))})
    return ids


# ---- context windows -------------------------------------------------------------
# Used when the provider does not report limits. (context, max output) in tokens.
KNOWN_LIMITS: list[tuple[str, int, int]] = [
    (r"^claude-(opus|sonnet|fable|mythos)-[5-9]", 1_000_000, 128_000),
    (r"^claude-haiku-5", 1_000_000, 128_000),
    (r"^claude-haiku-4", 200_000, 64_000),
    (r"^grok-4", 256_000, 64_000),
    (r"^grok-(code|build)", 256_000, 64_000),
    (r"^gpt-5", 400_000, 128_000),
    (r"^gemini-", 1_000_000, 64_000),
    (r"^glm-5", 200_000, 64_000),
    (r"^(deepseek)", 128_000, 32_000),
    (r"^nesi3$", 131_072, 32_768),
    (r"^(nesi|redarrow)", 245_760, 32_768),
]
DEFAULT_LIMITS = (128_000, 16_000)


def known_limits(model: str) -> tuple[int, int] | None:
    for rx, c, o in KNOWN_LIMITS:
        if re.search(rx, model):
            return c, o
    return None


def fetch_limits(kind: str, base_url: str, api_key: str, model: str, timeout: float = 15.0) -> dict | None:
    """Ask the provider for a model's context window. None if it does not say."""
    import httpx

    base = base_url.rstrip("/")

    def body(r) -> dict:  # a JSON object, or {} for anything else (HTML pages, lists, errors)
        try:
            data = r.json() if r.status_code == 200 else {}
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    try:
        if kind == "anthropic":
            r = httpx.get((base or "https://api.anthropic.com") + f"/v1/models/{model}",
                          headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}, timeout=timeout)
            d = body(r)
            if d.get("max_input_tokens"):
                return {"context": int(d["max_input_tokens"]), "output": int(d.get("max_tokens") or 0) or None}
            return None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # LiteLLM proxies (e.g. Nesi Labs): /model/info
        root = re.sub(r"/v1$", "", base)
        info_list = body(httpx.get(root + "/model/info", headers=headers, timeout=timeout)).get("data")
        is_litellm = isinstance(info_list, list)
        best = None
        for m in info_list if is_litellm else []:
            if not isinstance(m, dict):
                continue
            info = m.get("model_info") if isinstance(m.get("model_info"), dict) else {}
            if m.get("model_name") == model and info.get("max_input_tokens"):
                cand = {"context": int(info["max_input_tokens"]), "output": int(info.get("max_output_tokens") or 0) or None}
                best = cand if not best or cand["context"] < best["context"] else best  # several backends: the smallest is safe
        if best:
            return best
        # OpenRouter-style /models with context_length
        listed = body(httpx.get(base + "/models", headers=headers, timeout=timeout)).get("data")
        if isinstance(listed, list):
            for m in listed:
                if isinstance(m, dict) and m.get("id") == model:
                    c = m.get("context_length") or m.get("context_window") or m.get("max_context_length")
                    if c:
                        top = m.get("top_provider") or {}
                        return {"context": int(c), "output": int(top.get("max_completion_tokens") or 0) or None}
        # Not reported: measure it. An oversized request is rejected (not billed) and the error says the limit.
        if is_litellm:
            return probe_limit(base, headers, model)
    except Exception:
        return None
    return None


LIMIT_PATTERNS = [
    r"maximum (?:context|input|model|prompt) length is ([0-9][0-9,]*)",  # OpenAI, vLLM; xAI says "prompt"
    r"available context size \(([0-9][0-9,]*) tokens\)",
    r"max_model_len[^0-9]{0,10}([0-9][0-9,]*)",
    r"tokens? > ([0-9][0-9,]*) maximum",  # Anthropic: "prompt is too long: 1050000 tokens > 1000000 maximum"
    # Anthropic (older): "input length and `max_tokens` exceed context limit: 195000 + 64000 > 200000, decrease ..."
    r"exceed context limit: [0-9,]+ \+ [0-9,]+ > ([0-9][0-9,]*)",
    # Gemini: "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."
    r"maximum number of tokens allowed \(([0-9][0-9,]*)\)",
    # Mistral: "... too large for model with 131072 maximum context length"
    r"([0-9][0-9,]*) maximum context length",
    # llama.cpp: {"type": "exceed_context_size_error", "n_prompt_tokens": 10000, "n_ctx": 8192}
    r"\"n_ctx\"\s*:\s*([0-9]+)",
    # OpenAI: "Input tokens exceed the configured limit of 272000 tokens."
    r"exceeds? the configured limit of ([0-9][0-9,]*)",
    r"context (?:window|length) (?:is|of) (?:only )?([0-9][0-9,]*)",  # LM Studio: "context length of only 4096"
    r"exceeds? (?:the )?(?:model'?s? )?(?:maximum )?(?:context|token) (?:limit|length) (?:of )?([0-9][0-9,]*)",
]


def parse_limit(text: str) -> int | None:
    for rx in LIMIT_PATTERNS:
        m = re.search(rx, text or "", re.I)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


_probed: set[tuple[str, str]] = set()  # (endpoint, model) probes that told us nothing: never sent again


def probe_limit(base: str, headers: dict, model: str, timeout: float = 60.0) -> dict | None:
    """Send ~300-600k tokens with max_tokens=1 to self-hosted/proxied models and read the limit from the rejection
    (not billed). Only used when the provider does not publish the context window. If the model ACCEPTS the probe
    (a window that big, and the request is billed), the result is None — the table/default is used — and the probe
    is not repeated while the server runs."""
    import httpx

    key = (base, model)
    if key in _probed:
        return None
    try:
        r = httpx.post(base + "/chat/completions", headers=headers, timeout=httpx.Timeout(timeout, connect=15.0),
                       json={"model": model, "messages": [{"role": "user", "content": "hola " * 300_000}], "max_tokens": 1})
    except Exception:
        _probed.add(key)
        return None
    limit = parse_limit(r.text) if r.status_code >= 400 else None
    if not limit:
        _probed.add(key)
        return None
    return {"context": limit, "output": None}


# Pre-filled forms for the "Añadir proveedor" dialog.
TEMPLATES = [
    {"name": "nesilabs", "label": "Nesi Labs", "kind": "openai", "base_url": "https://api.nesilabs.com/v1"},
    {"name": "anthropic", "label": "Anthropic (Claude)", "kind": "anthropic", "base_url": "https://api.anthropic.com"},
    {"name": "openai", "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1"},
    {"name": "openrouter", "label": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1"},
    {"name": "gemini", "label": "Google Gemini", "kind": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    {"name": "zai", "label": "Z.ai (GLM)", "kind": "openai", "base_url": "https://api.z.ai/api/paas/v4"},
    {"name": "xai", "label": "xAI (Grok)", "kind": "openai", "base_url": "https://api.x.ai/v1"},
    {"name": "deepseek", "label": "DeepSeek", "kind": "openai", "base_url": "https://api.deepseek.com/v1"},
    {"name": "ollama", "label": "Ollama (local)", "kind": "openai", "base_url": "http://localhost:11434/v1"},
    {"name": "custom", "label": "Otro (compatible con OpenAI)", "kind": "openai", "base_url": ""},
]


__all__ = [
    "Conversation", "LLMError", "ModelOptions", "Provider", "ToolCall", "ToolResult", "ToolSpec",
    "Turn", "Usage", "PROVIDERS", "get_provider", "split_model", "provider_available", "KINDS",
    "TEMPLATES", "build_provider", "list_models", "server_provider_config", "split_any",
    "fetch_limits", "known_limits", "DEFAULT_LIMITS", "parse_limit", "probe_limit",
]

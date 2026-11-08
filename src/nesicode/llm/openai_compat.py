"""Any OpenAI-compatible chat-completions endpoint (OpenAI, OpenRouter, Ollama, Gemini,
Z.ai/GLM, vLLM, LM Studio...). Non-streaming; tool calls via the `tools` field."""

from __future__ import annotations

import json
import time

import httpx

from .base import (
    Conversation,
    EventSink,
    LLMError,
    ModelOptions,
    Provider,
    ToolCall,
    ToolResult,
    Turn,
    Usage,
)

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class OpenAICompatConversation(Conversation):
    def __init__(self, provider: "OpenAICompatProvider", model, system, tools, options):
        super().__init__(model, system, tools, options)
        self.provider = provider
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text or "(no answer)"})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            content = ("ERROR: " if r.is_error else "") + (r.content or "(empty)")
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})

    def mask_tool_results(self, keep_last: int, marker, *, keep_chars: int = 0, apply: bool = True) -> int:
        idx = [i for i, m in enumerate(self.messages) if m.get("role") == "tool"]
        last_reply = max((i for i, m in enumerate(self.messages) if m.get("role") == "assistant"), default=-1)
        pending = sum(1 for i in idx if i > last_reply)  # results the model has not seen yet: never masked
        sizes = [len(c) if isinstance(c := self.messages[i].get("content"), str) else 0 for i in idx]
        freed = 0
        for k in self.maskable(sizes, pending, keep_last, keep_chars):
            m = self.messages[idx[k]]
            old = m.get("content")
            if not isinstance(old, str):
                continue
            new = marker(m.get("tool_call_id", ""), old)
            if new is not None and len(new) < len(old):
                freed += len(old) - len(new)
                if apply:
                    m["content"] = new
        return freed

    def step(self, on_event: EventSink | None = None) -> Turn:
        body: dict = {"model": self.model, "messages": self.messages}
        if self.options.max_tokens:
            # Reserve output explicitly: vLLM otherwise trims it silently near the limit.
            body["max_completion_tokens" if self.provider.name == "openai" else "max_tokens"] = self.options.max_tokens
        if self.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in self.tools
            ]
        data = self.provider.post("/chat/completions", body, cancel=self.options.cancel)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.provider.name}: {str(data)[:300]}") from exc

        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                err = "" if isinstance(args, dict) else "arguments must be a JSON object"
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError as exc:
                args, err = {}, f"invalid JSON arguments: {exc}"
            calls.append(ToolCall(id=tc.get("id") or f"call_{len(calls)}", name=fn.get("name", ""), arguments=args, parse_error=err))

        assistant: dict = {"role": "assistant", "content": text}
        if msg.get("tool_calls"):
            assistant["tool_calls"] = msg["tool_calls"]
        self.messages.append(assistant)

        if text and on_event:
            on_event("text", text)
        u = data.get("usage") or {}
        usage = Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))
        finish = choice.get("finish_reason") or "stop"
        stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}.get(finish, finish)
        if calls and stop == "end_turn":
            stop = "tool_use"
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return Turn(text=text, tool_calls=calls, stop_reason=stop, usage=usage, thinking=str(thinking))


class OpenAICompatProvider(Provider):
    def __init__(self, name: str, base_url: str, api_key: str = "", extra_headers: dict | None = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def post(self, path: str, body: dict, cancel=None) -> dict:
        last = ""
        for attempt in range(4):
            try:
                resp = self._send(path, body, cancel)
            except httpx.HTTPError as exc:
                last = f"connection error: {exc}"
            else:
                if resp.status_code < 400:
                    return resp.json()
                last = f"HTTP {resp.status_code}: {resp.text[:1500]}"
                if resp.status_code not in RETRY_STATUS:
                    break
            if cancel is not None and cancel.wait(min(2 ** attempt, 20)):
                raise LLMError("cancelled by the user")
            if cancel is None:
                time.sleep(min(2 ** attempt, 20))
        raise LLMError(f"{self.name} request failed: {last}")

    def _send(self, path: str, body: dict, cancel=None) -> httpx.Response:
        """POST; with `cancel`, the request runs on a helper thread so Stop returns at once
        (the abandoned request is closed in the background)."""
        if cancel is None:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
                return client.post(self.base_url + path, json=body, headers=self.headers)
        import threading

        box: dict = {}
        client = httpx.Client(timeout=httpx.Timeout(600.0, connect=15.0))

        def work():
            try:
                box["resp"] = client.post(self.base_url + path, json=body, headers=self.headers)
            except BaseException as exc:  # noqa: BLE001 — handed back to the caller
                box["exc"] = exc
            finally:
                client.close()

        t = threading.Thread(target=work, daemon=True, name="nesi-llm-request")
        t.start()
        while t.is_alive():
            t.join(0.25)
            if cancel.is_set() and t.is_alive():
                try:
                    client.close()  # aborts the connection in flight
                except Exception:
                    pass
                raise LLMError("cancelled by the user")
        if "exc" in box:
            raise box["exc"]
        return box["resp"]

    def conversation(self, model, system, tools, options: ModelOptions) -> Conversation:
        return OpenAICompatConversation(self, model, system, tools, options)

    def transcribe(self, model: str, audio: bytes, filename: str, language: str = "") -> str:
        data = {"model": model}
        if language:
            data["language"] = language
        headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
                r = client.post(self.base_url + "/audio/transcriptions", data=data,
                                files={"file": (filename, audio, "audio/wav")}, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: could not reach the transcription endpoint: {exc}") from exc
        if r.status_code >= 400:
            raise LLMError(f"{self.name} transcription failed: HTTP {r.status_code}: {r.text[:500]}")
        try:
            return str(r.json().get("text", "")).strip()
        except ValueError:
            return r.text.strip()

    def generate_image(self, model: str, prompt: str) -> tuple[bytes, str]:
        import base64

        body = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json"}
        for attempt in range(3):
            try:
                data = self.post("/images/generations", body)
                break
            except LLMError as exc:
                text = str(exc).lower()
                wrong_backend = "language model" in text or "not available on this endpoint" in text or "does not support image" in text
                if not wrong_backend:
                    raise
                if attempt == 2:  # a load-balanced group keeps landing on a text-only model
                    raise LLMError(
                        f"NOT RETRYABLE — the image model '{model}' of {self.name} is answering with a text-only model "
                        "(it cannot create images). Do not try again with other prompts: tell the user to choose another "
                        "model in Project settings → Models by role → Image generation (e.g. an xAI grok-imagine or "
                        "OpenAI gpt-image model), or to fix that model group at the provider."
                    ) from exc
        try:
            item = data["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected image response from {self.name}: {str(data)[:300]}") from exc
        if item.get("b64_json"):
            raw = base64.b64decode(item["b64_json"])
        elif item.get("url"):
            with httpx.Client(timeout=120) as c:
                r = c.get(item["url"])
                r.raise_for_status()
                raw = r.content
        else:
            raise LLMError(f"{self.name} returned no image")
        mime = item.get("mime_type") or ("image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else
                                         "image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/webp")
        return raw, mime

    def vision(self, model: str, prompt: str, image: bytes, mime: str) -> tuple[str, Usage]:
        import base64

        url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        data = self.post("/chat/completions", {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}},
        ]}]})
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected vision response from {self.name}: {str(data)[:300]}") from exc
        u = data.get("usage") or {}
        return text, Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))

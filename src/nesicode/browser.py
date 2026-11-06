"""Bridge to the user's own Chrome through the Nesicode extension (WebSocket).

The extension connects to /api/browser/ws with a per-user pairing token. The agent (running in a
worker thread) calls `hub.call(user_id, cmd, args)`, which is forwarded to the extension and awaited.
The extension only exposes tabs the agent opened (its "Nesicode" tab group) or that the user shared.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time

from .llm import ToolSpec


class BrowserHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.sockets: dict[int, object] = {}  # user_id -> WebSocket
        self.loops: dict[int, asyncio.AbstractEventLoop] = {}
        self.info: dict[int, dict] = {}
        self.pending: dict[str, asyncio.Future] = {}
        self.ids = itertools.count(1)

    # -- connection lifecycle (called from the server's event loop)
    def attach(self, user_id: int, ws, hello: dict) -> None:
        with self.lock:
            self.sockets[user_id] = ws
            self.loops[user_id] = asyncio.get_running_loop()
            self.info[user_id] = {**hello, "since": time.time()}

    def detach(self, user_id: int, ws) -> None:
        with self.lock:
            if self.sockets.get(user_id) is ws:
                self.sockets.pop(user_id, None)
                self.info.pop(user_id, None)

    def on_message(self, msg: dict) -> None:
        fut = self.pending.pop(str(msg.get("id")), None)
        if fut and not fut.done():
            fut.set_result(msg)

    def connected(self, user_id: int) -> bool:
        return user_id in self.sockets

    def status(self, user_id: int) -> dict:
        return {"connected": self.connected(user_id), **self.info.get(user_id, {})}

    # -- calls from agent threads
    def call(self, user_id: int, cmd: str, args: dict | None = None, timeout: float = 45) -> dict:
        ws = self.sockets.get(user_id)
        loop = self.loops.get(user_id)
        if ws is None or loop is None:
            raise RuntimeError("Chrome no está conectado: abre la extensión de Nesicode en Chrome")

        async def go():
            rid = str(next(self.ids))
            fut = loop.create_future()
            self.pending[rid] = fut
            try:
                await ws.send_json({"id": rid, "cmd": cmd, "args": args or {}})
                return await asyncio.wait_for(fut, timeout)
            finally:
                self.pending.pop(rid, None)

        msg = asyncio.run_coroutine_threadsafe(go(), loop).result(timeout + 5)
        if not msg.get("ok"):
            raise RuntimeError(str(msg.get("error") or "error en Chrome"))
        return msg.get("result") or {}


hub = BrowserHub()

_TAB = {"tab_id": {"type": "integer", "description": "Tab id from chrome_tabs / chrome_open. Default: the last tab used."}}

TOOLS: dict[str, ToolSpec] = {
    "chrome_tabs": ToolSpec(
        "chrome_tabs", "List the tabs in the user's 'Nesicode' tab group: the only tabs you may use (the ones you opened "
        "and the ones the user dragged into the group).",
        {"type": "object", "properties": {}},
    ),
    "chrome_open": ToolSpec(
        "chrome_open", "Open a URL in the user's own Chrome (their real browser, with their logged-in sessions). "
        "Opens a new tab in the 'Nesicode' tab group unless tab_id is given.",
        {"type": "object", "properties": {"url": {"type": "string"}, **_TAB}, "required": ["url"]},
    ),
    "chrome_read": ToolSpec(
        "chrome_read", "Read a tab: title, URL, visible text and the interactive elements, each with a `ref` "
        "to use with chrome_click / chrome_type. Read again after the page changes.",
        {"type": "object", "properties": {**_TAB}},
    ),
    "chrome_click": ToolSpec(
        "chrome_click", "Click an element by its ref (from chrome_read).",
        {"type": "object", "properties": {"ref": {"type": "string"}, **_TAB}, "required": ["ref"]},
    ),
    "chrome_type": ToolSpec(
        "chrome_type", "Type text into an input/textarea/editable element by ref; submit=true presses Enter.",
        {"type": "object", "properties": {
            "ref": {"type": "string"}, "text": {"type": "string"}, "submit": {"type": "boolean"}, **_TAB,
        }, "required": ["ref", "text"]},
    ),
    "chrome_scroll": ToolSpec(
        "chrome_scroll", "Scroll the page up or down by about one screen.",
        {"type": "object", "properties": {"direction": {"type": "string", "enum": ["up", "down"]}, **_TAB}},
    ),
    "chrome_back": ToolSpec(
        "chrome_back", "Go back in the tab's history.", {"type": "object", "properties": {**_TAB}},
    ),
    "chrome_close": ToolSpec(
        "chrome_close", "Close a tab of the 'Nesicode' group (e.g. ones you opened and no longer need).",
        {"type": "object", "properties": {**_TAB}},
    ),
    "chrome_screenshot": ToolSpec(
        "chrome_screenshot", "Take a screenshot of a tab (it is brought to the front). It is shown to the user and "
        "saved in the workspace; look at it with view_image if you need to see it.",
        {"type": "object", "properties": {**_TAB}},
    ),
}

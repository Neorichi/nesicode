"""Bridge to the user's own Chrome through the Nesicode extension (WebSocket).

The extension connects to /api/browser/ws with a per-user pairing token. The agent (running in a
worker thread) calls `hub.call(user_id, cmd, args)`, which is forwarded to the extension and awaited.
The extension only exposes tabs the agent opened (its "Nesicode" tab group) or that the user shared.

Each user has their own channel: a command gets a random, unguessable id, and only a reply that comes back
on the same socket (same user) the command was sent to can settle it. Another user's extension can't answer
(or deny) someone else's commands, even on a multi-user server.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import secrets
import threading
import time

from .llm import ToolSpec


def _settle(fut: asyncio.Future, result=None, exc: BaseException | None = None) -> None:
    """Resolve a future from any thread (it belongs to the socket's event loop)."""
    def go():
        if not fut.done():
            fut.set_exception(exc) if exc is not None else fut.set_result(result)

    try:
        same = asyncio.get_running_loop() is fut.get_loop()
    except RuntimeError:
        same = False
    if same:
        go()
    else:
        fut.get_loop().call_soon_threadsafe(go)


class BrowserHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.sockets: dict[int, object] = {}  # user_id -> WebSocket
        self.loops: dict[int, asyncio.AbstractEventLoop] = {}
        self.info: dict[int, dict] = {}
        # user_id -> {request id: (socket the command went to, future)}: one channel per user
        self.pending: dict[int, dict[str, tuple[object, asyncio.Future]]] = {}
        # the receive loop of each attached socket -> (user_id, ws): who is talking when on_message is called
        self.senders: dict[asyncio.Task, tuple[int, object]] = {}

    # -- connection lifecycle (called from the server's event loop)
    def attach(self, user_id: int, ws, hello: dict) -> None:
        with self.lock:
            self.sockets[user_id] = ws
            self.loops[user_id] = asyncio.get_running_loop()
            self.info[user_id] = {**hello, "since": time.time()}
            task = asyncio.current_task()
            if task is not None:  # the socket's receive loop runs in this task: its messages come from (user, ws)
                self.senders[task] = (user_id, ws)

    def detach(self, user_id: int, ws) -> None:
        with self.lock:
            if self.sockets.get(user_id) is ws:
                self.sockets.pop(user_id, None)
                self.info.pop(user_id, None)
            for task, (_, w) in list(self.senders.items()):
                if w is ws:
                    self.senders.pop(task, None)
            chan = self.pending.get(user_id) or {}
            lost = [(rid, fut) for rid, (w, fut) in chan.items() if w is ws]
            for rid, _ in lost:
                chan.pop(rid, None)
        for _, fut in lost:  # their extension is gone: fail now instead of making the agent wait for the timeout
            _settle(fut, exc=RuntimeError("Chrome se desconectó antes de responder: vuelve a abrir la extensión de Nesicode"))

    def on_message(self, msg: dict, user_id: int | None = None, ws=None) -> None:
        """A reply from an extension. It only settles a command that was sent to that same socket of that same
        user. The sender is (user_id, ws) when given, else the socket whose receive loop is the current task."""
        if ws is None:
            try:
                sender = self.senders.get(asyncio.current_task())
            except RuntimeError:  # no running loop: nobody to attribute the message to
                sender = None
            if sender is None or (user_id is not None and sender[0] != user_id):
                return
            user_id, ws = sender
        if user_id is None:
            return
        rid = str(msg.get("id") or "")
        with self.lock:
            chan = self.pending.get(user_id) or {}
            entry = chan.get(rid)
            if entry is None or entry[0] is not ws:
                return  # unknown id, or a command that went to another socket: ignore it
            chan.pop(rid, None)
        _settle(entry[1], result=msg)

    def connected(self, user_id: int) -> bool:
        return user_id in self.sockets

    def status(self, user_id: int) -> dict:
        return {"connected": self.connected(user_id), **self.info.get(user_id, {})}

    # -- calls from agent threads
    def call(self, user_id: int, cmd: str, args: dict | None = None, timeout: float = 45,
             cancel: threading.Event | None = None) -> dict:
        """Run a command in the user's Chrome. `cancel` (e.g. the run's Stop) abandons the wait."""
        with self.lock:
            ws = self.sockets.get(user_id)
            loop = self.loops.get(user_id)
        if ws is None or loop is None:
            raise RuntimeError("Chrome no está conectado: abre la extensión de Nesicode en Chrome")

        async def go():
            rid = secrets.token_urlsafe(18)  # unguessable: never a counter another user could predict
            fut = loop.create_future()
            with self.lock:
                if self.sockets.get(user_id) is not ws:
                    raise RuntimeError("Chrome se desconectó: vuelve a abrir la extensión de Nesicode")
                self.pending.setdefault(user_id, {})[rid] = (ws, fut)
            try:
                await ws.send_json({"id": rid, "cmd": cmd, "args": args or {}})
                return await asyncio.wait_for(fut, timeout)
            except TimeoutError:
                raise RuntimeError(f"Chrome no respondió en {timeout:.0f} s") from None
            finally:
                with self.lock:
                    chan = self.pending.get(user_id)
                    if chan is not None:
                        chan.pop(rid, None)
                        if not chan:
                            self.pending.pop(user_id, None)

        cf = asyncio.run_coroutine_threadsafe(go(), loop)
        deadline = time.monotonic() + timeout + 5
        try:
            while not concurrent.futures.wait([cf], timeout=min(0.25 if cancel is not None else timeout + 5,
                                                                max(0.0, deadline - time.monotonic()))).done:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelado: el usuario pulsó Detener")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Chrome no respondió en {timeout:.0f} s")
            msg = cf.result()
        except BaseException:
            cf.cancel()  # no-op when it already finished; otherwise stops waiting on the loop too
            raise
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

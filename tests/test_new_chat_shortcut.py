"""Ctrl+T (or ⌘T) opens a new conversation in the current project — the key test of the shortcut, bundled with the
esbuild Vite ships, as the other web tests do."""

from test_fix_web_ui import pytestmark, run_ts  # noqa: F401  (pytestmark: skipped without node/esbuild)


def test_ctrl_or_cmd_t_and_nothing_else(tmp_path):
    out = run_ts(tmp_path, """
      const k = (key: string, m: Partial<Record<'ctrlKey'|'metaKey'|'shiftKey'|'altKey', boolean>>) =>
        L.isNewChatShortcut({ key, ctrlKey: false, metaKey: false, shiftKey: false, altKey: false, ...m })
      out.ctrl = k('t', { ctrlKey: true })
      out.cmd = k('t', { metaKey: true })
      out.caps = k('T', { ctrlKey: true })
      out.plain = k('t', {})
      out.shift = k('t', { ctrlKey: true, shiftKey: true })
      out.alt = k('t', { metaKey: true, altKey: true })
      out.other = k('n', { ctrlKey: true })
    """)
    assert out == {"ctrl": True, "cmd": True, "caps": True, "plain": False, "shift": False, "alt": False, "other": False}

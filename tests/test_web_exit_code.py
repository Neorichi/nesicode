"""The chat shows a terminal command's output without its «exit code: 0» line (that line is for the model); a
failed command shows its code. Bundles web/src/logic.ts with the esbuild Vite ships, as the other web tests do."""

from test_fix_web_ui import run_ts, pytestmark  # noqa: F401  (pytestmark: skipped without node/esbuild)


def test_the_exit_code_shows_only_when_the_command_failed(tmp_path):
    out = run_ts(tmp_path, """
      out.ok = L.splitExit('bash', 'exit code: 0\\nhola\\n')
      out.fail = L.splitExit('bash', 'exit code: 1\\nboom\\n')
      out.crlf = L.splitExit('bash', 'exit code: 0\\r\\nwin\\r\\n')
      out.empty = L.splitExit('bash', 'exit code: 0\\n')
      out.other = L.splitExit('read_file', 'exit code: 0\\nnot a header here')
      out.none = L.splitExit('bash', undefined)
    """)
    assert out["ok"] == {"code": 0, "body": "hola\n"}
    assert out["fail"] == {"code": 1, "body": "boom\n"}
    assert out["crlf"] == {"code": 0, "body": "win\r\n"}
    assert out["empty"] == {"code": 0, "body": ""}
    assert out["other"] == {"code": None, "body": "exit code: 0\nnot a header here"}
    assert out["none"] == {"code": None, "body": ""}

"""nesicode command line.

    nesicode serve [--host 127.0.0.1] [--port 8420]
    nesicode run "task" [--workspace .] [--model anthropic/claude-opus-5-5]
    nesicode harness "task" [--workspace .] [--k 1] [--mode parallel|sequential] [--apply]
    nesicode demo        offline end-to-end demo with the mock provider
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import tempfile
import threading
from pathlib import Path

# The settings (nesicode.config) are loaded by the commands that use them, never when this module is imported:
# `nesicode desktop` has to say it is the desktop app first (desktop.main sets NESICODE_DESKTOP), or its data would go
# to ./data and the app folder's .env would never be read — while its log and WebKit storage still go to the app folder.

CONFLICT_WHY = {
    "changed": "cambió en el proyecto después de empezar la ejecución",
    "exists": "ya existe en el proyecto con otro contenido",
    "unsafe": "sale del proyecto por un enlace simbólico",
    "error": "no se pudo escribir",
}


def _print_event(e: dict) -> None:
    t = e.get("type")
    if t == "assistant":
        print(f"\n\033[1m{e['text']}\033[0m")
    elif t == "tool_call":
        print(f"  → {e['name']} {json.dumps(e['args'], ensure_ascii=False)[:160]}")
    elif t == "tool_result" and e.get("is_error"):
        print(f"  ✗ {e['output'][:200]}")
    elif t == "error":
        print(f"\033[31mError: {e['text']}\033[0m")
    elif t == "phase":
        print(f"\n== {e['phase']}" + (f" (ronda {e['round']})" if e.get("round") else ""))
    elif t == "stage" and e.get("status") == "done":
        ch = e["changes"]
        print(f"  [{e['label']}] +{len(ch['added'])} ~{len(ch['modified'])} -{len(ch['removed'])}"
              + (" (fallback)" if e["fell_back"] else "") + (" KEEP_BASE" if e["kept_base"] else ""))
    elif t == "rollout" and e.get("status") != "running":
        print(f"  rollout {e['label']}: {e['status']}, entregado={e['delivered']}, pasos={e['steps']}, check={e['check_code']}")
    elif t == "selected":
        print(f"  seleccionado: {e['label']}")


def _safe_console() -> None:
    """A character the console cannot show (Windows: cp1252 when output is redirected) prints as '?' instead of
    raising inside the agent loop and ending the run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


@contextlib.contextmanager
def _ctrl_c_stops(cancel: threading.Event):
    """First Ctrl+C stops the run cleanly (like the web's Stop: running commands are killed, rollouts wind
    down, workspace copies are removed); a second one exits at once."""

    def handler(signum, frame):
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        print("\nParando... (Ctrl+C otra vez para salir ya)", file=sys.stderr, flush=True)

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread: Ctrl+C keeps its default behaviour
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous if previous is not None else signal.default_int_handler)


def _default_model() -> str:
    from .config import settings

    return settings.default_executor_model


def _limits(spec: str) -> tuple[int, int]:
    """(context, output) tokens of `provider/model`: the known table, else the defaults (as the server does)."""
    from .llm import DEFAULT_LIMITS, known_limits

    return known_limits(spec.split("/", 1)[-1]) or DEFAULT_LIMITS


def _load_harness(path: str | None) -> dict:
    from .harness import base_harness

    # utf-8 whatever the locale (Windows: cp1252); -sig also accepts files saved with a BOM
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else base_harness()


def _print_applied(ch: dict) -> None:
    print(f"aplicado: +{len(ch.get('added', []))} ~{len(ch.get('modified', []))} -{len(ch.get('removed', []))}")
    for c in ch.get("conflicts", []):
        side = f"; la versión propuesta está en {c['side_file']}" if c.get("side_file") else ""
        print(f"  conflicto, no se tocó {c['path']}: {CONFLICT_WHY.get(c['reason'], c['reason'])}{side}")


def cmd_serve(a) -> None:
    import uvicorn

    from .config import settings

    if settings.sandbox == "local":
        print("Aviso: NESICODE_SANDBOX=local ejecuta bash en este host. Para varios usuarios usa NESICODE_SANDBOX=docker.")
    if not settings.auth_enabled and a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Aviso: NESICODE_AUTH=false y escuchando en {a.host}: cualquiera que llegue a este puerto entra sin login.")
    uvicorn.run("nesicode.server.app:app", host=a.host, port=a.port, reload=a.reload)


def cmd_run(a) -> None:
    from .agent import Agent, Workspace

    cancel = threading.Event()
    model = a.model or _default_model()
    context, output = _limits(model)
    agent = Agent(model=model, harness=_load_harness(a.harness), workspace=Workspace(a.workspace),
                  emit=lambda e: e.get("_persist", True) and _print_event(e), max_steps=a.max_steps, cancel=cancel,
                  context_limit=context, output_limit=output)
    with _ctrl_c_stops(cancel):
        res = agent.run(a.task)
    print(f"\n[{res.status}] pasos={res.steps} uso={res.usage.to_dict()}")
    sys.exit(0 if res.status in ("done", "max_steps") else 130 if res.status == "cancelled" else 1)


def cmd_harness(a) -> None:
    from .harnessing import HarnessingConfig, HarnessingRun

    ws = Path(a.workspace).resolve()
    out = Path(a.out).resolve() if a.out else Path(tempfile.mkdtemp(prefix="nesicode-run-"))
    if out == ws or ws in out.parents or out in ws.parents:
        print(f"Error: --out ({out}) tiene que quedar fuera de --workspace ({ws}): guarda copias del proyecto "
              "y dentro de él se copiaría a sí mismo.", file=sys.stderr)
        sys.exit(2)
    model = a.model or _default_model()
    improver = a.improver or model
    (context, output), (i_context, i_output) = _limits(model), _limits(improver)
    cfg = HarnessingConfig(task=a.task, executor_model=model, improver_model=improver, k=a.k, mode=a.mode,
                           check_command=a.check or "", domain=a.domain, context_limit=context, output_limit=output,
                           improver_context=i_context, improver_output=i_output)
    cancel = threading.Event()
    hr = HarnessingRun(cfg, base_harness=_load_harness(a.harness), source_ws=ws, run_dir=out, emit=_print_event,
                       cancel=cancel)
    try:
        with _ctrl_c_stops(cancel):
            result = hr.run()
    except RuntimeError as exc:
        if str(exc) != "cancelled":
            raise
        print(f"\nCancelado: no se aplicó nada. Trayectorias y diffs en {out}", file=sys.stderr)
        sys.exit(130)
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "adopted_harness.json").write_text(json.dumps(result["adopted_harness"], indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    if a.apply:
        _print_applied(hr.apply_selected(result["selected"], ws))
    hr.cleanup_workspaces(keep=result["selected"], snapshot=True)  # one copy: the chosen result
    print(f"\nResultado en {out}  (elegido: rollouts/{result['selected']}/ws, harness adoptado: adopted_harness.json, "
          f"coste: {result['usage']})")
    if result["harness_diff"]:
        print(result["harness_diff"])


def cmd_demo(a) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="nesicode-demo-"))
    (tmp / "ws").mkdir()
    (tmp / "ws" / "README.md").write_text("# Demo\n", encoding="utf-8")
    a2 = argparse.Namespace(task="Write a short report about this repository", workspace=str(tmp / "ws"),
                            model="mock/demo", improver=None, k=a.k, mode=a.mode, check="test -f NESICODE_RESULT.md",
                            domain="demo", harness=None, out=str(tmp / "run"), apply=False)
    cmd_harness(a2)


def cmd_desktop(a) -> None:
    from .desktop import main as desktop_main

    desktop_main()


def main(argv: list[str] | None = None) -> None:
    _safe_console()
    p = argparse.ArgumentParser(prog="nesicode")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="arranca la web")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8420)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("run", help="ejecuta el agente una vez")
    r.add_argument("task")
    r.add_argument("--workspace", default=".")
    r.add_argument("--model", help="proveedor/modelo (por defecto NESICODE_EXECUTOR_MODEL)")
    r.add_argument("--harness", help="JSON {path: content}")
    r.add_argument("--max-steps", type=int, default=60)
    r.set_defaults(fn=cmd_run)

    h = sub.add_parser("harness", help="Harness² sobre una tarea")
    h.add_argument("task")
    h.add_argument("--workspace", default=".")
    h.add_argument("--model", help="proveedor/modelo (por defecto NESICODE_EXECUTOR_MODEL)")
    h.add_argument("--improver")
    h.add_argument("--k", type=int, default=1)
    h.add_argument("--mode", choices=["parallel", "sequential"], default="parallel")
    h.add_argument("--check", help="comando de verificación (p. ej. 'uv run pytest -q')")
    h.add_argument("--domain", default="general")
    h.add_argument("--harness")
    h.add_argument("--out", help="carpeta del resultado (fuera del workspace)")
    h.add_argument("--apply", action="store_true", help="aplica los cambios del resultado elegido")
    h.set_defaults(fn=cmd_harness)

    dk = sub.add_parser("desktop", help="abre nesicode en una ventana de escritorio")
    dk.set_defaults(fn=cmd_desktop)

    d = sub.add_parser("demo", help="demo offline con el proveedor mock")
    d.add_argument("--k", type=int, default=1)
    d.add_argument("--mode", choices=["parallel", "sequential"], default="parallel")
    d.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

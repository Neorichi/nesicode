"""nesicode command line.

    nesicode serve [--host 127.0.0.1] [--port 8420]
    nesicode run "task" [--workspace .] [--model anthropic/claude-opus-5-5]
    nesicode harness "task" [--workspace .] [--k 1] [--mode parallel|sequential] [--apply]
    nesicode demo        offline end-to-end demo with the mock provider
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .config import settings


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


def cmd_serve(a) -> None:
    import uvicorn

    if settings.sandbox == "local":
        print("Aviso: NESICODE_SANDBOX=local ejecuta bash en este host. Para varios usuarios usa NESICODE_SANDBOX=docker.")
    if not settings.auth_enabled and a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Aviso: NESICODE_AUTH=false y escuchando en {a.host}: cualquiera que llegue a este puerto entra sin login.")
    uvicorn.run("nesicode.server.app:app", host=a.host, port=a.port, reload=a.reload)


def cmd_run(a) -> None:
    from .agent import Agent, Workspace
    from .harness import base_harness

    h = json.loads(Path(a.harness).read_text()) if a.harness else base_harness()
    res = Agent(model=a.model, harness=h, workspace=Workspace(a.workspace), emit=lambda e: e.get("_persist", True) and _print_event(e),
                max_steps=a.max_steps).run(a.task)
    print(f"\n[{res.status}] pasos={res.steps} uso={res.usage.to_dict()}")
    sys.exit(0 if res.status in ("done", "max_steps") else 1)


def cmd_harness(a) -> None:
    from .harness import base_harness
    from .harnessing import HarnessingConfig, HarnessingRun

    h = json.loads(Path(a.harness).read_text()) if a.harness else base_harness()
    cfg = HarnessingConfig(task=a.task, executor_model=a.model, improver_model=a.improver or a.model, k=a.k,
                           mode=a.mode, check_command=a.check or "", domain=a.domain)
    out = Path(a.out or tempfile.mkdtemp(prefix="nesicode-run-"))
    hr = HarnessingRun(cfg, base_harness=h, source_ws=Path(a.workspace), run_dir=out, emit=_print_event)
    result = hr.run()
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / "adopted_harness.json").write_text(json.dumps(result["adopted_harness"], indent=1, ensure_ascii=False))
    if a.apply:
        print("aplicado:", hr.apply_selected(result["selected"], Path(a.workspace)))
    print(f"\nResultado en {out}  (harness adoptado: adopted_harness.json, coste: {result['usage']})")
    if result["harness_diff"]:
        print(result["harness_diff"])


def cmd_demo(a) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="nesicode-demo-"))
    (tmp / "ws").mkdir()
    (tmp / "ws" / "README.md").write_text("# Demo\n")
    a2 = argparse.Namespace(task="Write a short report about this repository", workspace=str(tmp / "ws"),
                            model="mock/demo", improver=None, k=a.k, mode=a.mode, check="test -f NESICODE_RESULT.md",
                            domain="demo", harness=None, out=str(tmp / "run"), apply=False)
    cmd_harness(a2)


def cmd_desktop(a) -> None:
    from .desktop import main as desktop_main

    desktop_main()


def main(argv: list[str] | None = None) -> None:
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
    r.add_argument("--model", default=settings.default_executor_model)
    r.add_argument("--harness", help="JSON {path: content}")
    r.add_argument("--max-steps", type=int, default=60)
    r.set_defaults(fn=cmd_run)

    h = sub.add_parser("harness", help="Harness² sobre una tarea")
    h.add_argument("task")
    h.add_argument("--workspace", default=".")
    h.add_argument("--model", default=settings.default_executor_model)
    h.add_argument("--improver")
    h.add_argument("--k", type=int, default=1)
    h.add_argument("--mode", choices=["parallel", "sequential"], default="parallel")
    h.add_argument("--check", help="comando de verificación (p. ej. 'uv run pytest -q')")
    h.add_argument("--domain", default="general")
    h.add_argument("--harness")
    h.add_argument("--out")
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

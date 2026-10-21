"""Local app startup; keeps optional HTTP dependencies out of legacy commands."""
from pathlib import Path


def launch(args):
    from aiohttp import web
    from agentos.execution.cli import overrides
    from agentos.workbench.server import create_app
    app = create_app(args.cwd, overrides=overrides(args), state_dir=args.state_dir, env_file=args.env_file)
    print(f"Workbench: http://{args.host}:{args.port}/\nMonitor: http://{args.host}:{args.port}/monitor/", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0


def add_command(subparsers):
    from agentos.execution.cli import options
    app = subparsers.add_parser("app", help="Start the local workbench over the shared execution service")
    app.add_argument("--cwd", default=".")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=9410)
    app.add_argument("--env-file", type=Path, help="Explicit server-only credential file; never sent to the browser")
    app.add_argument("--state-dir", type=Path, help="Persistent job metadata directory")
    options(app)
    app.set_defaults(func=launch)

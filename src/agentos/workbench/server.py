"""Thin local HTTP workbench over the shared execution service."""
from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import os
import re
import stat
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from aiohttp import web

from agentos.dashboard.broadcast import EventBroadcaster
from agentos.dashboard.server import DashboardApp
from agentos.events.types import RuntimeEvent
from agentos.execution.config import FIELDS, resolve_config
from agentos.execution.service import execute

APP_DEFAULTS = {"model": "deepseek-flash", "base_url": "https://api.deepseek.com", "harness": "langgraph",
                "profile": "runtime", "thinking": "enabled", "reasoning_effort": "high", "max_tokens": None,
                "max_parallel": 2, "api_key_env": "DEEPSEEK_API_KEY", "context_optimization": "legacy"}
EDITABLE = {"model", "base_url", "harness"}
SAFE_CONFIG = FIELDS - {"api_key_env", "output"}
STORE = web.AppKey("workbench", object)
STATIC = Path(__file__).parent / "static"
PREVIEW_BYTES = 128 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def effective_config(cwd, overrides):
    cwd = Path(cwd).expanduser().resolve()
    configured = {}
    user = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "agentos/config.toml"
    for path in (user, cwd / ".agentos.toml"):
        if path.is_file():
            configured.update(tomllib.loads(path.read_text()))
    fallback = {key: value for key, value in APP_DEFAULTS.items() if key not in configured}
    return resolve_config(cwd, {**fallback, **overrides})


class Workbench:
    def __init__(self, cwd, *, overrides=None, state_dir=None, env_file=None, executor=None):
        self.overrides = dict(overrides or {})
        self.defaults = effective_config(cwd, self.overrides)
        if self.defaults["harness"] != "langgraph":
            raise ValueError("the workbench currently supports only LangGraph")
        self.cwd = Path(self.defaults["cwd"])
        self.api_key_env = self.defaults["api_key_env"]
        if env_file:
            from dotenv import dotenv_values
            path = Path(env_file).expanduser().resolve(strict=True)
            value = dotenv_values(path, interpolate=False).get(self.api_key_env)
            if value:
                os.environ[self.api_key_env] = value
        self.secret = os.environ.get(self.api_key_env, "")
        self.state_dir = Path(state_dir or self.cwd / ".agentos/workbench").expanduser().resolve()
        self.run_root = Path(self.defaults.get("output") or self.cwd / ".agentos/runs").expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._run_fd = os.open(self.run_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.executor = executor or execute
        self.jobs, self.histories = {}, {}
        self.active_task = None
        self.active_job = None
        self.broadcaster = EventBroadcaster()
        self.dashboard = DashboardApp(self.broadcaster,
            workload_matrix_path=Path(os.environ["DASHBOARD_WORKLOAD_MATRIX"]) if os.environ.get("DASHBOARD_WORKLOAD_MATRIX") else None)
        for path in sorted(self.state_dir.glob("job-*.json")):
            if path.is_symlink():
                continue
            try:
                job = json.loads(path.read_text())
                if job["id"] != path.stem or not re.fullmatch(r"job-[a-f0-9]{32}", job["id"]):
                    continue
                self.jobs[job["id"]] = job
                if job["state"] in {"queued", "running"}:
                    summary = self._summary_if_present(job)
                    if summary:
                        job.update(summary=summary, state="completed" if summary["workflow_status"] == "completed" else "failed", error=summary.get("error"))
                    else:
                        job.update(state="failed", error="Server restarted during this job; it was not resumed.")
                    self.save(job)
            except (OSError, ValueError, KeyError):
                continue
        if self.jobs:
            latest = max(self.jobs.values(), key=lambda job: job["created_at"])
            for event in self.events(latest):
                self.broadcaster.publish(RuntimeEvent.from_dict(event))

    def clean(self, value):
        raw = json.dumps(value, ensure_ascii=False)
        if self.secret:
            raw = raw.replace(json.dumps(self.secret, ensure_ascii=False)[1:-1], "[REDACTED]")
        return json.loads(raw)

    def public_config(self, config):
        return self.clean({key: value for key, value in config.items()
                           if key in SAFE_CONFIG or key in {"cwd", "context_window", "generation_limit"}})

    def public_job(self, job):
        return self.clean({key: value for key, value in job.items() if not key.startswith("_")})

    def save(self, job):
        target = self.state_dir / (job["id"] + ".json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.clean(job), ensure_ascii=False, indent=2))
        temporary.replace(target)

    def get(self, job_id):
        if job_id not in self.jobs:
            raise web.HTTPNotFound(reason="Job not found")
        return self.jobs[job_id]

    def _open_run_file(self, job, parts):
        run_id = job.get("run_id")
        if not run_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise FileNotFoundError("No archived run is available")
        fd = os.dup(self._run_fd)
        try:
            components = [run_id, *parts]
            for component in components[:-1]:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = next_fd
            result = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            info = os.fstat(result)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os.close(result)
                raise ValueError("Archived links and special files are not accessible")
            return os.fdopen(result, "rb")
        finally:
            os.close(fd)

    def _run_json(self, job, name):
        with self._open_run_file(job, [name]) as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Run metadata exceeds the workbench read limit")
        return json.loads(data)

    def _summary_if_present(self, job):
        try:
            return self._run_json(job, "summary.json")
        except (OSError, ValueError):
            return None

    def events(self, job):
        if job["id"] not in self.histories:
            result = []
            if job.get("run_id"):
                try:
                    with self._open_run_file(job, ["events.jsonl"]) as stream:
                        for line in stream:
                            try:
                                result.append(self.clean(RuntimeEvent.from_dict(json.loads(line)).to_dict()))
                            except (ValueError, KeyError):
                                continue
                except OSError:
                    pass
            self.histories[job["id"]] = result
        return self.histories[job["id"]]

    def submit(self, payload):
        if self.active_task is not None and not self.active_task.done():
            raise web.HTTPConflict(reason="Another job is still running; wait or cancel it first")
        if not isinstance(payload, dict) or set(payload) - {"task", "cwd", "config"}:
            raise ValueError("Expected task, cwd and editable config fields")
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip() or len(task) > 50000:
            raise ValueError("Task must contain 1–50000 characters")
        values = payload.get("config") or {}
        if not isinstance(values, dict) or set(values) - EDITABLE:
            raise ValueError("Browser config can edit only model, base_url and harness")
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("Model, endpoint and harness must be nonempty strings")
        cwd = payload.get("cwd") or str(self.cwd)
        if not isinstance(cwd, str):
            raise ValueError("cwd must be a directory path")
        candidate_cwd = Path(cwd).expanduser()
        if not candidate_cwd.is_absolute():
            candidate_cwd = self.cwd / candidate_cwd
        config = effective_config(candidate_cwd, {**self.overrides, **values, "api_key_env": self.api_key_env,
                                      "output": str(self.run_root)})
        if config["harness"] != "langgraph":
            raise ValueError("Only the LangGraph harness is currently supported")
        host = urlsplit(config["base_url"]).hostname
        if not self.secret and host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Model credential is not configured. Start app with --env-file or set the server credential environment variable.")
        job = {"id": "job-" + uuid.uuid4().hex, "run_id": None, "state": "queued", "task": task,
               "cwd": config["cwd"], "config": self.public_config(config), "created_at": now(),
               "summary": None, "error": None}
        self.jobs[job["id"]] = job
        self.histories[job["id"]] = []
        self.broadcaster._history.clear()
        self.save(job)
        self.active_job = job["id"]
        self.active_task = asyncio.create_task(self._execute(job, config))
        self.active_task.add_done_callback(lambda task: self._settle_cancel(job, task))
        return job

    def _settle_cancel(self, job, task):
        if task.cancelled() and job["state"] in {"queued", "running"}:
            job.update(state="cancelled", error="Job cancelled", finished_at=now())
            self.save(job)

    async def _execute(self, job, config):
        job.update(state="running", started_at=now())
        self.save(job)
        def listener(event):
            safe = self.clean(event)
            run_id = safe.get("data", {}).get("run_id")
            if run_id:
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
                    raise ValueError("Execution emitted an invalid run ID")
                if job["run_id"] and job["run_id"] != run_id:
                    raise ValueError("Execution emitted inconsistent run IDs")
                if job["run_id"] is None:
                    job["run_id"] = run_id
                    self.save(job)
            self.histories[job["id"]].append(safe)
            self.broadcaster.publish(RuntimeEvent.from_dict(safe))
        try:
            summary = await self.executor(job["task"], cwd=job["cwd"],
                overrides={key: config[key] for key in FIELDS if key in config}, listener=listener)
            if job["run_id"] != summary["run_id"]:
                raise ValueError("Summary run ID does not match the execution event stream")
            job.update(summary=self.clean(summary), state="completed" if summary["workflow_status"] == "completed" else "failed", error=summary.get("error"))
        except asyncio.CancelledError:
            job.update(state="cancelled", error="Job cancelled; completed run summary may be unavailable")
            raise
        except Exception as exc:
            job.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            job["finished_at"] = now()
            self.save(job)

    async def cancel(self, job):
        if job["id"] == self.active_job and self.active_task and not self.active_task.done():
            job["cancel_requested"] = True
            self.save(job)
            self.active_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.active_task), 5)
            except (asyncio.CancelledError, TimeoutError):
                pass
            self._settle_cancel(job, self.active_task)
        return job

    def artifacts(self, job):
        try:
            manifest = self._run_json(job, "artifacts.json")
        except FileNotFoundError:
            return {"files": [], "answers": (job.get("summary") or {}).get("answers", {})}
        files = []
        for change in manifest.get("changes", []):
            name = change["path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
                continue
            row = {"path": name, "sha256": change.get("after_sha256"),
                   "before_sha256": change.get("before_sha256"), "deleted": change.get("after_sha256") is None}
            if not row["deleted"]:
                try:
                    with self._open_run_file(job, ["artifacts", "files", *relative.parts]) as stream:
                        row["size"] = os.fstat(stream.fileno()).st_size
                    row["available"] = True
                except (OSError, ValueError):
                    row["available"] = False
            else:
                row["available"] = False
            files.append(row)
        return {"files": files, "answers": (job.get("summary") or {}).get("answers", manifest.get("answers", {}))}

    def artifact_content(self, job, path):
        allowed = {row["path"]: row for row in self.artifacts(job)["files"]}
        if path not in allowed or not allowed[path].get("available"):
            raise web.HTTPNotFound(reason="Archived artifact is not available")
        with self._open_run_file(job, ["artifacts", "files", *PurePosixPath(path).parts]) as stream:
            size = os.fstat(stream.fileno()).st_size
            data = stream.read(PREVIEW_BYTES)
        try:
            content = codecs.getincrementaldecoder("utf-8")("strict").decode(data, final=size <= len(data))
            binary = "\x00" in content
        except UnicodeDecodeError:
            content, binary = None, True
        return {"path": path, "content": None if binary else content, "binary": binary,
                "size": size, "sha256": allowed[path]["sha256"], "sha256_scope": "recorded archived artifact",
                "preview_sha256": hashlib.sha256(data).hexdigest(), "truncated": size > len(data), "limit_bytes": PREVIEW_BYTES}

    async def close(self):
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
            await asyncio.gather(self.active_task, return_exceptions=True)
        os.close(self._run_fd)


@web.middleware
async def boundary(request, handler):
    store = request.app[STORE]
    try:
        host = urlsplit("http://" + request.host)
        address = request.transport.get_extra_info("sockname") if request.transport else None
        allowed_hosts = {"localhost", "127.0.0.1", "::1"}
        if address:
            allowed_hosts.add(address[0])
        if host.hostname not in allowed_hosts or host.username or host.password or host.path or host.query or host.fragment:
            raise web.HTTPForbidden(reason="Host is not allowed for this local workbench")
        origin = request.headers.get("Origin")
        expected = request.scheme + "://" + request.host
        if origin is not None and origin != expected:
            raise web.HTTPForbidden(reason="Cross-origin requests are not allowed")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if origin != expected:
                raise web.HTTPForbidden(reason="Mutating requests require the same Origin")
            if request.content_type != "application/json":
                raise web.HTTPUnsupportedMediaType(reason="Mutating requests require application/json")
        response = await handler(request)
    except web.HTTPException as exc:
        response = web.json_response(store.clean({"error": exc.reason}), status=exc.status)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        response = web.json_response(store.clean({"error": str(exc)}), status=400)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


def create_app(cwd=".", *, overrides=None, state_dir=None, env_file=None, executor=None):
    store = Workbench(cwd, overrides=overrides, state_dir=state_dir, env_file=env_file, executor=executor)
    app = web.Application(middlewares=[boundary], client_max_size=256 * 1024)
    app[STORE] = store
    def reply(value, status=200):
        return web.json_response(store.clean(value), status=status)
    async def config(request):
        return reply({"defaults": store.public_config(store.defaults), "harnesses": [{"id": "langgraph", "name": "LangGraph"}],
                      "credential_configured": bool(store.secret)})
    async def jobs(request):
        if request.method == "POST":
            return reply(store.public_job(store.submit(await request.json())), 202)
        return reply({"jobs": [store.public_job(job) for job in sorted(store.jobs.values(), key=lambda job: job["created_at"], reverse=True)]})
    async def job(request):
        return reply(store.public_job(store.get(request.match_info["job_id"])))
    async def events(request):
        job = store.get(request.match_info["job_id"])
        after, limit = int(request.query.get("after", 0)), int(request.query.get("limit", 500))
        if after < 0 or not 1 <= limit <= 500:
            raise ValueError("after must be nonnegative and limit must be 1–500")
        values = store.events(job)
        return reply({"events": values[after:after + limit], "next_cursor": min(len(values), after + limit),
                      "state": job["state"], "run_id": job["run_id"]})
    async def cancel(request):
        await request.json()
        return reply(store.public_job(await store.cancel(store.get(request.match_info["job_id"]))))
    async def artifacts(request):
        job = store.get(request.match_info["job_id"])
        return reply(await asyncio.to_thread(store.artifacts, job))
    async def file(request):
        job = store.get(request.match_info["job_id"])
        return reply(await asyncio.to_thread(store.artifact_content, job, request.query.get("path", "")))
    async def static(request):
        name = {"/": "index.html", "/index.html": "index.html", "/workbench.css": "workbench.css", "/workbench.js": "workbench.js"}[request.path]
        return web.FileResponse(STATIC / name)
    async def legacy(request):
        path = "/" if request.path in {"/monitor", "/monitor/"} else request.path
        status, headers, body = await store.dashboard.handle_request("GET", path)
        return web.Response(text=body, status=status, headers=headers)
    async def sse(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        stream = await store.dashboard.sse_generator()
        try:
            async for chunk in stream:
                await response.write(chunk.encode())
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            stream.close()
        return response
    app.router.add_get('/api/workbench/config', config)
    app.router.add_get('/api/workbench/jobs', jobs)
    app.router.add_post('/api/workbench/jobs', jobs)
    app.router.add_get('/api/workbench/jobs/{job_id}', job)
    app.router.add_get('/api/workbench/jobs/{job_id}/events', events)
    app.router.add_post('/api/workbench/jobs/{job_id}/cancel', cancel)
    app.router.add_get('/api/workbench/jobs/{job_id}/artifacts', artifacts)
    app.router.add_get('/api/workbench/jobs/{job_id}/artifacts/file', file)
    for path in ('/', '/index.html', '/workbench.css', '/workbench.js'):
        app.router.add_get(path, static)
    for path in ('/monitor', '/monitor/', '/style.css', '/app.js', '/director.css', '/director.js',
                 '/api/snapshot', '/api/cloud-benchmark', '/api/workload-matrix'):
        app.router.add_get(path, legacy)
    app.router.add_get('/api/events', sse)
    async def cleanup(app):
        await store.close()
    app.on_cleanup.append(cleanup)
    return app

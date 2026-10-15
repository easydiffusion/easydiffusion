# Architecture

This page describes how Easy Diffusion v4 is put together, grounded in the actual code. It covers the overall request flow, the pluggable backend-engine system, the process model, and where everything lives in the repository.

Easy Diffusion is a browser UI plus a local HTTP server that manages one or more *rendering engines*. The UI (plain HTML/JS, no build step) talks to a FastAPI server over REST. The server owns the task queue, render threads, and image-saving logic, and delegates the actual image generation and filtering to a pluggable **backend engine**. Four engines currently exist (`ed_classic`, `ed_diffusers`, `webui`, and `sdkit3`), each behind the same small interface (see [`backend-engines.md`](backend-engines.md)).

The server is launched by uvicorn. `scripts/check_modules.py::launch_uvicorn` computes the bind address and port (default `127.0.0.1:9000`) from the config and calls `uvicorn.run("main:server_api", ...)`. The entry module `ui/main.py` then wires everything up:

```py
from easydiffusion import model_manager, app, server, bucket_manager
from easydiffusion.server import server_api  # required for uvicorn

app.init()               # read config, create plugin dirs, start the backend
server.init()            # mount static dirs and register REST routes
model_manager.init()     # scan/create model folders, download defaults if needed
app.init_render_threads()# load server plugins and start one render thread per device
bucket_manager.init()    # initialise the SQLite bucket store
```

## The pluggable backend system

Backends live in `ui/easydiffusion/backends/`. `ui/easydiffusion/backend_manager.py` discovers them at startup:

- `find_valid_backends(root_dir)` lists candidate modules in the `backends/` directory (and also scans the installation root, so plugins can ship backends).
- `is_valid_backend(file_path)` parses each candidate with Python's `ast` module and accepts it only if the module assigns a top-level dictionary named `ed_info`:

```py
def is_valid_backend(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        node = ast.parse(file.read())

    # Check for presence of a dictionary named 'ed_info'
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "ed_info":
                    return True
    return False
```

- `start_backend()` reads `config["backend"]`, imports the matching module, and calls its `start_backend()`.

The backend interface contract (as implemented in `backends/sdkit_common.py` and `backends/webui_common.py`) is a set of module-level functions and the `ed_info` metadata dict:

| Member | Purpose |
| --- | --- |
| `ed_info` | Metadata dict (`name`, `version`, `type: "backend"`) that marks a module as a backend |
| `create_context()` | Create a render context for a device (in-process backends return an `sdkit.Context`) |
| `start_backend()` / `stop_backend()` | Bring the engine up / shut it down |
| `install_backend()` / `uninstall_backend()` / `is_installed()` | First-run installation and removal of the engine's packages/binaries |
| `ping(timeout)` | Check the engine is alive and responsive |
| `load_model(context, ...)` / `unload_model(...)` | Load models on demand into a context |
| `set_options(context, ...)` | Per-task options (output format/quality, clip skip, streaming, etc.) |
| `generate_images(context, ...)` / `filter_images(context, ...)` | The actual rendering and filtering work |
| `get_url()` | URL of a spawned engine's HTTP endpoint (out-of-process backends) |
| `stop_rendering(context)` | Cooperative cancel signal for a running render |
| `refresh_models()` / `list_controlnet_filters()` | Model list refresh; enumerate supported ControlNet filters |
| `convert_ED_controlnet_filter_name(filter)` | Map Easy Diffusion filter names to engine-specific names |

## Process model

There are two execution models, depending on the engine:

**In-process (sdkit v1/v2 - `ed_classic`, `ed_diffusers`).** The engine's packages are pip-installed into the server's environment (`scripts/check_modules.py` pins e.g. `stable-diffusion-sdkit` 2.1.5 and sdkit 2.1.1). `create_context()` calls `create_sdkit_context(use_diffusers=...)` and returns an `sdkit.Context` per render device; `runtime.py` keeps this thread-local context per render thread. Model loading, generation, and filtering all happen as direct function calls on the context.

**Spawned process (`webui`, `sdkit3`).** Both engines run the actual renderer in a separate process and talk to it over HTTP:

- `webui.py` (the v3.5 engine) creates a dedicated conda env, git-clones Easy Diffusion's fork of Forge (`https://github.com/easydiffusion/stable-diffusion-webui-forge.git`) at a pinned commit (`WEBUI_COMMIT`), and starts it. `webui_common.do_start_backend(was_still_installing, run_fn)` supervises the process, including restarting it if it dies during startup, and `webui_common` functions such as `generate_images()` and `filter_images()` translate Easy Diffusion requests into Forge's `/sdapi/v1/*` calls.
- `sdkit3.py` (the v4 engine) downloads prebuilt native binaries of sdkit v3 (built on stable-diffusion.cpp/ggml) from the easydiffusion/sdkit GitHub releases, and runs the `sdkit` server binary from `<install dir>/backends/sdkit3/<target>/`. Startup flags are derived from the VRAM usage level (e.g. `--offload-to-cpu`, VAE tiling) and user overrides in `backend_config`. The HTTP bridge reuses `webui_common` with `WEBUI_API_PREFIX = "/v1"` and `USE_SDKIT3_API = True`.

In both out-of-process cases the server-side code (`task_manager`, `runtime`, task classes) is unchanged - only the backend module's implementation of the contract differs. See [`backend-engines.md`](backend-engines.md) for per-engine details and installation.

## Request flow at a glance

```
Browser UI (ui/index.html + ui/media/js)
        │  REST (POST /render, /filter, GET /ping, /image/stream/…)
        ▼
FastAPI server (ui/easydiffusion/server.py)
        │  validates & parses into typed request models (types.py)
        ▼
task_manager.py ── per-device render threads (thread_render → runtime.py)
        │  picks tasks from tasks_queue
        ▼
RenderTask.run() (ui/easydiffusion/tasks/render_images.py)
        │  load models → backend.generate_images → backend.filter_images
        ▼
utils/save_utils.py ── images saved to disk; response streamed back to the UI
```

The full walk-through, including queueing, overload protection, cancellation, and live previews, is in [`task-lifecycle.md`](task-lifecycle.md).

## Where things live

| Path | Contents |
| --- | --- |
| `scripts/` | Startup chain: `start.sh`/`.cmd` → `bootstrap.*` → `on_env_start.*` → `on_sd_start.*`, plus `check_modules.py`, `ensure_torchruntime.py`, `get_config.py`, and the backend test scripts (`test_*.py`) |
| `ui/index.html`, `ui/media/` | The frontend: plain HTML/JS/CSS, no build step ([`frontend.md`](frontend.md)) |
| `ui/easydiffusion/` | The server package: `server.py`, `app.py`, `task_manager.py`, `runtime.py`, `types.py`, `backend_manager.py`, `model_manager/`, `tasks/`, `utils/`, `device_manager.py`, `bucket_manager.py` |
| `ui/easydiffusion/backends/` | The four engine backends |
| `ui/plugins/ui/` | Core UI plugins (`*.plugin.js`) |
| `ui/easydiffusion/easydb/` | SQLAlchemy models/crud for the SQLite database (`Bucket`, `BucketFile` tables) |
| `models/` (created at runtime) | Model folders per type |
| `NSIS/`, `build.bat`, `build.sh` | Installer build ([`release-process.md`](release-process.md)) |

## State

Two small persistence/state layers exist besides the config file:

- **easydb** (`ui/easydiffusion/easydb/`): a SQLite database managed by `bucket_manager.py` (`SQLALCHEMY_DATABASE_URL = "sqlite:///" + <install dir>/bucket/bucket.db`), storing `Bucket` and `BucketFile` rows - a generic key/blob store used for caching assets.
- **In-memory caches** (`task_manager.py`): `session_cache` and `task_cache` (`DataCache` instances with TTLs) plus `tasks_queue` and `weak_thread_data` for render-thread bookkeeping. Task data survives only for `TASK_TTL` (30 minutes) after activity - see [`task-lifecycle.md`](task-lifecycle.md).

## The v4 rewrite

Separately from the engines above, the **`v4` branch** is rewriting the server internals: a new `easydiffusion/` package with a cleaner `/v1` REST API, user-scoped tasks, a `ConfigManager`, a task-agnostic `Workers` system, and a pytest test suite. It is work-in-progress and not in production; the pages in this KB describe the production (`main`) architecture unless stated otherwise. See [`v4.md`](v4.md) for what changes and why.

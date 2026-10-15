# Task lifecycle

This page traces a generation request from the REST endpoint to the saved image, through the task queue, per-device render threads, and the backend engine. It complements [`architecture.md`](architecture.md) (overall structure) and [`backend-engines.md`](backend-engines.md) (where `generate_images` actually runs).

The modules involved:

| Module | Role |
| --- | --- |
| `ui/easydiffusion/types.py` | Pydantic request models (`GenerateImageRequest`, `FilterImageRequest`, `TaskData`, …) and legacy-request conversion |
| `ui/easydiffusion/server.py` | REST endpoints; validates and enqueues tasks |
| `ui/easydiffusion/task_manager.py` | Queueing, sessions, render threads, overload protection, TTLs |
| `ui/easydiffusion/runtime.py` | The per-thread render context bound to a device |
| `ui/easydiffusion/tasks/render_images.py`, `tasks/filter_images.py` | Task implementations (model loading, generation, filters) |
| `ui/easydiffusion/utils/save_utils.py` | Image and metadata saving |

## 1. Request types

`GenerateImageRequest` carries the rendering parameters (prompt, negative prompt, seed, width/height, `num_outputs`, `num_inference_steps`, `guidance_scale`, `init_image`/`init_image_mask`, `control_image`, `controlnet_filter`, `ref_images`, sampler/scheduler names, …). `TaskData` (and its subclass `RenderTaskData`) carries everything else the server needs: session id, VRAM level, model selections (`use_stable_diffusion_model`, `use_vae_model`, `use_lora_model`, `use_controlnet_model`, …), post-processing options (`use_face_correction`, `use_upscale`, `block_nsfw`, `filters`/`filter_params`), and streaming preferences (`stream_image_progress`). `SaveToDiskData` carries the save path and metadata format.

Older clients may still send the pre-3.x field layout (`use_face_correction: "GFPGANv1.4"`, `use_upscale`, `mask`, …). `convert_legacy_render_req_to_new()` in `types.py` translates that layout into the current model/filter/param structure - including turning `block_nsfw` and upscaler choices into entries of the `filters` list - and `convert_legacy_controlnet_filter_name()` maps old ControlNet filter names to the active engine's names.

## 2. Endpoint handling

`POST /render` (`render_internal` in `server.py`) parses the request dict into the typed models:

```py
render_req: GenerateImageRequest = GenerateImageRequest.parse_obj(req)
task_data: RenderTaskData = RenderTaskData.parse_obj(req)
models_data: ModelsData = ModelsData.parse_obj(req)
output_format: OutputFormatData = OutputFormatData.parse_obj(req)
save_data: SaveToDiskData = SaveToDiskData.parse_obj(req)
```

It then enforces server policy (a configured `force_save_path` overrides the client's save path) and constructs a `RenderTask`, enqueueing it via `enqueue_task()`. `POST /filter` does the same for `FilterTask` - plus, for convenience filters, it copies upscaler model selections into `filter_params`.

The response immediately returns the queue position, task id, and the streaming URL (`/image/stream/{task_id}`); the UI polls `GET /ping` for session/task status and streams previews from that URL.

## 3. Queueing and overload protection

`task_manager.enqueue_task(task)`:

- Refuses work if no render thread is alive (`ChildProcessError` → HTTP 500).
- Counts the session's pending tasks and rejects with `ConnectionRefusedError` (→ HTTP 503 *Service Unavailable*) when pending tasks exceed `MAX_OVERLOAD_ALLOWED_RATIO × <number of render threads>` (ratio = 2). This is the overload protection that stops a browser from queueing unbounded work.
- Appends the task to `tasks_queue` and sets `idle_event`, waking render threads.

Each task and session live in TTL caches (`task_cache`, `session_cache`) - `DataCache` entries expire after `TASK_TTL` (30 minutes) without activity. `GET /ping` and stream/stop/temp-image endpoints refresh the TTLs via `get_cached_task(..., update_ttl=True)`, so results stay retrievable for half an hour after a session goes quiet, then get garbage-collected (`DataCache.clean()` runs on each render-thread loop iteration).

## 4. Render threads

One render thread runs per active device (`update_render_threads()` → `start_render_thread(device)` → `thread_render(device)`). Which devices are active comes from `config["render_devices"]` (or "auto") via `device_manager.get_device_delta()`.

`thread_render()` first calls `runtime.init(device)`, which creates the backend context for that device and stores it in the thread-local `runtime.context` (`stop_processing`, `temp_images`, model-load error buckets, …). It then loads the default models and settles into a loop:

- `thread_get_next_task()` picks the next task under the manager lock (`LOCK_TIMEOUT` = 15 s - a task fails with a lock error rather than deadlocking). Tasks requesting a specific `render_device` wait for that thread; plain tasks are skipped by the CPU thread while other render devices are alive (CPU is a last resort).
- The task runs under its own `task.lock`; `task.run()` streams progress chunks into `task.buffer_queue`, which `GET /image/stream/{task_id}` drains.
- The `ServerStates` symbol (`Init`, `LoadingModel`, `Online`, `Rendering`, `Unavailable`) tracks what the UI sees on `/ping`.
- `weak_thread_data` holds per-thread device info keyed weakly, so dead threads and their data are collectible.

## 5. Running a render task

`RenderTask.run()` (in `tasks/render_images.py`) does the following on the render thread:

1. Applies server policy: if `block_nsfw` is set in the server config, it forces `task_data.block_nsfw = True` (see [`configuration.md`](configuration.md#protected-keys)).
2. Resolves and (re)loads models - `model_manager.resolve_model_paths()`, `reload_models_if_necessary()` (force-reloading when the VRAM level or clip-skip changes) - and fails fast if a model didn't load (`fail_if_models_did_not_load`).
3. Calls the backend: `backend.set_options(...)` (output format/quality, clip skip, streaming flags) then `backend.generate_images(context, callback=..., output_type="base64", **req.dict())`. The step callback writes each preview to `task.temp_images[i]` and emits a progress event with preview paths of the form `/image/tmp/{task_id}/{i}` - that's the live preview.
4. Applies post-processing: unless the user stopped the task, `backend.filter_images(context, images, filters, filter_params, ...)` runs the requested filters (face correction, upscalers, ControlNet pre/post handling), and `filter_nsfw()` blurs any NSFW-flagged output when `block_nsfw` is on. (Earlier in the task, `filter_nsfw()` is called once on a dummy image to make sure the checker model is loaded.)
5. Saves to disk when requested (`save_images_to_disk()` in `utils/save_utils.py`, with `force_save_path` already substituted), then puts the JSON response on `buffer_queue`.

`FilterTask` (in `tasks/filter_images.py`) follows the same skeleton minus the generation step.

## 6. Stop / cancel

Cancellation is cooperative:

- `GET /image/stop` with a task id sets `task.error = StopAsyncIteration("... stop requested.")`. Without a task id, it sets a global `current_state_error = StopAsyncIteration`.
- During generation, the step callback (`step_callback` in `RenderTask.run`) checks for these errors on every diffusion step and calls `backend.stop_rendering(context)` to interrupt the engine.
- A cancelled task still completes its loop iteration; `task_manager` logs it as *cancelled*, releases the lock, and keeps the response/status available in `task_cache` until TTL expiry.

## 7. Image saving

`utils/save_utils.py` builds save paths from a folder format string (default `$id`, configurable via the `folder_format` setting) with placeholders: `$id` (session id), `$p` (prompt, first 50 chars), `$s` (seed), and time placeholders (`$yyyy $MM $dd $HH $mm $ss`). Filenames are the sanitized prompt plus an image number; metadata (prompt, seed, model, etc.) is written per the chosen `metadata_output_format` (`txt`, `json`, or embedded). `force_save_path` in the server config pins the output folder and removes the client's ability to change or disable auto-save.

## Where engine differences hide

Everything above is engine-independent: the task classes only call the backend contract (`generate_images`, `filter_images`, `set_options`, `stop_rendering`). In-process engines (v2.0/v3.0) run these directly on an sdkit context; the webui and sdkit3 engines translate them to HTTP calls into their spawned processes. See [`backend-engines.md`](backend-engines.md) and [`features.md`](features.md) for per-engine availability of the options flowing through here.

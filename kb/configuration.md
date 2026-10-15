# Configuration

How Easy Diffusion is configured: the `config.yaml` file, the HTTP config API, legacy-key migration, protected keys, and Cloudflare tunnel support. Startup-time behaviour of these settings is covered in [`install-and-startup.md`](install-and-startup.md).

## The config.yaml file

The config file lives at `<install dir>/config.yaml` (created from `scripts/config.yaml.sample` logic; the runtime default set is `APP_CONFIG_DEFAULTS` in `ui/easydiffusion/app.py`). Keys, grouped:

| Key | Values | Effect |
| --- | --- | --- |
| `net.listen_port` | port number (default `9000`) | HTTP port the server binds to |
| `net.listen_to_network` | `true`/`false` | `false` binds to `127.0.0.1` only; `true` binds to all interfaces (`0.0.0.0`, or `net.bind_ip` if set) |
| `models_dir` | path | Where model folders are scanned (default `<install dir>/models`) |
| `update_branch` | `main` / `beta` | Self-update channel (see [`release-process.md`](release-process.md#branch-flow)) |
| `backend` | `ed_diffusers` (default) / `ed_classic` / `webui` / `sdkit3` | Which engine loads at startup (see [`backend-engines.md`](backend-engines.md#switching-engines)) |
| `render_devices` | `auto` or a device list | Which GPUs get render threads; `auto` picks all comparable GPUs (see `device_manager.py`) |
| `vram_usage_level` | `low` / `balanced` / `medium` | VRAM-saving behaviours (affects engine flags and model reloads) |
| `ui.open_browser_on_start` | `true`/`false` | Whether a browser tab opens on each start |
| `force_save_path` | path | Pins the output folder; clients can't change or disable auto-save |
| `force_save_metadata` | `txt` / `json` / `embed` / combos | Forces metadata saving format |
| `folder_format` | format string | Output folder naming (placeholders like `$id`, `$p`, `$s`, `$yyyy` - see [`task-lifecycle.md`](task-lifecycle.md#7-image-saving)) |
| `block_nsfw` | `true`/`false` | Server-side NSFW blur - see [Protected keys](#protected-keys) |
| `backend_config` | dict | Per-engine overrides - see [below](#backend_config-overrides) |

Environment variables provide startup fallbacks when the keys are missing: `SD_UI_BIND_PORT` / `SD_UI_BIND_IP` are read by `getConfig()` when `net` isn't configured. The shell-level overrides `scripts/config.sh` / `config.bat` and `scripts/user_config.sh` / `user_config.bat` are sourced during startup and are the place for runtime-specific environment variables (e.g. ROCm settings - see the wiki's [`Config-settings`](https://github.com/easydiffusion/easydiffusion/wiki/Config-settings) page).

## The config API

Two endpoints serve configuration to the UI:

- **`GET /get/{key}`** - the generic key-value reader (`read_web_data_internal`). Keys: `app_config` (the full merged config, plus a defaulted `models_dir`), `system_info` (devices, hosts, output folder, backend URL), `models`, `modifiers`, `ui_plugins`. Any other key is a 404 (and the bare `/get` returns HTTP 418, an easter egg).
- **`POST /app_config`** - the writer (`set_app_config_internal`, request model `SetAppConfigRequest`). It accepts explicit fields (`update_branch`, `render_devices`, `ui_open_browser_on_start`, `listen_to_network`, `listen_port`, `backend`, `backend_platform`, `models_dir`, `vram_usage_level`) plus any additional properties the client sends (`extra=Extra.allow`) - unknown properties are copied into the config as-is. The merged dict is written back to `config.yaml` via `app.setConfig()` (which preserves the sample file's comments using `ruamel.yaml`). If render devices changed, `app.update_render_threads()` applies them without a restart; the engine choice requires a restart.

The Settings tab is just a client of these two endpoints.

## Legacy-key migration

The config loader has been carried through several formats, and `getConfig()` (in `ui/easydiffusion/app.py`) migrates old installations transparently:

1. **Legacy file location.** Older installs kept the config inside `scripts/`; if `scripts/config.yaml` exists, it is moved to `<install dir>/config.yaml` (the same migration also exists in the standalone `scripts/get_config.py` helper used by startup scripts).
2. **config.json → config.yaml.** If no `config.yaml` exists but an old `config.json` does, it is loaded, re-saved in YAML format, and the JSON file is replaced by a `config.json.txt` note pointing at the new location.
3. **`use_v3_engine` → `backend`.** Configs from the v2→v3 era may have a boolean `use_v3_engine` instead of an engine name. If `backend` is missing:
   ```py
   if "use_v3_engine" in config:
       config["backend"] = "ed_diffusers" if config["use_v3_engine"] else "ed_classic"
   else:
       config["backend"] = "ed_diffusers"
   ```
   `set_app_config_internal` keeps writing the mirror (`config["use_v3_engine"] = req.backend == "ed_diffusers"`) so pre-3.5 code paths keep working.

## Protected keys

`PROTECTED_CONFIG_KEYS = ("block_nsfw",)` in `server.py` - this key can't be changed through the HTTP API's free-form extras. It's enforced the other way too: even if a client sends `block_nsfw: false` with a render request, `RenderTask.run()` overrides it when the server config says so:

```py
if config.get("block_nsfw", False):  # override if set on the server
    self.task_data.block_nsfw = True
```

So an operator who enables `block_nsfw` in `config.yaml` (or edits it only directly in the file) prevents API clients and UI users from disabling NSFW blurring.

## backend_config overrides

`backend_config` carries engine-specific settings applied at startup:

- `backend_config.COMMANDLINE_ARGS` - extra CLI arguments for the spawned engines (e.g. `--opt-split-attention` for the webui engine; parsed into argv by `sdkit3.py` for the v4 engine).
- `backend_config.platform` - forces the v4 engine's backend selection (e.g. the Vulkan option in Settings; `auto` removes the override).
- `backend_config.version` / `backend_config.auto_update` - pin or refresh the sdkit3 binary version (see [`backend-engines.md`](backend-engines.md#how-each-engine-is-installed-and-upgraded)).
- Arbitrary keys are exported as environment variables by `check_modules.py` (`apply_backend_config_env_overrides`), e.g. `FORCE_FULL_PRECISION=1` to work around GPU precision bugs.

## Cloudflare tunnel

For sharing the UI beyond the LAN, `POST /tunnel/cloudflare/start` and `POST /tunnel/cloudflare/stop` manage a Cloudflare quick tunnel via `pycloudflared` (`CloudflareTunnel` class in `server.py`). When active, the tunnel URL is included in `/ping` responses (`response["cloudflare"]`). This exposes your Easy Diffusion instance to the internet - see the wiki's usage notes before enabling it.

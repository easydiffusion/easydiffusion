# Install and startup

How Easy Diffusion installs itself and what happens every time it starts. The shipped installer is deliberately thin - it contains only the bootstrap scripts; the conda environment, Python packages, the UI code, and the models are all downloaded on first run (progress is tracked in `scripts/install_status.txt`).

## The script chain

`start.*` → `bootstrap.*` → `on_env_start.*` → `on_sd_start.*` → `check_modules.py`:

**1. `scripts/start.sh` / `Start Stable Diffusion UI.cmd`.** The user-facing entry point. It refuses to run from a source checkout (the installer is the supported path), then calls `scripts/bootstrap.sh`/`bootstrap.bat` and hands over to `on_env_start`.

**2. `scripts/bootstrap.sh` / `bootstrap.bat`.** Prepares the package manager: if neither the current nor the legacy install env has a conda, it downloads micromamba and installs `conda`, `python=3.9` (and `git` if missing) into `installer_files/env`. Legacy installs (old `stable-diffusion/` folder) are detected and preserved via `LEGACY_INSTALL_ENV_DIR`.

**3. `scripts/on_env_start.sh` / `on_env_start.bat`.** Sets up the environment variables and fetches the application code:

- Sources `scripts/config.sh` / `user_config.sh` (and the `.bat` equivalents) for user overrides, and reads `update_branch` via `scripts/get_config.py` to pick the channel (default `main`).
- If `sd-ui-files/` already exists (previous run), it updates it in place: `git stash` → `git reset --hard` → checkout `origin/$update_branch` → `git pull`. Otherwise it clones `https://github.com/easydiffusion/easydiffusion.git` into `sd-ui-files/` at the configured branch.
- Copies the fresh code into the install dir: `rm -rf ui; cp -Rf sd-ui-files/ui .`, plus the scripts (`on_sd_start`, `bootstrap`, `check_modules.py`, `get_config.py`, `config.yaml.sample`, `webui_console.py`, `ensure_torchruntime.py`, `functions.sh`, …).
- `exec`s `scripts/on_sd_start.*`.

This copy step is what the developer workflow comments out (see [`development.md`](development.md#development-environment)).

**4. `scripts/on_sd_start.sh` / `on_sd_start.bat`.** Activates the conda env (activating the legacy `stable-diffusion/env` instead when that exists), performs compatibility fixes (e.g. disabling legacy `src`/`ldm` folders, DLL copies on Windows), then runs:

- **`python scripts/ensure_torchruntime.py`** - ensures `torchruntime` matches the pinned spec (`torchruntime~=2.4.0`, pip-installing it if needed). torchruntime is the component that installs the right `torch`/`torchvision` build for the detected GPU vendor.
- **`python scripts/check_modules.py --launch-uvicorn`** - the package check and server launch (below).

**5. `scripts/check_modules.py`.** Two jobs:

- `update_modules()` - pip-installs/updates the pinned packages (`modules_to_check` in the script: e.g. sdkit 2.1.1, `stable-diffusion-sdkit` 2.1.5, uvicorn 0.19.0, fastapi, pycloudflared, sqlalchemy 2.0.19, onnxruntime, gfpgan, …). Legacy sdkit installs (≤ 2.0.15.x) are held at diffusers 0.21.4 to avoid a slow huggingface-hub upgrade chain.
- `launch_uvicorn()` - writes the completion markers to `install_status.txt`, runs `torchruntime.configure()`, applies `backend_config` environment overrides, computes the bind IP/port from config (default `127.0.0.1:9000`; `net.listen_to_network` widens it, `net.bind_ip` overrides), sets `SD_UI_PATH` to the `ui/` folder, and launches `uvicorn.run("main:server_api", ...)`.

From there the startup continues in Python: `ui/main.py` calls `app.init()` (read config → `backend_manager.start_backend()`, which installs/updates/starts the configured engine - see [`backend-engines.md`](backend-engines.md#how-each-engine-is-installed-and-upgraded)), `server.init()`, `model_manager.init()` (create model folders, download the default SD 1.5 / GFPGAN / RealESRGAN / VAE models if missing), and `app.init_render_threads()` (load server plugins, start one render thread per device).

## Self-update

There is no separate updater program - updating *is* startup:

- The application code (UI + scripts) is updated by the `on_env_start` git pull of `sd-ui-files`, on whatever branch `update_branch` selects (`main` = stable, `beta` = the beta channel; see [`release-process.md`](release-process.md#branch-flow)).
- Python packages are bumped by `check_modules.update_modules()` against the pinned versions on every start.
- The v4 engine updates its own binaries from the sdkit release manifests when `backend_config.auto_update` is on (see [`backend-engines.md`](backend-engines.md#how-each-engine-is-installed-and-upgraded)).

## The Windows long-path fix

Long install paths can break model file storage on Windows. Two mitigations ship:

- The NSIS installer (`NSIS/sdui.nsi`) checks the registry value `LongPathsEnabled` when the chosen install path exceeds 30 characters; if support is off it offers to enable it (needs admin), otherwise it refuses paths over 30 characters.
- `scripts/win_enable_long_filepaths.ps1` does the same manually (sets `LongPathsEnabled = 1` under `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`). It's shipped for manual/admin use - no startup script invokes it.

## What happens on each startup (timeline)

1. Bootstrap checks the conda/git tooling (fast no-op after first install).
2. `on_env_start` git-updates `sd-ui-files` from the configured channel and refreshes the install dir's `ui/` and scripts.
3. `on_sd_start` activates the env; `ensure_torchruntime.py` keeps torchruntime current.
4. `check_modules.py` verifies/updates pip packages and applies `backend_config` overrides.
5. The configured backend engine's `is_installed()` / `update_backend()` runs (e.g. the v4 engine re-fetches its manifest unless `auto_update` is off; the webui engine ensures its clone exists).
6. The server boots on port 9000, models load on the render threads, and the browser opens.

Related pages: [`configuration.md`](configuration.md) (where the settings above come from), [`backend-engines.md`](backend-engines.md), [`release-process.md`](release-process.md).

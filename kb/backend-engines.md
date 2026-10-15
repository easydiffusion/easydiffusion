# Backend engines

Easy Diffusion ships multiple *rendering engines* ("backends"). Exactly one engine is active at a time; you can switch between them in the **Settings** tab. This matters mostly for hardware and model compatibility - newer engines support newer model families, while older engines remain available so users can keep rendering with their existing models.

All four engines implement the same backend contract (see [`architecture.md`](architecture.md#the-pluggable-backend-system)), so the rest of the server - task queue, render threads, image saving - is engine-independent.

## Engine comparison

| Engine (UI label) | Backend code | Underlying stack | Runs as | Typical models |
| --- | --- | --- | --- | --- |
| **v4 (very experimental)** (`sdkit3`) | `ui/easydiffusion/backends/sdkit3.py` | sdkit v3 - C++ [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) / ggml | Spawned process, HTTP | Z-Image, Flux 1, Flux2-Klein, Lens, PiD, Ernie Image, quantized (GGUF) models |
| **v3.5** (`webui`) | `ui/easydiffusion/backends/webui.py` + `webui_common.py` | Easy Diffusion's fork of [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) at a pinned commit, own conda env | Spawned process, HTTP | SD 1.x / 2.x / XL, Flux, Chroma, LyCORIS, plus A1111/Forge extensions |
| **v3.0** (`ed_diffusers`) | `ui/easydiffusion/backends/ed_diffusers.py` | sdkit v2 (diffusers-based) | In-process | SD 1.x / 2.x / XL |
| **v2.0** (`ed_classic`) | `ui/easydiffusion/backends/ed_classic.py` | sdkit v1 (`stable-diffusion-sdkit`, LDM-based) | In-process | SD 1.x / 2.x |

The labels come from the `backend` parameter definition in `ui/media/js/parameters.js`; the engine shipped as default for new configs is `ed_diffusers` (v3.0), matching `APP_CONFIG_DEFAULTS["backend"]` in `ui/easydiffusion/app.py`. This is a legacy-compatible default, **not a recommendation** - there is no recommended engine right now, and the v4 engine (sdkit3) is the one slated to become the default going forward (see [`v4.md`](v4.md)).

> **Developer note:** sdkit v2 lives on the `main` branch of the [sdkit repo](https://github.com/easydiffusion/sdkit), and sdkit v3 lives on its `v3` branch (for now). The v3.5 engine is a fork of Forge, cloned at a specific commit (`WEBUI_COMMIT` in `webui.py`) rather than tracking upstream releases.

## How each engine is installed and upgraded

**v2.0 / v3.0 (`ed_classic`, `ed_diffusers`).** These are ordinary pip packages installed into the server's Python environment by `scripts/check_modules.py`. `update_modules()` pins sdkit 2.1.1 and `stable-diffusion-sdkit` 2.1.5; legacy installs on sdkit ≤ 2.0.15.x are capped at diffusers 0.21.4. Upgrades happen automatically through the normal startup path (see [`install-and-startup.md`](install-and-startup.md#what-happens-on-each-startup)) - there is no per-engine installer step.

**v3.5 (`webui`).** The first time this engine is selected, `webui.py::install_backend()` creates a dedicated conda environment (`conda create --prefix webui/system`), installs Python 3.10 and git into it, git-clones the pinned Forge fork into `webui/webui/`, and uses torchruntime to install the right torch build inside that env. Because the repo is cloned at a fixed commit, engine upgrades happen when the Easy Diffusion release bumps `WEBUI_COMMIT`. `webui_common.do_start_backend()` then supervises the spawned Forge process (including restarting it if it dies during startup).

**v4 (`sdkit3`).** No Python packages are involved - this engine ships as prebuilt native binaries. `sdkit3.py::update_backend()`:

1. Computes a *target* from the OS, CPU, and GPU vendor (`get_target()`), e.g. a Windows/NVIDIA build.
2. Fetches `<target>-manifest.json` from the GitHub releases of the easydiffusion/sdkit repo (`BACKEND_BINARY_URL_BASE`, default version `DEFAULT_BACKEND_VERSION = "v3.3.0"`, overridable via `backend_config.version` in `config.yaml`).
3. Downloads each file listed in the manifest (as tarballs) into `backends/sdkit3/<target>/`, verifying sha256 checksums and skipping files that are already up to date.
4. `start_backend()` then runs the `sdkit` server binary from that folder, adding flags derived from the VRAM usage level (`--offload-to-cpu`, `--vae-tiling`, VAE tile size for "balanced") plus any user `COMMANDLINE_ARGS` from `backend_config`. `auto_update` in `backend_config` controls whether the manifest is re-fetched on every start (default: yes).

If a manifest is missing for the user's platform, the backend asks the user to request the build on Discord or the sdkit issue tracker.

## Feature gating

Not every feature exists on every engine, and the UI hides options that the active engine doesn't support. The mechanism is simple:

- Elements in `ui/index.html` carry a `data-feature-keys` attribute listing the engines that support them, using keys of the form `backend_<engine>`:

```html
<button id="embeddings-button" class="tertiaryButton smallButton gated-feature"
        data-feature-keys="backend_ed_diffusers backend_webui backend_sdkit3">+ Embedding</button>
```

- After loading the config, `ui/media/js/parameters.js` reads `config.backend`, builds the key `backend_` + backend name, and toggles every `.gated-feature` element's `display` accordingly. For example, embeddings are hidden on the v2.0 engine, and Clip Skip is only offered on the v3.0/v3.5/v4 engines.
- Non-gated adaptations also exist in code, e.g. `parameters.js` switches the image-size step size (64 px for `ed_classic`, 8 px otherwise).

This is why switching engines changes what the Settings and generation screens offer, without any build step. Per-feature availability details are in [`features.md`](features.md).

## Switching engines

In the **Settings** tab, the *Engine to use* dropdown writes the choice via `POST /app_config`. Server-side (`set_app_config_internal` in `ui/easydiffusion/server.py`), this sets `config["backend"]` to the selected module name - and also mirrors the old `use_v3_engine` key for backwards compatibility with pre-3.5 configs:

```py
config["use_v3_engine"] = req.backend == "ed_diffusers"
config["backend"] = req.backend
```

Conversely, `app.getConfig()` migrates old configs: if `backend` is missing but `use_v3_engine` is present, it maps back to `ed_diffusers` or `ed_classic`. After saving, Easy Diffusion must be restarted so `backend_manager.start_backend()` loads the newly selected engine.

Typical reasons to switch: a model family only renders on a newer engine (e.g. Flux on v3.5/v4), or an older engine is needed for models that the newer stack handles differently. When features differ between engines, [`features.md`](features.md) notes which engines support each one; the v4 engine itself is described in [`v4.md`](v4.md#the-sdkit3-engine).

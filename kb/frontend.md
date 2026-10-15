# Frontend

Easy Diffusion's UI is plain HTML/JS/CSS with **no build step and no framework**. Everything the browser runs is shipped as-is:

- `ui/index.html` - the single page (generation screen, settings, image gallery)
- `ui/media/js/` - the JavaScript: `main.js` (wiring and task creation), `parameters.js` (settings definitions and config handling), `engine.js` (the `SD` API: task classes, `/ping` polling, streaming), `task-manager.js` (multi-task queue handling in the browser), `image-modifiers.js`, `searchable-models.js` + `multi-model-selector.js` (model dropdowns), `image-editor.js` (inpainting), `themes.js`, `auto-save.js`, `plugins.js`, `dnd.js`, plus vendored libraries (jQuery, JSZip, marked, …)
- `ui/media/css/` - stylesheets; theming is handled by `ui/media/js/themes.js` (theme selection and dark/light detection) rather than a preprocessor

Because there's no bundler, changes to these files are visible on refresh - which is also why the developer workflow is just symlinking the repo's `ui/` folder into an installed copy (see [`development.md`](development.md)).

## How the frontend talks to the server

The UI is a thin client over the server's REST API ([`architecture.md`](architecture.md)); the main endpoints it uses:

| Endpoint | Used by | Purpose |
| --- | --- | --- |
| `GET /ping` (`?session_id=`) | `engine.js` | Poll server state, device list, and per-session task statuses |
| `POST /render` | `engine.js` (`RenderTask`) | Queue a generation task; response carries the task id and stream URL |
| `POST /filter` | `engine.js` (`FilterTask`) | Queue a filter-only task |
| `GET /image/stream/{task_id}` | `engine.js` (stream reader) | Consume the task's JSON progress stream (step events + final response) |
| `GET /image/tmp/{task_id}/{i}` | progress events | Fetch live-preview images referenced in step events |
| `GET /image/stop?task=` | `engine.js` | Cancel a task |
| `GET /get/app_config` / `POST /app_config` | `parameters.js` | Read and save settings (including the engine choice) |
| `GET /get/models` | `searchable-models.js` | Model list for the dropdowns (optionally scanning for malicious files) |
| `GET /get/modifiers` | `image-modifiers.js` | Image-modifier list (built-in + custom) |
| `GET /get/system_info` | `engine.js` (`SD.getSystemInfo`) | Devices, hosts, output folder info |
| `GET /get/ui_plugins` | `plugins.js` | List of core + user plugin scripts to load |
| `POST /model/merge` | `model-tools` plugin | Model merging |
| `POST /tunnel/cloudflare/*`, `POST /package/tensorrt` | settings UI | Cloudflare tunnel and TensorRT package install |

Live progress and cancellation are the interesting bits: each queued task gets a stream URL, `engine.js` keeps a reader open on it, and step events reference preview images served from `/image/tmp/...`. The server side of this is described in [`task-lifecycle.md`](task-lifecycle.md).

## Adapting to the active engine

The UI must render differently depending on which backend engine is configured. Two mechanisms in `parameters.js` do this:

1. **Feature gating.** Elements in `index.html` declare `data-feature-keys="backend_<engine> ..."`. After the config loads, the UI builds `backend_ + config.backend` and shows/hides every `.gated-feature` element:

```js
const currentBackendKey = "backend_" + config.backend
document.querySelectorAll('.gated-feature').forEach((element) => {
    const featureKeys = element.getAttribute('data-feature-keys').split(' ')
    if (featureKeys.includes(currentBackendKey)) {
        element.style.display = getDefaultDisplay(element)
    } else {
        element.style.display = 'none'
    }
});
```

2. **Code-level adjustments.** A handful of behaviors branch on the engine directly, e.g. `parameters.js` sets the image-size step to 64 px for `ed_classic` and 8 px otherwise, and the settings panel includes per-engine parameters (like `backend_platform` for the v4 engine's Vulkan option).

The engine labels and defaults themselves live in the `backend` parameter definition in `parameters.js`. See [`backend-engines.md`](backend-engines.md#feature-gating) for the backend side of this.

## The plugin system

Plugins are JavaScript (or Python, for the server side) files that extend the UI without touching core code:

- **Core plugins** live in `ui/plugins/ui/` as `*.plugin.js` files (Autoscroll, Modifiers-dnd, Modifiers-wheel, custom-modifiers, image-editor-improvements, lora-prompt-parser, model-tools, modifiers-toggle, release-notes, snow, tiled-image-download). They're served by the server under `/plugins/core/`.
- **User plugins** are dropped into `plugins/ui/` (frontend) and `plugins/server/` (server) in the *installation* directory (not the repo). These directories are created on startup by `app.init()` and served under `/plugins/user/`.

Loading flow:

- The server answers `GET /get/ui_plugins` with the plugin list: `app.getUIPlugins()` scans the core and user UI-plugin directories (`UI_PLUGINS_SOURCES`) for `*.plugin.js` files, skipping user plugins that shadow a core plugin of the same name.
- `ui/media/js/plugins.js` fetches that list and injects the scripts. Plugins integrate by registering into the `PLUGINS` catalog (`PLUGIN_API_VERSION = "1.0"`) - e.g. `PLUGINS['IMAGE_INFO_BUTTONS']` appends extra buttons to the per-image button menu built in `main.js` (which already includes *Use as Input*, *Make Similar Images*, *Upscale*, *Fix Faces*, …), plus hooks like `GET_PROMPTS_HOOK`, `MODIFIERS_LOAD`, `TASK_CREATE`, and `OUTPUTS_FORMATS`.
- **Server plugins** are Python files (`*_plugin.py` or `*_plugin/` packages) in `plugins/server/`, loaded by `app.load_server_plugins()` after render threads init. A server plugin can hook engine internals - the documented example is overriding `sdkit.generate.image_generator.get_cond_and_uncond` to change how prompts are processed. Frontend and server plugin mechanisms are independent; a server plugin needs no UI code and vice versa.

The existing core plugins are good reading for how each hook works; the plugin API surface is also documented for users on the wiki ([`UI-Plugins`](https://github.com/easydiffusion/easydiffusion/wiki/UI-Plugins)). To write your own, start from [`development.md`](development.md#how-to-write-plugins) for the setup and [`features.md`](features.md) for the features you'd extend.

> The v4 rewrite on the `v4` branch keeps this frontend and its plugin compatibility (the UI was adapted to the new API data format while retaining the old JS-side data structures for plugins). See [`v4.md`](v4.md).

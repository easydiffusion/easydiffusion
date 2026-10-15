# Easy Diffusion - developer knowledge base

Easy Diffusion is a Stable Diffusion frontend that ships as a **1-click installer**: a browser-based web UI served by a local FastAPI server, which hosts pluggable **rendering engines** (its own sdkit-based engines plus a Forge fork and a native C++ engine) behind a common interface - handling model management, a task queue with per-device render threads, live previews, and image saving.

This `kb/` is for developers working on the codebase. It documents the production architecture on `main`; user documentation lives on the [wiki](https://github.com/easydiffusion/easydiffusion/wiki), and day-to-day discussion happens on [Discord](https://discord.com/invite/u9yhsFmEkB).

| Page | What it covers |
| --- | --- |
| [`architecture.md`](architecture.md) | Overall structure: UI → FastAPI server → pluggable backends, the backend contract, process model, repo layout |
| [`backend-engines.md`](backend-engines.md) | The four engines (v2.0, v3.0, v3.5, v4), how each installs/upgrades, feature gating, engine switching |
| [`task-lifecycle.md`](task-lifecycle.md) | Request flow end-to-end: queueing, render threads, overload protection, cancellation, live preview, saving |
| [`frontend.md`](frontend.md) | The no-build UI, how it talks to the server, engine adaptation, and the plugin system |
| [`models.md`](models.md) | Model folders, detection/classification, per-type notes, merging, per-engine model families |
| [`configuration.md`](configuration.md) | `config.yaml`, the config API, legacy-key migration, protected keys, Cloudflare tunnels |
| [`install-and-startup.md`](install-and-startup.md) | The thin installer, startup script chain, torchruntime, self-update, long-path fix |
| [`features.md`](features.md) | Feature tour (prompting, generation modes, filters, conveniences) with per-engine availability |
| [`development.md`](development.md) | Dev environment setup, conventions, plugin authoring pointers |
| [`testing.md`](testing.md) | Existing tests, how to run them, pre-PR checks (and the lack of a working UI suite) |
| [`release-process.md`](release-process.md) | Branch flow, installer builds, versioning, sdkit3 binary releases |
| [`v4.md`](v4.md) | The work-in-progress v4 rewrite: new `/v1` API, workers, config manager, tests |

**Suggested reading order for a new developer:** 1. [`architecture.md`](architecture.md) → 2. [`backend-engines.md`](backend-engines.md) → 3. [`task-lifecycle.md`](task-lifecycle.md) → 4. [`development.md`](development.md). Then read the pages matching what you're working on; [`v4.md`](v4.md) is an appendix for anyone following the in-progress rewrite.

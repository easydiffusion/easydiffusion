# Features

A tour of Easy Diffusion's feature set, noting where each is available per engine. Shared internals (queueing, streaming, saving) are described once in [`task-lifecycle.md`](task-lifecycle.md) and only referenced here. How per-engine hiding works is in [`backend-engines.md`](backend-engines.md#feature-gating).

## Prompting

- **Image modifiers** (`Image Modifiers` panel): style keywords attached to prompts. The built-in list ships in `ui/modifiers.json`; users can add their own by dropping images into the `modifiers/` folder in the installation directory - a file named `flower.portrait.png` (portrait extensions: `.portrait`, `_portrait`, ` portrait`, `-portrait`; landscape analogously) provides per-aspect-ratio thumbnails. Loading and merging is `app.get_image_modifiers()` on the server, `image-modifiers.js` in the UI.
- **Negative prompts** and **weighted prompt syntax** follow the standard Stable Diffusion conventions - documented for users in [`Writing-prompts`](https://github.com/easydiffusion/easydiffusion/wiki/Writing-prompts) and [`Prompt-Syntax`](https://github.com/easydiffusion/easydiffusion/wiki/Prompt-Syntax).
- **Matrix/sets syntax** (`prompt A [variant1|variant2]`, matrix separators) expands one request into many tasks - `getPrompts()` in `main.js` performs the expansion client-side, then `task-manager.js` schedules the resulting batch.
- **Multiple-prompts file**: the *Load from a file* button (`promptsFromFileBtn`) reads a text file of prompts (one per line) and creates a task per line.

Available on all engines (prompt expansion happens before the backend sees anything).

## Generation modes

- **Text-to-image** - the base mode, all engines.
- **Image-to-image** - supply an *Initial Image* (via `init_image`), optionally scaled by *prompt strength*; all engines.
- **Inpainting** - mask-based editing via the built-in image editor (`image-editor.js`, `init_image_mask`); all engines (the v3.5 engine had dedicated fixes for mask resizing/output size - see `CHANGES.md` 3.5.12/3.5.4).
- **ControlNet** - guide generation with a control image and a preprocessor filter. Available on the v3.0, v3.5 and v4 engines (the v2.0 engine lacks it - the model selector is feature-gated). Each engine names its preprocessors differently; the request's `controlnet_filter` is translated through the backend contract (`list_controlnet_filters()` / `convert_ED_controlnet_filter_name()`), and the dropdown options in `index.html` are gated per engine (e.g. several pose/face filters are v3.5-only).
- **Reference images** - `ref_images` on the request carries reference images for vision-capable models; currently exposed only on the **v4** engine (`data-feature-keys="backend_sdkit3"`).

## Post-processing filters

Applied by the backend after generation (or standalone via `POST /filter`):

- **Face correction** - *Fix incorrect faces and eyes* (`use_face_correction`): GFPGAN or CodeFormer model. The *Upscale Faces* sub-option (CodeFormer face resolution improvement) is v3.0-only in the current UI.
- **Upscaling** - *Scale up by* (`use_upscale` + `upscale_amount`): RealESRGAN (default x4plus and anime models auto-download), plus the additional samplers added in v3.5 (SwinIR, ScuNET, Nearest, Lanczos, ESRGAN - see `CHANGES.md`).
- **NSFW checker** - when `block_nsfw` is on (server config; see [`configuration.md`](configuration.md#protected-keys)), flagged images are blurred before display and saving.
- Filters chain: a single task can request face correction + upscaling + NSFW blur; the legacy request converter composes the `filters` list for old clients.

## Convenience features

- **Task queue** - queue many tasks; the server renders them across per-device threads with overload protection and cancellation. Internals in [`task-lifecycle.md`](task-lifecycle.md#3-queueing-and-overload-protection).
- **Live preview** - watch the image form during diffusion (step events stream through `/image/stream/{task_id}`, previews fetched from `/image/tmp/...`).
- **Use as Input** - send a finished image back as the next task's init image, for iterative (loopback-style) refinement (`onUseAsInputClick` in `main.js`).
- **Make Similar Images** - creates five image-to-image variations of the selected image (prompt strength 0.7, fresh random seeds; `onMakeSimilarClick`).
- **Draw another 25 steps** - continue an image's diffusion (`onContinueDrawingClick`).
- **Undo/Redo of filters** - per-image filter undo buffer in `main.js`.
- **Save to disk** - auto-save rendered images with folder-format placeholders (`$id`, `$p`, `$s`, time placeholders) and metadata formats (`txt`/`json`/embedded); the server can pin the folder via `force_save_path` ([`configuration.md`](configuration.md)).
- **Auto-save of settings** - UI settings persist in the browser (`auto-save.js`), with a "reset settings" path.
- **Multi-GPU** - multiple render threads on multiple devices via `render_devices` config (see [`configuration.md`](configuration.md) and the wiki's [`Run-on-Multiple-GPUs`](https://github.com/easydiffusion/easydiffusion/wiki/Run-on-Multiple-GPUs)).
- **Sampler/scheduler choice** - per-engine: the sampler dropdown options are feature-gated in `index.html` (e.g. DPM++ options on v3.5), and the scheduler picker exists only on the v3.5/v4 engines.

For the model-related features (LoRAs, embeddings, merging, upscaler models) see [`models.md`](models.md); for engine internals see [`backend-engines.md`](backend-engines.md).

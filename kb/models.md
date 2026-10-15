# Models

How Easy Diffusion discovers, classifies, and serves models, and which model families each engine can render. See [`backend-engines.md`](backend-engines.md) for the engines themselves.

## The models folder

By default models live under `<install dir>/models/`, overridable via the `models_dir` config key (see [`configuration.md`](configuration.md#the-config-yaml-file)). Within it, one folder per model type - the list lives in `ui/easydiffusion/model_manager/__init__.py` (`KNOWN_MODEL_TYPES`):

- `stable-diffusion/` - the main models
- `lora/`, `embeddings/`, `controlnet/`, `vae/`, `hypernetwork/`, `text-encoder/`
- Face-correction and upscaling models: `codeformer/`, `gfpgan/`, `realesrgan/`

`get_model_dirs(model_type)` returns the possible directories per type: `models/<type>`, plus a WebUI-compatible alternate name from `ALTERNATE_FOLDER_NAMES` (e.g. `Stable-diffusion`, `VAE`, `Lora`, `ControlNet`, `Codeformer`, `GFPGAN`, `RealESRGAN`, `text_encoder`) when that folder also exists - so drop-in folders from A1111/Forge installs work too.

Recognised file extensions per type are in `MODEL_EXTENSIONS` (e.g. `.ckpt`/`.safetensors` for stable-diffusion, also `.gguf` for stable-diffusion and text encoders on newer engines, `.pt`/`.bin`/`.safetensors` for embeddings).

The UI's model pickers are `ui/media/js/searchable-models.js` (searchable dropdown) and `multi-model-selector.js` (multi-select, e.g. for stacking LoRAs); they're populated from `GET /get/models`, which triggers `backend.refresh_models()` and returns `model_manager.list_models()`.

## How models are detected and classified

`list_models()` (in `ui/easydiffusion/model_manager/list_models.py`) walks the model directories per type, and for each file attaches metadata:

- `m["model"]` / `m["name"]` - the path-without-extension, used as the model identifier everywhere (requests, config).
- `m["tags"]` - the model type plus, for `stable-diffusion`, the detected model class (see below), which the UI uses to group and filter models.

The class detection is `identify_model_type()` in `ui/easydiffusion/utils/model_identifier.py`. Instead of trusting filenames, it reads the file's key names - the safetensors header (or the GGUF header for `.gguf` files) - and matches them against `CHECKPOINT_KEY_NAMES` entries derived from diffusers' own single-file heuristics. That's how models get tagged `v1`, `v2`, `xl_base`, `sd3`, `flux`, `sana`, `hidream`, and so on in the dropdown regardless of what the file is called.

Config selection (which YAML/config a model gets when loaded) is delegated to the active engine: the `model_params` passed with each request (e.g. clip-skip, TensorRT build options for the v3.0 engine) and the engine's own loader handle it; Easy Diffusion itself only resolves paths and detects types.

## VAEs

`models/vae/` holds `.vae.pt`/`.ckpt`/`.safetensors` checkpoints (SD 1.x/2.x VAEs, the Flux `ae` VAE appears prefilled in `PREFILLED_MODELS`). Selected per-task via `use_vae_model`; like all model selections it's part of `ModelsData` and resolved by `model_manager.resolve_model_paths()` at task time.

## LoRAs / LyCORIS

LoRAs go into `models/lora/` (or `Lora/`). Multiple LoRAs can be applied per task (`use_lora_model` accepts a list, with per-LoRA `lora_alpha` weights in `GenerateImageRequest`). LyCORIS variants (LoCon, Hada, IA3, Lokr) are supported by the v3.5 engine (see the v3.5 section of `CHANGES.md`); older engines only handle plain LoRAs.

## Textual-inversion embeddings

Embeddings live in `models/embeddings/` (recognized extensions: `.pt`, `.bin`, `.safetensors`). The token used to trigger an embedding in a prompt is derived from the filename (`get_embedding_token()` from sdkit's embedding loader, applied in `set_model_metadata`), and `use_embeddings_model` selects them per task. Embeddings are unavailable on the v2.0 engine - the `+ Embedding` button in the UI is feature-gated to the newer engines.

## ControlNet models

ControlNet models go into `models/controlnet/` (or `ControlNet/`). A request pairs the model (`use_controlnet_model`) with a *filter name* (`controlnet_filter` / `control_filter_to_apply`) identifying the preprocessor (canny, openpose, depth, …). Because each engine names its preprocessors differently, the backend contract includes `list_controlnet_filters()` and `convert_ED_controlnet_filter_name()` - `types.py` converts legacy ED filter names to the active engine's names before dispatch. The v3.5 engine brought 42 new ControlNet filters (see `CHANGES.md` v3.5).

## Upscaler and face-correction models

Three directories, three purposes: `models/codeformer/` and `models/gfpgan/` power *Fix Faces* (`use_face_correction`), and `models/realesrgan/` powers *Upscale* (`use_upscale`, with `upscale_amount`). Default models (GFPGANv1.4, RealESRGAN_x4plus and the anime variant, CodeFormer) are auto-downloaded on first run (`DEFAULT_MODELS` in `model_manager/__init__.py`). The NSFW checker is *not* a user model - it's fetched at runtime by `ui/easydiffusion/utils/nsfw_checker.py`.

## Text encoders

Newer model families (Flux etc.) use separate text encoders, stored in `models/text-encoder/` (`t5xxl_fp16`, `clip_l`, `clip_g` - prefilled entries in `PREFILLED_MODELS`). Selection is via `use_text_encoder_model`, and only the newer engines use them.

## Model merging

`POST /model/merge` (server: `model_merge_internal` in `server.py`) calls sdkit's `train.merge_models` on two selected stable-diffusion models, producing a third at a given `ratio` (with optional fp16). The UI for this lives in the `model-tools` core plugin (`ui/plugins/ui/model-tools.plugin.js`), including the guidance about merging only compatible model families.

## Model families per engine

Documented model support per engine (grounded in the Settings note in `ui/media/js/parameters.js` and the `CHANGES.md` entries):

- **v2.0** - SD 1.x / 2.x (LDM-based sdkit v1).
- **v3.0** - SD 1.x / 2.x, SDXL (diffusers-based sdkit v2).
- **v3.5** - the broadest set: SDXL, SD3, Flux, Chroma (including quantized bnb/nf4 variants), LyCORIS (LoCon, Hada, IA3, Lokr), plus A1111/Forge extension ecosystem. See the v3.5 major changes in `CHANGES.md`.
- **v4** - Z-Image, Flux 1, Flux2-Klein, Lens, PiD, Ernie Image / Ernie Image Turbo (added across 4.0.0 → 4.4.0; 4.4.0 = sdkit v3.3).

When a model only renders on some engines, it's usually because the underlying diffusion library gained support for it - switch engines from Settings (see [`backend-engines.md`](backend-engines.md#switching-engines)).

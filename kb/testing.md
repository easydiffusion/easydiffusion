# Testing

What test coverage exists, how to run it, and what to check before submitting a PR. For the dev environment, see [`development.md`](development.md).

## Backend tests (main branch)

Three unittest-style scripts live in `scripts/`; they run from a repo checkout:

| Script | Covers |
| --- | --- |
| `scripts/test_backend_config_overrides.py` | `check_modules.apply_backend_config_env_overrides()` - `backend_config` keys (e.g. `COMMANDLINE_ARGS` as string or list) map correctly onto environment variables |
| `scripts/test_webui_alternate_field_names.py` | UI/labelling consistency for the alternate field names ("Guidance Scale (CFG)", "Prompt Strength (Denoising Strength)") across the UI label, task summary, text import, and metadata export |
| `scripts/test_webui_face_filters.py` | Payloads sent to the Forge backend for face filters - face-only filters (GFPGAN/CodeFormer) must not trigger an implicit upscale, and explicit upscale settings are still forwarded |

Run them with unittest, e.g. from the repo root:

```sh
python -m unittest scripts/test_backend_config_overrides.py
python -m unittest scripts/test_webui_alternate_field_names.py
python -m unittest scripts/test_webui_face_filters.py
```

(The scripts locate `check_modules.py` via `importlib` relative to their own path, so they don't depend on a working install.)

`ui/easydiffusion/backends/test_backend.py` is a manual harness for exercising a backend directly - useful while debugging an engine, but not part of an automated suite.

## Integration tests (v4 branch only)

A real test suite is being built as part of the v4 rewrite on the `v4` branch - pytest-based, in `tests/`:

- `tests/integration/test_worker_management.py` - the full workflow of worker lifecycle: workers starting on app startup, and dynamically changing (via `PUT /v1/config`) when configuration changes, including device selection and overload behaviour.
- Unit tests alongside: `tests/test_config.py`, `tests/test_server.py`, `tests/test_workers.py`, `tests/test_models.py`, `tests/test_device_utils.py`, with fixtures in `tests/conftest.py` and `tests/support.py` (a dummy backend used to run tasks end-to-end without real models).

To run them: check out the `v4` branch, install pytest, and run `pytest tests/` from the repo root. This suite only exists on `v4`; it is part of the rewrite described in [`v4.md`](v4.md), not of the production branch documented elsewhere in this KB.

## UI tests

**There is currently no working UI test suite.** The Jasmine-based UI test setup that once existed is broken and unused - do not rely on it, extend it, or treat it as an option when making UI changes. UI changes are verified manually in the browser (refresh after each change, per the [`development.md`](development.md) workflow).

## Before a PR

1. Run the three backend test scripts above (they're fast and dependency-light).
2. If you touched backend-facing code, smoke-test every affected engine. The engine adapters share code (`sdkit_common.py` for the in-process engines, `webui_common.py` reused by both `webui` and `sdkit3`), so a change in a "shared" file can regress multiple engines at once - switch engines in Settings and render one image on each affected engine.
3. If you touched shared task code (`task_manager`, `types`, request conversion), check both the current and the legacy request field names - the legacy conversion paths (`convert_legacy_render_req_to_new` etc.) are exercised by real clients.
4. Update `CHANGES.md` ([`release-process.md`](release-process.md#version-scheme)) and discuss anything non-obvious on Discord first ([`development.md`](development.md#conventions)).

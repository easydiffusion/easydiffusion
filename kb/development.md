# Development

Setting up a development environment, the project's conventions, and where to go next. For architecture background, read [`architecture.md`](architecture.md) first.

## Repo tour

A condensed version of the layout table in [`architecture.md`](architecture.md#where-things-live): `ui/easydiffusion/` is the server package, `ui/` (index.html + `media/`) the frontend, `scripts/` the startup chain and backend tests, `ui/easydiffusion/backends/` the engine adapters, `ui/plugins/ui/` the core UI plugins, `NSIS/` + `build.bat`/`build.sh` the installer build.

## Development environment

The project has no build system and no dev-server tooling; the workflow from `CONTRIBUTING.md` is: keep a normal (installer-based) Easy Diffusion installation running, and point its `ui/` folder at your repo clone so edits are visible on browser refresh.

Steps (from `CONTRIBUTING.md`; use `.sh` or `.bat` per your OS):

1. Install Easy Diffusion normally using the usual installation process, and confirm you can generate images on `localhost:9000`.
2. Clone the repository somewhere else (`git clone https://github.com/easydiffusion/easydiffusion.git`).
3. Stop the server, and edit the *installed copy's* `scripts/on_env_start.sh` (or `on_env_start.bat`) to comment out the lines that overwrite the install's code from `sd-ui-files`:

   ```sh
   # rm -rf ui
   # cp -Rf sd-ui-files/ui .
   # cp sd-ui-files/scripts/on_sd_start.sh scripts/
   # cp sd-ui-files/scripts/start.sh .
   ```

4. Comment out the top of the installed copy's `scripts/on_sd_start.*` that copies `on_env_start` back over your edit (e.g. `@rem @copy sd-ui-files\scripts\on_env_start.bat scripts\ /Y`).
5. Delete the install's `ui/` folder, then symlink the repo's one in:

   ```sh
   ln -s /projects/easydiffusion-repo/ui /projects/easydiffusion-install/ui
   ```

   or on Windows:

   ```bat
   mklink /J \projects\easydiffusion-install\ui \projects\easydiffusion-repo\ui
   ```

6. Start the installed app again - changes in the repo's `ui/` folder now run live.

Backend (server Python) changes live in `ui/easydiffusion/`, which the install imports from the `ui` folder you just symlinked - so the same trick covers server code too. Installer/startup-script changes are tested by running the relevant scripts, or by walking through the installer build ([`release-process.md`](release-process.md#building-installers)).

## Conventions

- **Update `CHANGES.md` in every PR** that users would notice, with a dated, per-engine version entry (see [`release-process.md`](release-process.md#version-scheme) for the format).
- **Discussion happens on Discord** ([invite](https://discord.com/invite/u9yhsFmEkB)); issues on GitHub for bugs/features.
- Run the backend test scripts and smoke-test other engines before submitting - see [`testing.md`](testing.md).

## How to write plugins

Plugin authoring is covered in [`frontend.md`](frontend.md#the-plugin-system) (loading mechanism, `PLUGINS` catalog hooks like `IMAGE_INFO_BUTTONS`, server plugins in `plugins/server/`). The core plugins in `ui/plugins/ui/` are the reference examples; the user-facing guide is the wiki's [`UI-Plugins`](https://github.com/easydiffusion/easydiffusion/wiki/UI-Plugins) page.

## Related pages

- [`testing.md`](testing.md) - what tests exist and what to check before a PR.
- [`release-process.md`](release-process.md) - branches, installers, versioning.
- [`v4.md`](v4.md) - the in-progress server rewrite, which has its own (pytest) test suite.

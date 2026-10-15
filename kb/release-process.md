# Release process

How Easy Diffusion gets from `main` to users' machines: branch flow, installer builds, versioning, and the special-case sdkit3 backend binaries. See also [`install-and-startup.md`](install-and-startup.md) for what users' installs do at startup.

## Branch flow

- **`main`** is the stable channel; **`beta`** is the experimental one.
- Features land in `beta` first, are tested by beta users, and are merged into `main` once proven. The current setup (since the v3.5/v4 engine merge, per `CHANGES.md`: *"Merge v3.5 and v4 engine into the main branch. This brings back the streamlined release process of landing new features first in beta, and then merging into the main branch after testing."*) applies this to all engine work on the main line.
- Users pick a channel with the `update_branch` config key (`main` or `beta`): `on_env_start` pulls and checks out that branch at every startup ([`install-and-startup.md`](install-and-startup.md#self-update)). The UI exposes this as the *Beta Channel* setting; releases announce on Discord (`#beta` for beta feedback).

## Building installers

Installer scripts: `build.bat` (Windows), `build.sh` (Linux/macOS). **Build on the matching OS** - never build the Windows installer under WSL; an Ubuntu VM is fine for the Linux build on a Windows host.

**Windows (`build.bat`):**

1. Verifies makensis (NSIS-BI) is on the PATH, and asks for an output folder.
2. Stages the thin installer into the output folder (`scripts/on_env_start.bat`, `config.yaml.sample`, the launcher `.cmd`, licenses, NSIS assets).
3. Runs `bootstrap.bat` there (conda/git env), then launches the installer's own `Start Stable Diffusion UI.cmd` to produce a full working installation (packages, UI code, models).
4. Packages the *working installation* with NSIS (`NSIS/sdui.nsi`, with torch uninstalled from the env first so the installer stays thin - see the script's prompts).

The `NSIS/` folder needs the NSIS plugins `amd64-unicode`, `x86-ansi`, `x86-unicode` in `nsis/Plugins` for the build to succeed.

**Linux/macOS (`build.sh`):** stages `dist/linux-mac/easy-diffusion/` (bootstrap/on_env_start/start scripts, config sample, licenses) and zips it as `Easy-Diffusion-Linux.zip` / `Easy-Diffusion-Mac.zip`.

Because the staged installer is thin, the "installation" performed in step 3 of the Windows flow is what actually assembles the app - the resulting NSIS installer is that working installation minus torch.

## dist/ output and uploading

- Windows: the NSIS-built installer in the output folder you gave `build.bat`.
- Linux/macOS: the zip files inside `dist/`.

Both are uploaded as assets to a new GitHub Release on the easydiffusion/easydiffusion repo. Release notes point users to the changelog; the user-facing download links live on the website.

## Version scheme

`CHANGES.md` is the release record, with **dated, per-engine version entries** - every PR that changes user-visible behaviour adds an entry under the engine(s) it affects, e.g.:

```
* 4.5.1 - 24 Jul 2026 - Improve Easy Diffusion's startup time by removing unnecessary steps or running them in parallel.
* 3.5.15 - 24 Jul 2026 - Improve Easy Diffusion's startup time by removing unnecessary steps or running them in parallel.
```

Engine versions advance independently (v4 engine: 4.0.0, 4.1.0, …; webui engine: 3.5.x; and so on), which is also what the Settings screen and the UI version labels surface. This convention is mandatory for PRs (see [`development.md`](development.md#conventions)).

## sdkit3 backend binary releases

The v4 engine's binaries are released separately from Easy Diffusion itself:

1. The **sdkit repo** (easydiffusion/sdkit, `v3` branch - see [`backend-engines.md`](backend-engines.md#engine-comparison)) publishes a release containing per-target builds, each accompanied by a `<target>-manifest.json` (target = platform + GPU-vendor combination, computed by `sdkit3.py::get_target()`).
2. The manifest lists the files (tarballs) and their sha256 hashes; `sdkit3.py::update_backend()` fetches it from `https://github.com/easydiffusion/sdkit/releases/download/<version>/<target>-manifest.json`, verifies and downloads the files into `backends/sdkit3/<target>/`.
3. To ship a new sdkit3 build through Easy Diffusion: publish the sdkit release with the manifests, then bump `DEFAULT_BACKEND_VERSION` in `ui/easydiffusion/backends/sdkit3.py` (users can also pin a version via `backend_config.version` in `config.yaml`).

So a v4-engine feature release typically involves *both* a sdkit release (engine code) and an Easy Diffusion change (the version bump / integration), each with their own `CHANGES.md` entry.

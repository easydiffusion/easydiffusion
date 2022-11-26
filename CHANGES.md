# What's new?

## v2.4
### Major Changes
- **Automatic scanning for malicious model files** - using `picklescan`. Thanks @JeLuf
- **Support for custom VAE models**. You can place your VAE files in the `models/vae` folder, and refresh the browser page to use them. More info: https://github.com/cmdr2/stable-diffusion-ui/wiki/VAE-Variational-Auto-Encoder
- **Experimental support for multiple GPUs!** It should work automatically. Just open one browser tab per GPU, and spread your tasks across your GPUs. For e.g. open our UI in two browser tabs if you have two GPUs. You can customize which GPUs it should use in the "Settings" tab, otherwise let it automatically pick the best GPUs. Thanks @madrang . More info: https://github.com/cmdr2/stable-diffusion-ui/wiki/Run-on-Multiple-GPUs
- **Cleaner UI design** - Show settings and help in new tabs, instead of dropdown popups (which were buggy). Thanks @mdiller
- **Progress bar.** Thanks @mdiller
- **Custom Image Modifiers** - You can now save your custom image modifiers! Your saved modifiers can include special characters like `{}, (), [], |`
- Drag and Drop **text files generated from previously saved images**, and copy settings to clipboard. Thanks @madrang
- Paste settings from clipboard. Thanks @JeLuf
- Bug fixes to reduce the chances of tasks crashing during long multi-hour runs (chrome can put long-running background tabs to sleep). Thanks @JeLuf and @madrang
- **Improved documentation.** Thanks @JeLuf and @jsuelwald
- Improved the codebase for dealing with system settings and UI settings. Thanks @mdiller
- Help instructions next to some setttings, and in the tab
- Show system info in the settings tab
- Keyboard shortcut: Ctrl+Enter to start a task
- Configuration to prevent the browser from opening on startup
- Lots of minor bug fixes
- A `What's New?` tab in the UI

### Detailed changelog
* 2.4.15 - 25 Nov 2022 - Experimental support for SD 2.0. Uses lots of memory, not optimized, probably GPU-only.
* 2.4.14 - 22 Nov 2022 - Change the backend to a custom fork of Stable Diffusion
* 2.4.13 - 21 Nov 2022 - Change the modifier weight via mouse wheel, drag to reorder selected modifiers, and some more modifier-related fixes. Thanks @patriceac
* 2.4.12 - 21 Nov 2022 - Another fix for improving how long images take to generate. Reduces the time taken for an enqueued task to start processing.
* 2.4.11 - 21 Nov 2022 - Installer improvements: avoid crashing if the username contains a space or special characters, allow moving/renaming the folder after installation on Windows, whitespace fix on git apply
* 2.4.11 - 21 Nov 2022 - Validate inputs before submitting the Image request
* 2.4.11 - 19 Nov 2022 - New system settings to manage the network config (port number and whether to only listen on localhost)
* 2.4.11 - 19 Nov 2022 - Address a regression in how long images take to generate. Use the previous code for moving a model to CPU. This improves things by a second or two per image, but we still have a regression (investigating).
* 2.4.10 - 18 Nov 2022 - Textarea for negative prompts. Thanks @JeLuf
* 2.4.10 - 18 Nov 2022 - Improved design for Settings, and rounded toggle buttons instead of checkboxes for a more modern look. Thanks @mdiller
* 2.4.9 - 18 Nov 2022 - Add Picklescan - a scanner for malicious model files. If it finds a malicious file, it will halt the web application and alert the user. Thanks @JeLuf
* 2.4.8 - 18 Nov 2022 - A `Use these settings` button to use the settings from a previously generated image task. Thanks @patriceac
* 2.4.7 - 18 Nov 2022 - Don't crash if a VAE file fails to load
* 2.4.7 - 17 Nov 2022 - Fix a bug where Face Correction (GFPGAN) would fail on cuda:N (i.e. GPUs other than cuda:0), as well as fail on CPU if the system had an incompatible GPU.
* 2.4.6 - 16 Nov 2022 - Fix a regression in VRAM usage during startup, which caused 'Out of Memory' errors when starting on GPUs with 4gb (or less) VRAM
* 2.4.5 - 16 Nov 2022 - Add checkbox for "Open browser on startup".
* 2.4.5 - 16 Nov 2022 - Add a directory for core plugins that ship with Stable Diffusion UI by default.
* 2.4.5 - 16 Nov 2022 - Add a "What's New?" tab as a core plugin, which fetches the contents of CHANGES.md from the app's release branch.

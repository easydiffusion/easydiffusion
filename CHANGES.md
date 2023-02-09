# What's new?

## v2.5
### Major Changes
- **Nearly twice as fast** - significantly faster speed of image generation. We're now pretty close to automatic1111's speed. Code contributions are welcome to make our project even faster: https://github.com/easydiffusion/sdkit/#is-it-fast
- **Full support for Stable Diffusion 2.1 (including CPU)** - supports loading v1.4 or v2.0 or v2.1 models seamlessly. No need to enable "Test SD2", and no need to add `sd2_` to your SD 2.0 model file names. Works on CPU as well.
- **Memory optimized Stable Diffusion 2.1** - you can now use Stable Diffusion 2.1 models, with the same low VRAM optimizations that we've always had for SD 1.4. Please note, the SD 2.0 and 2.1 models require more GPU and System RAM, as compared to the SD 1.4 and 1.5 models.
- **6 new samplers!** - explore the new samplers, some of which can generate great images in less than 10 inference steps!
- **Model Merging** - You can now merge two models (`.ckpt` or `.safetensors`) and output `.ckpt` or `.safetensors` models, optionally in `fp16` precision. Details: https://github.com/cmdr2/stable-diffusion-ui/wiki/Model-Merging
- **Fast loading/unloading of VAEs** - No longer needs to reload the entire Stable Diffusion model, each time you change the VAE
- **Database of known models** - automatically picks the right configuration for known models. E.g. we automatically detect and apply "v" parameterization (required for some SD 2.0 models), and "fp32" attention precision (required for some SD 2.1 models).
- **Color correction for img2img** - an option to preserve the color profile (histogram) of the initial image. This is especially useful if you're getting red-tinted images after inpainting/masking.
- **Three GPU Memory Usage Settings** - `High` (fastest, maximum VRAM usage), `Balanced` (default - almost as fast, significantly lower VRAM usage), `Low` (slowest, very low VRAM usage). The `Low` setting is applied automatically for GPUs with less than 4 GB of VRAM.
- **Find models in sub-folders** - This allows you to organize your models into sub-folders inside `models/stable-diffusion`, instead of keeping them all in a single folder.
- **Save metadata as JSON** - You can now save the metadata files as either text or json files (choose in the Settings tab).
- **Major rewrite of the code** - Most of the codebase has been reorganized and rewritten, to make it more manageable and easier for new developers to contribute features. We've separated our core engine into a new project called `sdkit`, which allows anyone to easily integrate Stable Diffusion (and related modules like GFPGAN etc) into their programming projects (via a simple `pip install sdkit`): https://github.com/easydiffusion/sdkit/
- **Name change** - Last, and probably the least, the UI is now called "Easy Diffusion". It indicates the focus of this project - an easy way for people to play with Stable Diffusion.

Our focus continues to remain on an easy installation experience, and an easy user-interface. While still remaining pretty powerful, in terms of features and speed.

### Detailed changelog
* 2.5.15 - 8 Feb 2023 - Allow using 'balanced' VRAM usage mode on GPUs with 4 GB or less of VRAM. This mode used to be called 'Turbo' in the previous version.
* 2.5.14 - 8 Feb 2023 - Fix broken auto-save settings. We renamed `sampler` to `sampler_name`, which caused old settings to fail.
* 2.5.14 - 6 Feb 2023 - Simplify the UI for merging models, and some other minor UI tweaks. Better error reporting if a model failed to load.
* 2.5.14 - 3 Feb 2023 - Fix the 'Make Similar Images' button, which was producing incorrect images (weren't very similar).
* 2.5.13 - 1 Feb 2023 - Fix the remaining GPU memory leaks, including a better fix (more comprehensive) for the change in 2.5.12 (27 Jan).
* 2.5.12 - 27 Jan 2023 - Fix a memory leak, which made the UI unresponsive after an out-of-memory error. The allocated memory is now freed-up after an error.
* 2.5.11 - 25 Jan 2023 - UI for Merging Models. Thanks @JeLuf. More info: https://github.com/cmdr2/stable-diffusion-ui/wiki/Model-Merging
* 2.5.10 - 24 Jan 2023 - Reduce the VRAM usage for img2img in 'balanced' mode (without reducing the rendering speed), to make it similar to v2.4 of this UI.
* 2.5.9 - 23 Jan 2023 - Fix a bug where img2img would produce poorer-quality images for the same settings, as compared to version 2.4 of this UI.
* 2.5.9 - 23 Jan 2023 - Reduce the VRAM usage for 'balanced' mode (without reducing the rendering speed), to make it similar to v2.4 of the UI.
* 2.5.8 - 17 Jan 2023 - Fix a bug where 'Low' VRAM usage would consume a LOT of VRAM (on higher-end GPUs). Also fixed a bug that caused out-of-memory errors on SD 2.1-768 models, on 'high' VRAM usage setting.
* 2.5.7 - 16 Jan 2023 - Fix a bug where VAE files ending with .vae.pt weren't getting displayed. Thanks Madrang, rbertus2000 and JeLuf.
* 2.5.6 - 10 Jan 2023 - `Fill` tool for the Image Editor, to allow filling areas with color (or the entire image). And some bug fixes to the Image Editor. Thanks @mdiller.
* 2.5.6 - 10 Jan 2023 - Find Stable Diffusion models in sub-folders inside `models/stable-diffusion`. This allows you to organize your models into sub-folders, instead of keeping them all in a single folder. Thanks @JeLuf.
* 2.5.5 - 9 Jan 2023 - Lots of bug fixes. Thanks @patriceac and @JeLuf.
* 2.5.4 - 29 Dec 2022 - Press Esc key on the keyboard to close the Image Editor. Thanks @patriceac.
* 2.5.4 - 29 Dec 2022 - Lots of bug fixes in the UI. Thanks @patriceac.
* 2.5.4 - 28 Dec 2022 - Full support for running tasks in parallel on multiple GPUs. Warning: 'Euler Ancestral', 'DPM2 Ancestral' and 'DPM++ 2s Ancestral' may produce slight variations in the image (if run in parallel), so we recommend using the other samplers.
* 2.5.3 - 27 Dec 2022 - Fix broken drag-and-drop for text metadata files (as well as paste in clipboard).
* 2.5.3 - 27 Dec 2022 - Allow upscaling by 2x as well as 4x.
* 2.5.3 - 27 Dec 2022 - Fix broken renders on a second GPU.
* 2.5.3 - 26 Dec 2022 - Add a `Remove` button on each image. Thanks @JeLuf.
* 2.5.2 - 26 Dec 2022 - Fix broken inpainting if using non-square target images.
* 2.5.2 - 26 Dec 2022 - Fix a bug where an incorrect model config would get used for some SD 2.1 models.
* 2.5.2 - 26 Dec 2022 - Slight performance and memory improvement while rendering using SD 2.1 models.
* 2.5.1 - 25 Dec 2022 - Allow custom config yaml files for models. You can put a config file (`.yaml`) next to the model file, with the same name as the model. For e.g. if you put `robo-diffusion-v2-base.yaml` next to `robo-diffusion-v2-base.ckpt`, it'll automatically use that config file.
* 2.5.1 - 25 Dec 2022 - Fix broken rendering for SD 2.1-768 models. Fix broken rendering SD 2.0 safetensor models.
* 2.5.0 - 25 Dec 2022 - Major new release! Nearly twice as fast, Full support for SD 2.1 (including low GPU RAM optimizations), 6 new samplers, Model Merging, Fast loading/unloading of VAEs, Database of known models, Color correction for img2img, Three GPU Memory Usage Settings, Save metadata as JSON, Major rewrite of the code, Name change.

## v2.4
### Major Changes
- **Allow reordering the task queue** (by dragging and dropping tasks). Thanks @madrang
- **Automatic scanning for malicious model files** - using `picklescan`, and support for `safetensor` model format. Thanks @JeLuf
- **Image Editor** - for drawing simple images for guiding the AI. Thanks @mdiller
- **Use pre-trained hypernetworks** - for improving the quality of images. Thanks @C0bra5
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
- Ask for a confimation before clearing the results pane or stopping a render task. The dialog can be skipped by holding down the shift key while clicking on the button.
- Show the network addresses of the server in the systems setting dialog
- Support loading models in the safetensor format, for improved safety

### Detailed changelog
* 2.4.24 - 9 Jan 2022 - Urgent fix for failures on old/long-term-support browsers. Thanks @JeLuf.
* 2.4.23/22 - 29 Dec 2022 - Allow rolling back from the upcoming v2.5 change (in beta).
* 2.4.21 - 23 Dec 2022 - Speed up image creation, by removing a delay (regression) of 4-5 seconds between clicking the `Make Image` button and calling the server.
* 2.4.20 - 22 Dec 2022 - `Pause All` button to pause all the pending tasks. Thanks @JeLuf
* 2.4.20 - 22 Dec 2022 - `Undo`/`Redo` buttons in the image editor. Thanks @JeLuf
* 2.4.20 - 22 Dec 2022 - Drag handle to reorder the tasks. This fixed a bug where the metadata was no longer selectable (for copying). Thanks @JeLuf
* 2.4.19 - 17 Dec 2022 - Add Undo/Redo buttons in the Image Editor. Thanks @JeLuf
* 2.4.19 - 10 Dec 2022 - Show init img in task list
* 2.4.19 - 7 Dec 2022 - Use pre-trained hypernetworks while generating images. Thanks @C0bra5
* 2.4.19 - 6 Dec 2022 - Allow processing new tasks first. Thanks @madrang
* 2.4.19 - 6 Dec 2022 - Allow reordering the task queue (by dragging tasks). Thanks @madrang
* 2.4.19 - 6 Dec 2022 - Re-organize the code, to make it easier to write user plugins. Thanks @madrang
* 2.4.18 - 5 Dec 2022 - Make JPEG Output quality user controllable. Thanks @JeLuf
* 2.4.18 - 5 Dec 2022 - Support loading models in the safetensor format, for improved safety. Thanks @JeLuf
* 2.4.18 - 1 Dec 2022 - Image Editor, for drawing simple images for guiding the AI. Thanks @mdiller
* 2.4.18 - 1 Dec 2022 - Disable an image modifier temporarily by right-clicking it. Thanks @patriceac
* 2.4.17 - 30 Nov 2022 - Scroll to generated image. Thanks @patriceac
* 2.4.17 - 30 Nov 2022 - Show the network addresses of the server in the systems setting dialog. Thanks @JeLuf
* 2.4.17 - 30 Nov 2022 - Fix a bug where GFPGAN wouldn't work properly when multiple GPUs tried to run it at the same time. Thanks @madrang
* 2.4.17 - 30 Nov 2022 - Confirm before stopping or clearing all the tasks. Thanks @JeLuf
* 2.4.16 - 29 Nov 2022 - Bug fixes for SD 2.0 - remove the need for patching, default to SD 1.4 model if trying to load an SD2 model in SD1.4.
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

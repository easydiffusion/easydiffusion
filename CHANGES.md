# What's new?

## v2.5
### Major Changes
- **Nearly twice as fast** - significantly faster speed of image generation. Code contributions are welcome to make our project even faster: https://github.com/easydiffusion/sdkit/#is-it-fast
- **Mac M1/M2 support** - Experimental support for Mac M1/M2. Thanks @michaelgallacher, @JeLuf and vishae.
- **AMD support for Linux** - Experimental support for AMD GPUs on Linux. Thanks @DianaNites and @JeLuf.
- **Full support for Stable Diffusion 2.1 (including CPU)** - supports loading v1.4 or v2.0 or v2.1 models seamlessly. No need to enable "Test SD2", and no need to add `sd2_` to your SD 2.0 model file names. Works on CPU as well.
- **Memory optimized Stable Diffusion 2.1** - you can now use Stable Diffusion 2.1 models, with the same low VRAM optimizations that we've always had for SD 1.4. Please note, the SD 2.0 and 2.1 models require more GPU and System RAM, as compared to the SD 1.4 and 1.5 models.
- **11 new samplers!** - explore the new samplers, some of which can generate great images in less than 10 inference steps! We've added the Karras and UniPC samplers. Thanks @Schorny for the UniPC samplers.
- **Model Merging** - You can now merge two models (`.ckpt` or `.safetensors`) and output `.ckpt` or `.safetensors` models, optionally in `fp16` precision. Details: https://github.com/easydiffusion/easydiffusion/wiki/Model-Merging . Thanks @JeLuf.
- **Fast loading/unloading of VAEs** - No longer needs to reload the entire Stable Diffusion model, each time you change the VAE
- **Database of known models** - automatically picks the right configuration for known models. E.g. we automatically detect and apply "v" parameterization (required for some SD 2.0 models), and "fp32" attention precision (required for some SD 2.1 models).
- **Color correction for img2img** - an option to preserve the color profile (histogram) of the initial image. This is especially useful if you're getting red-tinted images after inpainting/masking.
- **Three GPU Memory Usage Settings** - `High` (fastest, maximum VRAM usage), `Balanced` (default - almost as fast, significantly lower VRAM usage), `Low` (slowest, very low VRAM usage). The `Low` setting is applied automatically for GPUs with less than 4 GB of VRAM.
- **Find models in sub-folders** - This allows you to organize your models into sub-folders inside `models/stable-diffusion`, instead of keeping them all in a single folder. Thanks @patriceac and @ogmaresca.
- **Custom Modifier Categories** - Ability to create custom modifiers with thumbnails, and custom categories (and hierarchy of categories). Details: https://github.com/easydiffusion/easydiffusion/wiki/Custom-Modifiers . Thanks @ogmaresca.
- **Embed metadata, or save as TXT/JSON** - You can now embed the metadata directly into the images, or save them as text or json files (choose in the Settings tab). Thanks @patriceac.
- **Major rewrite of the code** - Most of the codebase has been reorganized and rewritten, to make it more manageable and easier for new developers to contribute features. We've separated our core engine into a new project called `sdkit`, which allows anyone to easily integrate Stable Diffusion (and related modules like GFPGAN etc) into their programming projects (via a simple `pip install sdkit`): https://github.com/easydiffusion/sdkit/
- **Name change** - Last, and probably the least, the UI is now called "Easy Diffusion". It indicates the focus of this project - an easy way for people to play with Stable Diffusion.

Our focus continues to remain on an easy installation experience, and an easy user-interface. While still remaining pretty powerful, in terms of features and speed.

### Detailed changelog
* 3.0.1 - 18 Aug 2023 - Rotate an image if EXIF rotation is present. For e.g. this is common in images taken with a smartphone.
* 3.0.1 - 18 Aug 2023 - Resize control images to the task dimensions, to avoid memory errors with high-res control images.
* 3.0.1 - 18 Aug 2023 - Show controlnet filter preview in the task entry.
* 3.0.1 - 18 Aug 2023 - Fix drag-and-drop and 'Use these Settings' for LoRA and ControlNet.
* 3.0.1 - 18 Aug 2023 - Auto-save LoRA models and strengths.
* 3.0.1 - 17 Aug 2023 - Automatically use the correct yaml config file for custom SDXL models, even if a yaml file isn't present in the folder.
* 3.0.1 - 17 Aug 2023 - Fix broken embeddings with SDXL.
* 3.0.1 - 16 Aug 2023 - Fix broken LoRA with SDXL.
* 3.0.1 - 15 Aug 2023 - Fix broken seamless tiling.
* 3.0.1 - 15 Aug 2023 - Fix textual inversion embeddings not working in `low` VRAM usage mode.
* 3.0.1 - 15 Aug 2023 - Fix for custom VAEs not working in `low` VRAM usage mode.
* 3.0.1 - 14 Aug 2023 - Slider to change the image dimensions proportionally (in Image Settings). Thanks @JeLuf.
* 3.0.1 - 14 Aug 2023 - Show an error to the user if an embedding isn't compatible with the model, instead of failing silently without informing the user. Thanks @JeLuf.
* 3.0.1 - 14 Aug 2023 - Disable watermarking for SDXL img2img. Thanks @AvidGameFan.
* 3.0.0 - 3 Aug 2023 - Enabled diffusers for everyone by default. The old v2 engine can be used by disabling the "Use v3 engine" option in the Settings tab.
* 2.5.48 - 1 Aug 2023 - (beta-only) Full support for ControlNets. You can select a control image to guide the AI. You can pick a filter to pre-process the image, and one of the known (or custom) controlnet models. Supports `OpenPose`, `Canny`, `Straight Lines`, `Depth`, `Line Art`, `Scribble`, `Soft Edge`, `Shuffle` and `Segment`.
* 2.5.47 - 30 Jul 2023 - An option to use `Strict Mask Border` while inpainting, to avoid touching areas outside the mask. But this might show a slight outline of the mask, which you will have to touch up separately.
* 2.5.47 - 29 Jul 2023 - (beta-only) Fix long prompts with SDXL.
* 2.5.47 - 29 Jul 2023 - (beta-only) Fix red dots in some SDXL images.
* 2.5.47 - 29 Jul 2023 - Significantly faster `Fix Faces` and `Upscale` buttons (on the image). They no longer need to generate the image from scratch, instead they just upscale/fix the generated image in-place.
* 2.5.47 - 28 Jul 2023 - Lots of internal code reorganization, in preparation for supporting Controlnets. No user-facing changes.
* 2.5.46 - 27 Jul 2023 - (beta-only) Full support for SD-XL models (base and refiner)!
* 2.5.45 - 24 Jul 2023 - (beta-only) Hide the samplers that won't be supported in the new diffusers version.
* 2.5.45 - 22 Jul 2023 - (beta-only) Fix the recently-broken inpainting models.
* 2.5.45 - 16 Jul 2023 - (beta-only) Fix the image quality of LoRAs, which had degraded in v2.5.44.
* 2.5.44 - 15 Jul 2023 - (beta-only) Support for multiple LoRA files.
* 2.5.43 - 9 Jul 2023 - (beta-only) Support for loading Textual Inversion embeddings. You can find the option in the Image Settings panel. Thanks @JeLuf.
* 2.5.43 - 9 Jul 2023 - Improve the startup time of the UI.
* 2.5.42 - 4 Jul 2023 - Keyboard shortcuts for the Image Editor. Thanks @JeLuf.
* 2.5.42 - 28 Jun 2023 - Allow dropping images from folders to use as an Initial Image.
* 2.5.42 - 26 Jun 2023 - Show a popup for Image Modifiers, allowing a larger screen space, better UX on mobile screens, and more room for us to develop and improve the Image Modifiers panel. Thanks @Hakorr.
* 2.5.42 - 26 Jun 2023 - (beta-only) Show a welcome screen for users of the diffusers beta, with instructions on how to use the new prompt syntax, and known bugs. Thanks @JeLuf.
* 2.5.42 - 26 Jun 2023 - Use YAML files for config. You can now edit the `config.yaml` file (using a text editor, like Notepad). This file is present inside the Easy Diffusion folder, and is easier to read and edit (for humans) than JSON. Thanks @JeLuf.
* 2.5.41 - 24 Jun 2023 - (beta-only) Fix broken inpainting in low VRAM usage mode.
* 2.5.41 - 24 Jun 2023 - (beta-only) Fix a recent regression where the LoRA would not get applied when changing SD models.
* 2.5.41 - 23 Jun 2023 - Fix a regression where latent upscaler stopped working on PCs without a graphics card.
* 2.5.41 - 20 Jun 2023 - Automatically fix black images if fp32 attention precision is required in diffusers.
* 2.5.41 - 19 Jun 2023 - Another fix for multi-gpu rendering (in all VRAM usage modes).
* 2.5.41 - 13 Jun 2023 - Fix multi-gpu bug with "low" VRAM usage mode while generating images.
* 2.5.41 - 12 Jun 2023 - Fix multi-gpu bug with CodeFormer.
* 2.5.41 - 6 Jun 2023 - Allow changing the strength of CodeFormer, and slightly improved styling of the CodeFormer options.
* 2.5.41 - 5 Jun 2023 - Allow sharing an Easy Diffusion instance via https://try.cloudflare.com/ . You can find this option at the bottom of the Settings tab. Thanks @JeLuf.
* 2.5.41 - 5 Jun 2023 - Show an option to download for tiled images. Shows a button on the generated image. Creates larger images by tiling them with the image generated by Easy Diffusion. Thanks @JeLuf.
* 2.5.41 - 5 Jun 2023 - (beta-only) Allow LoRA strengths between -2 and 2. Thanks @ogmaresca.
* 2.5.40 - 5 Jun 2023 - Reduce the VRAM usage of Latent Upscaling when using "balanced" VRAM usage mode.
* 2.5.40 - 5 Jun 2023 - Fix the "realesrgan" key error when using CodeFormer with more than 1 image in a batch.
* 2.5.40 - 3 Jun 2023 - Added CodeFormer as another option for fixing faces and eyes. CodeFormer tends to perform better than GFPGAN for many images. Thanks @patriceac for the implementation, and for contacting the CodeFormer team (who were supportive of it being integrated into Easy Diffusion).
* 2.5.39 - 25 May 2023 - (beta-only) Seamless Tiling - make seamlessly tiled images, e.g. rock and grass textures. Thanks @JeLuf.
* 2.5.38 - 24 May 2023 - Better reporting of errors, and show an explanation if the user cannot disable the "Use CPU" setting.
* 2.5.38 - 23 May 2023 - Add Latent Upscaler as another option for upscaling images. Thanks @JeLuf for the implementation of the Latent Upscaler model.
* 2.5.37 - 19 May 2023 - (beta-only) Two more samplers: DDPM and DEIS. Also disables the samplers that aren't working yet in the Diffusers version. Thanks @ogmaresca.
* 2.5.37 - 19 May 2023 - (beta-only) Support CLIP-Skip. You can set this option under the models dropdown. Thanks @JeLuf.
* 2.5.37 - 19 May 2023 - (beta-only) More VRAM optimizations for all modes in diffusers. The VRAM usage for diffusers in "low" and "balanced" should now be equal or less than the non-diffusers version. Performs softmax in half precision, like sdkit does.
* 2.5.36 - 16 May 2023 - (beta-only) More VRAM optimizations for "balanced" VRAM usage mode.
* 2.5.36 - 11 May 2023 - (beta-only) More VRAM optimizations for "low" VRAM usage mode.
* 2.5.36 - 10 May 2023 - (beta-only) Bug fix for "meta" error when using a LoRA in 'low' VRAM usage mode.
* 2.5.35 - 8 May 2023 - Allow dragging a zoomed-in image (after opening an image with the "expand" button). Thanks @ogmaresca.
* 2.5.35 - 3 May 2023 - (beta-only) First round of VRAM Optimizations for the "Test Diffusers" version. This change significantly reduces the amount of VRAM used by the diffusers version during image generation. The VRAM usage is still not equal to the "non-diffusers" version, but more optimizations are coming soon.
* 2.5.34 - 22 Apr 2023 - Don't start the browser in an incognito new profile (on Windows). Thanks @JeLuf.
* 2.5.33 - 21 Apr 2023 - Install PyTorch 2.0 on new installations (on Windows and Linux).
* 2.5.32 - 19 Apr 2023 - Automatically check for black images, and set full-precision if necessary (for attn). This means custom models based on Stable Diffusion v2.1 will just work, without needing special command-line arguments or editing of yaml config files.
* 2.5.32 - 18 Apr 2023 - Automatic support for AMD graphics cards on Linux. Thanks @DianaNites and @JeLuf.
* 2.5.31 - 10 Apr 2023 - Reduce VRAM usage while upscaling.
* 2.5.31 - 6 Apr 2023 - Allow seeds upto `4,294,967,295`. Thanks @ogmaresca.
* 2.5.31 - 6 Apr 2023 - Buttons to show the previous/next image in the image popup. Thanks @ogmaresca.
* 2.5.30 - 5 Apr 2023 - Fix a bug where the JPEG image quality wasn't being respected when embedding the metadata into it. Thanks @JeLuf.
* 2.5.30 - 1 Apr 2023 - (beta-only) Slider to control the strength of the LoRA model.
* 2.5.30 - 28 Mar 2023 - Refactor task entry config to use a generating method. Added ability for plugins to easily add to this. Removed confusing sentence from `contributing.md`
* 2.5.30 - 28 Mar 2023 - Allow the user to undo the deletion of tasks or images, instead of showing a pop-up each time. The new `Undo` button will be present at the top of the UI. Thanks @JeLuf.
* 2.5.30 - 28 Mar 2023 - Support saving lossless WEBP images. Thanks @ogmaresca.
* 2.5.30 - 28 Mar 2023 - Lots of bug fixes for the UI (Read LoRA flag in metadata files, new prompt weight format with scrollwheel, fix overflow with lots of tabs, clear button in image editor, shorter filenames in download). Thanks @patriceac, @JeLuf and @ogmaresca.
* 2.5.29 - 27 Mar 2023 - (beta-only) Fix a bug where some non-square images would fail while inpainting with a `The size of tensor a must match size of tensor b` error.
* 2.5.29 - 27 Mar 2023 - (beta-only) Fix the `incorrect number of channels` error, when given a PNG image with an alpha channel in `Test Diffusers`.
* 2.5.29 - 27 Mar 2023 - (beta-only) Fix broken inpainting in `Test Diffusers`.
* 2.5.28 - 24 Mar 2023 - (beta-only) Support for weighted prompts and long prompt lengths (not limited to 77 tokens). This change requires enabling the `Test Diffusers` setting in beta (in the Settings tab), and restarting the program.
* 2.5.27 - 21 Mar 2023 - (beta-only) LoRA support, accessible by enabling the `Test Diffusers` setting (in the Settings tab in the UI). This change switches the internal engine to diffusers (if the `Test Diffusers` setting is enabled). If the `Test Diffusers` flag is disabled, it'll have no impact for the user.
* 2.5.26 - 15 Mar 2023 - Allow styling the buttons displayed on an image. Update the API to allow multiple buttons and text labels in a single row. Thanks @ogmaresca.
* 2.5.26 - 15 Mar 2023 - View images in full-screen, by either clicking on the image, or clicking the "Full screen" icon next to the Seed number on the image. Thanks @ogmaresca for the internal API.
* 2.5.25 - 14 Mar 2023 - Button to download all the images, and all the metadata as a zip file. This is available at the top of the UI, as well as on each image. Thanks @JeLuf.
* 2.5.25 - 14 Mar 2023 - Lots of UI tweaks and bug fixes. Thanks @patriceac and @JeLuf.
* 2.5.24 - 11 Mar 2023 - Button to load an image mask from a file.
* 2.5.24 - 10 Mar 2023 - Logo change. Image credit: @lazlo_vii.
* 2.5.23 - 8 Mar 2023 - Experimental support for Mac M1/M2. Thanks @michaelgallacher, @JeLuf and vishae!
* 2.5.23 - 8 Mar 2023 - Ability to create custom modifiers with thumbnails, and custom categories (and hierarchy of categories). More details - https://github.com/easydiffusion/easydiffusion/wiki/Custom-Modifiers . Thanks @ogmaresca.
* 2.5.22 - 28 Feb 2023 - Minor styling changes to UI buttons, and the models dropdown.
* 2.5.22 - 28 Feb 2023 - Lots of UI-related bug fixes. Thanks @patriceac.
* 2.5.21 - 22 Feb 2023 - An option to control the size of the image thumbnails. You can use the `Display options` in the top-right corner to change this. Thanks @JeLuf.
* 2.5.20 - 20 Feb 2023 - Support saving images in WEBP format (which consumes less disk space, with similar quality). Thanks @ogmaresca.
* 2.5.20 - 18 Feb 2023 - A setting to block NSFW images from being generated. You can enable this setting in the Settings tab.
* 2.5.19 - 17 Feb 2023 - Initial support for server-side plugins. Currently supports overriding the `get_cond_and_uncond()` function.
* 2.5.18 - 17 Feb 2023 - 5 new samplers! UniPC samplers, some of which produce images in less than 15 steps. Thanks @Schorny.
* 2.5.16 - 13 Feb 2023 - Searchable dropdown for models. This is useful if you have a LOT of models. You can type part of the model name, to auto-search through your models. Thanks @patriceac for the feature, and @AssassinJN for help in UI tweaks!
* 2.5.16 - 13 Feb 2023 - Lots of fixes and improvements to the installer. First round of changes to add Mac support. Thanks @JeLuf.
* 2.5.16 - 13 Feb 2023 - UI bug fixes for the inpainter editor. Thanks @patriceac.
* 2.5.16 - 13 Feb 2023 - Fix broken task reorder. Thanks @JeLuf.
* 2.5.16 - 13 Feb 2023 - Remove a task if all the images inside it have been removed. Thanks @AssassinJN.
* 2.5.16 - 10 Feb 2023 - Embed metadata into the JPG/PNG images, if selected in the "Settings" tab (under "Metadata format"). Thanks @patriceac.
* 2.5.16 - 10 Feb 2023 - Sort models alphabetically in the models dropdown. Thanks @ogmaresca.
* 2.5.16 - 10 Feb 2023 - Support multiple GFPGAN models. Download new GFPGAN models into the `models/gfpgan` folder, and refresh the UI to use it. Thanks @JeLuf.
* 2.5.16 - 10 Feb 2023 - Allow a server to enforce a fixed directory path to save images. This is useful if the server is exposed to a lot of users. This can be set in the `config.json` file as `force_save_path: "/path/to/fixed/save/dir"`. E.g. `force_save_path: "D:/user_images"`. Thanks @JeLuf.
* 2.5.16 - 10 Feb 2023 - The "Make Images" button now shows the correct amount of images it'll create when using operators like `{}` or `|`. For e.g. if the prompt is `Photo of a {woman, man}`, then the button will say `Make 2 Images`. Thanks @JeLuf.
* 2.5.16 - 10 Feb 2023 - A bunch of UI-related bug fixes. Thanks @patriceac.
* 2.5.15 - 8 Feb 2023 - Allow using 'balanced' VRAM usage mode on GPUs with 4 GB or less of VRAM. This mode used to be called 'Turbo' in the previous version.
* 2.5.14 - 8 Feb 2023 - Fix broken auto-save settings. We renamed `sampler` to `sampler_name`, which caused old settings to fail.
* 2.5.14 - 6 Feb 2023 - Simplify the UI for merging models, and some other minor UI tweaks. Better error reporting if a model failed to load.
* 2.5.14 - 3 Feb 2023 - Fix the 'Make Similar Images' button, which was producing incorrect images (weren't very similar).
* 2.5.13 - 1 Feb 2023 - Fix the remaining GPU memory leaks, including a better fix (more comprehensive) for the change in 2.5.12 (27 Jan).
* 2.5.12 - 27 Jan 2023 - Fix a memory leak, which made the UI unresponsive after an out-of-memory error. The allocated memory is now freed-up after an error.
* 2.5.11 - 25 Jan 2023 - UI for Merging Models. Thanks @JeLuf. More info: https://github.com/easydiffusion/easydiffusion/wiki/Model-Merging
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
- **Support for custom VAE models**. You can place your VAE files in the `models/vae` folder, and refresh the browser page to use them. More info: https://github.com/easydiffusion/easydiffusion/wiki/VAE-Variational-Auto-Encoder
- **Experimental support for multiple GPUs!** It should work automatically. Just open one browser tab per GPU, and spread your tasks across your GPUs. For e.g. open our UI in two browser tabs if you have two GPUs. You can customize which GPUs it should use in the "Settings" tab, otherwise let it automatically pick the best GPUs. Thanks @madrang . More info: https://github.com/easydiffusion/easydiffusion/wiki/Run-on-Multiple-GPUs
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

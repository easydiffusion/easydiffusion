# Stable Diffusion UI v2
### A simple 1-click way to install and use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on your own computer. No dependencies or technical knowledge required.

<p float="left">
  <a href="#installation"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/develop/media/download-win.png" width="200" /></a>
  <a href="#installation"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/develop/media/download-linux.png" width="200" /></a>
</p>

[![Discord Server](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.com/invite/u9yhsFmEkB) (for support, and development discussion) | [Troubleshooting guide for common problems](Troubleshooting.md)

️‍🔥🎉 **New!** Face Correction (GFPGAN) and Upscaling (RealESRGAN) is now available!

# Features in the new v2 Version:
- **No Dependencies or Technical Knowledge Required**: 1-click install for Windows 10/11 and Linux. *No dependencies*, no need for WSL or Docker or Conda or technical setup. Just download and run!
- **Face Correction (GFPGAN) and Upscaling (RealESRGAN)**
- **Image Modifiers**: A library of *modifier tags* like *"Realistic"*, *"Pencil Sketch"*, *"ArtStation"* etc. Experiment with various styles quickly.
- **New UI**: with cleaner design
- Supports "*Text to Image*" and "*Image to Image*"
- **NSFW Setting**: A setting in the UI to control *NSFW content*
- **Use CPU setting**: If you don't have a compatible graphics card, but still want to run it on your CPU.
- **Auto-updater**: Gets you the latest improvements and bug-fixes to a rapidly evolving project.

![Screenshot](media/shot-v8.jpg?raw=true)

# System Requirements
1. Windows 10/11, or Linux. Experimental support for Mac is coming soon.
2. An NVIDIA graphics card, preferably with 6GB or more of VRAM. But if you don't have a compatible graphics card, you can still use it with a "Use CPU" setting. It'll be very slow, but it should still work.

You do not need anything else. You do not need WSL, Docker or Conda. The installer will take care of it.

# Installation
1. **Download** [for Windows](https://github.com/cmdr2/stable-diffusion-ui/releases/download/v2.05/stable-diffusion-ui-win64.zip) or [for Linux](https://github.com/cmdr2/stable-diffusion-ui/releases/download/v2.05/stable-diffusion-ui-linux.tar.xz).

2. **Extract**:
  - For Windows: After unzipping the file, please move the `stable-diffusion-ui` folder to your `C:` (or any drive like D: at the top root level). For e.g. `C:\stable-diffusion-ui`. This will avoid a common problem with Windows (of file path length limits).
  - For Linux: After extracting the .tar.xz file, please open a terminal, and go to the `stable-diffusion-ui` directory.

3. **Run**:
  - For Windows: `Start Stable Diffusion UI.cmd` by double-clicking it.
  - For Linux: In the terminal, run `./start.sh` (or `bash start.sh`)

This will automatically install Stable Diffusion, set it up, and start the interface. No additional steps are needed.

**To Uninstall:** Just delete the `stable-diffusion-ui` folder, to uninstall all the downloaded packages.


# Usage
Open http://localhost:9000 in your browser (after running step 3 previously).

## With a text description
1. Enter a text prompt, like `a photograph of an astronaut riding a horse` in the textbox.
2. Press `Make Image`. This will take some time, depending on your system's processing power.
3. See the image generated using your prompt.

## With an image
1. Click `Browse..` next to `Initial Image`. Select your desired image.
2. An optional text prompt can help you further describe the kind of image you want to generate.
3. Press `Make Image`. See the image generated using your prompt.

You can use Face Correction or Upscaling to improve the image further.

**Pro tip:** You can also click `Use as Input` on a generated image, to use it as the input image for your next generation. This can be useful for sequentially refining the generated image with a single click.

**Another tip:** Images with the same aspect ratio of your generated image work best. E.g. 1:1 if you're generating images sized 512x512.

## Problems? Troubleshooting
Please try the common [troubleshooting](Troubleshooting.md) steps. If that doesn't fix it, please ask on the [discord server](https://discord.com/invite/u9yhsFmEkB), or [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues).

# Advanced Settings
You can also set the configuration like `seed`, `width`, `height`, `num_outputs`, `num_inference_steps` and `guidance_scale` using the 'show' button next to 'Advanced settings'.

Use the same `seed` number to get the same image for a certain prompt. This is useful for refining a prompt without losing the basic image design. Enable the `random images` checkbox to get random images.

![Screenshot of advanced settings](media/config-v5.jpg?raw=true)

# What is this? Why no Docker?
This version is a 1-click installer. You don't need WSL or Docker or anything beyond a working NVIDIA GPU with an updated driver. You don't need to use the command-line at all. Even if you don't have a compatible GPU, you can run it on your CPU (albeit very slowly).

It'll download the necessary files from the original [Stable Diffusion](https://github.com/CompVis/stable-diffusion) git repository, and set it up. It'll then start the browser-based interface like before.

The NSFW option is currently off (temporarily), so it'll allow NSFW images, for those people who are unable to run their prompts without hitting the NSFW filter incorrectly.

# Bugs reports and code contributions welcome
If there are any problems or suggestions, please feel free to ask on the [discord server](https://discord.com/invite/u9yhsFmEkB) or [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues).

Also, please feel free to submit a pull request, if you have any code contributions in mind. Join the [discord server](https://discord.com/invite/u9yhsFmEkB) for development-related discussions, and for helping other users.

# Disclaimer
The authors of this project are not responsible for any content generated using this interface.

This license of this software forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please read [the license](LICENSE).

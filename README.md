# Stable Diffusion UI
### A simple way to install and use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on your own computer

# Features in the new v2 Version! Try the development build
- 1-click install for Windows 10 and 11. **No dependencies**, no need for WSL or Docker or Conda. Just download and run!
- A library of **modifier tags** like *"Realistic"*, *"Pencil Sketch"*, *"ArtStation"* etc. Experiment with various styles quickly.
- **New UI** with cleaner design
- Supports "**Text to Image**" and "**Image to Image**"
- A setting in the UI to control **NSFW content**

<img src="https://github.com/cmdr2/stable-diffusion-ui/raw/v2/media/shot-v7.jpg" height="600" />

# System Requirements
1. Windows 10 or 11. Support for Linux and (experimentally) Mac is coming soon.
2. An NVIDIA graphics card, preferably with 8GB or more of VRAM. Working is being done to reduce the memory requirements.

You no longer need anything else. No need for WSL or Docker or Conda. The installer will take care of it.

# Installation
1. Download [for Windows](https://drive.google.com/file/d/1cEuOcb9OaldXcc2XzEMqEdvAr_w-KJ_p/view?usp=sharing) (this will be hosted on GitHub in the future).

2. After unzipping the file, please run `stable-diffusion-ui.cmd` by double-clicking it.

3. This will automatically install Stable Diffusion, set it up, and start the interface. No additional steps are needed.

The installation will take some time, due to the large number of dependencies. It'll install the original [Stable Diffusion](https://github.com/CompVis/stable-diffusion) git repository.

This version is currently only built for Windows 10 and 11. WSL is not required.

Note: This version is not optimized, so it may cause `CUDA Out of Memory` errors. Please try reducing your image size for now.

**I would love to know if this works (or fails). Please [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues/26), thanks!**

### What is this? Why no Docker?
This version is a 1-click installer. You don't need WSL or Docker or anything beyond a working NVIDIA GPU with an updated driver. You don't need to use the command-line at all.

It'll download the necessary files from the original [Stable Diffusion](https://github.com/CompVis/stable-diffusion) git repository, and set it up. It'll then start the browser-based interface like before.

An NSFW option is present in the interface, for those people who are unable to run their prompts without hitting the NSFW filter incorrectly.

# Bugs reports and code contributions welcome
If there are any problems or suggestions, please feel free to [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues/26).

Also, please feel free to submit a pull request, if you have any code contributions in mind.

# Disclaimer
The authors of this project are not responsible for any content generated using this interface.

This license of this software forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please read [the license](LICENSE).

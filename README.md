# Stable Diffusion UI
### Easiest way to install and use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on your own computer. No dependencies or technical knowledge required. 1-click install, powerful features.

[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB) (for support, and development discussion) | [Troubleshooting guide for common problems](Troubleshooting.md)

----

## Step 1: Download the installer

<p float="left">
  <a href="#installation"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/develop/media/download-win.png" width="200" /></a>
  <a href="#installation"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/develop/media/download-linux.png" width="200" /></a>
</p>

## Step 2: Run the program
- On Windows: Double-click `Start Stable Diffusion UI.cmd`
- On Linux: Run `./start.sh` in a terminal

## Step 3: There is no step 3!
It's simple to get started. You don't need to install or struggle with Python, Anaconda, Docker etc.

The installer will take care of whatever is needed. A friendly [Discord community](https://discord.com/invite/u9yhsFmEkB) will help you if you face any problems.

----

# Easy for new users, powerful features for advanced users
### Features:
- **No Dependencies or Technical Knowledge Required**: 1-click install for Windows 10/11 and Linux. *No dependencies*, no need for WSL or Docker or Conda or technical setup. Just download and run!
- **Clutter-free UI**: a friendly and simple UI, while providing a lot of powerful features
- Supports "*Text to Image*" and "*Image to Image*"
- **Custom Models**: Use your own `.ckpt` file, by placing it inside the `models/stable-diffusion` folder!
- **Live Preview**: See the image as the AI is drawing it
- **Task Queue**: Queue up all your ideas, without waiting for the current task to finish
- **In-Painting**: Specify areas of your image to paint into
- **Face Correction (GFPGAN) and Upscaling (RealESRGAN)**
- **Image Modifiers**: A library of *modifier tags* like *"Realistic"*, *"Pencil Sketch"*, *"ArtStation"* etc. Experiment with various styles quickly.
- **Loopback**: Use the output image as the input image for the next img2img task
- **Negative Prompt**: Specify aspects of the image to *remove*.
- **Attention/Emphasis:** () in the prompt increases the model's attention to enclosed words, and [] decreases it
- **Weighted Prompts:** Use weights for specific words in your prompt to change their importance, e.g. `red:2.4 dragon:1.2`
- **Prompt Matrix:** (in beta) Quickly create multiple variations of your prompt, e.g. `a photograph of an astronaut riding a horse | illustration | cinematic lighting`
- **Lots of Samplers:** ddim, plms, heun, euler, euler_a, dpm2, dpm2_a, lms
- **Multiple Prompts File:** Queue multiple prompts by entering one prompt per line, or by running a text file
- **NSFW Setting**: A setting in the UI to control *NSFW content*
- **JPEG/PNG output**
- **Save generated images to disk**
- **Use CPU setting**: If you don't have a compatible graphics card, but still want to run it on your CPU.
- **Auto-updater**: Gets you the latest improvements and bug-fixes to a rapidly evolving project.
- **Low Memory Usage**: Creates 512x512 images with less than 4GB of VRAM!
- **Developer Console**: A developer-mode for those who want to modify their Stable Diffusion code, and edit the conda environment.

### Easy for new users:
![Screenshot of the initial UI](media/shot-v10-simple.jpg?raw=true)

### Powerful features for advanced users:
![Screenshot of advanced settings](media/shot-v10.jpg?raw=true)

### Live Preview
Useful for judging (and stopping) an image quickly, without waiting for it to finish rendering.

![live-512](https://user-images.githubusercontent.com/844287/192097249-729a0a1e-a677-485e-9ccc-16a9e848fabe.gif)

### Task Queue
![Screenshot of task queue](media/task-queue-v1.jpg?raw=true)

# System Requirements
1. Windows 10/11, or Linux. Experimental support for Mac is coming soon.
2. An NVIDIA graphics card, preferably with 4GB or more of VRAM. If you don't have a compatible graphics card, it'll automatically run in the slower "CPU Mode".
3. Minimum 8 GB of RAM and 25GB of disk space.

You don't need to install or struggle with Python, Anaconda, Docker etc. The installer will take care of whatever is needed.

# Installation
1. **Download** [for Windows](https://github.com/cmdr2/stable-diffusion-ui/releases/download/v2.3.5/stable-diffusion-ui-windows.zip) or [for Linux](https://github.com/cmdr2/stable-diffusion-ui/releases/download/v2.3.5/stable-diffusion-ui-linux.zip).

2. **Extract**:
  - For Windows: After unzipping the file, please move the `stable-diffusion-ui` folder to your `C:` (or any drive like D:, at the top root level), e.g. `C:\stable-diffusion-ui`. This will avoid a common problem with Windows (file path length limits).
  - For Linux: After extracting the .tar.xz file, please open a terminal, and go to the `stable-diffusion-ui` directory.

3. **Run**:
  - For Windows: `Start Stable Diffusion UI.cmd` by double-clicking it.
  - For Linux: In the terminal, run `./start.sh` (or `bash start.sh`)

This will automatically install Stable Diffusion, set it up, and start the interface. No additional steps are needed.

**To Uninstall:** Just delete the `stable-diffusion-ui` folder to uninstall all the downloaded packages.

# How to use?
Please use our [guide](https://github.com/cmdr2/stable-diffusion-ui/wiki/How-to-Use) to understand how to use the features in this UI.

# Bugs reports and code contributions welcome
If there are any problems or suggestions, please feel free to ask on the [discord server](https://discord.com/invite/u9yhsFmEkB) or [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues).

We could really use help on these aspects (click to view tasks that need your help):
* [User Interface](https://github.com/users/cmdr2/projects/1/views/1)
* [Engine](https://github.com/users/cmdr2/projects/3/views/1)
* [Installer](https://github.com/users/cmdr2/projects/4/views/1)
* [Documentation](https://github.com/users/cmdr2/projects/5/views/1)

If you have any code contributions in mind, please feel free to say Hi to us on the [discord server](https://discord.com/invite/u9yhsFmEkB). We use the Discord server for development-related discussions, and for helping users.

# Disclaimer
The authors of this project are not responsible for any content generated using this interface.

The license of this software forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation, or target vulnerable groups. For the full list of restrictions please read [the license](LICENSE). You agree to these terms by using this software.

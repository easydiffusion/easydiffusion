# Easy Diffusion 2.5
### The easiest way to install and use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on your own computer.

Does not require technical knowledge, does not require pre-installed software. 1-click install, powerful features, friendly community.

[Installation guide](#step-1-download-and-extract-the-installer) | [Troubleshooting guide](https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting) | <sub>[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)</sub> <sup>(for support queries, and development discussions)</sup>

![t2i](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/assets/stable-samples/txt2img/768/merged-0006.png)

# Step 1: Download and extract the installer
Click the download button for your operating system:

<p float="left">
  <a href="https://github.com/cmdr2/stable-diffusion-ui/releases/download/v2.5.15/stable-diffusion-ui-windows.zip"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/download-win.png" width="200" /></a>
  <a href="https://github.com/cmdr2/stable-diffusion-ui/releases/download/v2.5.15/stable-diffusion-ui-linux.zip"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/download-linux.png" width="200" /></a>
</p>

## On Windows:
1. Unzip/extract the folder `stable-diffusion-ui` which should be in your downloads folder, unless you changed your default downloads destination.
2. Move the `stable-diffusion-ui` folder to your `C:` drive (or any other drive like `D:`, at the top root level). `C:\stable-diffusion-ui` or `D:\stable-diffusion-ui` as examples. This will avoid a common problem with Windows (file path length limits).
## On Linux:
1. Unzip/extract the folder `stable-diffusion-ui` which should be in your downloads folder, unless you changed your default downloads destination.
2. Open a terminal window, and navigate to the `stable-diffusion-ui` directory.

# Step 2: Run the program
## On Windows: 
Double-click `Start Stable Diffusion UI.cmd`.
If Windows SmartScreen prevents you from running the program click `More info` and then `Run anyway`.
## On Linux: 
Run `./start.sh` (or `bash start.sh`) in a terminal.

The installer will take care of whatever is needed. If you face any problems, you can join the friendly [Discord community](https://discord.com/invite/u9yhsFmEkB) and ask for assistance.

# Step 3: There is no Step 3. It's that simple!

**To Uninstall:** Just delete the `stable-diffusion-ui` folder to uninstall all the downloaded packages.

----

# Easy for new users, powerful features for advanced users
## Features:

### User experience
- **Hassle-free installation**: Does not require technical knowledge, does not require pre-installed software. Just download and run!
- **Clutter-free UI**: A friendly and simple UI, while providing a lot of powerful features.
- **Task Queue**: Queue up all your ideas, without waiting for the current task to finish.
- **Intelligent Model Detection**: Automatically figures out the YAML config file to use for the chosen model (via a models database).
- **Live Preview**: See the image as the AI is drawing it.
- **Image Modifiers**: A library of *modifier tags* like *"Realistic"*, *"Pencil Sketch"*, *"ArtStation"* etc. Experiment with various styles quickly.
- **Multiple Prompts File**: Queue multiple prompts by entering one prompt per line, or by running a text file.
- **Save generated images to disk**: Save your images to your PC!
- **UI Themes**: Customize the program to your liking.
- **Searchable models dropdown**: organize your models into sub-folders, and search through them in the UI.

### Image generation
- **Supports**: "*Text to Image*" and "*Image to Image*".
- **19 Samplers**: `ddim`, `plms`, `heun`, `euler`, `euler_a`, `dpm2`, `dpm2_a`, `lms`, `dpm_solver_stability`, `dpmpp_2s_a`, `dpmpp_2m`, `dpmpp_sde`, `dpm_fast`, `dpm_adaptive`, `unipc_snr`, `unipc_tu`, `unipc_tq`, `unipc_snr_2`, `unipc_tu_2`.
- **In-Painting**: Specify areas of your image to paint into.
- **Simple Drawing Tool**: Draw basic images to guide the AI, without needing an external drawing program.
- **Face Correction (GFPGAN)**
- **Upscaling (RealESRGAN)**
- **Loopback**: Use the output image as the input image for the next img2img task.
- **Negative Prompt**: Specify aspects of the image to *remove*.
- **Attention/Emphasis**: () in the prompt increases the model's attention to enclosed words, and [] decreases it.
- **Weighted Prompts**: Use weights for specific words in your prompt to change their importance, e.g. `red:2.4 dragon:1.2`.
- **Prompt Matrix**: Quickly create multiple variations of your prompt, e.g. `a photograph of an astronaut riding a horse | illustration | cinematic lighting`.
- **1-click Upscale/Face Correction**: Upscale or correct an image after it has been generated.
- **Make Similar Images**: Click to generate multiple variations of a generated image.
- **NSFW Setting**: A setting in the UI to control *NSFW content*.
- **JPEG/PNG/WEBP output**: Multiple file formats.

### Advanced features
- **Custom Models**: Use your own `.ckpt` or `.safetensors` file, by placing it inside the `models/stable-diffusion` folder!
- **Stable Diffusion 2.1 support**
- **Merge Models**
- **Use custom VAE models**
- **Use pre-trained Hypernetworks**
- **Use custom GFPGAN models**
- **UI Plugins**: Choose from a growing list of [community-generated UI plugins](https://github.com/cmdr2/stable-diffusion-ui/wiki/UI-Plugins), or write your own plugin to add features to the project!

### Performance and security
- **Fast**: Creates a 512x512 image with euler_a in 5 seconds, on an NVIDIA 3060 12GB.
- **Low Memory Usage**: Create 512x512 images with less than 3 GB of GPU RAM, and 768x768 images with less than 4 GB of GPU RAM!
- **Use CPU setting**: If you don't have a compatible graphics card, but still want to run it on your CPU.
- **Multi-GPU support**: Automatically spreads your tasks across multiple GPUs (if available), for faster performance!
- **Auto scan for malicious models**: Uses picklescan to prevent malicious models.
- **Safetensors support**: Support loading models in the safetensor format, for improved safety.
- **Auto-updater**: Gets you the latest improvements and bug-fixes to a rapidly evolving project.
- **Developer Console**: A developer-mode for those who want to modify their Stable Diffusion code, and edit the conda environment.

**(and a lot more)**

----

## Easy for new users:
![Screenshot of the initial UI](https://user-images.githubusercontent.com/844287/217043152-29454d15-0387-4228-b70d-9a4b84aeb8ba.png)


## Powerful features for advanced users:
![Screenshot of advanced settings](https://user-images.githubusercontent.com/844287/217042588-fc53c975-bacd-4a9c-af88-37408734ade3.png)


## Live Preview
Useful for judging (and stopping) an image quickly, without waiting for it to finish rendering.

![live-512](https://user-images.githubusercontent.com/844287/192097249-729a0a1e-a677-485e-9ccc-16a9e848fabe.gif)

## Task Queue
![Screenshot of task queue](https://user-images.githubusercontent.com/844287/217043984-0b35f73b-1318-47cb-9eed-a2a91b430490.png)



# System Requirements
1. Windows 10/11, or Linux. Experimental support for Mac is coming soon.
2. An NVIDIA graphics card, preferably with 4GB or more of VRAM. If you don't have a compatible graphics card, it'll automatically run in the slower "CPU Mode".
3. Minimum 8 GB of RAM and 25GB of disk space.

You don't need to install or struggle with Python, Anaconda, Docker etc. The installer will take care of whatever is needed.

----

# How to use?
Please refer to our [guide](https://github.com/cmdr2/stable-diffusion-ui/wiki/How-to-Use) to understand how to use the features in this UI.

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

The license of this software forbids you from sharing any content that:
- Violates any laws.
- Produces any harm to a person or persons.
- Disseminates (spreads) any personal information that would be meant for harm.
- Spreads misinformation.
- Target vulnerable groups. 

For the full list of restrictions please read [the License](LICENSE). You agree to these terms by using this software.

# Easy Diffusion 3.0
### The easiest way to install and use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on your computer.

Does not require technical knowledge, does not require pre-installed software. 1-click install, powerful features, friendly community.

️‍🔥🎉 **New!** Support for SDXL, ControlNet, multiple LoRA files, embeddings (and a lot more) have been added!

[Installation guide](#installation) | [Troubleshooting guide](https://github.com/easydiffusion/easydiffusion/wiki/Troubleshooting) | [User guide](https://github.com/easydiffusion/easydiffusion/wiki) | <sub>[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)</sub> <sup>(for support queries, and development discussions)</sup>

---
![262597678-11089485-2514-4a11-88fb-c3acc81fc9ec](https://github.com/easydiffusion/easydiffusion/assets/844287/050b5e15-e909-45bf-8162-a38234830e38)


# Installation
Click the download button for your operating system:

<p float="left">
  <a href="https://github.com/cmdr2/stable-diffusion-ui/releases/latest/download/Easy-Diffusion-Linux.zip"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/download-linux.png" width="200" /></a>
  <a href="https://github.com/cmdr2/stable-diffusion-ui/releases/latest/download/Easy-Diffusion-Mac.zip"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/download-mac.png" width="200" /></a>
  <a href="https://github.com/cmdr2/stable-diffusion-ui/releases/latest/download/Easy-Diffusion-Windows.exe"><img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/download-win.png" width="200" /></a>
</p>

**Hardware requirements:**
- **Windows:** NVIDIA graphics card¹ (minimum 2 GB RAM), or run on your CPU.
- **Linux:** NVIDIA¹ or AMD² graphics card (minimum 2 GB RAM), or run on your CPU.
- **Mac:** M1 or M2, or run on your CPU.
- Minimum 8 GB of system RAM.
- Atleast 25 GB of space on the hard disk.

¹) [CUDA Compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) level of 3.7 or higher required.

²) ROCm 5.2 support required.

The installer will take care of whatever is needed. If you face any problems, you can join the friendly [Discord community](https://discord.com/invite/u9yhsFmEkB) and ask for assistance.

## On Windows:
1. Run the downloaded `Easy-Diffusion-Windows.exe` file.
2. Run `Easy Diffusion` once the installation finishes. You can also start from your Start Menu, or from your desktop (if you created a shortcut).

If Windows SmartScreen prevents you from running the program click `More info` and then `Run anyway`.

**Tip:** On Windows 10, please install at the top level in your drive, e.g. `C:\EasyDiffusion` or `D:\EasyDiffusion`. This will avoid a common problem with Windows 10 (file path length limits).

## On Linux/Mac:
1. Unzip/extract the folder `easy-diffusion` which should be in your downloads folder, unless you changed your default downloads destination.
2. Open a terminal window, and navigate to the `easy-diffusion` directory.
3. Run `./start.sh` (or `bash start.sh`) in a terminal.

# To remove/uninstall:
Just delete the `EasyDiffusion` folder to uninstall all the downloaded packages.

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

### Powerful image generation
- **Supports**: "*Text to Image*", "*Image to Image*" and "*InPainting*"
- **ControlNet**: For advanced control over the image, e.g. by setting the pose or drawing the outline for the AI to fill in.
- **16 Samplers**: `PLMS`, `DDIM`, `DEIS`, `Heun`, `Euler`, `Euler Ancestral`, `DPM2`, `DPM2 Ancestral`, `LMS`, `DPM Solver`, `DPM++ 2s Ancestral`, `DPM++ 2m`, `DPM++ 2m SDE`, `DPM++ SDE`, `DDPM`, `UniPC`.
- **Stable Diffusion XL and 2.1**: Generate higher-quality images using the latest Stable Diffusion XL models.
- **Textual Inversion Embeddings**: For guiding the AI strongly towards a particular concept.
- **Simple Drawing Tool**: Draw basic images to guide the AI, without needing an external drawing program.
- **Face Correction (GFPGAN)**
- **Upscaling (RealESRGAN)**
- **Loopback**: Use the output image as the input image for the next image task.
- **Negative Prompt**: Specify aspects of the image to *remove*.
- **Attention/Emphasis**: `+` in the prompt increases the model's attention to enclosed words, and `-` decreases it. E.g. `apple++ falling from a tree`.
- **Weighted Prompts**: Use weights for specific words in your prompt to change their importance, e.g. `(red)2.4 (dragon)1.2`.
- **Prompt Matrix**: Quickly create multiple variations of your prompt, e.g. `a photograph of an astronaut riding a horse | illustration | cinematic lighting`.
- **Prompt Set**: Quickly create multiple variations of your prompt, e.g. `a photograph of an astronaut on the {moon,earth}`
- **1-click Upscale/Face Correction**: Upscale or correct an image after it has been generated.
- **Make Similar Images**: Click to generate multiple variations of a generated image.
- **NSFW Setting**: A setting in the UI to control *NSFW content*.
- **JPEG/PNG/WEBP output**: Multiple file formats.

### Advanced features
- **Custom Models**: Use your own `.ckpt` or `.safetensors` file, by placing it inside the `models/stable-diffusion` folder!
- **Stable Diffusion XL and 2.1 support**
- **Merge Models**
- **Use custom VAE models**
- **Textual Inversion Embeddings**
- **ControlNet**
- **Use custom GFPGAN models**
- **UI Plugins**: Choose from a growing list of [community-generated UI plugins](https://github.com/easydiffusion/easydiffusion/wiki/UI-Plugins), or write your own plugin to add features to the project!

### Performance and security
- **Fast**: Creates a 512x512 image with euler_a in 5 seconds, on an NVIDIA 3060 12GB.
- **Low Memory Usage**: Create 512x512 images with less than 2 GB of GPU RAM, and 768x768 images with less than 3 GB of GPU RAM!
- **Use CPU setting**: If you don't have a compatible graphics card, but still want to run it on your CPU.
- **Multi-GPU support**: Automatically spreads your tasks across multiple GPUs (if available), for faster performance!
- **Auto scan for malicious models**: Uses picklescan to prevent malicious models.
- **Safetensors support**: Support loading models in the safetensor format, for improved safety.
- **Auto-updater**: Gets you the latest improvements and bug-fixes to a rapidly evolving project.
- **Developer Console**: A developer-mode for those who want to modify their Stable Diffusion code, modify packages, and edit the conda environment.

**(and a lot more)**

----

## Easy for new users, powerful features for advanced users:
![image](https://github.com/easydiffusion/easydiffusion/assets/844287/efbbac9f-42ce-4aef-8625-fd23c74a8241)

## Task Queue
![Screenshot of task queue](https://user-images.githubusercontent.com/844287/217043984-0b35f73b-1318-47cb-9eed-a2a91b430490.png)


----

# How to use?
Please refer to our [guide](https://github.com/easydiffusion/easydiffusion/wiki/How-to-Use) to understand how to use the features in this UI.

# Bugs reports and code contributions welcome
If there are any problems or suggestions, please feel free to ask on the [discord server](https://discord.com/invite/u9yhsFmEkB) or [file an issue](https://github.com/easydiffusion/easydiffusion/issues).

If you have any code contributions in mind, please feel free to say Hi to us on the [discord server](https://discord.com/invite/u9yhsFmEkB). We use the Discord server for development-related discussions, and for helping users.

# Credits
* Stable Diffusion: https://github.com/Stability-AI/stablediffusion
* CodeFormer: https://github.com/sczhou/CodeFormer (license: https://github.com/sczhou/CodeFormer/blob/master/LICENSE)
* GFPGAN: https://github.com/TencentARC/GFPGAN
* RealESRGAN: https://github.com/xinntao/Real-ESRGAN
* k-diffusion: https://github.com/crowsonkb/k-diffusion
* Code contributors and artists on the cmdr2 UI: https://github.com/cmdr2/stable-diffusion-ui and Discord (https://discord.com/invite/u9yhsFmEkB)
* Lots of contributors on the internet

# Disclaimer
The authors of this project are not responsible for any content generated using this interface.

The license of this software forbids you from sharing any content that:
- Violates any laws.
- Produces any harm to a person or persons.
- Disseminates (spreads) any personal information that would be meant for harm.
- Spreads misinformation.
- Target vulnerable groups. 

For the full list of restrictions please read [the License](LICENSE). You agree to these terms by using this software.

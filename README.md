# Stable Diffusion UI - v2 (beta)
### A simple way to install and use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on your own computer (Win 10/11, Linux). No dependencies or technical knowledge required.

[![Discord Server](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.com/invite/u9yhsFmEkB) (for support and development discussion)

# Features in the new v2 Version:
- **No Dependencies or Technical Knowledge Required**: 1-click install for Windows 10/11 and Linux. *No dependencies*, need for WSL, Docker, Conda, or even technical setup. Just download and run!
- **Image Modifiers**: A library of *modifier tags* like *"Realistic"*, *"Pencil Sketch"*, *"ArtStation"* etc. Experiment with various styles quickly.
- **New UI**: Cleaner design thats easy to use
- Supports "*Text to Image*" and "*Image to Image*"
- **NSFW Setting**: A setting in the UI to control *NSFW content*
- **Use CPU setting**: If you don't have a compatible graphics card, but still want to run it on your CPU.

![Screenshot](media/shot-v8.jpg?raw=true)

# System Requirements
1. Windows 10/11 or Linux ***(NOTE: Experimental support for Mac is coming soon)***
2. An NVIDIA graphics card containing 4GB+ VRAM ***(INFO: Some settings will not function correctly with under 6GB VRAM | "Use CPU" setting can be used if no compatible graphics card is available, although this will result in lower performance and longer run times)***
3. 8GB+ RAM ***(INFO: 8GB is barebones, its reccemended to have more ram on the system)***

# Installation

**These steps will automatically install Stable Diffusion, set it up, and start the interface. No additional steps are needed**

**I would love to know if this works (or fails). If you run into any issues please [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues/26) or come visit us on the [Discord Server](https://discord.com/invite/u9yhsFmEkB)**

## Via Zip
1. Download [for Windows](https://drive.google.com/file/d/1MY5gzsQHV_KREbYs3gw33QL4gGIlQRqj/view?usp=sharing) or [for Linux](https://drive.google.com/file/d/1Gwz1LVQUCart8HhCjrmXkS6TWKbTsLsR/view?usp=sharing)

2. Extract:
    - For Windows: After unzipping the file, please move the `stable-diffusion-ui` folder to your `C:` (or any drive like D: at the top root level). For e.g. `C:\stable-diffusion-ui`. This will avoid a common problem with Windows (of file path length limits).
    - For Linux: After extracting the .tar.xz file, please open a terminal, and go to the `stable-diffusion-ui` directory.

3. Run:
    - For Windows: `Start Stable Diffusion UI.cmd` by double-clicking it ***(NOTE: The installation will take some time, due to the large number of dependencies. It'll install the original [Stable Diffusion](https://github.com/CompVis/stable-diffusion) git repository)***
    - For Linux: In the terminal, run `./start.sh`

4. Sit back, relax, and enjoy automatic updates ***(NOTE: Updates will only be triggered by launching the server | there will be an option to opt out of updates in the future)***


## Via GitHub

***(NOTE: This is not currently supported)***

# What is this? Why no Docker?
This version is a 1-click installer. You only need an NVIDIA GPU with an updated driver, unlike previous versions.

It'll download the necessary files from the original [Stable Diffusion](https://github.com/CompVis/stable-diffusion) git repository, and set it up. It'll then start the browser-based interface like before.

The NSFW option is currently off so it'll allow NSFW images ***(NOTE: This is temporary for those people who are unable to run their prompts without hitting the NSFW filter incorrectly)***

# Bugs Reports and Code Contributions are Welcome
If there are any problems or suggestions, please feel free to [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues/26).

Also, please feel free to submit a pull request, if you have any code contributions in mind. Join the [discord server](https://discord.com/invite/u9yhsFmEkB) for development-related discussions and for helping other users.

# Disclaimer
The authors of this project are not responsible for any content generated using this interface.

This license of this software forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please read [the license](LICENSE).

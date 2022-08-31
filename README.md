# Stable Diffusion UI
### A simple way to install and use [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion) on your own computer

---

# What does this do?
Two things:
1. Automatically downloads and installs Stable Diffusion on your own computer (no need to mess with conda or environments)
2. Gives you a simple browser-based UI to talk to your local Stable Diffusion. Enter text prompts and view the generated image. No API keys required.

All the processing will happen on your computer locally, it does not transmit your prompts or process on any remote server.

<p float="left">
  <img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/shot-v3a.jpg" height="500" />
  <img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/shot-v6a.jpg" height="500" />
</p>


# System Requirements
1. Computer capable of running Stable Diffusion.
2. Linux or Windows 11 (with [WSL](https://docs.microsoft.com/en-us/windows/wsl/install)) or Windows 10 v2004+ (Build 19041+) with [WSL](https://docs.microsoft.com/en-us/windows/wsl/install).
3. Requires (a) [Docker](https://docs.docker.com/engine/install/), (b) [docker-compose v1.29](https://docs.docker.com/compose/install/), and (c) [nvidia-container-toolkit](https://stackoverflow.com/a/58432877).

**Important:** If you're using Windows, please install docker inside your [WSL](https://docs.microsoft.com/en-us/windows/wsl/install)'s Linux. Install docker for the Linux distro in your WSL. **Don't install Docker for Windows.**

# Installation
1. Clone this repository: `git clone https://github.com/cmdr2/stable-diffusion-ui.git` or [download the zip file](https://github.com/cmdr2/stable-diffusion-ui/archive/refs/heads/main.zip) and unzip.
2. Open your terminal, and in the project directory run: `docker-compose up &` (warning: this will take some time during the first run, since it'll download Stable Diffusion's [docker image](https://replicate.com/stability-ai/stable-diffusion), nearly 17 GiB)
3. Open http://localhost:9000 in your browser. That's it!

If you're getting errors, please check the [Troubleshooting](https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting) page.

To stop the server, please run `docker-compose down`

# Usage
Open http://localhost:9000 in your browser (after running `docker-compose up &` from step 2 previously).

## With a text description
1. Enter a text prompt, like `a photograph of an astronaut riding a horse` in the textbox.
2. Press `Make Image`. This will take some time, depending on your system's processing power.
3. See the image generated using your prompt.

## With an image
1. Click `Browse..` next to `Initial Image`. Select your desired image.
2. An optional text prompt can help you further describe the kind of image you want to generate.
3. Press `Make Image`. See the image generated using your prompt.

You can also set an `Image Mask` for telling Stable Diffusion to draw in only the black areas in your image mask. White areas in your mask will be ignored.

**Pro tip:** You can also click `Use as Input` on a generated image, to use it as the input image for your next generation. This can be useful for sequentially refining the generated image with a single click.

**Another tip:** Images with the same aspect ratio of your generated image work best. E.g. 1:1 if you're generating images sized 512x512.

## Problems?
Please [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues) if this did not work for you (after trying the common [troubleshooting](#troubleshooting) steps)!

# Advanced Settings
You can also set the configuration like `seed`, `width`, `height`, `num_outputs`, `num_inference_steps` and `guidance_scale` using the 'show' button next to 'Advanced settings'.

Use the same `seed` number to get the same image for a certain prompt. This is useful for refining a prompt without losing the basic image design. Enable the `random images` checkbox to get random images.

![Screenshot of advanced settings](media/config-v3.jpg?raw=true)

# Troubleshooting
The [Troubleshooting wiki page](https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting) contains some common errors and their solutions. Please check that, and if it doesn't work, feel free to [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues).

# Behind the scenes
This project is a quick way to get started with Stable Diffusion. You do not need to have Stable Diffusion already installed, and do not need any API keys. This project will automatically download Stable Diffusion's docker image, the first time it is run.

This project runs Stable Diffusion in a docker container behind the scenes, using Stable Diffusion's [Docker image](https://replicate.com/stability-ai/stable-diffusion) on replicate.com.

# Bugs reports and code contributions welcome
If there are any problems or suggestions, please feel free to [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues).

Also, please feel free to submit a pull request, if you have any code contributions in mind.

# Disclaimer
The authors of this project are not responsible for any content generated using this interface.

This license of this software forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please read [the license](LICENSE).

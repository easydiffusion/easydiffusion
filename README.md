A simple way to install and use [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion) on your local computer. Provides a browser UI for generating images from text prompts. Just enter your text prompt, and see the generated image.

# What does this do?
Two things:
1. Automatically downloads and installs Stable Diffusion on your local computer (no need to mess with conda or environments)
2. Gives you a simple browser-based UI to talk to your local Stable Diffusion. Enter text prompts and view the generated image. No API keys required.

All the processing will happen on your local computer, it does not transmit your prompts or process on any remote server.

<img src="https://github.com/cmdr2/stable-diffusion-ui/raw/main/media/shot-v2.jpg" height="500" alt="Screenshot of tool">

# System Requirements
1. Linux or Windows 11 (with [WSL](https://docs.microsoft.com/en-us/windows/wsl/install)). And if your computer can run [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion).
2. Requires [Docker](https://docs.docker.com/engine/install/), [docker-compose v1.29](https://docs.docker.com/compose/install/), [nvidia-container-toolkit](https://stackoverflow.com/a/58432877) and [Python](https://www.python.org/downloads/) (3.6 or higher).

# Installation
1. Clone this repository: `git clone git@github.com:cmdr2/stable-diffusion-ui.git` or [download the zip file](https://github.com/cmdr2/stable-diffusion-ui/archive/refs/heads/main.zip) and unzip.
2. Open your terminal, and in the project directory run: `docker-compose up &` (warning: this will take some time during the first run, since it'll download Stable Diffusion's [docker image](https://replicate.com/stability-ai/stable-diffusion), nearly 17 GiB)
3. Open http://localhost:8000 in your browser. That's it!

# Usage
1. Open http://localhost:8000 in your browser (after running `docker-compose up &` from step 2 previously).
2. Enter a text prompt, like `a photograph of an astronaut riding a horse` in the textbox.
3. Press `Make Image`. This will take some time, depending on your system's processing power.
4. See the image generated using your prompt.

Please [let me know](mailto:sd@cmdr2.org) if this worked (or didn't) for you!

To stop the server, please run `docker-compose down`

# Advanced Settings
You can also set the configuration like `seed`, `width`, `height`, `num_outputs`, `num_inference_steps` and `guidance_scale` using the 'show' button next to 'Advanced settings'.

Use the same `seed` number to get the same image for a certain prompt. This is useful for refining a prompt without losing the basic image design. Use a seed of `-1` to get random images.

![Screenshot of advanced settings](media/config-v2.jpg?raw=true)

# Behind the scenes
This project is a quick way to get started with Stable Diffusion. You do not need to have Stable Diffusion already installed, and do not need any API keys. This project will automatically download Stable Diffusion's docker image, the first time it is run.

This project runs Stable Diffusion in a docker container behind the scenes, using Stable Diffusion's [official Docker image](https://replicate.com/stability-ai/stable-diffusion) on replicate.com.

# Bugs reports and code contributions welcome
This was built in a few hours for fun. So if there are any problems or suggestions, please feel free to [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues) or [email me](mailto:sd@cmdr2.org).

Also, please feel free to submit a pull request, if you have any code contributions in mind.

# Disclaimer
I (and the authors of this project) are not responsible for any images generated using this interface.

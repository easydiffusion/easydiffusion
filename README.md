A simple browser UI for generating images from text prompts, using [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion). Designed for running locally on your computer. Just enter the text prompt, and see the generated image.

# What does this do?
Two things:
1. Automatically downloads and installs Stable Diffusion on your local computer (no need to mess with conda or environments)
2. Gives you a simple browser-based UI to talk to your local Stable Diffusion. Enter text prompts and view the generated image. No API keys required.

All the processing will happen on your local computer, it does not transmit your prompts or process on any remote server.

![Screenshot of tool](shot.jpg?raw=true)

# System Requirements
1. Requires [Docker](https://docs.docker.com/engine/install/) and [Python](https://www.python.org/downloads/) (3.6 or higher).
2. Linux or Windows 11 (with [WSL](https://docs.microsoft.com/en-us/windows/wsl/install)). Basically if your system can run [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion).

# Installation
1. Download [Quick UI](https://github.com/cmdr2/stable-diffusion-ui/archive/refs/heads/main.zip) (this project)
2. Unzip: `unzip main.zip`
3. Enter: `cd stable-diffusion-ui-main`
4. Install dependencies: `pip install fastapi uvicorn` (this is the framework and server used by this UI project)
5. Run: `./server.sh` (warning: this will take a while the first time, since it'll download Stable Diffusion's [docker image](https://replicate.com/stability-ai/stable-diffusion), nearly 17 GiB)
6. Open `http://localhost:8000` in your browser

# Usage
1. Open `http://localhost:8000` in your browser
2. Enter a text prompt, like `a photograph of an astronaut riding a horse` in the textbox.
3. Press `Make Image`. This will take a while, depending on your system's processing power.
4. See the image generated using your prompt. If there's an error, the status message at the top will show 'error' in red.

# Behind the scenes
This project is a quick way to get started with Stable Diffusion. You do not need to have Stable Diffusion already installed, and do not need any API keys. This project will automatically download Stable Diffusion's docker image, the first time it is run.

This project runs Stable Diffusion in a docker container behind the scenes, using Stable Diffusion's [official Docker image](https://replicate.com/stability-ai/stable-diffusion) on replicate.com.

# Bugs reports and code contributions welcome
This was built in a few hours for fun. So if there are any problems, please feel free to file an issue.

Also, please feel free to submit a pull request, if you have any code contributions in mind.

# Disclaimer
I am not responsible for any images generated using this interface.
A simple browser UI for generating images from text prompts, using [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion). Designed for running locally on your computer.

This project runs Stable Diffusion in a docker container behind the scenes, using Stable Diffusion's official Docker image on replicate.com.

![Screenshot of tool](shot1.jpeg?raw=true)

# System Requirements
1. Requires `docker` and `python3.6` (or higher).
2. Linux or Windows 11 (with WSL) or macOS. Basically if your system can run [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion).

# Installation
1. Download [Quick UI](https://github.com/cmdr2/stable-diffusion-ui/archive/refs/heads/main.zip)
2. Unzip: `unzip main.zip`
3. Enter: `cd stable-diffusion-ui`
4. Install dependencies: `pip install fastapi uvicorn` (this is the framework and server used by QuickUI)
5. Run: `./server.sh`
6. Open `http://localhost:8000` in your browser

# Usage
1. Open `http://localhost:8000` in your browser
2. Enter a text prompt, like `a photograph of an astronaut riding a horse` in the textbox.
3. Press `Make Image`. This will take a while, depending on your system's processing power.
4. See the image generated using your prompt. If there's an error, the status message at the top will show 'error' in red.

# Bugs reports and code contributions welcome
This was built in a few hours for fun. So if there are any problems, please feel free to file an issue.

Also, if you have any code contributions, please feel to submit a pull request.
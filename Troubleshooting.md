Common issues and their solutions. If these solutions don't work, please feel free to ask at the [discord server](https://discord.com/invite/u9yhsFmEkB) or [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues).

## RuntimeError: CUDA out of memory
This can happen if your PC has less than 6GB of VRAM.

Try disabling the "Turbo mode" setting under "Advanced Settings", since that takes an additional 1 GB of VRAM (to increase the speed).

Additionally, a common reason for this error is that you're using an initial image larger than 768x768 pixels. Try using a smaller initial image.

Also try generating smaller sized images.

## No ldm found, or antlr4 or any other missing module, or ClobberError: This transaction has incompatible packages due to a shared path
On Windows, please ensure that you had placed the `stable-diffusion-ui` folder after unzipping to the root of C: or D: (or any drive). For e.g. `C:\stable-diffusion-ui`. **Note:** This has to be done **before** you start the installation process. If you have already installed (and are facing this error), please delete the installed folder, and start fresh by unzipping and placing the folder at the top of your drive.

This error can also be caused if you already have conda/miniconda/anaconda installed, due to package conflicts. Please open your Anaconda Prompt, and run `conda clean --all` to clean up unused packages.

If nothing works, this could be due to a corrupted installation. Please try reinstalling this, by deleting the installed folder, and unzipping from the downloaded zip file.

## Killed uvicorn server:app --app-dir ... --port 9000 --host 0.0.0.0
This happens if your PC ran out of RAM. Stable Diffusion requires a lot of RAM, and requires atleast 10 GB of RAM to work well. You can also try closing all other applications before running Stable Diffusion UI.

## Green image generated
This usually happens if you're running NVIDIA 1650 or 1660 Super. To solve this, please close and run the Stable Diffusion command on your computer. If you're using the older Docker-based solution (v1), please upgrade to v2: https://github.com/cmdr2/stable-diffusion-ui/tree/v2#installation

If you're still seeing this error, please try enabling "Full Precision" under "Advanced Settings" in the Stable Diffusion UI.

## './docker-compose.yml' is invalid:
> ERROR: The Compose file './docker-compose.yml' is invalid because:
> services.stability-ai.deploy.resources.reservations value Additional properties are not allowed ('devices' was unexpected)

Please ensure you have `docker-compose` version 1.29 or higher. Check `docker-compose --version`, and if required [update it to 1.29](https://docs.docker.com/compose/install/). (Thanks [HVRyan](https://github.com/HVRyan))

## RuntimeError: Found no NVIDIA driver on your system:
If you have an NVIDIA GPU and the latest [NVIDIA driver](http://www.nvidia.com/Download/index.aspx), please ensure that you've installed [nvidia-container-toolkit](https://stackoverflow.com/a/58432877). (Thanks [u/exintrovert420](https://www.reddit.com/user/exintrovert420/))

## Some other process is already running at port 9000 / port 9000 could not be bound
You can override the port used. Please change `docker-compose.yml` inside the project directory, and update the line `9000:9000` to `1337:9000` (where 1337 is whichever port number you want).

After doing this, please restart your server, by running `./server restart`.

After this, you can access the server at `http://localhost:1337` (where 1337 is the new port you specified earlier).

## RuntimeError: CUDA error: unknown error
Please ensure that you have an NVIDIA GPU and the latest [NVIDIA driver](http://www.nvidia.com/Download/index.aspx), and that you've installed [nvidia-container-toolkit](https://stackoverflow.com/a/58432877).

Also, if you are using WSL (Windows), please ensure you have the latest WSL kernel by running `wsl --shutdown` and then `wsl --update`. (Thanks [AndrWeisR](https://github.com/AndrWeisR))

## 'chcp' is not recognized as an internal or external command, operable program or batch file.
Your PATH variable is missing the entry C:\windows\system32. Click on the Windows logo in the task bar ("start menu"), type "env", and click on "Edit the system environment variables". Then, click on the "Environment Variables" button near the bottom of the "System properties" window. Find the "PATH" entry in the "System variables" section, click on it, then click on the "Edit..." button beneath it. Click on the "New" button and type C:\Windows\System32 into the new line at the end of the list. Click the various OK buttons to close all the windows you just opened and rerun the installation.

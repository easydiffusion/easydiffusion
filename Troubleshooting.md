Common issues and their solutions. If these solutions don't work, please feel free to ask at the [discord server](https://discord.com/invite/u9yhsFmEkB) or [file an issue](https://github.com/cmdr2/stable-diffusion-ui/issues).

## Green image generated
This usually happens if you're running NVIDIA 1650 or 1660 Super. To solve this, please close and run the Stable Diffusion command on your computer. If you're using the older Docker-based solution (v1), please upgrade to v2: https://github.com/cmdr2/stable-diffusion-ui/tree/v2#installation

## No module found
This can happen if you're hitting the Windows file path length limitation. To solve this, please upgrade to v2 by following the installation steps here: https://github.com/cmdr2/stable-diffusion-ui/tree/v2#installation , and ensure that you've placed the `stable-diffusion-ui` folder in `C:` or `D:` (or any top-level location).

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
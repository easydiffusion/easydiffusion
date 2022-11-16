Hi there, these instructions are meant for the developers of this project.

If you only want to use the Stable Diffusion UI, you've downloaded the wrong file. In that case, please download and follow the instructions at https://github.com/cmdr2/stable-diffusion-ui#installation

Thanks

# For developers:

If you would like to contribute to this project, there is a discord for dicussion:
[![Discord Server](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.com/invite/u9yhsFmEkB)

## Development environment for UI (frontend and server) changes
This is in-flux, but one way to get a development environment running for editing the UI of this project is:
(swap `.sh` or `.bat` in instructions depending on your environment, and be sure to adjust any paths to match where you're working)

1) Install the project to a new location using the [usual installation process](https://github.com/cmdr2/stable-diffusion-ui#installation), e.g. to `/projects/stable-diffusion-ui-archive`
2) Start the newly installed project, and check that you can view and generate images on `localhost:9000`
3) Next, please clone the project repository using `git clone` (e.g. to `/projects/stable-diffusion-ui-repo`)
4) Close the server (started in step 2), and edit `/projects/stable-diffusion-ui-archive/scripts/on_env_start.sh` (or `on_env_start.bat`)
5) Comment out the lines near the bottom that copies the `files/ui` folder, e.g:
  
  for `.sh`
```
# rm -rf ui
# cp -Rf sd-ui-files/ui .
# cp sd-ui-files/scripts/on_sd_start.sh scripts/
# cp sd-ui-files/scripts/start.sh .
``` 
for `.bat`
```
REM @xcopy sd-ui-files\ui ui /s /i /Y
REM @copy sd-ui-files\scripts\on_sd_start.bat scripts\ /Y
REM @copy "sd-ui-files\scripts\Start Stable Diffusion UI.cmd" . /Y
``` 
6) Next, comment out the line at the top of `/projects/stable-diffusion-ui-archive/scripts/on_sd_start.sh` (or `on_sd_start.bat`) that copies `on_env_start`. For e.g. `@rem @copy sd-ui-files\scripts\on_env_start.bat scripts\ /Y`
8) Delete the current `ui` folder at `/projects/stable-diffusion-ui-archive/ui`
9) Now make a symlink between the repository clone (where you will be making changes) and this archive (where you will be running stable diffusion):
`ln -s /projects/stable-diffusion-ui-repo/ui /projects/stable-diffusion-ui-archive/ui`
or for Windows
`mklink /J \projects\stable-diffusion-ui-archive\ui \projects\stable-diffusion-ui-repo\ui` (link name first, source repo dir second)
9) Run the project again (like in step 2) and ensure you can still use the UI.
10) Congrats, now any changes you make in your repo `ui` folder are linked to this running archive of the app and can be previewed in the browser.

Check the `ui/frontend/build/README.md` for instructions on running and building the React code.

## Development environment for Installer changes
Build the Windows installer using Windows, and the Linux installer using Linux. Don't mix the two, and don't use WSL. An Ubuntu VM is fine for building the Linux installer on a Windows host.

1. Run `build.bat` or `./build.sh` depending on whether you're in Windows or Linux.
2. Make a new GitHub release and upload the Windows and Linux installer builds created inside the `dist` folder.

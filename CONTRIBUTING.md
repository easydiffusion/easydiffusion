Hi there,
These instructions are meant for developers of this project.

If you only want to use the Stable Diffusion UI, you've downloaded the wrong file.

Please download and follow the instructions at https://github.com/cmdr2/stable-diffusion-ui#installation
Thanks

# For developers:

If you would like to contribute to this project, there is a discord for dicussion:
[![Discord Server](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.com/invite/u9yhsFmEkB)

## Development environment
This is in-flux, but one way to get a development environment running for editing the UI of this project is:
(swap `.sh` or `.bat` in instructions depending on your environment, and be sure to adjust any paths to match where you're working)

1) `git clone` the repository, e.g. to `/projects/stable-diffusion-ui-repo`
2) Download the pre-built end user archive from the link on github, and extract it, e.g. to `/projects/stable-diffusion-ui-archive`
3) `cd /projects/stable-diffusion-ui-archive` and run the script to set up and start the project, e.g. `start.sh`
4) Check you can view and generate images on `localhost:9000`
5) Close the server, and edit `/projects/stable-diffusion-ui-archive/scripts/on_env_start.sh`
6) Comment out the line near the bottom that copies the `files/ui` folder, e.g. `cp -Rf sd-ui-files/ui ui` for `.sh` or `@xcopy sd-ui-files\ui ui /s /i /Y` for `.bat`
7) Delete the current `ui` folder at `/projects/stable-diffusion-ui-archive/ui`
8) Now make a symlink between the repository clone (where you will be making changes) and this archive (where you will be running stable diffusion):
`ln -s /projects/stable-diffusion-ui-repo/ui /projects/stable-diffusion-ui-archive/ui`
or for Windows
`mklink /projects/stable-diffusion-ui-repo/ui /projects/stable-diffusion-ui-archive/ui`
9) Run the archive again `start.sh` and ensure you can still use the UI.
10) Congrats, now any changes you make in your repo `ui` folder are linked to this running archive of the app and can be previewed in the browser.

Check the `ui/frontend/build/README.md` for instructions on running and building the React code.

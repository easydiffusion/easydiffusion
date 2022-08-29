@echo. & echo "Stable Diffusion UI" & echo.

@cd ..

@set new_install=F

@if not exist "stable-diffusion\" (
    @set new_install=T
    @echo. & echo "Downloading Stable Diffusion.." & echo.
    @call git clone https://github.com/CompVis/stable-diffusion.git
)

@cd stable-diffusion

@if not exist "env\" (
    @echo. & echo "Downloading packages necessary for Stable Diffusion.." & echo. & echo "***** This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient ***** .." & echo.

    @call conda env create --prefix env -f environment.yaml
    @call conda activate .\env

    @echo. & echo "Downloading packages necessary for Stable Diffusion UI.." & echo.

    @call conda install -c conda-forge -y --prefix env uvicorn fastapi
) else (
    @call conda activate .\env
)

@if not exist "sd-v1-4.ckpt" (
    @echo. & echo "Downloading data files (weights) for Stable Diffusion.." & echo.
    @call curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt
)

@if "%new_install%"=="T" (
    @echo. & echo "Testing your new installation of Stable Diffusion.." & echo.

    python scripts\txt2img.py --prompt "photo of an astronaut riding a motorcycle" --W 256 --H 256 --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 1

    @if not exist "outputs\txt2img-samples\samples" (
        @echo. & echo "There was an error while running Stable Diffusion. Please check the troubleshooting guide (https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting) for common issues. If that doesn't work, please feel free to file an issue at: https://github.com/cmdr2/stable-diffusion-ui/issues" & echo.
        @pause
        @exit /b
    )
)

@echo. & echo "Ready to rock!" & echo.

@cd ..\ui

@uvicorn server:app --port 9000
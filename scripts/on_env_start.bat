@echo. & echo "Stable Diffusion UI" & echo.

@cd ..

@set new_install=F

@if not exist "stable-diffusion\" (
    @set new_install=T
    @echo. & echo "Downloading Stable Diffusion.." & echo.
    @call git clone https://github.com/basujindal/stable-diffusion.git
)

@cd stable-diffusion

@if not exist "env\" (
    @echo. & echo "Downloading packages necessary for Stable Diffusion.." & echo. & echo "***** This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient ***** .." & echo.

    @call conda env create --prefix env -f environment.yaml
    @call conda activate .\env

    @echo. & echo "Downloading packages necessary for Stable Diffusion UI.." & echo.

    @call conda install -c conda-forge -y --prefix env uvicorn fastapi

    @rem "Check if everything was installed, by running it once more"

    @call conda env update --prefix .\env --file environment.yaml
) else (
    @call conda activate .\env
)

@if not exist "sd-v1-4.ckpt" (
    @echo. & echo "Downloading data files (weights) for Stable Diffusion.." & echo.
    @call curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt
)

@if "%new_install%"=="T" (
    @echo. & echo "Checking your new installation of Stable Diffusion.." & echo.

    @call conda env update --prefix .\env --file environment.yaml
)

@echo. & echo "Ready to rock!" & echo.

@set SD_UI_PATH=%cd%\..\ui

@uvicorn server:app --app-dir "%SD_UI_PATH%" --port 9000 --host 0.0.0.0

@pause
@echo off

@>nul grep -c "sd_git_cloned" scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Stable Diffusion's git repository was already installed. Updating.."

    @cd stable-diffusion

    @call git reset --hard
    @call git pull

    @cd ..
) else (
    @echo. & echo "Downloading Stable Diffusion.." & echo.

    @call git clone https://github.com/basujindal/stable-diffusion.git && (
        @echo sd_git_cloned >> scripts\install_status.txt
    ) || (
        @echo "Error downloading Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues"
        pause
        @exit /b
    )
)

@cd stable-diffusion

@>nul grep -c "conda_sd_env_created" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Packages necessary for Stable Diffusion were already installed"
) else (
    @echo. & echo "Downloading packages necessary for Stable Diffusion.." & echo. & echo "***** This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient ***** .." & echo.

    @rmdir /s /q .\env

    @call conda env create --prefix env -f environment.yaml && (
        @echo conda_sd_env_created >> ..\scripts\install_status.txt
    ) || (
        echo "Error installing the packages necessary for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues"
        pause
        exit /b
    )
)

@call conda activate .\env

@>nul grep -c "conda_sd_ui_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    echo "Packages necessary for Stable Diffusion UI were already installed"
) else (
    @echo. & echo "Downloading packages necessary for Stable Diffusion UI.." & echo.

    @call conda install -c conda-forge -y --prefix env uvicorn fastapi && (
        @echo conda_sd_ui_deps_installed >> ..\scripts\install_status.txt
    ) || (
        echo "Error installing the packages necessary for Stable Diffusion UI. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues"
        pause
        exit /b
    )
)

@if exist "sd-v1-4.ckpt" (
    for %%I in ("sd-v1-4.ckpt") do if %%~zI GTR 400000000 (
        echo "Data files (weights) necessary for Stable Diffusion were already downloaded"
    ) else (
        echo. & echo "The model file present at %cd%\sd-v1-4.ckpt is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "sd-v1-4.ckpt"
    )
)

@if not exist "sd-v1-4.ckpt" (
    @echo. & echo "Downloading data files (weights) for Stable Diffusion.." & echo.

    @call curl -L -k https://me.cmdr2.org/stable-diffusion-ui/sd-v1-4.ckpt > sd-v1-4.ckpt

    @if exist "sd-v1-4.ckpt" (
        for %%I in ("sd-v1-4.ckpt") do if %%~zI LSS 400000000 (
            echo. & echo "Error: The downloaded model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo.
        pause
        exit /b
    )

    @echo sd_weights_downloaded >> ..\scripts\install_status.txt
    @echo sd_install_complete >> ..\scripts\install_status.txt
)

@echo. & echo "Stable Diffusion is ready!" & echo.

@cd ..
@set SD_UI_PATH=%cd%\ui
@cd stable-diffusion

@uvicorn server:app --app-dir "%SD_UI_PATH%" --port 9000 --host 0.0.0.0

@pause
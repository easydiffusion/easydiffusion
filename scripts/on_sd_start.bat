@echo off

@REM Caution, this file will make your eyes and brain bleed. It's such an unholy mess.
@REM Note to self: Please rewrite this in Python. For the sake of your own sanity.

@>nul grep -c "sd_git_cloned" scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Stable Diffusion's git repository was already installed. Updating.."

    @cd stable-diffusion

    @call git reset --hard
    @call git pull
    @call git checkout d154155d4c0b43e13ec1f00eb72b7ff9d522fcf9

    @cd ..
) else (
    @echo. & echo "Downloading Stable Diffusion.." & echo.

    @call git clone https://github.com/basujindal/stable-diffusion.git && (
        @echo sd_git_cloned >> scripts\install_status.txt
    ) || (
        @echo "Error downloading Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        @exit /b
    )

    @cd stable-diffusion
    @call git checkout d154155d4c0b43e13ec1f00eb72b7ff9d522fcf9
    @cd ..
)

@cd stable-diffusion

@>nul grep -c "conda_sd_env_created" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Packages necessary for Stable Diffusion were already installed"

    @call conda activate .\env
) else (
    @echo. & echo "Downloading packages necessary for Stable Diffusion.." & echo. & echo "***** This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient ***** .." & echo.

    @rmdir /s /q .\env

    @call conda env create --prefix env -f environment.yaml || (
        @echo. & echo "Error installing the packages necessary for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @call conda activate .\env

    for /f "tokens=*" %%a in ('python -c "import torch; import ldm; import transformers; import numpy; import antlr4; print(42)"') do if "%%a" NEQ "42" (
        @echo. & echo "Dependency test failed! Error installing the packages necessary for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @echo conda_sd_env_created >> ..\scripts\install_status.txt
)

@>nul grep -c "conda_sd_gfpgan_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Packages necessary for GFPGAN (Face Correction) were already installed"
) else (
    @echo. & echo "Downloading packages necessary for GFPGAN (Face Correction).." & echo.

    @call pip install -e git+https://github.com/TencentARC/GFPGAN#egg=GFPGAN || (
        @echo. & echo "Error installing the packages necessary for GFPGAN (Face Correction). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    for /f "tokens=*" %%a in ('python -c "from gfpgan import GFPGANer; print(42)"') do if "%%a" NEQ "42" (
        @echo. & echo "Dependency test failed! Error installing the packages necessary for GFPGAN (Face Correction). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @echo conda_sd_gfpgan_deps_installed >> ..\scripts\install_status.txt
)

@>nul grep -c "conda_sd_esrgan_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Packages necessary for ESRGAN (Resolution Upscaling) were already installed"
) else (
    @echo. & echo "Downloading packages necessary for ESRGAN (Resolution Upscaling).." & echo.

    @call pip install -e git+https://github.com/xinntao/Real-ESRGAN#egg=realesrgan || (
        @echo. & echo "Error installing the packages necessary for ESRGAN (Resolution Upscaling). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    for /f "tokens=*" %%a in ('python -c "from basicsr.archs.rrdbnet_arch import RRDBNet; from realesrgan import RealESRGANer; print(42)"') do if "%%a" NEQ "42" (
        @echo. & echo "Dependency test failed! Error installing the packages necessary for ESRGAN (Resolution Upscaling). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @echo conda_sd_esrgan_deps_installed >> ..\scripts\install_status.txt
)

@>nul grep -c "conda_sd_ui_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    echo "Packages necessary for Stable Diffusion UI were already installed"
) else (
    @echo. & echo "Downloading packages necessary for Stable Diffusion UI.." & echo.

    @call conda install -c conda-forge -y --prefix env uvicorn fastapi || (
        echo "Error installing the packages necessary for Stable Diffusion UI. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        exit /b
    )
)

call WHERE uvicorn > .tmp
@>nul grep -c "uvicorn" .tmp
@if "%ERRORLEVEL%" NEQ "0" (
    @echo. & echo "UI packages not found! Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
    pause
    exit /b
)

@>nul grep -c "conda_sd_ui_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" NEQ "0" (
    @echo conda_sd_ui_deps_installed >> ..\scripts\install_status.txt
)



@if exist "sd-v1-4.ckpt" (
    for %%I in ("sd-v1-4.ckpt") do if "%%~zI" EQU "4265380512" (
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
        for %%I in ("sd-v1-4.ckpt") do if "%%~zI" NEQ "4265380512" (
            echo. & echo "Error: The downloaded model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@if exist "GFPGANv1.3.pth" (
    for %%I in ("GFPGANv1.3.pth") do if "%%~zI" EQU "348632874" (
        echo "Data files (weights) necessary for GFPGAN (Face Correction) were already downloaded"
    ) else (
        echo. & echo "The GFPGAN model file present at %cd%\GFPGANv1.3.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "GFPGANv1.3.pth"
    )
)

@if not exist "GFPGANv1.3.pth" (
    @echo. & echo "Downloading data files (weights) for GFPGAN (Face Correction).." & echo.

    @call curl -L -k https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth > GFPGANv1.3.pth

    @if exist "GFPGANv1.3.pth" (
        for %%I in ("GFPGANv1.3.pth") do if "%%~zI" NEQ "348632874" (
            echo. & echo "Error: The downloaded GFPGAN model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for GFPGAN (Face Correction). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for GFPGAN (Face Correction). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@if exist "RealESRGAN_x4plus.pth" (
    for %%I in ("RealESRGAN_x4plus.pth") do if "%%~zI" EQU "67040989" (
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus were already downloaded"
    ) else (
        echo. & echo "The GFPGAN model file present at %cd%\RealESRGAN_x4plus.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "RealESRGAN_x4plus.pth"
    )
)

@if not exist "RealESRGAN_x4plus.pth" (
    @echo. & echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus.." & echo.

    @call curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth > RealESRGAN_x4plus.pth

    @if exist "RealESRGAN_x4plus.pth" (
        for %%I in ("RealESRGAN_x4plus.pth") do if "%%~zI" NEQ "67040989" (
            echo. & echo "Error: The downloaded ESRGAN x4plus model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@if exist "RealESRGAN_x4plus_anime_6B.pth" (
    for %%I in ("RealESRGAN_x4plus_anime_6B.pth") do if "%%~zI" EQU "17938799" (
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus_anime were already downloaded"
    ) else (
        echo. & echo "The GFPGAN model file present at %cd%\RealESRGAN_x4plus_anime_6B.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "RealESRGAN_x4plus_anime_6B.pth"
    )
)

@if not exist "RealESRGAN_x4plus_anime_6B.pth" (
    @echo. & echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime.." & echo.

    @call curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth > RealESRGAN_x4plus_anime_6B.pth

    @if exist "RealESRGAN_x4plus_anime_6B.pth" (
        for %%I in ("RealESRGAN_x4plus_anime_6B.pth") do if "%%~zI" NEQ "17938799" (
            echo. & echo "Error: The downloaded ESRGAN x4plus_anime model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@>nul grep -c "sd_install_complete" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" NEQ "0" (
    @echo sd_weights_downloaded >> ..\scripts\install_status.txt
    @echo sd_install_complete >> ..\scripts\install_status.txt
)

@echo. & echo "Stable Diffusion is ready!" & echo.

@cd ..
@set SD_UI_PATH=%cd%\ui
@cd stable-diffusion

@uvicorn server:app --app-dir "%SD_UI_PATH%" --port 9000 --host 0.0.0.0

@pause
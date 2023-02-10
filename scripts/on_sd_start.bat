@echo off

@REM Caution, this file will make your eyes and brain bleed. It's such an unholy mess.
@REM Note to self: Please rewrite this in Python. For the sake of your own sanity.

@copy sd-ui-files\scripts\on_env_start.bat scripts\ /Y
@copy sd-ui-files\scripts\bootstrap.bat scripts\ /Y
@copy sd-ui-files\scripts\check_modules.py scripts\ /Y

if exist "%cd%\profile" (
    set USERPROFILE=%cd%\profile
)

@rem set the correct installer path (current vs legacy)
if exist "%cd%\installer_files\env" (
    set INSTALL_ENV_DIR=%cd%\installer_files\env
)
if exist "%cd%\stable-diffusion\env" (
    set INSTALL_ENV_DIR=%cd%\stable-diffusion\env
)

@mkdir tmp
@set TMP=%cd%\tmp
@set TEMP=%cd%\tmp

@rem activate the installer env
call conda activate
@if "%ERRORLEVEL%" NEQ "0" (
       @echo. & echo "Error activating conda for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
       pause
       exit /b
)

@REM remove the old version of the dev console script, if it's still present
if exist "Open Developer Console.cmd" del "Open Developer Console.cmd"

@call python -c "import os; import shutil; frm = 'sd-ui-files\\ui\\hotfix\\9c24e6cd9f499d02c4f21a033736dabd365962dc80fe3aeb57a8f85ea45a20a3.26fead7ea4f0f843f6eb4055dfd25693f1a71f3c6871b184042d4b126244e142'; dst = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'transformers', '9c24e6cd9f499d02c4f21a033736dabd365962dc80fe3aeb57a8f85ea45a20a3.26fead7ea4f0f843f6eb4055dfd25693f1a71f3c6871b184042d4b126244e142'); shutil.copyfile(frm, dst) if os.path.exists(dst) else print(''); print('Hotfixed broken JSON file from OpenAI');"

@rem create the stable-diffusion folder, to work with legacy installations
if not exist "stable-diffusion" mkdir stable-diffusion
cd stable-diffusion

@rem activate the old stable-diffusion env, if it exists
if exist "env" (
    call conda activate .\env
)

@rem disable the legacy src and ldm folder (otherwise this prevents installing gfpgan and realesrgan)
if exist src rename src src-old
if exist ldm rename ldm ldm-old

if not exist "..\models\stable-diffusion" mkdir "..\models\stable-diffusion"
if not exist "..\models\gfpgan" mkdir "..\models\gfpgan"
if not exist "..\models\realesrgan" mkdir "..\models\realesrgan"
if not exist "..\models\vae" mkdir "..\models\vae"

@rem migrate the legacy models to the correct path (if already downloaded)
if exist "sd-v1-4.ckpt" move sd-v1-4.ckpt ..\models\stable-diffusion\
if exist "custom-model.ckpt" move custom-model.ckpt ..\models\stable-diffusion\
if exist "GFPGANv1.3.pth" move GFPGANv1.3.pth ..\models\gfpgan\
if exist "RealESRGAN_x4plus.pth" move RealESRGAN_x4plus.pth ..\models\realesrgan\
if exist "RealESRGAN_x4plus_anime_6B.pth" move RealESRGAN_x4plus_anime_6B.pth ..\models\realesrgan\

@rem install torch and torchvision
call python ..\scripts\check_modules.py torch torchvision
if "%ERRORLEVEL%" EQU "0" (
    echo "torch and torchvision have already been installed."
) else (
    echo "Installing torch and torchvision.."

    @REM prevent from using packages from the user's home directory, to avoid conflicts
    set PYTHONNOUSERSITE=1
    set PYTHONPATH=%INSTALL_ENV_DIR%\lib\site-packages

    call python -m pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 || (
        echo "Error installing torch. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        exit /b
    )
)

set PATH=C:\Windows\System32;%PATH%

@rem install/upgrade sdkit
call python ..\scripts\check_modules.py sdkit sdkit.models ldm transformers numpy antlr4 gfpgan realesrgan
if "%ERRORLEVEL%" EQU "0" (
    echo "sdkit is already installed."

    @rem skip sdkit upgrade if in developer-mode
    if not exist "..\src\sdkit" (
        @REM prevent from using packages from the user's home directory, to avoid conflicts
        set PYTHONNOUSERSITE=1
        set PYTHONPATH=%INSTALL_ENV_DIR%\lib\site-packages

        call python -m pip install --upgrade sdkit==1.0.35 -q || (
            echo "Error updating sdkit"
        )
    )
) else (
    echo "Installing sdkit: https://pypi.org/project/sdkit/"

    @REM prevent from using packages from the user's home directory, to avoid conflicts
    set PYTHONNOUSERSITE=1
    set PYTHONPATH=%INSTALL_ENV_DIR%\lib\site-packages

    call python -m pip install sdkit || (
        echo "Error installing sdkit. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        exit /b
    )
)

call python -c "from importlib.metadata import version; print('sdkit version:', version('sdkit'))"

@rem upgrade stable-diffusion-sdkit
call python -m pip install --upgrade stable-diffusion-sdkit -q || (
    echo "Error updating stable-diffusion-sdkit"
)
call python -c "from importlib.metadata import version; print('stable-diffusion version:', version('stable-diffusion-sdkit'))"

@rem install rich
call python ..\scripts\check_modules.py rich
if "%ERRORLEVEL%" EQU "0" (
    echo "rich has already been installed."
) else (
    echo "Installing rich.."

    set PYTHONNOUSERSITE=1
    set PYTHONPATH=%INSTALL_ENV_DIR%\lib\site-packages

    call python -m pip install rich || (
        echo "Error installing rich. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        exit /b
    )
)

set PATH=C:\Windows\System32;%PATH%

call python ..\scripts\check_modules.py uvicorn fastapi
@if "%ERRORLEVEL%" EQU "0" (
    echo "Packages necessary for Stable Diffusion UI were already installed"
) else (
    @echo. & echo "Downloading packages necessary for Stable Diffusion UI.." & echo.

    set PYTHONNOUSERSITE=1
    set PYTHONPATH=%INSTALL_ENV_DIR%\lib\site-packages

    @call conda install -c conda-forge -y uvicorn fastapi || (
        echo "Error installing the packages necessary for Stable Diffusion UI. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        exit /b
    )
)

call WHERE uvicorn > .tmp
@>nul findstr /m "uvicorn" .tmp
@if "%ERRORLEVEL%" NEQ "0" (
    @echo. & echo "UI packages not found! Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
    pause
    exit /b
)

@>nul findstr /m "conda_sd_ui_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" NEQ "0" (
    @echo conda_sd_ui_deps_installed >> ..\scripts\install_status.txt
)

@if exist "..\models\stable-diffusion\sd-v1-4.ckpt" (
    for %%I in ("..\models\stable-diffusion\sd-v1-4.ckpt") do if "%%~zI" EQU "4265380512" (
        echo "Data files (weights) necessary for Stable Diffusion were already downloaded. Using the HuggingFace 4 GB Model."
    ) else (
        for %%J in ("..\models\stable-diffusion\sd-v1-4.ckpt") do if "%%~zJ" EQU "7703807346" (
            echo "Data files (weights) necessary for Stable Diffusion were already downloaded. Using the HuggingFace 7 GB Model."
        ) else (
            for %%K in ("..\models\stable-diffusion\sd-v1-4.ckpt") do if "%%~zK" EQU "7703810927" (
                echo "Data files (weights) necessary for Stable Diffusion were already downloaded. Using the Waifu Model."
            ) else (
                echo. & echo "The model file present at models\stable-diffusion\sd-v1-4.ckpt is invalid. It is only %%~zK bytes in size. Re-downloading.." & echo.
                del "..\models\stable-diffusion\sd-v1-4.ckpt"
            )
        )
    )
)

@if not exist "..\models\stable-diffusion\sd-v1-4.ckpt" (
    @echo. & echo "Downloading data files (weights) for Stable Diffusion.." & echo.

    @call curl -L -k https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt > ..\models\stable-diffusion\sd-v1-4.ckpt

    @if exist "..\models\stable-diffusion\sd-v1-4.ckpt" (
        for %%I in ("..\models\stable-diffusion\sd-v1-4.ckpt") do if "%%~zI" NEQ "4265380512" (
            echo. & echo "Error: The downloaded model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@if exist "..\models\gfpgan\GFPGANv1.3.pth" (
    for %%I in ("..\models\gfpgan\GFPGANv1.3.pth") do if "%%~zI" EQU "348632874" (
        echo "Data files (weights) necessary for GFPGAN (Face Correction) were already downloaded"
    ) else (
        echo. & echo "The GFPGAN model file present at models\gfpgan\GFPGANv1.3.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "..\models\gfpgan\GFPGANv1.3.pth"
    )
)

@if not exist "..\models\gfpgan\GFPGANv1.3.pth" (
    @echo. & echo "Downloading data files (weights) for GFPGAN (Face Correction).." & echo.

    @call curl -L -k https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth > ..\models\gfpgan\GFPGANv1.3.pth

    @if exist "..\models\gfpgan\GFPGANv1.3.pth" (
        for %%I in ("..\models\gfpgan\GFPGANv1.3.pth") do if "%%~zI" NEQ "348632874" (
            echo. & echo "Error: The downloaded GFPGAN model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for GFPGAN (Face Correction). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for GFPGAN (Face Correction). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@if exist "..\models\realesrgan\RealESRGAN_x4plus.pth" (
    for %%I in ("..\models\realesrgan\RealESRGAN_x4plus.pth") do if "%%~zI" EQU "67040989" (
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus were already downloaded"
    ) else (
        echo. & echo "The RealESRGAN model file present at models\realesrgan\RealESRGAN_x4plus.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "..\models\realesrgan\RealESRGAN_x4plus.pth"
    )
)

@if not exist "..\models\realesrgan\RealESRGAN_x4plus.pth" (
    @echo. & echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus.." & echo.

    @call curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth > ..\models\realesrgan\RealESRGAN_x4plus.pth

    @if exist "..\models\realesrgan\RealESRGAN_x4plus.pth" (
        for %%I in ("..\models\realesrgan\RealESRGAN_x4plus.pth") do if "%%~zI" NEQ "67040989" (
            echo. & echo "Error: The downloaded ESRGAN x4plus model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@if exist "..\models\realesrgan\RealESRGAN_x4plus_anime_6B.pth" (
    for %%I in ("..\models\realesrgan\RealESRGAN_x4plus_anime_6B.pth") do if "%%~zI" EQU "17938799" (
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus_anime were already downloaded"
    ) else (
        echo. & echo "The RealESRGAN model file present at models\realesrgan\RealESRGAN_x4plus_anime_6B.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "..\models\realesrgan\RealESRGAN_x4plus_anime_6B.pth"
    )
)

@if not exist "..\models\realesrgan\RealESRGAN_x4plus_anime_6B.pth" (
    @echo. & echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime.." & echo.

    @call curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth > ..\models\realesrgan\RealESRGAN_x4plus_anime_6B.pth

    @if exist "..\models\realesrgan\RealESRGAN_x4plus_anime_6B.pth" (
        for %%I in ("RealESRGAN_x4plus_anime_6B.pth") do if "%%~zI" NEQ "17938799" (
            echo. & echo "Error: The downloaded ESRGAN x4plus_anime model file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)



@if exist "..\models\vae\vae-ft-mse-840000-ema-pruned.ckpt" (
    for %%I in ("..\models\vae\vae-ft-mse-840000-ema-pruned.ckpt") do if "%%~zI" EQU "334695179" (
        echo "Data files (weights) necessary for the default VAE (sd-vae-ft-mse-original) were already downloaded"
    ) else (
        echo. & echo "The default VAE (sd-vae-ft-mse-original) file present at models\vae\vae-ft-mse-840000-ema-pruned.ckpt is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "..\models\vae\vae-ft-mse-840000-ema-pruned.ckpt"
    )
)

@if not exist "..\models\vae\vae-ft-mse-840000-ema-pruned.ckpt" (
    @echo. & echo "Downloading data files (weights) for the default VAE (sd-vae-ft-mse-original).." & echo.

    @call curl -L -k https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt > ..\models\vae\vae-ft-mse-840000-ema-pruned.ckpt

    @if exist "..\models\vae\vae-ft-mse-840000-ema-pruned.ckpt" (
        for %%I in ("..\models\vae\vae-ft-mse-840000-ema-pruned.ckpt") do if "%%~zI" NEQ "334695179" (
            echo. & echo "Error: The downloaded default VAE (sd-vae-ft-mse-original) file was invalid! Bytes downloaded: %%~zI" & echo.
            echo. & echo "Error downloading the data files (weights) for the default VAE (sd-vae-ft-mse-original). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
            pause
            exit /b
        )
    ) else (
        @echo. & echo "Error downloading the data files (weights) for the default VAE (sd-vae-ft-mse-original). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )
)

@>nul findstr /m "sd_install_complete" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" NEQ "0" (
    @echo sd_weights_downloaded >> ..\scripts\install_status.txt
    @echo sd_install_complete >> ..\scripts\install_status.txt
)

@echo. & echo "Stable Diffusion is ready!" & echo.

@set SD_DIR=%cd%

set PYTHONPATH=%INSTALL_ENV_DIR%\lib\site-packages
echo PYTHONPATH=%PYTHONPATH%

call where python
call python --version

@cd ..
@set SD_UI_PATH=%cd%\ui
@cd stable-diffusion

@rem set any overrides
set HF_HUB_DISABLE_SYMLINKS_WARNING=true

@if NOT DEFINED SD_UI_BIND_PORT set SD_UI_BIND_PORT=9000
@if NOT DEFINED SD_UI_BIND_IP set SD_UI_BIND_IP=0.0.0.0
@uvicorn main:server_api --app-dir "%SD_UI_PATH%" --port %SD_UI_BIND_PORT% --host %SD_UI_BIND_IP% --log-level error


@pause

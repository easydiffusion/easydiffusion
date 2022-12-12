@echo off

@REM Caution, this file will make your eyes and brain bleed. It's such an unholy mess.
@REM Note to self: Please rewrite this in Python. For the sake of your own sanity.

@copy sd-ui-files\scripts\on_env_start.bat scripts\ /Y
@copy sd-ui-files\scripts\bootstrap.bat scripts\ /Y

if exist "%cd%\profile" (
    set USERPROFILE=%cd%\profile
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

if NOT DEFINED test_sd2 set test_sd2=N

@>nul findstr /m "sd_git_cloned" scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Stable Diffusion's git repository was already installed. Updating.."

    @cd stable-diffusion

    @call git remote set-url origin https://github.com/easydiffusion/diffusion-kit.git

    @call git reset --hard
    @call git pull

    if "%test_sd2%" == "N" (
        @call git -c advice.detachedHead=false checkout 7f32368ed1030a6e710537047bacd908adea183a
    )
    if "%test_sd2%" == "Y" (
        @call git -c advice.detachedHead=false checkout 733a1f6f9cae9b9a9b83294bf3281b123378cb1f
    )

    @cd ..
) else (
    @echo. & echo "Downloading Stable Diffusion.." & echo.

    @call git clone https://github.com/easydiffusion/diffusion-kit.git stable-diffusion && (
        @echo sd_git_cloned >> scripts\install_status.txt
    ) || (
        @echo "Error downloading Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        @exit /b
    )

    @cd stable-diffusion
    @call git -c advice.detachedHead=false checkout 7f32368ed1030a6e710537047bacd908adea183a

    @cd ..
)

@cd stable-diffusion

@>nul findstr /m "conda_sd_env_created" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Packages necessary for Stable Diffusion were already installed"

    @call conda activate .\env
) else (
    @echo. & echo "Downloading packages necessary for Stable Diffusion.." & echo. & echo "***** This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient ***** .." & echo.

    @rmdir /s /q .\env

    @REM prevent conda from using packages from the user's home directory, to avoid conflicts
    @set PYTHONNOUSERSITE=1

    set USERPROFILE=%cd%\profile

    set PYTHONPATH=%cd%;%cd%\env\lib\site-packages

    @call conda env create --prefix env -f environment.yaml || (
        @echo. & echo "Error installing the packages necessary for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @call conda activate .\env

    for /f "tokens=*" %%a in ('python -c "import torch; import ldm; import transformers; import numpy; import antlr4; print(42)"') do if "%%a" NEQ "42" (
        @echo. & echo "Dependency test failed! Error installing the packages necessary for Stable Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @echo conda_sd_env_created >> ..\scripts\install_status.txt
)

set PATH=C:\Windows\System32;%PATH%

@>nul findstr /m "conda_sd_gfpgan_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Packages necessary for GFPGAN (Face Correction) were already installed"
) else (
    @echo. & echo "Downloading packages necessary for GFPGAN (Face Correction).." & echo.

    @set PYTHONNOUSERSITE=1

    set USERPROFILE=%cd%\profile

    set PYTHONPATH=%cd%;%cd%\env\lib\site-packages

    for /f "tokens=*" %%a in ('python -c "from gfpgan import GFPGANer; print(42)"') do if "%%a" NEQ "42" (
        @echo. & echo "Dependency test failed! Error installing the packages necessary for GFPGAN (Face Correction). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @echo conda_sd_gfpgan_deps_installed >> ..\scripts\install_status.txt
)

@>nul findstr /m "conda_sd_esrgan_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Packages necessary for ESRGAN (Resolution Upscaling) were already installed"
) else (
    @echo. & echo "Downloading packages necessary for ESRGAN (Resolution Upscaling).." & echo.

    @set PYTHONNOUSERSITE=1

    set USERPROFILE=%cd%\profile

    set PYTHONPATH=%cd%;%cd%\env\lib\site-packages

    for /f "tokens=*" %%a in ('python -c "from basicsr.archs.rrdbnet_arch import RRDBNet; from realesrgan import RealESRGANer; print(42)"') do if "%%a" NEQ "42" (
        @echo. & echo "Dependency test failed! Error installing the packages necessary for ESRGAN (Resolution Upscaling). Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!" & echo.
        pause
        exit /b
    )

    @echo conda_sd_esrgan_deps_installed >> ..\scripts\install_status.txt
)

@>nul findstr /m "conda_sd_ui_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    echo "Packages necessary for Stable Diffusion UI were already installed"
) else (
    @echo. & echo "Downloading packages necessary for Stable Diffusion UI.." & echo.

    @set PYTHONNOUSERSITE=1

    set USERPROFILE=%cd%\profile

    set PYTHONPATH=%cd%;%cd%\env\lib\site-packages

    @call conda install -c conda-forge -y --prefix env uvicorn fastapi || (
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

@>nul 2>nul call python -m picklescan --help
@if "%ERRORLEVEL%" NEQ "0" (
    @echo. & echo Picklescan not found. Installing
    @call pip install picklescan || (
        echo "Error installing the picklescan package necessary for Stable Diffusion UI. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        exit /b
    )
)

@>nul 2>nul call python -c "import safetensors"
@if "%ERRORLEVEL%" NEQ "0" (
    @echo. & echo SafeTensors not found. Installing
    @call pip install safetensors || (
        echo "Error installing the safetensors package necessary for Stable Diffusion UI. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues" & echo "Thanks!"
        pause
        exit /b
    )
)

@>nul findstr /m "conda_sd_ui_deps_installed" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" NEQ "0" (
    @echo conda_sd_ui_deps_installed >> ..\scripts\install_status.txt
)



if not exist "..\models\vae" mkdir "..\models\vae"

@if exist "sd-v1-4.ckpt" (
    for %%I in ("sd-v1-4.ckpt") do if "%%~zI" EQU "4265380512" (
        echo "Data files (weights) necessary for Stable Diffusion were already downloaded. Using the HuggingFace 4 GB Model."
    ) else (
        for %%J in ("sd-v1-4.ckpt") do if "%%~zJ" EQU "7703807346" (
            echo "Data files (weights) necessary for Stable Diffusion were already downloaded. Using the HuggingFace 7 GB Model."
        ) else (
            for %%K in ("sd-v1-4.ckpt") do if "%%~zK" EQU "7703810927" (
                echo "Data files (weights) necessary for Stable Diffusion were already downloaded. Using the Waifu Model."
            ) else (
                echo. & echo "The model file present at %cd%\sd-v1-4.ckpt is invalid. It is only %%~zK bytes in size. Re-downloading.." & echo.
                del "sd-v1-4.ckpt"
            )
        )
    )
)

@if not exist "sd-v1-4.ckpt" (
    @echo. & echo "Downloading data files (weights) for Stable Diffusion.." & echo.

    @call curl -L -k https://me.cmdr2.org/stable-diffusion-ui/sd-v1-4.ckpt > sd-v1-4.ckpt

    @if exist "sd-v1-4.ckpt" (
        for %%I in ("sd-v1-4.ckpt") do if "%%~zI" NEQ "4265380512" (
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



@if exist "RealESRGAN_x4plus.pth" (
    for %%I in ("RealESRGAN_x4plus.pth") do if "%%~zI" EQU "67040989" (
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus were already downloaded"
    ) else (
        echo. & echo "The RealESRGAN model file present at %cd%\RealESRGAN_x4plus.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "RealESRGAN_x4plus.pth"
    )
)

@if not exist "RealESRGAN_x4plus.pth" (
    @echo. & echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus.." & echo.

    @call curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth > RealESRGAN_x4plus.pth

    @if exist "RealESRGAN_x4plus.pth" (
        for %%I in ("RealESRGAN_x4plus.pth") do if "%%~zI" NEQ "67040989" (
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



@if exist "RealESRGAN_x4plus_anime_6B.pth" (
    for %%I in ("RealESRGAN_x4plus_anime_6B.pth") do if "%%~zI" EQU "17938799" (
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus_anime were already downloaded"
    ) else (
        echo. & echo "The RealESRGAN model file present at %cd%\RealESRGAN_x4plus_anime_6B.pth is invalid. It is only %%~zI bytes in size. Re-downloading.." & echo.
        del "RealESRGAN_x4plus_anime_6B.pth"
    )
)

@if not exist "RealESRGAN_x4plus_anime_6B.pth" (
    @echo. & echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime.." & echo.

    @call curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth > RealESRGAN_x4plus_anime_6B.pth

    @if exist "RealESRGAN_x4plus_anime_6B.pth" (
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

if "%test_sd2%" == "Y" (
    @call pip install open_clip_torch==2.0.2
)

@>nul findstr /m "sd_install_complete" ..\scripts\install_status.txt
@if "%ERRORLEVEL%" NEQ "0" (
    @echo sd_weights_downloaded >> ..\scripts\install_status.txt
    @echo sd_install_complete >> ..\scripts\install_status.txt
)

@echo. & echo "Stable Diffusion is ready!" & echo.

@set SD_DIR=%cd%

@cd env\lib\site-packages
@set PYTHONPATH=%SD_DIR%;%cd%
@cd ..\..\..
@echo PYTHONPATH=%PYTHONPATH%

call where python
call python --version

@cd ..
@set SD_UI_PATH=%cd%\ui
@cd stable-diffusion

@rem
@rem Rewrite easy-install.pth. This fixes the installation if the user has relocated the SDUI installation
@rem
>env\Lib\site-packages\easy-install.pth echo %cd%\src\taming-transformers
>>env\Lib\site-packages\easy-install.pth echo %cd%\src\clip
>>env\Lib\site-packages\easy-install.pth echo %cd%\src\gfpgan
>>env\Lib\site-packages\easy-install.pth echo %cd%\src\realesrgan

@if NOT DEFINED SD_UI_BIND_PORT set SD_UI_BIND_PORT=9000
@if NOT DEFINED SD_UI_BIND_IP set SD_UI_BIND_IP=0.0.0.0
@uvicorn server:server_api --app-dir "%SD_UI_PATH%" --port %SD_UI_BIND_PORT% --host %SD_UI_BIND_IP% --log-level error


@pause

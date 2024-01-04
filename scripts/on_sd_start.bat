@echo off

@REM Caution, this file will make your eyes and brain bleed. It's such an unholy mess.
@REM Note to self: Please rewrite this in Python. For the sake of your own sanity.

@copy sd-ui-files\scripts\on_env_start.bat scripts\ /Y
@copy sd-ui-files\scripts\check_modules.py scripts\ /Y
@copy sd-ui-files\scripts\get_config.py scripts\ /Y
@copy sd-ui-files\scripts\config.yaml.sample scripts\ /Y

if exist "%cd%\profile" (
    set HF_HOME=%cd%\profile\.cache\huggingface
    set HF_ENDPOINT=https://hf-mirror.com
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
       @echo. & echo "Error activating conda for Easy Diffusion. Sorry about that, please try to:" & echo "  1. Run this installer again." & echo "  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/easydiffusion/easydiffusion/wiki/Troubleshooting" & echo "  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB" & echo "  4. If that doesn't solve the problem, please file an issue at https://github.com/easydiffusion/easydiffusion/issues" & echo "Thanks!" & echo.
       pause
       exit /b
)

@REM remove the old version of the dev console script, if it's still present
if exist "Open Developer Console.cmd" del "Open Developer Console.cmd"
if exist "ui\plugins\ui\merge.plugin.js" del "ui\plugins\ui\merge.plugin.js"

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


if not exist "%INSTALL_ENV_DIR%\DLLs\libssl-1_1-x64.dll"    copy "%INSTALL_ENV_DIR%\Library\bin\libssl-1_1-x64.dll"    "%INSTALL_ENV_DIR%\DLLs\"
if not exist "%INSTALL_ENV_DIR%\DLLs\libcrypto-1_1-x64.dll" copy "%INSTALL_ENV_DIR%\Library\bin\libcrypto-1_1-x64.dll" "%INSTALL_ENV_DIR%\DLLs\"

cd ..

@rem set any overrides
set HF_HUB_DISABLE_SYMLINKS_WARNING=true

@rem install or upgrade the required modules
set PATH=C:\Windows\System32;%PATH%

@REM prevent from using packages from the user's home directory, to avoid conflicts
set PYTHONNOUSERSITE=1
set PYTHONPATH=%INSTALL_ENV_DIR%\lib\site-packages
echo PYTHONPATH=%PYTHONPATH%

@rem Download the required packages
call where python
call python --version

call python scripts\check_modules.py --launch-uvicorn
pause
exit /b


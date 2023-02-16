@echo off

@echo "Hi there, what you are running is meant for the developers of this project, not for users." & echo.
@echo "If you only want to use the Stable Diffusion UI, you've downloaded the wrong file."
@echo "Please download and follow the instructions at https://github.com/cmdr2/stable-diffusion-ui#installation" & echo.
@echo "If you are actually a developer of this project, please type Y and press enter" & echo.

set /p answer=Are you a developer of this project (Y/N)?
if /i "%answer:~,1%" NEQ "Y" exit /b

mkdir dist\win\stable-diffusion-ui\scripts
@REM mkdir dist\linux-mac\stable-diffusion-ui\scripts

@rem copy the installer files for Windows

copy scripts\on_env_start.bat dist\win\stable-diffusion-ui\scripts\
copy scripts\bootstrap.bat dist\win\stable-diffusion-ui\scripts\
copy scripts\config.yaml.sample dist\win\stable-diffusion-ui\scripts\config.yaml
copy "scripts\Start Stable Diffusion UI.cmd" dist\win\stable-diffusion-ui\
copy LICENSE dist\win\stable-diffusion-ui\
copy "CreativeML Open RAIL-M License" dist\win\stable-diffusion-ui\
copy "How to install and run.txt" dist\win\stable-diffusion-ui\
echo. > dist\win\stable-diffusion-ui\scripts\install_status.txt

@rem copy the installer files for Linux and Mac

@REM copy scripts\on_env_start.sh dist\linux-mac\stable-diffusion-ui\scripts\
@REM copy scripts\bootstrap.sh dist\linux-mac\stable-diffusion-ui\scripts\
@REM copy scripts\start.sh dist\linux-mac\stable-diffusion-ui\
@REM copy LICENSE dist\linux-mac\stable-diffusion-ui\
@REM copy "CreativeML Open RAIL-M License" dist\linux-mac\stable-diffusion-ui\
@REM copy "How to install and run.txt" dist\linux-mac\stable-diffusion-ui\
@REM echo. > dist\linux-mac\stable-diffusion-ui\scripts\install_status.txt

@rem make the zip

cd dist\win
call powershell Compress-Archive -Path stable-diffusion-ui -DestinationPath ..\stable-diffusion-ui-windows.zip
cd ..\..

@REM cd dist\linux-mac
@REM call powershell Compress-Archive -Path stable-diffusion-ui -DestinationPath ..\stable-diffusion-ui-linux.zip
@REM call powershell Compress-Archive -Path stable-diffusion-ui -DestinationPath ..\stable-diffusion-ui-mac.zip
@REM cd ..\..

echo "Build ready. Upload the zip files inside the 'dist' folder."

pause

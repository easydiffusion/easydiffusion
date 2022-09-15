@echo off

@echo "Hi there, what you are running is meant for the developers of this project, not for users." & echo.
@echo "If you only want to use the Stable Diffusion UI, you've downloaded the wrong file."
@echo "Please download and follow the instructions at https://github.com/cmdr2/stable-diffusion-ui#installation" & echo.
@echo "If you are actually a developer of this project, please type Y and press enter" & echo.

set /p answer=Are you a developer of this project (Y/N)?
if /i "%answer:~,1%" NEQ "Y" exit /b

@set PYTHONNOUSERSITE=1

@mkdir dist\stable-diffusion-ui

@echo "Downloading components for the installer.."

@call conda env create --prefix installer -f environment.yaml
@call conda activate .\installer

@echo "Creating a distributable package.."

@call conda install -c conda-forge -y conda-pack
@call conda pack --n-threads -1 --prefix installer --format tar

@cd dist\stable-diffusion-ui
@mkdir installer

@call tar -xf ..\..\installer.tar -C installer

@mkdir scripts

@copy ..\..\scripts\on_env_start.bat scripts\
@copy "..\..\scripts\Start Stable Diffusion UI.cmd" .
@copy ..\..\LICENSE .
@copy "..\..\CreativeML Open RAIL-M License" .
@copy "..\..\How to install and run.txt" .
@echo "" > scripts\install_status.txt

@echo "Build ready. Zip the 'dist\stable-diffusion-ui' folder."

@echo "Cleaning up.."

@cd ..\..

@rmdir /s /q installer

@del installer.tar
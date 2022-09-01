@mkdir dist\stable-diffusion-ui

@echo "Downloading components for the installer.."

@call conda env create --prefix installer -f environment.yaml
@call conda activate .\installer

@echo "Setting up startup scripts.."

@mkdir installer\etc\conda\activate.d
@copy scripts\post_activate.bat installer\etc\conda\activate.d\

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
@copy ..\..\CreativeML Open RAIL-M License .
@copy "..\..\How to install and run.txt" .
@xcopy ..\..\ui ui /s /i

@echo "Build ready. Zip the 'dist\stable-diffusion-ui' folder."

@echo "Cleaning up.."

@cd ..\..

@rmdir /s /q installer

@del installer.tar
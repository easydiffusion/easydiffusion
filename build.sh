#!/bin/bash

printf "Hi there, what you are running is meant for the developers of this project, not for users.\n\n"
printf "If you only want to use the Stable Diffusion UI, you've downloaded the wrong file.\n"
printf "Please download and follow the instructions at https://github.com/cmdr2/stable-diffusion-ui#installation\n\n"
printf "If you are actually a developer of this project, please type Y and press enter\n\n"

read -p "Are you a developer of this project (Y/N) " yn
case $yn in
    [Yy]* ) ;;
    * ) exit;;
esac

mkdir -p dist/win/stable-diffusion-ui
mkdir -p dist/linux-mac/stable-diffusion-ui

# copy the installer files for Windows

cp scripts/on_env_start.bat dist/win/stable-diffusion-ui/scripts/
cp scripts/bootstrap.bat dist/win/stable-diffusion-ui/scripts/
cp "scripts/Start Stable Diffusion UI.cmd" dist/win/stable-diffusion-ui/
cp LICENSE dist/win/stable-diffusion-ui/
cp "CreativeML Open RAIL-M License" dist/win/stable-diffusion-ui/
cp "/How to install and run.txt" dist/win/stable-diffusion-ui/
echo. > dist/win/stable-diffusion-ui/scripts/install_status.txt

# copy the installer files for Linux and Mac

cp scripts/on_env_start.sh dist/linux-mac/stable-diffusion-ui/scripts/
cp scripts/bootstrap.sh dist/linux-mac/stable-diffusion-ui/scripts/
cp scripts/start.sh dist/linux-mac/stable-diffusion-ui/
cp LICENSE dist/linux-mac/stable-diffusion-ui/
cp "CreativeML Open RAIL-M License" dist/linux-mac/stable-diffusion-ui/
cp "/How to install and run.txt" dist/linux-mac/stable-diffusion-ui/
echo. > dist/linux-mac/stable-diffusion-ui/scripts/install_status.txt

# make the zip

cd dist/win
zip -r ../stable-diffusion-ui-win-x64.zip stable-diffusion-ui
cd ../..

cd dist/linux-mac
zip -r ../stable-diffusion-ui-linux-x64.zip stable-diffusion-ui
zip -r ../stable-diffusion-ui-linux-arm64.zip stable-diffusion-ui
zip -r ../stable-diffusion-ui-mac-x64.zip stable-diffusion-ui
zip -r ../stable-diffusion-ui-mac-arm64.zip stable-diffusion-ui
cd ../..

echo "Build ready. Upload the zip files inside the 'dist' folder."

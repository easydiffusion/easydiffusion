#!/bin/bash

printf "Hi there, what you are running is meant for the developers of this project, not for users.\n\n"
printf "If you only want to use Easy Diffusion, you've downloaded the wrong file.\n"
printf "Please download and follow the instructions at https://github.com/easydiffusion/easydiffusion#installation \n\n"
printf "If you are actually a developer of this project, please type Y and press enter\n\n"

read -p "Are you a developer of this project (Y/N) " yn
case $yn in
    [Yy]* ) ;;
    * ) exit;;
esac

mkdir -p dist/linux-mac/easy-diffusion/scripts

# copy the installer files for Linux and Mac

cp scripts/on_env_start.sh dist/linux-mac/easy-diffusion/scripts/
cp scripts/bootstrap.sh dist/linux-mac/easy-diffusion/scripts/
cp scripts/functions.sh dist/linux-mac/easy-diffusion/scripts/
cp scripts/config.yaml.sample dist/linux-mac/easy-diffusion/scripts/config.yaml.sample
cp scripts/start.sh dist/linux-mac/easy-diffusion/
cp LICENSE dist/linux-mac/easy-diffusion/
cp "CreativeML Open RAIL-M License" dist/linux-mac/easy-diffusion/
cp "How to install and run.txt" dist/linux-mac/easy-diffusion/
echo "" > dist/linux-mac/easy-diffusion/scripts/install_status.txt

# set the permissions
chmod u+x dist/linux-mac/easy-diffusion/scripts/on_env_start.sh
chmod u+x dist/linux-mac/easy-diffusion/scripts/bootstrap.sh
chmod u+x dist/linux-mac/easy-diffusion/start.sh

# make the zip

cd dist/linux-mac
zip -r ../Easy-Diffusion-Linux.zip easy-diffusion
zip -r ../Easy-Diffusion-Mac.zip easy-diffusion
cd ../..

echo "Build ready. Upload the zip files inside the 'dist' folder."

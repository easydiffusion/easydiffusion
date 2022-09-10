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

mkdir -p dist/stable-diffusion-ui

echo "Downloading components for the installer.."

source ~/miniconda3/etc/profile.d/conda.sh

conda install -c conda-forge -y conda-pack

conda env create --prefix installer -f environment.yaml
conda activate ./installer

echo "Creating a distributable package.."

conda pack --n-threads -1 --prefix installer --format tar

cd dist/stable-diffusion-ui
mkdir installer

tar -xf ../../installer.tar -C installer

mkdir scripts

cp ../../scripts/on_env_start.sh scripts/
cp ../../scripts/start.sh .
cp ../../LICENSE .
cp "../../CreativeML Open RAIL-M License" .
cp "../../How to install and run.txt" .

chmod u+x start.sh

echo "Build ready. Zip the 'dist/stable-diffusion-ui' folder."

echo "Cleaning up.."

cd ../..

rm -rf installer

rm installer.tar
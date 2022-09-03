#!/bin/bash

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
cp "../../scripts/start.sh" .
cp ../../LICENSE .
cp "../../CreativeML Open RAIL-M License" .
cp "../../How to install and run.txt" .

echo "Build ready. Zip the 'dist/stable-diffusion-ui' folder."

echo "Cleaning up.."

cd ../..

rm -rf installer

rm installer.tar
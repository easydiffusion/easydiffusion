#!/bin/bash

mkdir -p dist/stable-diffusion-ui

echo "Downloading components for the installer.."

conda env create --prefix installer -f environment.yaml
conda activate ./installer

echo "Setting up startup scripts.."

mkdir -p installer/etc/conda/activate.d
cp scripts/post_activate.sh installer/etc/conda/activate.d/

echo "Creating a distributable package.."

conda install -c conda-forge -y conda-pack
conda pack --n-threads -1 --prefix installer --format tar

cd dist/stable-diffusion-ui
mkdir installer

tar -xf ../../installer.tar -C installer

chmod u+x installer/bin/activate

mkdir scripts

cp ../../scripts/on_env_start.sh scripts/
cp "../../scripts/Start Stable Diffusion UI.sh" .
cp ../../LICENSE .
cp "../../CreativeML Open RAIL-M License" .
cp "../../How to install and run.txt" .

echo "Build ready. Zip the 'dist/stable-diffusion-ui' folder."

echo "Cleaning up.."

cd ../..

rm -rf installer

rm installer.tar
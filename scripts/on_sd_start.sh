#!/bin/bash

cp sd-ui-files/scripts/functions.sh scripts/
cp sd-ui-files/scripts/on_env_start.sh scripts/
cp sd-ui-files/scripts/bootstrap.sh scripts/
cp sd-ui-files/scripts/check_modules.py scripts/
cp sd-ui-files/scripts/get_config.py scripts/
cp sd-ui-files/scripts/config.yaml.sample scripts/
cp sd-ui-files/scripts/webui_console.py scripts/
     

source ./scripts/functions.sh

# activate the installer env
export CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # avoids the 'shell not initialized' error

conda activate || fail "Failed to activate conda"

# hack to fix conda 4.14 on older installations
cp $CONDA_BASEPATH/condabin/conda $CONDA_BASEPATH/bin/conda

# remove the old version of the dev console script, if it's still present
if [ -e "open_dev_console.sh" ]; then
    rm "open_dev_console.sh"
fi

if [ -e "ui/plugins/ui/merge.plugin.js" ]; then
    rm "ui/plugins/ui/merge.plugin.js"
fi

# set the correct installer path (current vs legacy)
if [ -e "installer_files/env" ]; then
    export INSTALL_ENV_DIR="$(pwd)/installer_files/env"
fi
if [ -e "stable-diffusion/env" ]; then
    export INSTALL_ENV_DIR="$(pwd)/stable-diffusion/env"
fi

# create the stable-diffusion folder, to work with legacy installations
if [ ! -e "stable-diffusion" ]; then mkdir stable-diffusion; fi
cd stable-diffusion

# activate the old stable-diffusion env, if it exists
if [ -e "env" ]; then
    conda activate ./env || fail "conda activate failed"
fi

# disable the legacy src and ldm folder (otherwise this prevents installing gfpgan and realesrgan)
if [ -e "src" ]; then mv src src-old; fi
if [ -e "ldm" ]; then mv ldm ldm-old; fi

# this is outside check_modules.py to ensure that the required version of torchruntime is present
python -m pip install -q "torchruntime>=1.16.2"

cd ..
# Download the required packages
python scripts/check_modules.py --launch-uvicorn

read -p "Press any key to continue"

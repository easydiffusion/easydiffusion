#!/bin/bash

cp sd-ui-files/scripts/functions.sh scripts/
cp sd-ui-files/scripts/on_env_start.sh scripts/
cp sd-ui-files/scripts/bootstrap.sh scripts/
cp sd-ui-files/scripts/check_modules.py scripts/
cp sd-ui-files/scripts/get_config.py scripts/
cp sd-ui-files/scripts/config.yaml.sample scripts/

source ./scripts/functions.sh

# activate the installer env
CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # avoids the 'shell not initialized' error

conda activate || fail "Failed to activate conda"

# remove the old version of the dev console script, if it's still present
if [ -e "open_dev_console.sh" ]; then
    rm "open_dev_console.sh"
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

# Download the required packages
if ! python ../scripts/check_modules.py; then
    read -p "Press any key to continue"
    exit 1
fi

if ! command -v uvicorn &> /dev/null; then
    fail "UI packages not found!"
fi

if [ `grep -c sd_install_complete ../scripts/install_status.txt` -gt "0" ]; then
    echo sd_weights_downloaded >> ../scripts/install_status.txt
    echo sd_install_complete >> ../scripts/install_status.txt
fi

printf "\n\nEasy Diffusion installation complete, starting the server!\n\n"

SD_PATH=`pwd`

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$INSTALL_ENV_DIR/lib/python3.8/site-packages"
echo "PYTHONPATH=$PYTHONPATH"

which python
python --version

cd ..
export SD_UI_PATH=`pwd`/ui
export ED_BIND_PORT="$( python scripts/get_config.py --default=9000 net listen_port )"
case "$( python scripts/get_config.py --default=False net listen_to_network )" in
    "True")
        export ED_BIND_IP=$( python scripts/get_config.py --default=0.0.0.0) net bind_ip)
        ;;
    "False")
        export ED_BIND_IP=127.0.0.1
        ;;
esac
cd stable-diffusion

uvicorn main:server_api --app-dir "$SD_UI_PATH" --port "$ED_BIND_PORT" --host "$ED_BIND_IP" --log-level error

read -p "Press any key to continue"

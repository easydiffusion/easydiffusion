#!/bin/bash

source ./scripts/functions.sh

cp sd-ui-files/scripts/on_env_start.sh scripts/
cp sd-ui-files/scripts/bootstrap.sh scripts/
cp sd-ui-files/scripts/check_modules.py scripts/

# activate the installer env
CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # avoids the 'shell not initialized' error

conda activate || fail "Failed to activate conda"

# remove the old version of the dev console script, if it's still present
if [ -e "open_dev_console.sh" ]; then
    rm "open_dev_console.sh"
fi

python -c "import os; import shutil; frm = 'sd-ui-files/ui/hotfix/9c24e6cd9f499d02c4f21a033736dabd365962dc80fe3aeb57a8f85ea45a20a3.26fead7ea4f0f843f6eb4055dfd25693f1a71f3c6871b184042d4b126244e142'; dst = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'transformers', '9c24e6cd9f499d02c4f21a033736dabd365962dc80fe3aeb57a8f85ea45a20a3.26fead7ea4f0f843f6eb4055dfd25693f1a71f3c6871b184042d4b126244e142'); shutil.copyfile(frm, dst) if os.path.exists(dst) else print(''); print('Hotfixed broken JSON file from OpenAI');" 

# Caution, this file will make your eyes and brain bleed. It's such an unholy mess.
# Note to self: Please rewrite this in Python. For the sake of your own sanity.

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

mkdir -p "../models/stable-diffusion"
mkdir -p "../models/gfpgan"
mkdir -p "../models/realesrgan"
mkdir -p "../models/vae"

# migrate the legacy models to the correct path (if already downloaded)
if [ -e "sd-v1-4.ckpt" ]; then mv sd-v1-4.ckpt ../models/stable-diffusion/; fi
if [ -e "custom-model.ckpt" ]; then mv custom-model.ckpt ../models/stable-diffusion/; fi
if [ -e "GFPGANv1.3.pth" ]; then mv GFPGANv1.3.pth ../models/gfpgan/; fi
if [ -e "RealESRGAN_x4plus.pth" ]; then mv RealESRGAN_x4plus.pth ../models/realesrgan/; fi
if [ -e "RealESRGAN_x4plus_anime_6B.pth" ]; then mv RealESRGAN_x4plus_anime_6B.pth ../models/realesrgan/; fi

# install torch and torchvision
if python ../scripts/check_modules.py torch torchvision; then
    echo "torch and torchvision have already been installed."
else
    echo "Installing torch and torchvision.."

    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$INSTALL_ENV_DIR/lib/python3.8/site-packages"

    if python -m pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 ; then
        echo "Installed."
    else
        fail "torch install failed" 
    fi
fi

# install/upgrade sdkit
if python ../scripts/check_modules.py sdkit sdkit.models ldm transformers numpy antlr4 gfpgan realesrgan ; then
    echo "sdkit is already installed."

    # skip sdkit upgrade if in developer-mode
    if [ ! -e "../src/sdkit" ]; then
        export PYTHONNOUSERSITE=1
        export PYTHONPATH="$INSTALL_ENV_DIR/lib/python3.8/site-packages"

        python -m pip install --upgrade sdkit -q
    fi
else
    echo "Installing sdkit: https://pypi.org/project/sdkit/"

    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$INSTALL_ENV_DIR/lib/python3.8/site-packages"

    if python -m pip install sdkit ; then
        echo "Installed."
    else
        fail "sdkit install failed"
    fi
fi

python -c "from importlib.metadata import version; print('sdkit version:', version('sdkit'))"

# upgrade stable-diffusion-sdkit
python -m pip install --upgrade stable-diffusion-sdkit -q
python -c "from importlib.metadata import version; print('stable-diffusion version:', version('stable-diffusion-sdkit'))"

# install rich
if python ../scripts/check_modules.py rich; then
    echo "rich has already been installed."
else
    echo "Installing rich.."

    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$INSTALL_ENV_DIR/lib/python3.8/site-packages"

    if python -m pip install rich ; then
        echo "Installed."
    else
        fail "Install failed for rich"
    fi
fi

if python ../scripts/check_modules.py uvicorn fastapi ; then
    echo "Packages necessary for Stable Diffusion UI were already installed"
else
    printf "\n\nDownloading packages necessary for Stable Diffusion UI..\n\n"

    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$INSTALL_ENV_DIR/lib/python3.8/site-packages"

    if conda install -c conda-forge -y uvicorn fastapi ; then
        echo "Installed. Testing.."
    else
        fail "'conda install uvicorn' failed" 
    fi

    if ! command -v uvicorn &> /dev/null; then
        fail "UI packages not found!"
    fi
fi

if [ -f "../models/stable-diffusion/sd-v1-4.ckpt" ]; then
    model_size=`find "../models/stable-diffusion/sd-v1-4.ckpt" -printf "%s"`

    if [ "$model_size" -eq "4265380512" ] || [ "$model_size" -eq "7703807346" ] || [ "$model_size" -eq "7703810927" ]; then
        echo "Data files (weights) necessary for Stable Diffusion were already downloaded"
    else
        printf "\n\nThe model file present at models/stable-diffusion/sd-v1-4.ckpt is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm ../models/stable-diffusion/sd-v1-4.ckpt
    fi
fi

if [ ! -f "../models/stable-diffusion/sd-v1-4.ckpt" ]; then
    echo "Downloading data files (weights) for Stable Diffusion.."

    curl -L -k https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt > ../models/stable-diffusion/sd-v1-4.ckpt

    if [ -f "../models/stable-diffusion/sd-v1-4.ckpt" ]; then
        model_size=`find "../models/stable-diffusion/sd-v1-4.ckpt" -printf "%s"`
        if [ ! "$model_size" == "4265380512" ]; then
	    fail "The downloaded model file was invalid! Bytes downloaded: $model_size"
        fi
    else
        fail "Error downloading the data files (weights) for Stable Diffusion"
    fi
fi


if [ -f "../models/gfpgan/GFPGANv1.3.pth" ]; then
    model_size=`find "../models/gfpgan/GFPGANv1.3.pth" -printf "%s"`

    if [ "$model_size" -eq "348632874" ]; then
        echo "Data files (weights) necessary for GFPGAN (Face Correction) were already downloaded"
    else
        printf "\n\nThe model file present at models/gfpgan/GFPGANv1.3.pth is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm ../models/gfpgan/GFPGANv1.3.pth
    fi
fi

if [ ! -f "../models/gfpgan/GFPGANv1.3.pth" ]; then
    echo "Downloading data files (weights) for GFPGAN (Face Correction).."

    curl -L -k https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth > ../models/gfpgan/GFPGANv1.3.pth

    if [ -f "../models/gfpgan/GFPGANv1.3.pth" ]; then
        model_size=`find "../models/gfpgan/GFPGANv1.3.pth" -printf "%s"`
        if [ ! "$model_size" -eq "348632874" ]; then
            fail "The downloaded GFPGAN model file was invalid! Bytes downloaded: $model_size"
        fi
    else
        fail "Error downloading the data files (weights) for GFPGAN (Face Correction)."
    fi
fi


if [ -f "../models/realesrgan/RealESRGAN_x4plus.pth" ]; then
    model_size=`find "../models/realesrgan/RealESRGAN_x4plus.pth" -printf "%s"`

    if [ "$model_size" -eq "67040989" ]; then
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus were already downloaded"
    else
        printf "\n\nThe model file present at models/realesrgan/RealESRGAN_x4plus.pth is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm ../models/realesrgan/RealESRGAN_x4plus.pth
    fi
fi

if [ ! -f "../models/realesrgan/RealESRGAN_x4plus.pth" ]; then
    echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus.."

    curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth > ../models/realesrgan/RealESRGAN_x4plus.pth

    if [ -f "../models/realesrgan/RealESRGAN_x4plus.pth" ]; then
        model_size=`find "../models/realesrgan/RealESRGAN_x4plus.pth" -printf "%s"`
        if [ ! "$model_size" -eq "67040989" ]; then
            fail "The downloaded ESRGAN x4plus model file was invalid! Bytes downloaded: $model_size"
        fi
    else
        fail "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus"
    fi
fi


if [ -f "../models/realesrgan/RealESRGAN_x4plus_anime_6B.pth" ]; then
    model_size=`find "../models/realesrgan/RealESRGAN_x4plus_anime_6B.pth" -printf "%s"`

    if [ "$model_size" -eq "17938799" ]; then
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus_anime were already downloaded"
    else
        printf "\n\nThe model file present at models/realesrgan/RealESRGAN_x4plus_anime_6B.pth is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm ../models/realesrgan/RealESRGAN_x4plus_anime_6B.pth
    fi
fi

if [ ! -f "../models/realesrgan/RealESRGAN_x4plus_anime_6B.pth" ]; then
    echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime.."

    curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth > ../models/realesrgan/RealESRGAN_x4plus_anime_6B.pth

    if [ -f "../models/realesrgan/RealESRGAN_x4plus_anime_6B.pth" ]; then
        model_size=`find "../models/realesrgan/RealESRGAN_x4plus_anime_6B.pth" -printf "%s"`
        if [ ! "$model_size" -eq "17938799" ]; then
            fail "The downloaded ESRGAN x4plus_anime model file was invalid! Bytes downloaded: $model_size"
        fi
    else
        fail "Error downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime."
    fi
fi


if [ -f "../models/vae/vae-ft-mse-840000-ema-pruned.ckpt" ]; then
    model_size=`find ../models/vae/vae-ft-mse-840000-ema-pruned.ckpt -printf "%s"`

    if [ "$model_size" -eq "334695179" ]; then
        echo "Data files (weights) necessary for the default VAE (sd-vae-ft-mse-original) were already downloaded"
    else
        printf "\n\nThe model file present at models/vae/vae-ft-mse-840000-ema-pruned.ckpt is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm ../models/vae/vae-ft-mse-840000-ema-pruned.ckpt
    fi
fi

if [ ! -f "../models/vae/vae-ft-mse-840000-ema-pruned.ckpt" ]; then
    echo "Downloading data files (weights) for the default VAE (sd-vae-ft-mse-original).."

    curl -L -k https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt > ../models/vae/vae-ft-mse-840000-ema-pruned.ckpt

    if [ -f "../models/vae/vae-ft-mse-840000-ema-pruned.ckpt" ]; then
        model_size=`find ../models/vae/vae-ft-mse-840000-ema-pruned.ckpt -printf "%s"`
        if [ ! "$model_size" -eq "334695179" ]; then
            printf "\n\nError: The downloaded default VAE (sd-vae-ft-mse-original) file was invalid! Bytes downloaded: $model_size\n\n"
            printf "\n\nError downloading the data files (weights) for the default VAE (sd-vae-ft-mse-original). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
            read -p "Press any key to continue"
            exit
        fi
    else
        printf "\n\nError downloading the data files (weights) for the default VAE (sd-vae-ft-mse-original). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi
fi

if [ `grep -c sd_install_complete ../scripts/install_status.txt` -gt "0" ]; then
    echo sd_weights_downloaded >> ../scripts/install_status.txt
    echo sd_install_complete >> ../scripts/install_status.txt
fi

printf "\n\nStable Diffusion is ready!\n\n"

SD_PATH=`pwd`

export PYTHONPATH="$INSTALL_ENV_DIR/lib/python3.8/site-packages"
echo "PYTHONPATH=$PYTHONPATH"

which python
python --version

cd ..
export SD_UI_PATH=`pwd`/ui
cd stable-diffusion

uvicorn main:server_api --app-dir "$SD_UI_PATH" --port ${SD_UI_BIND_PORT:-9000} --host ${SD_UI_BIND_IP:-0.0.0.0} --log-level error

read -p "Press any key to continue"

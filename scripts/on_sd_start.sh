#!/bin/bash

cp sd-ui-files/scripts/on_env_start.sh scripts/

source installer/etc/profile.d/conda.sh

# Caution, this file will make your eyes and brain bleed. It's such an unholy mess.
# Note to self: Please rewrite this in Python. For the sake of your own sanity.

if [ -e "scripts/install_status.txt" ] && [ `grep -c sd_git_cloned scripts/install_status.txt` -gt "0" ]; then
    echo "Stable Diffusion's git repository was already installed. Updating.."

    cd stable-diffusion

    git reset --hard
    git pull
    git checkout d154155d4c0b43e13ec1f00eb72b7ff9d522fcf9

    cd ..
else
    printf "\n\nDownloading Stable Diffusion..\n\n"

    if git clone https://github.com/basujindal/stable-diffusion.git ; then
        echo sd_git_cloned >> scripts/install_status.txt
    else
        printf "\n\nError downloading Stable Diffusion. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    cd stable-diffusion
    git checkout d154155d4c0b43e13ec1f00eb72b7ff9d522fcf9
    cd ..
fi

cd stable-diffusion

if [ `grep -c conda_sd_env_created ../scripts/install_status.txt` -gt "0" ]; then
    echo "Packages necessary for Stable Diffusion were already installed"

    conda activate ./env
else
    printf "\n\nDownloading packages necessary for Stable Diffusion..\n"
    printf "\n\n***** This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient ***** ..\n\n"

    if conda env create --prefix env --force -f environment.yaml ; then
        echo "Installed. Testing.."
    else
        printf "\n\nError installing the packages necessary for Stable Diffusion. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    conda activate ./env

    out_test=`python -c "import torch; import ldm; import transformers; import numpy; import antlr4; print(42)"`
    if [ "$out_test" != "42" ]; then
        printf "\n\nDependency test failed! Error installing the packages necessary for Stable Diffusion. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    echo conda_sd_env_created >> ../scripts/install_status.txt
fi

if [ `grep -c conda_sd_gfpgan_deps_installed ../scripts/install_status.txt` -gt "0" ]; then
    echo "Packages necessary for GFPGAN (Face Correction) were already installed"
else
    printf "\n\nDownloading packages necessary for GFPGAN (Face Correction)..\n"

    if pip install -e git+https://github.com/TencentARC/GFPGAN#egg=GFPGAN ; then
        echo "Installed. Testing.."
    else
        printf "\n\nError installing the packages necessary for GFPGAN (Face Correction). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    out_test=`python -c "from gfpgan import GFPGANer; print(42)"`
    if [ "$out_test" != "42" ]; then
        printf "\n\nDependency test failed! Error installing the packages necessary for GFPGAN (Face Correction). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    echo conda_sd_gfpgan_deps_installed >> ../scripts/install_status.txt
fi

if [ `grep -c conda_sd_esrgan_deps_installed ../scripts/install_status.txt` -gt "0" ]; then
    echo "Packages necessary for ESRGAN (Resolution Upscaling) were already installed"
else
    printf "\n\nDownloading packages necessary for ESRGAN (Resolution Upscaling)..\n"

    if pip install -e git+https://github.com/xinntao/Real-ESRGAN#egg=realesrgan ; then
        echo "Installed. Testing.."
    else
        printf "\n\nError installing the packages necessary for ESRGAN (Resolution Upscaling). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    out_test=`python -c "from basicsr.archs.rrdbnet_arch import RRDBNet; from realesrgan import RealESRGANer; print(42)"`
    if [ "$out_test" != "42" ]; then
        printf "\n\nDependency test failed! Error installing the packages necessary for ESRGAN (Resolution Upscaling). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    echo conda_sd_esrgan_deps_installed >> ../scripts/install_status.txt
fi

if [ `grep -c conda_sd_ui_deps_installed ../scripts/install_status.txt` -gt "0" ]; then
    echo "Packages necessary for Stable Diffusion UI were already installed"
else
    printf "\n\nDownloading packages necessary for Stable Diffusion UI..\n\n"

    if conda install -c conda-forge --prefix ./env -y uvicorn fastapi ; then
        echo "Installed. Testing.."
    else
        printf "\n\nError installing the packages necessary for Stable Diffusion UI. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    if ! command -v uvicorn &> /dev/null; then
        printf "\n\nUI packages not found! Error installing the packages necessary for Stable Diffusion UI. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi

    echo conda_sd_ui_deps_installed >> ../scripts/install_status.txt
fi



if [ -f "sd-v1-4.ckpt" ]; then
    model_size=`ls -l sd-v1-4.ckpt | awk '{print $5}'`

    if [ "$model_size" -eq "4265380512" ]; then
        echo "Data files (weights) necessary for Stable Diffusion were already downloaded"
    else
        printf "\n\nThe model file present at $PWD/sd-v1-4.ckpt is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm sd-v1-4.ckpt
    fi
fi

if [ ! -f "sd-v1-4.ckpt" ]; then
    echo "Downloading data files (weights) for Stable Diffusion.."

    curl -L -k https://me.cmdr2.org/stable-diffusion-ui/sd-v1-4.ckpt > sd-v1-4.ckpt

    if [ -f "sd-v1-4.ckpt" ]; then
        model_size=`ls -l sd-v1-4.ckpt | awk '{print $5}'`
        if [ ! "$model_size" -eq "4265380512" ]; then
            printf "\n\nError: The downloaded model file was invalid! Bytes downloaded: $model_size\n\n"
            printf "\n\nError downloading the data files (weights) for Stable Diffusion. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
            read -p "Press any key to continue"
            exit
        fi
    else
        printf "\n\nError downloading the data files (weights) for Stable Diffusion. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi
fi


if [ -f "GFPGANv1.3.pth" ]; then
    model_size=`ls -l GFPGANv1.3.pth | awk '{print $5}'`

    if [ "$model_size" -eq "348632874" ]; then
        echo "Data files (weights) necessary for GFPGAN (Face Correction) were already downloaded"
    else
        printf "\n\nThe model file present at $PWD/GFPGANv1.3.pth is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm GFPGANv1.3.pth
    fi
fi

if [ ! -f "GFPGANv1.3.pth" ]; then
    echo "Downloading data files (weights) for GFPGAN (Face Correction).."

    curl -L -k https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth > GFPGANv1.3.pth

    if [ -f "GFPGANv1.3.pth" ]; then
        model_size=`ls -l GFPGANv1.3.pth | awk '{print $5}'`
        if [ ! "$model_size" -eq "348632874" ]; then
            printf "\n\nError: The downloaded GFPGAN model file was invalid! Bytes downloaded: $model_size\n\n"
            printf "\n\nError downloading the data files (weights) for GFPGAN (Face Correction). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
            read -p "Press any key to continue"
            exit
        fi
    else
        printf "\n\nError downloading the data files (weights) for GFPGAN (Face Correction). Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi
fi


if [ -f "RealESRGAN_x4plus.pth" ]; then
    model_size=`ls -l RealESRGAN_x4plus.pth | awk '{print $5}'`

    if [ "$model_size" -eq "67040989" ]; then
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus were already downloaded"
    else
        printf "\n\nThe model file present at $PWD/RealESRGAN_x4plus.pth is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm RealESRGAN_x4plus.pth
    fi
fi

if [ ! -f "RealESRGAN_x4plus.pth" ]; then
    echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus.."

    curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth > RealESRGAN_x4plus.pth

    if [ -f "RealESRGAN_x4plus.pth" ]; then
        model_size=`ls -l RealESRGAN_x4plus.pth | awk '{print $5}'`
        if [ ! "$model_size" -eq "67040989" ]; then
            printf "\n\nError: The downloaded ESRGAN x4plus model file was invalid! Bytes downloaded: $model_size\n\n"
            printf "\n\nError downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
            read -p "Press any key to continue"
            exit
        fi
    else
        printf "\n\nError downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi
fi


if [ -f "RealESRGAN_x4plus_anime_6B.pth" ]; then
    model_size=`ls -l RealESRGAN_x4plus_anime_6B.pth | awk '{print $5}'`

    if [ "$model_size" -eq "17938799" ]; then
        echo "Data files (weights) necessary for ESRGAN (Resolution Upscaling) x4plus_anime were already downloaded"
    else
        printf "\n\nThe model file present at $PWD/RealESRGAN_x4plus_anime_6B.pth is invalid. It is only $model_size bytes in size. Re-downloading.."
        rm RealESRGAN_x4plus_anime_6B.pth
    fi
fi

if [ ! -f "RealESRGAN_x4plus_anime_6B.pth" ]; then
    echo "Downloading data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime.."

    curl -L -k https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth > RealESRGAN_x4plus_anime_6B.pth

    if [ -f "RealESRGAN_x4plus_anime_6B.pth" ]; then
        model_size=`ls -l RealESRGAN_x4plus_anime_6B.pth | awk '{print $5}'`
        if [ ! "$model_size" -eq "17938799" ]; then
            printf "\n\nError: The downloaded ESRGAN x4plus_anime model file was invalid! Bytes downloaded: $model_size\n\n"
            printf "\n\nError downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
            read -p "Press any key to continue"
            exit
        fi
    else
        printf "\n\nError downloading the data files (weights) for ESRGAN (Resolution Upscaling) x4plus_anime. Sorry about that, please try to:\n  1. Run this installer again.\n  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md\n  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB\n  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\nThanks!\n\n"
        read -p "Press any key to continue"
        exit
    fi
fi


if [ `grep -c sd_install_complete ../scripts/install_status.txt` -gt "0" ]; then
    echo sd_weights_downloaded >> ../scripts/install_status.txt
    echo sd_install_complete >> ../scripts/install_status.txt
fi

printf "\n\nStable Diffusion is ready!\n\n"

cd ..
export SD_UI_PATH=`pwd`/ui
cd stable-diffusion

uvicorn server:app --app-dir "$SD_UI_PATH" --port 9000 --host 0.0.0.0

read -p "Press any key to continue"
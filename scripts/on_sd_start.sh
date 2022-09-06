source installer/etc/profile.d/conda.sh

if [ `grep -c sd_git_cloned scripts/install_status.txt` -gt "0" ]; then
    echo "Stable Diffusion's git repository was already installed. Updating.."

    cd stable-diffusion

    git reset --hard
    git pull

    cd ..
else
    printf "\n\nDownloading Stable Diffusion..\n\n"

    if git clone https://github.com/basujindal/stable-diffusion.git ; then
        echo sd_git_cloned >> scripts/install_status.txt
    else
        printf "\n\nError downloading Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
        read -p "Press any key to continue"
        exit
    fi
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
        printf "\n\nError installing the packages necessary for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
        read -p "Press any key to continue"
        exit
    fi

    conda activate ./env

    out_test=`python -c "import torch; import ldm; import transformers; import numpy; print(42)"`
    if [ "$out_test" != "42" ]; then
        printf "\n\nDependency test failed! Error installing the packages necessary for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
        read -p "Press any key to continue"
        exit
    fi

    echo conda_sd_env_created >> ../scripts/install_status.txt
fi

if [ `grep -c conda_sd_ui_deps_installed ../scripts/install_status.txt` -gt "0" ]; then
    echo "Packages necessary for Stable Diffusion UI were already installed"
else
    printf "\n\nDownloading packages necessary for Stable Diffusion UI..\n\n"

    if conda install -c conda-forge --prefix ./env -y uvicorn fastapi ; then
        echo "Installed. Testing.."
    else
        printf "\n\nError installing the packages necessary for Stable Diffusion UI. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
        read -p "Press any key to continue"
        exit
    fi

    if ! command -v uvicorn &> /dev/null; then
        printf "\n\nUI packages not found! Error installing the packages necessary for Stable Diffusion UI. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
        read -p "Press any key to continue"
        exit
    fi

    echo conda_sd_ui_deps_installed >> ../scripts/install_status.txt
fi

if [ -f "sd-v1-4.ckpt" ]; then
    model_size=`ls -l sd-v1-4.ckpt | awk '{print $5}'`

    if [ "$model_size" -gt "40000000000" ]; then
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
        if [ "$model_size" -lt "40000000000" ]; then
            printf "\n\nError: The downloaded model file was invalid! Bytes downloaded: $model_size\n\n"
            printf "\n\nError downloading the data files (weights) for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
            read -p "Press any key to continue"
            exit
        fi
    else
        printf "\n\nError downloading the data files (weights) for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
        read -p "Press any key to continue"
        exit
    fi

    echo sd_weights_downloaded >> ../scripts/install_status.txt
    echo sd_install_complete >> ../scripts/install_status.txt
fi

printf "\n\nStable Diffusion is ready!\n\n"

cd ..
export SD_UI_PATH=`pwd`/ui
cd stable-diffusion

uvicorn server:app --app-dir "$SD_UI_PATH" --port 9000 --host 0.0.0.0

read -p "Press any key to continue"
if [ `grep -c sd_git_cloned scripts/install_status.txt` -gt "0" ]; then
    echo "Stable Diffusion's git repository was already installed. Updating.."

    cd stable-diffusion

    git reset --hard
    git pull

    cd ..
else
    echo "\nDownloading Stable Diffusion..\n"

    if git clone https://github.com/basujindal/stable-diffusion.git ; then
        echo sd_git_cloned >> scripts/install_status.txt
    else
        echo "\nError downloading Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n"
        read -p "Press any key to continue"
        exit
    fi
fi

cd stable-diffusion

if [ `grep -c conda_sd_env_created ../scripts/install_status.txt` -gt "0" ]; then
    echo "Packages necessary for Stable Diffusion were already installed"
else
    echo "Downloading packages necessary for Stable Diffusion.."
    echo "***** This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient ***** .."

    rm -rf ./env

    if conda env create --prefix env -f environment.yaml ; then
        echo conda_sd_env_created >> ../scripts/install_status.txt
    else
        echo "\nError installing the packages necessary for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n"
        read -p "Press any key to continue"
        exit
    fi
fi

conda activate ./env

if [ `grep -c conda_sd_ui_deps_installed ../scripts/install_status.txt` -gt "0" ]; then
    echo "Packages necessary for Stable Diffusion UI were already installed"
else
    echo "\nDownloading packages necessary for Stable Diffusion UI..\n"

    if conda install -c conda-forge -y --prefix env uvicorn fastapi ; then
        echo conda_sd_ui_deps_installed >> ../scripts/install_status.txt
    else
        echo "\nError installing the packages necessary for Stable Diffusion UI. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n"
        read -p "Press any key to continue"
        exit
    fi
fi

if [ -f "sd-v1-4.ckpt" ]; then
    echo "Data files (weights) necessary for Stable Diffusion were already downloaded"
else
    echo "Downloading data files (weights) for Stable Diffusion.."

    curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt

    if [ ! -f "sd-v1-4.ckpt" ]; then
        echo "\nError downloading the data files (weights) for Stable Diffusion. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n"
        read -p "Press any key to continue"
        exit
    fi

    echo sd_weights_downloaded >> ../scripts/install_status.txt
    echo sd_install_complete >> ../scripts/install_status.txt
fi

echo "\nStable Diffusion is ready!\n"

export SD_UI_PATH=`pwd`/../ui

uvicorn server:app --app-dir "%SD_UI_PATH%" --port 9000 --host 0.0.0.0

read -p "Press any key to continue"
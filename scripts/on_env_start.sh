printf "\n\nStable Diffusion UI\n\n"

if [ -f "scripts/install_status.txt" ] && [ `grep -c sd_ui_git_cloned scripts/install_status.txt` -gt "0" ]; then
    echo "Stable Diffusion UI's git repository was already installed. Updating.."

    cd sd-ui-files

    git reset --hard
    git pull

    cd ..
else
    printf "\n\nDownloading Stable Diffusion UI..\n\n"

    if git clone https://github.com/cmdr2/stable-diffusion-ui.git sd-ui-files ; then
        echo sd_ui_git_cloned >> scripts/install_status.txt
    else
        printf "\n\nError downloading Stable Diffusion UI. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues\n\n"
        read -p "Press any key to continue"
        exit
    fi
fi

cp -Rf sd-ui-files/ui ui
cp -Rf sd-ui-files/scripts/* scripts/
cp "scripts/start.sh" .

./scripts/on_sd_start.sh

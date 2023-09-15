#!/bin/bash

source ./scripts/functions.sh

printf "\n\nEasy Diffusion - v3\n\n"

export PYTHONNOUSERSITE=y

if [ -f "scripts/config.sh" ]; then
    source scripts/config.sh
fi

if [ -f "scripts/user_config.sh" ]; then
    source scripts/user_config.sh
fi

export PYTHONPATH=$(pwd)/installer_files/env/lib/python3.8/site-packages:$(pwd)/stable-diffusion/env/lib/python3.8/site-packages

if [ -f "scripts/get_config.py" ]; then
   export update_branch="$( python scripts/get_config.py --default=main update_branch )"
fi

if [ "$update_branch" == "" ]; then
    export update_branch="main"
fi

if [ -f "scripts/install_status.txt" ] && [ `grep -c sd_ui_git_cloned scripts/install_status.txt` -gt "0" ]; then
    echo "Easy Diffusion's git repository was already installed. Updating from $update_branch.."

    cd sd-ui-files

    git add -A .
    git stash
    git reset --hard
    git -c advice.detachedHead=false checkout "$update_branch"
    git pull

    cd ..
else
    printf "\n\nDownloading Easy Diffusion..\n\n"
    printf "Using the $update_branch channel\n\n"

    if git clone -b "$update_branch" https://github.com/easydiffusion/easydiffusion.git sd-ui-files ; then
        echo sd_ui_git_cloned >> scripts/install_status.txt
    else
        fail "git clone failed"
    fi
fi

rm -rf ui
cp -Rf sd-ui-files/ui .
cp sd-ui-files/scripts/on_sd_start.sh scripts/
cp sd-ui-files/scripts/bootstrap.sh scripts/
cp sd-ui-files/scripts/check_modules.py scripts/
cp sd-ui-files/scripts/get_config.py scripts/
cp sd-ui-files/scripts/config.yaml.sample scripts/
cp sd-ui-files/scripts/start.sh .
cp sd-ui-files/scripts/developer_console.sh .
cp sd-ui-files/scripts/functions.sh scripts/

exec ./scripts/on_sd_start.sh

#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -f "on_sd_start.bat" ]; then
    echo ================================================================================
    echo
    echo !!!! WARNING !!!!
    echo
    echo It looks like you\'re trying to run the installation script from a source code
    echo download. This will not work.
    echo
    echo Recommended: Please close this window and download the installer from
    echo https://stable-diffusion-ui.github.io/docs/installation/
    echo
    echo ================================================================================
    echo
    read
    exit 1
fi


# set legacy installer's PATH, if it exists
if [ -e "installer" ]; then export PATH="$(pwd)/installer/bin:$PATH"; fi

# Setup the packages required for the installer
scripts/bootstrap.sh || exit 1

# set new installer's PATH, if it downloaded any packages
if [ -e "installer_files/env" ]; then export PATH="$(pwd)/installer_files/env/bin:$PATH"; fi

# Test the bootstrap
which git
git --version || exit 1

which conda
conda --version || exit 1

# Download the rest of the installer and UI
chmod +x scripts/*.sh
scripts/on_env_start.sh

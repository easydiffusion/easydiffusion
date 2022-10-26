#!/bin/bash

# set legacy installer's PATH, if it exists
if [ -e "installer" ]; then export PATH="$(pwd)/installer/bin:$PATH"; fi

# Setup the packages required for the installer
scripts/bootstrap.sh

# set new installer's PATH, if it downloaded any packages
if [ -e "installer_files/env" ]; then export PATH="$(pwd)/installer_files/env/bin:$PATH"; fi

# Test the bootstrap
which git
git --version

which conda
conda --version

# Download the rest of the installer and UI
scripts/on_env_start.sh

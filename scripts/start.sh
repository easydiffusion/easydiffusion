#!/bin/bash

# set legacy installer's PATH, if it exists
if [ -e "installer" ]; then export PATH="$(pwd)/installer/bin:$PATH"; fi

# Setup the packages required for the installer
scripts/bootstrap.sh

# set new installer's PATH, if it downloaded any packages
if [ -e "installer_files/env" ]; then export PATH="$(pwd)/installer_files/env/bin:$PATH"; fi

# activate the installer env
CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # avoids the 'shell not initialized' error

conda activate

# Test the bootstrap
which git
git --version

which python
python --version

which conda
conda --version

# Download the rest of the installer and UI
scripts/on_env_start.sh

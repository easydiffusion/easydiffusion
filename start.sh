#!/bin/bash

echo "Stable Diffusion UI - v2.5"
echo ""

export START_CMD_FILENAME="start.sh"
export SD_BASE_DIR=$(pwd)

echo "Working in $SD_BASE_DIR"

# Setup the packages required for the installer
installer/bootstrap/bootstrap.sh

# Test the bootstrap
git --version
python --version

# Download the rest of the installer and UI
installer/installer/start.sh

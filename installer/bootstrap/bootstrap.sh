#!/bin/bash

# This file initializes micromamba and activates the env.
# A similar bootstrap file needs to exist for each platform (win, linux, macOS)
# Ready to hand-over to the platform-independent installer after this (written in python).

OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Linux*)     OS_NAME="linux";;
    Darwin*)    OS_NAME="mac";;
    *)          echo "Unknown OS: $OS_NAME! This only runs on Linux or Mac" && exit
esac

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x64";;
    arm64*)     OS_ARCH="arm64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This only runs on x86_64 or arm64" && exit
esac

export MAMBA_ROOT_PREFIX=$SD_BASE_DIR/env/mamba
INSTALL_ENV_DIR=$SD_BASE_DIR/env/installer_env
INSTALLER_YAML_FILE=$SD_BASE_DIR/installer/yaml/installer-environment.yaml
MICROMAMBA_BINARY_FILE=$SD_BASE_DIR/installer/bin/micromamba_${OS_NAME}_${OS_ARCH}

# initialize the mamba dir
mkdir -p "$MAMBA_ROOT_PREFIX"

cp "$MICROMAMBA_BINARY_FILE" "$MAMBA_ROOT_PREFIX/micromamba"

# test the mamba binary
echo "Micromamba version:"
"$MAMBA_ROOT_PREFIX/micromamba" --version

# run the shell hook
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"

# create the installer env
if [ ! -e "$INSTALL_ENV_DIR" ]; then
    micromamba create -y --prefix "$INSTALL_ENV_DIR" -f "$INSTALLER_YAML_FILE"
fi

# activate
micromamba activate "$INSTALL_ENV_DIR"

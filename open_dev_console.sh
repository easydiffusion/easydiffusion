#!/bin/bash

export SD_BASE_DIR=`pwd`
export MAMBA_ROOT_PREFIX="$SD_BASE_DIR/env/mamba"
export INSTALL_ENV_DIR="$SD_BASE_DIR/env/installer_env"

eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"

micromamba activate "$INSTALL_ENV_DIR"

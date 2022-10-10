#!/bin/bash

if [ "$0" == "bash" ]; then
  echo "Opening Stable Diffusion UI - Developer Console.."
  echo ""

  export SD_BASE_DIR=`pwd`
  export MAMBA_ROOT_PREFIX="$SD_BASE_DIR/env/mamba"
  export INSTALL_ENV_DIR="$SD_BASE_DIR/env/installer_env"
  export PROJECT_ENV_DIR="$SD_BASE_DIR/env/project_env"

  eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"

  micromamba activate "$INSTALL_ENV_DIR"
  micromamba activate "$PROJECT_ENV_DIR"
else
  bash --init-file developer_console.sh
fi
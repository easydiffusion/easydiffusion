#!/bin/bash

if [ "$0" == "bash" ]; then
  echo "Opening Stable Diffusion UI - Developer Console.."
  echo ""

  source installer/bin/activate

  conda-unpack

  conda --version
  git --version

  conda activate ./stable-diffusion/env
else
  bash --init-file open_dev_console.sh
fi
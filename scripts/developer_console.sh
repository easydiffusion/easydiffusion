#!/bin/bash

if [ "$0" == "bash" ]; then
  echo "Opening Stable Diffusion UI - Developer Console.."
  echo ""

  # set legacy and new installer's PATH, if they exist
  if [ -e "installer" ]; then export PATH="$(pwd)/installer/bin:$PATH"; fi
  if [ -e "installer_files/env" ]; then export PATH="$(pwd)/installer_files/env/bin:$PATH"; fi

  # test the environment
  echo "Environment Info:"
  which git
  git --version

  which python
  python --version

  which conda
  conda --version

  # activate the environment
  CONDA_BASEPATH=$(conda info --base)
  source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

  conda activate ./stable-diffusion/env
else
  bash --init-file developer_console.sh
fi
#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$0" == "bash" ]; then
  echo "Opening Stable Diffusion UI - Developer Console.."
  echo ""

  # set legacy and new installer's PATH, if they exist
  if [ -e "installer" ]; then export PATH="$(pwd)/installer/bin:$PATH"; fi
  if [ -e "installer_files/env" ]; then export PATH="$(pwd)/installer_files/env/bin:$PATH"; fi

  # activate the installer env
  CONDA_BASEPATH=$(conda info --base)
  source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # avoids the 'shell not initialized' error

  conda activate

  # test the environment
  echo "Environment Info:"
  which git
  git --version

  which conda
  conda --version

  echo ""

  # activate the environment
  CONDA_BASEPATH=$(conda info --base)
  source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

  conda activate ./stable-diffusion/env

  which python
  python --version

  echo ""
else
  file_name=$(basename "${BASH_SOURCE[0]}")
  bash --init-file "$file_name"
fi

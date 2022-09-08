#!/bin/bash

conda-unpack

source $CONDA_PREFIX/etc/profile.d/conda.sh

conda --version
git --version

cd $CONDA_PREFIX/../scripts

./on_env_start.sh
#!/bin/bash

source installer/bin/activate

conda-unpack

conda --version
git --version

scripts/on_env_start.sh

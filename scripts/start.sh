#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

source installer/bin/activate

conda-unpack

scripts/on_env_start.sh

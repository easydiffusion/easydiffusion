#!/bin/bash

source installer/bin/activate

conda-unpack

scripts/on_env_start.sh

#!/bin/bash

IMAGE_NAME=$1

if [ -z "$IMAGE_NAME" ]; then
  IMAGE_NAME="r8.im/stability-ai/stable-diffusion@sha256:06eb78b36068500c616a7f33c15e6fa40404f8e14b5bfad57ebe0c7fe0f6bdf1"
fi

echo "Using docker image $IMAGE_NAME"
docker run --name sd -d -p 5000:5000 --gpus all $IMAGE_NAME

uvicorn main:app --reload
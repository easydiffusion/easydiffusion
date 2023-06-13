FROM nvidia/cuda:latest
CMD nvidia-smi

# Use a multi-stage build
FROM debian:bullseye AS build

# install dependencies
RUN apt-get update
RUN apt-get install -y git bzip2 curl ffmpeg libsm6 libxext6

# add a non-root user and switch to this user
RUN useradd -ms /bin/bash stablediffuser
USER stablediffuser
WORKDIR /home/stablediffuser

# download the repo
RUN git clone https://github.com/anaximander2048/stable-diffusion-ui

# change DIR
WORKDIR /home/stablediffuser/stable-diffusion-ui

# comment the cd inside start.sh for not having issues executing bootstrap.sh
RUN sed -i 's/cd/\#cd/g' scripts/start.sh

# Specify the default command to run when a container is started from this image
CMD [ "bash", "scripts/start.sh" ]

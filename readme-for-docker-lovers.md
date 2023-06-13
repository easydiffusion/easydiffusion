# Hi awesome and cute Docker Lovers!
## This is how to play with the Docker version I made. Just copy-paste the following


```console
# Create the dir
mkdir stable-diffusion-docker

# Download the dockerfile
curl \
  -o stable-diffusion-docker/dockerfile \
  https://raw.githubusercontent.com/anaximander2048/stable-diffusion-ui/main/dockerfile

# Build the docker image
docker build stable-diffusion-docker -t stabledif:0.1

# Run the docker container
# remove --gpus=all if you are not intend to use GPU rendering
docker run -dp 9000:9000 --gpus=all stabledif:0.1
```

> **_NOTE:_** After the first run, you will have to wait some time (in my system is approximately 40')

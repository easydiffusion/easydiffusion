@echo "Check and install if necessary"

@if not exist "stable-diffusion\" (
    @echo "Downloading Stable Diffusion.."
    @call git clone https://github.com/CompVis/stable-diffusion.git
)

@cd stable-diffusion

@if not exist "env\" (
    @echo "Downloading packages necessary for Stable Diffusion.."

    @call conda env create --prefix env -f environment.yaml
    @call conda activate .\env

    @echo "Downloading packages necessary for Stable Diffusion UI.."

    @call conda install -c conda-forge -y --prefix env uvicorn fastapi
) else (
    @call conda activate .\env
)

@if not exist "sd-v1-4.ckpt" (
    @echo "Downloading data files (weights) for Stable Diffusion.."
    @call curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt
)

@rem "Start up the server"

@echo "Setting up and testing Stable Diffusion.."

python scripts\txt2img.py --prompt "photo of an astronaut riding a motorcycle" --W 256 --H 256 --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 1

@if exist "outputs\txt2img-samples\samples" (
    @echo "Ready to rock!"

    @cd ..\ui

    @uvicorn server:app --port 9000
) else (
    @echo "There was an error while running Stable Diffusion. Please check the troubleshooting guide () for common issues. If that doesn't work, please feel free to file an issue at: "
)
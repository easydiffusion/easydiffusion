# this script runs inside the legacy "stable-diffusion" folder

from sdkit.models import download_model, get_model_info_from_db
from sdkit.utils import hash_file_quick

import os
import shutil
from glob import glob

models_base_dir = os.path.abspath(os.path.join("..", "models"))

models_to_check = {
    "stable-diffusion": [
        {"file_name": "sd-v1-4.ckpt", "model_id": "1.4"},
    ],
    "gfpgan": [
        {"file_name": "GFPGANv1.4.pth", "model_id": "1.4"},
    ],
    "realesrgan": [
        {"file_name": "RealESRGAN_x4plus.pth", "model_id": "x4plus"},
        {"file_name": "RealESRGAN_x4plus_anime_6B.pth", "model_id": "x4plus_anime_6"},
    ],
    "vae": [
        {"file_name": "vae-ft-mse-840000-ema-pruned.ckpt", "model_id": "vae-ft-mse-840000-ema-pruned"},
    ],
}
MODEL_EXTENSIONS = {  # copied from easydiffusion/model_manager.py
    "stable-diffusion": [".ckpt", ".safetensors"],
    "vae": [".vae.pt", ".ckpt", ".safetensors"],
    "hypernetwork": [".pt", ".safetensors"],
    "gfpgan": [".pth"],
    "realesrgan": [".pth"],
    "lora": [".ckpt", ".safetensors"],
}


def download_if_necessary(model_type: str, file_name: str, model_id: str):
    model_path = os.path.join(models_base_dir, model_type, file_name)
    expected_hash = get_model_info_from_db(model_type=model_type, model_id=model_id)["quick_hash"]

    other_models_exist = any_model_exists(model_type)
    known_model_exists = os.path.exists(model_path)
    known_model_is_corrupt = known_model_exists and hash_file_quick(model_path) != expected_hash

    if known_model_is_corrupt or (not other_models_exist and not known_model_exists):
        print("> download", model_type, model_id)
        download_model(model_type, model_id, download_base_dir=models_base_dir)


def init():
    migrate_legacy_model_location()

    for model_type, models in models_to_check.items():
        for model in models:
            try:
                download_if_necessary(model_type, model["file_name"], model["model_id"])
            except:
                fail(model_type)

        print(model_type, "model(s) found.")


### utilities
def any_model_exists(model_type: str) -> bool:
    extensions = MODEL_EXTENSIONS.get(model_type, [])
    for ext in extensions:
        if any(glob(f"{models_base_dir}/{model_type}/**/*{ext}", recursive=True)):
            return True

    return False


def migrate_legacy_model_location():
    'Move the models inside the legacy "stable-diffusion" folder, to their respective folders'

    for model_type, models in models_to_check.items():
        for model in models:
            file_name = model["file_name"]
            if os.path.exists(file_name):
                dest_dir = os.path.join(models_base_dir, model_type)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(file_name, os.path.join(dest_dir, file_name))


def fail(model_name):
    print(
        f"""Error downloading the {model_name} model. Sorry about that, please try to:
1. Run this installer again.
2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting
3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues
Thanks!"""
    )
    exit(1)


### start

init()

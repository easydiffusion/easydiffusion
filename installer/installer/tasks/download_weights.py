import os

from installer import app, helpers

def run():
    fetch_model('Stable Diffusion', 'sd-v1-4.ckpt', model_dir_path=app.stable_diffusion_models_dir_path, download_url='https://me.cmdr2.org/stable-diffusion-ui/sd-v1-4.ckpt', expected_file_sizes=[4265380512, 7703807346, 7703810927])
    fetch_model('Face Correction (GFPGAN)', 'GFPGANv1.4.pth', model_dir_path=app.gfpgan_models_dir_path, download_url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', expected_file_sizes=[348632874])
    fetch_model('Resolution Upscale (RealESRGAN x4)', 'RealESRGAN_x4plus.pth', model_dir_path=app.realesrgan_models_dir_path, download_url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', expected_file_sizes=[67040989])
    fetch_model('Resolution Upscale (RealESRGAN x4_anime)', 'RealESRGAN_x4plus_anime_6B.pth', model_dir_path=app.realesrgan_models_dir_path, download_url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', expected_file_sizes=[17938799])

def fetch_model(model_type, file_name, model_dir_path, download_url, expected_file_sizes):
    os.makedirs(model_dir_path, exist_ok=True)

    file_path = os.path.join(model_dir_path, file_name)

    if model_exists(file_name, file_path, expected_file_sizes):
        helpers.log(f'Data files (weights) necessary for {model_type} were already downloaded')
        return

    helpers.log(f'Downloading data files (weights) for {model_type}..')

    helpers.run(f'curl -L -k "{download_url}" > "{file_path}"', log_the_cmd=True)

def model_exists(file_name, file_path, expected_file_sizes):
    legacy_file_path = os.path.join(app.stable_diffusion_repo_dir_path, file_name)

    file_exists = os.path.exists(file_path)
    legacy_file_exists = os.path.exists(legacy_file_path)

    if legacy_file_exists:
        file_size = os.path.getsize(legacy_file_path)
        if file_size in expected_file_sizes:
            return True

        helpers.log(f'{file_name} is invalid. Was only {file_size} bytes in size. Downloading again..')
        os.remove(legacy_file_path)

    if file_exists:
        file_size = os.path.getsize(file_path)
        if file_size in expected_file_sizes:
            return True

        helpers.log(f'{file_name} is invalid. Was only {file_size} bytes in size. Downloading again..')
        os.remove(file_path)

    return False

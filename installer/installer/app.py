import os
import json
import platform

# config
PROJECT_REPO_URL = 'https://github.com/cmdr2/stable-diffusion-ui.git'
DEFAULT_PROJECT_BRANCH = 'installer_new'
PROJECT_REPO_DIR_NAME = 'project_repo'

STABLE_DIFFUSION_REPO_URL = 'https://github.com/basujindal/stable-diffusion.git'
DEFAULT_STABLE_DIFFUSION_COMMIT = 'f6cfebffa752ee11a7b07497b8529d5971de916c'
STABLE_DIFFUSION_REPO_DIR_NAME = 'stable-diffusion'

PROJECT_ENV_DIR_NAME = 'project_env'

START_CMD_FILE_NAME = "Start Stable Diffusion UI.cmd" if platform.system() == "Windows" else "start.sh"
DEV_CONSOLE_CMD_FILE_NAME = "Developer Console.cmd" if platform.system() == "Windows" else "developer_console.sh"
CONFIG_FILE_NAME = 'config.json'

# top-level folders
ENV_DIR_NAME = 'env'
MODELS_DIR_NAME = 'models'

INSTALLER_DIR_NAME = 'installer'
UI_DIR_NAME = 'ui'
ENGINE_DIR_NAME = 'engine'


# env
SD_BASE_DIR = os.environ['SD_BASE_DIR']


# model folders
STABLE_DIFFUSION_MODELS_DIR_NAME = "stable-diffusion"
GFPGAN_MODELS_DIR_NAME = "gfpgan"
RealESRGAN_MODELS_DIR_NAME = "realesrgan"

# create references to dirs
env_dir_path = os.path.join(SD_BASE_DIR, ENV_DIR_NAME)

installer_dir_path = os.path.join(SD_BASE_DIR, INSTALLER_DIR_NAME)
ui_dir_path = os.path.join(SD_BASE_DIR, UI_DIR_NAME)
engine_dir_path = os.path.join(SD_BASE_DIR, ENGINE_DIR_NAME)

project_repo_dir_path = os.path.join(env_dir_path, PROJECT_REPO_DIR_NAME)
stable_diffusion_repo_dir_path = os.path.join(env_dir_path, STABLE_DIFFUSION_REPO_DIR_NAME)

project_env_dir_path = os.path.join(env_dir_path, PROJECT_ENV_DIR_NAME)

patches_dir_path = os.path.join(installer_dir_path, 'patches')

models_dir_path = os.path.join(SD_BASE_DIR, MODELS_DIR_NAME)
stable_diffusion_models_dir_path = os.path.join(models_dir_path, STABLE_DIFFUSION_MODELS_DIR_NAME)
gfpgan_models_dir_path = os.path.join(models_dir_path, GFPGAN_MODELS_DIR_NAME)
realesrgan_models_dir_path = os.path.join(models_dir_path, RealESRGAN_MODELS_DIR_NAME)


# useful functions
def get_config():
    config_path = os.path.join(SD_BASE_DIR, CONFIG_FILE_NAME)
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r") as f:
        return json.load(f)


# app context
config = get_config()
activated_env_dir_path = None

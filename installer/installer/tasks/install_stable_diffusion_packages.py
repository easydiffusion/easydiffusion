import os
import platform

from installer import app, helpers

def run():
    if is_valid_env():
        helpers.log("Packages necessary for Stable Diffusion were already installed")
        return

    log_installing_header()

    environment_file_path = get_environment_file_path()

    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'

    if not os.path.exists(app.project_env_dir_path):
        helpers.run(f'micromamba create --prefix {app.project_env_dir_path}', log_the_cmd=True)

    helpers.run(f'micromamba install -y --prefix {app.project_env_dir_path} -f {environment_file_path}', env=env, log_the_cmd=True)

    if is_valid_env():
        helpers.log("Installed the packages necessary for Stable Diffusion")

        app.activated_env_dir_path = app.project_env_dir_path # so that future `run()` invocations will run in the activated env
    else:
        helpers.fail_with_install_error(error_msg="Could not install the packages necessary for Stable Diffusion")

    apply_patches()

def apply_patches():
    gfpgan_repo_dir_path = os.path.join(app.stable_diffusion_repo_dir_path, 'src', 'gfpgan')
    helpers.apply_git_patches(gfpgan_repo_dir_path, patch_file_names=(
        "gfpgan_custom.patch",
    ))

def get_environment_file_path():
    environment_file_name = 'sd-environment-win-linux-nvidia.yaml'
    if platform.system() == 'Darwin':
        environment_file_name = 'sd-environment-mac-nvidia.yaml'

    return os.path.join(app.installer_dir_path, 'yaml', environment_file_name)

def log_installing_header():
    helpers.log('''

Downloading packages necessary for Stable Diffusion..

***** !! This will take some time (depending on the speed of the Internet connection) and may appear to be stuck, but please be patient *****

''')

def is_valid_env():
    return helpers.modules_exist_in_env(('torch', 'ldm', 'antlr4', 'transformers', 'numpy', 'gfpgan', 'realesrgan', 'basicsr'))

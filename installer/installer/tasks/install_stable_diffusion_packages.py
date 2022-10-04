import os
import shutil
import platform

from installer import app, helpers

def run():
    if is_valid_env():
        helpers.log("Packages necessary for Stable Diffusion were already installed")
        return

    log_installing_header()

    shutil.rmtree(app.project_env_dir_path, ignore_errors=True)

    environment_file_path = get_environment_file_path()

    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'

    if helpers.run(f'micromamba create --prefix {app.project_env_dir_path} -f {environment_file_path}', env=env) \
        and is_valid_env():

        helpers.log("Installed the packages necessary for Stable Diffusion")
    else:
        helpers.fail_with_install_error(error_msg="Could not install the packages necessary for Stable Diffusion")

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
    return helpers.modules_exist_in_env(('torch', 'ldm', 'antlr4', 'transformers', 'numpy'))

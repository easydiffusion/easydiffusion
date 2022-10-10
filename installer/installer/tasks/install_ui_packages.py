import os
import shutil
import platform

from installer import app, helpers

def run():
    if is_valid_env():
        helpers.log("Packages necessary for Stable Diffusion UI were already installed")
        return

    log_installing_header()

    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'

    helpers.run(f'micromamba install -y --prefix {app.project_env_dir_path} -c conda-forge uvicorn fastapi', env=env, log_the_cmd=True)

    if is_valid_env():
        helpers.log("Installed the packages necessary for Stable Diffusion UI")
    else:
        helpers.fail_with_install_error(error_msg="Could not install the packages necessary for Stable Diffusion UI")

def log_installing_header():
    helpers.log('''

Downloading packages necessary for Stable Diffusion UI..

''')

def is_valid_env():
    path = os.environ['PATH']
    path += ';' + os.path.join(app.project_env_dir_path, 'Scripts' if platform.system() == 'Windows' else 'bin')

    if shutil.which("uvicorn", path=path) is None:
        helpers.log("uvicorn not found!")
        return False

    return helpers.modules_exist_in_env(('uvicorn', 'fastapi'))

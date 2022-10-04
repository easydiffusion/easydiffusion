import argparse
import subprocess
import sys
import json
import os
import platform
import shutil

config_path = os.path.join('config.json')

if not os.path.exists('LICENSE'):
    print('Error: This script needs to be run from the root of the stable-diffusion-ui folder! Please cd to the correct folder, and run this again.')
    exit(1)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--symlink_dir", type=str, default=None, help="the absolute path to the project git repository (to link to)"
)
opt = parser.parse_args()

def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    for c in iter(lambda: p.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
        sys.stdout.flush()

    p.wait()

    return p.returncode == 0

def get_config():
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r") as f:
        return json.load(f)

def save_config(config):
    with open(config_path, "w") as f:
        json.dump(config, f)

# set the `is_developer_mode` flag to `true` in the config
config = get_config()
config['is_developer_mode'] = True
save_config(config)

print('set is_developer_mode=true in config.json')

# make the symlink, if requested
if opt.symlink_dir is not None:
    if not os.path.exists(opt.symlink_dir):
        print(f'Symlink directory "{opt.symlink_dir}" was not found! Are you sure it has been escaped correctly?')
        exit(1)

    installer_target_path = os.path.join(opt.symlink_dir, 'installer')
    ui_target_path = os.path.join(opt.symlink_dir, 'ui')
    engine_target_path = os.path.join(opt.symlink_dir, 'engine')

    shutil.rmtree('installer', ignore_errors=True)
    shutil.rmtree('ui', ignore_errors=True)
    shutil.rmtree('engine', ignore_errors=True)

    if not os.path.exists(ui_target_path) or not os.path.exists(installer_target_path) or not os.path.exists(engine_target_path):
        print('The target symlink directory does not contain the required {ui, installer, engine} folders. Are you sure it is the correct git repo for the project?')
        exit(1)

    if platform.system() == 'Windows':
        run(f'mklink /J "installer" "{installer_target_path}"')
        run(f'mklink /J "ui" "{ui_target_path}"')
        run(f'mklink /J "engine" "{engine_target_path}"')
    elif platform.system() in ('Linux', 'Darwin'):
        run(f'ln -s "{installer_target_path}" "installer"')
        run(f'ln -s "{ui_target_path}" "ui"')
        run(f'ln -s "{engine_target_path}" "engine"')

    print(f'Created symlinks! Your installation will now automatically use the files present in the repository at {opt.symlink_dir}')

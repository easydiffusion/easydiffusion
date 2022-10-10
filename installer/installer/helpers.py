from os import path
import subprocess

from installer import app

def run(cmd, run_in_folder=None, env=None, get_output=False, log_the_cmd=False):
    if app.activated_env_dir_path is not None and 'micromamba activate' not in cmd:
        cmd = f'micromamba activate "{app.activated_env_dir_path}" && {cmd}'

    if run_in_folder is not None:
        cmd = f'cd "{run_in_folder}" && {cmd}'

    if log_the_cmd:
        log('running: ' + cmd)

    if get_output:
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        p = subprocess.Popen(cmd, shell=True, env=env)

    out, err = p.communicate()

    if get_output:
        return out, err

def log(msg):
    print(msg)

def modules_exist_in_env(modules, env_dir_path=app.project_env_dir_path):
    if not path.exists(env_dir_path):
        return False

    check_modules_script_path = path.join(app.installer_dir_path, 'installer', 'check_modules.py')
    module_args = ' '.join(modules)
    check_modules_cmd = f'python "{check_modules_script_path}" {module_args}'

    if app.activated_env_dir_path != env_dir_path:
        activate_cmd = f'micromamba activate "{env_dir_path}"'
        check_modules_cmd = f'{activate_cmd} && {check_modules_cmd}'

    # activate and run the modules checker
    output, _ = run(check_modules_cmd, get_output=True)
    if 'Missing' in output:
        return False

    return True

def fail_with_install_error(error_msg):
    try:
        log(f'''

Error: {error_msg}. Sorry about that, please try to:
  1. Run this installer again.
  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md
  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues
Thanks!''')
    except:
        pass

    exit(1)

def apply_git_patches(repo_dir_path, patch_file_names):
    is_developer_mode = app.config.get('is_developer_mode', False)
    if is_developer_mode:
        return

    for patch_file_name in patch_file_names:
        patch_file_path = path.join(app.patches_dir_path, patch_file_name)
        run(f"git apply {patch_file_path}", run_in_folder=repo_dir_path)

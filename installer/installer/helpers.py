from os import path
import subprocess
import sys
import shutil
import time

from installer import app

def run(cmd, run_in_folder=None, get_output=False, write_to_log=True, env=None):
    if run_in_folder is not None:
        cmd = f'cd "{run_in_folder}" && {cmd}'

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, env=env)

    buf = bytearray()

    for c in iter(lambda: p.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
        sys.stdout.flush()

        buf.extend(c)

        if write_to_log and app.log_file is not None:
            app.log_file.write(c)
            app.log_file.flush()

    p.wait()

    if get_output:
        return p.returncode, buf.decode('utf-8')

    return p.returncode == 0

def log(msg):
    print(msg)

    app.log_file.write(bytes(msg + "\n", 'utf-8'))
    app.log_file.flush()

def modules_exist_in_env(modules, env_dir_path=app.project_env_dir_path):
    if not path.exists(env_dir_path):
        return False

    activate_cmd = f'micromamba activate "{env_dir_path}"'

    if not run(activate_cmd, write_to_log=False):
        return False

    check_modules_script_path = path.join(app.installer_dir_path, 'installer', 'check_modules.py')
    module_args = ' '.join(modules)
    check_modules_cmd = f'{activate_cmd} && python "{check_modules_script_path}" {module_args}'

    ret_code, output = run(check_modules_cmd, get_output=True, write_to_log=False)
    if ret_code != 0 or 'Missing' in output:
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

        ts = int(time.time())
        shutil.copy(app.LOG_FILE_NAME, f'error-{ts}.log')
    except:
        pass

    exit(1)

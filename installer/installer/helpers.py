import subprocess
import sys

from installer import app

def run(cmd, run_in_folder=None):
    if run_in_folder is not None:
        cmd = f'cd "{run_in_folder}" && {cmd}'

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    for c in iter(lambda: p.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
        sys.stdout.flush()

        if app.log_file is not None:
            app.log_file.write(c)
            app.log_file.flush()

    p.wait()

    return p.returncode == 0

def log(msg):
    print(msg)

    app.log_file.write(bytes(msg + "\n", 'utf-8'))
    app.log_file.flush()

def show_install_error(error_msg):
    log(f'''

Error: {error_msg}. Sorry about that, please try to:
  1. Run this installer again.
  2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md
  3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
  4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues
Thanks!''')

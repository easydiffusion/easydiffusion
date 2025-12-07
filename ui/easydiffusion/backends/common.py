import subprocess
import threading
import psutil


def read_output(pipe, prefix=""):
    while True:
        output = pipe.readline()
        if output:
            print(f"{prefix}{output.decode('utf-8')}", end="")
        else:
            break  # Pipe is closed, subprocess has likely exited


def run(cmds: list, cwd=None, env=None, stream_output=True, wait=True, output_prefix=""):
    p = subprocess.Popen(cmds, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if stream_output:
        output_thread = threading.Thread(target=read_output, args=(p.stdout, output_prefix))
        output_thread.start()

    if wait:
        p.wait()

    return p


# https://stackoverflow.com/a/25134985
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

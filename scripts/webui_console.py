import os
import platform
import subprocess


def configure_env(dir):
    env_entries = {
        "PATH": [
            f"{dir}",
            f"{dir}/bin",
            f"{dir}/Library/bin",
            f"{dir}/Scripts",
            f"{dir}/usr/bin",
        ],
        "PYTHONPATH": [
            f"{dir}",
            f"{dir}/lib/site-packages",
            f"{dir}/lib/python3.10/site-packages",
        ],
        "PYTHONHOME": [],
        "PY_LIBS": [
            f"{dir}/Scripts/Lib",
            f"{dir}/Scripts/Lib/site-packages",
            f"{dir}/lib",
            f"{dir}/lib/python3.10/site-packages",
        ],
        "PY_PIP": [f"{dir}/Scripts", f"{dir}/bin"],
    }

    if platform.system() == "Windows":
        env_entries["PATH"].append("C:/Windows/System32")
        env_entries["PATH"].append("C:/Windows/System32/wbem")
        env_entries["PYTHONNOUSERSITE"] = ["1"]
        env_entries["PYTHON"] = [f"{dir}/python"]
        env_entries["GIT"] = [f"{dir}/Library/bin/git"]
    else:
        env_entries["PATH"].append("/bin")
        env_entries["PATH"].append("/usr/bin")
        env_entries["PATH"].append("/usr/sbin")
        env_entries["PYTHONNOUSERSITE"] = ["y"]
        env_entries["PYTHON"] = [f"{dir}/bin/python"]
        env_entries["GIT"] = [f"{dir}/bin/git"]

    env = {}
    for key, paths in env_entries.items():
        paths = [p.replace("/", os.path.sep) for p in paths]
        paths = os.pathsep.join(paths)

        os.environ[key] = paths

    return env


def print_env_info():
    which_cmd = "where" if platform.system() == "Windows" else "which"

    python = "python"

    def locate_python():
        nonlocal python

        python = subprocess.getoutput(f"{which_cmd} python")
        python = python.split("\n")
        python = python[0].strip()
        print("python: ", python)

    locate_python()

    def run(cmd):
        with subprocess.Popen(cmd) as p:
            p.wait()

    run([which_cmd, "git"])
    run(["git", "--version"])
    run([which_cmd, "python"])
    run([python, "--version"])

    print(f"PATH={os.environ['PATH']}")

    # if platform.system() == "Windows":
    #     print(f"COMSPEC={os.environ['COMSPEC']}")
    #     print("")
    #     run("wmic path win32_VideoController get name,AdapterRAM,DriverDate,DriverVersion".split(" "))

    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")
    print("")


def open_dev_shell():
    if platform.system() == "Windows":
        subprocess.Popen("cmd").communicate()
    else:
        subprocess.Popen("bash").communicate()


if __name__ == "__main__":
    env_dir = os.path.abspath(os.path.join("webui", "system"))

    configure_env(env_dir)
    print_env_info()
    open_dev_shell()

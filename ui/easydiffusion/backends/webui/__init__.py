import os
import platform
import subprocess
import threading
from threading import local
import psutil

from easydiffusion.app import ROOT_DIR, getConfig

from . import impl
from .impl import (
    ping,
    load_model,
    unload_model,
    set_options,
    generate_images,
    filter_images,
    get_url,
    stop_rendering,
    refresh_models,
    list_controlnet_filters,
)


ed_info = {
    "name": "WebUI backend for Easy Diffusion",
    "version": (1, 0, 0),
    "type": "backend",
}

BACKEND_DIR = os.path.abspath(os.path.join(ROOT_DIR, "webui"))
SYSTEM_DIR = os.path.join(BACKEND_DIR, "system")
WEBUI_DIR = os.path.join(BACKEND_DIR, "webui")

backend_process = None


def install_backend():
    pass


def start_backend():
    config = getConfig()
    backend_config = config.get("backend_config", {})

    if not os.path.exists(BACKEND_DIR):
        install_backend()

    impl.WEBUI_HOST = backend_config.get("host", "localhost")
    impl.WEBUI_PORT = backend_config.get("port", "7860")

    env = dict(os.environ)
    env.update(get_env())

    def target():
        global backend_process

        cmd = "webui.bat" if platform.system() == "Windows" else "webui.sh"
        print("starting", cmd, WEBUI_DIR)
        backend_process = subprocess.Popen([cmd], shell=True, cwd=WEBUI_DIR, env=env)

    backend_thread = threading.Thread(target=target)
    backend_thread.start()


def stop_backend():
    global backend_process

    if backend_process:
        kill(backend_process.pid)

    backend_process = None


def uninstall_backend():
    pass


def create_context():
    context = local()

    # temp hack, throws an attribute not found error otherwise
    context.device = "cuda:0"
    context.half_precision = True
    context.vram_usage_level = None

    context.models = {}
    context.model_paths = {}
    context.model_configs = {}
    context.device_name = None
    context.vram_optimizations = set()
    context.vram_usage_level = "balanced"
    context.test_diffusers = False
    context.enable_codeformer = False

    return context


def get_env():
    dir = os.path.abspath(SYSTEM_DIR)

    if not os.path.exists(dir):
        raise RuntimeError("The system folder is missing!")

    config = getConfig()
    models_dir = config.get("models_dir", os.path.join(ROOT_DIR, "models"))
    embeddings_dir = os.path.join(models_dir, "embeddings")

    env_entries = {
        "PATH": [
            f"{dir}/git/bin",
            f"{dir}/python",
            f"{dir}/python/Library/bin",
            f"{dir}/python/Scripts",
            f"{dir}/python/Library/usr/bin",
        ],
        "PYTHONPATH": [
            f"{dir}/python",
            f"{dir}/python/lib/site-packages",
            f"{dir}/python/lib/python3.10/site-packages",
        ],
        "PYTHONHOME": [],
        "PY_LIBS": [f"{dir}/python/Scripts/Lib", f"{dir}/python/Scripts/Lib/site-packages"],
        "PY_PIP": [f"{dir}/python/Scripts"],
        "PIP_INSTALLER_LOCATION": [f"{dir}/python/get-pip.py"],
        "TRANSFORMERS_CACHE": [f"{dir}/transformers-cache"],
        "HF_HUB_DISABLE_SYMLINKS_WARNING": ["true"],
        "COMMANDLINE_ARGS": [f'--api --models-dir "{models_dir}" --embeddings-dir "{embeddings_dir}"'],
        "SKIP_VENV": ["1"],
        "SD_WEBUI_RESTARTING": ["1"],
        "PYTHON": [f"{dir}/python/python"],
        "GIT": [f"{dir}/git/bin/git"],
    }

    if platform.system() == "Windows":
        env_entries["PYTHONNOUSERSITE"] = ["1"]
    else:
        env_entries["PYTHONNOUSERSITE"] = ["y"]

    env = {}
    for key, paths in env_entries.items():
        paths = [p.replace("/", os.path.sep) for p in paths]
        paths = os.pathsep.join(paths)

        env[key] = paths

    return env


# https://stackoverflow.com/a/25134985
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

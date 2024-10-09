import os
import platform
import subprocess
import threading
from threading import local
import psutil
import time
import shutil

from easydiffusion.app import ROOT_DIR, getConfig
from easydiffusion.model_manager import get_model_dirs

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

WEBUI_REPO = "https://github.com/lllyasviel/stable-diffusion-webui-forge.git"
WEBUI_COMMIT = "f4d5e8cac16a42fa939e78a0956b4c30e2b47bb5"

BACKEND_DIR = os.path.abspath(os.path.join(ROOT_DIR, "webui"))
SYSTEM_DIR = os.path.join(BACKEND_DIR, "system")
WEBUI_DIR = os.path.join(BACKEND_DIR, "webui")

OS_NAME = platform.system()

MODELS_TO_OVERRIDE = {
    "stable-diffusion": "--ckpt-dir",
    "vae": "--vae-dir",
    "hypernetwork": "--hypernetwork-dir",
    "gfpgan": "--gfpgan-models-path",
    "realesrgan": "--realesrgan-models-path",
    "lora": "--lora-dir",
    "codeformer": "--codeformer-models-path",
    "embeddings": "--embeddings-dir",
    "controlnet": "--controlnet-dir",
}

backend_process = None
conda = "conda"


def locate_conda():
    global conda

    which = "where" if OS_NAME == "Windows" else "which"
    conda = subprocess.getoutput(f"{which} conda")
    conda = conda.split("\n")
    conda = conda[0].strip()
    print("conda: ", conda)


locate_conda()


def install_backend():
    print("Installing the WebUI backend..")

    # create the conda env
    run([conda, "create", "-y", "--prefix", SYSTEM_DIR], cwd=ROOT_DIR)

    print("Installing packages..")

    # install python 3.10 and git in the conda env
    run([conda, "install", "-y", "--prefix", SYSTEM_DIR, "-c", "conda-forge", "python=3.10", "git"], cwd=ROOT_DIR)

    # print info
    run_in_conda(["git", "--version"], cwd=ROOT_DIR)
    run_in_conda(["python", "--version"], cwd=ROOT_DIR)

    # clone webui
    run_in_conda(["git", "clone", WEBUI_REPO, WEBUI_DIR], cwd=ROOT_DIR)

    # install cpu-only torch if the PC doesn't have a graphics card (for Windows and Linux).
    # this avoids WebUI installing a CUDA version and trying to activate it
    if OS_NAME in ("Windows", "Linux") and not has_discrete_graphics_card():
        run_in_conda(["python", "-m", "pip", "install", "torch", "torchvision"], cwd=WEBUI_DIR)


def start_backend():
    config = getConfig()
    backend_config = config.get("backend_config", {})

    if not os.path.exists(BACKEND_DIR):
        install_backend()

    if backend_config.get("auto_update", True):
        run_in_conda(["git", "add", "-A", "."], cwd=WEBUI_DIR)
        run_in_conda(["git", "stash"], cwd=WEBUI_DIR)
        run_in_conda(["git", "reset", "--hard"], cwd=WEBUI_DIR)
        run_in_conda(["git", "fetch"], cwd=WEBUI_DIR)
        run_in_conda(["git", "-c", "advice.detachedHead=false", "checkout", WEBUI_COMMIT], cwd=WEBUI_DIR)

    # hack to prevent webui-macos-env.sh from overwriting the COMMANDLINE_ARGS env variable
    mac_webui_file = os.path.join(WEBUI_DIR, "webui-macos-env.sh")
    if os.path.exists(mac_webui_file):
        os.remove(mac_webui_file)

    impl.WEBUI_HOST = backend_config.get("host", "localhost")
    impl.WEBUI_PORT = backend_config.get("port", "7860")

    env = dict(os.environ)
    env.update(get_env())

    def restart_if_webui_dies_after_starting():
        has_started = False

        while True:
            try:
                impl.ping(timeout=1)
                has_started = True
            except (TimeoutError, ConnectionError):
                if has_started:  # process probably died
                    print("######################## WebUI probably died. Restarting...")
                    stop_backend()
                    backend_thread = threading.Thread(target=target)
                    backend_thread.start()
                    break
            except Exception:
                pass

            time.sleep(1)

    def target():
        global backend_process

        cmd = "webui.bat" if OS_NAME == "Windows" else "./webui.sh"

        print("starting", cmd, WEBUI_DIR)
        backend_process = run_in_conda([cmd], cwd=WEBUI_DIR, env=env, wait=False, output_prefix="[WebUI] ")

        restart_if_dead_thread = threading.Thread(target=restart_if_webui_dies_after_starting)
        restart_if_dead_thread.start()

        backend_process.wait()

    backend_thread = threading.Thread(target=target)
    backend_thread.start()

    start_proxy()


def start_proxy():
    # proxy
    from easydiffusion.server import server_api
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    import json

    URI_PREFIX = "/webui"

    webui_proxy = FastAPI(root_path=f"{URI_PREFIX}", docs_url="/swagger")

    @webui_proxy.get("{uri:path}")
    def proxy_get(uri: str, req: Request):
        if uri == "/openapi-proxy.json":
            uri = "/openapi.json"

        res = impl.webui_get(uri, headers=req.headers)

        content = res.content
        headers = dict(res.headers)

        if uri == "/docs":
            content = res.text.replace("url: '/openapi.json'", f"url: '{URI_PREFIX}/openapi-proxy.json'")
        elif uri == "/openapi.json":
            content = res.json()
            content["paths"] = {f"{URI_PREFIX}{k}": v for k, v in content["paths"].items()}
            content = json.dumps(content)

        if isinstance(content, str):
            content = bytes(content, encoding="utf-8")
            headers["content-length"] = str(len(content))

        # Return the same response back to the client
        return Response(content=content, status_code=res.status_code, headers=headers)

    @webui_proxy.post("{uri:path}")
    async def proxy_post(uri: str, req: Request):
        body = await req.body()
        res = impl.webui_post(uri, data=body, headers=req.headers)

        # Return the same response back to the client
        return Response(content=res.content, status_code=res.status_code, headers=dict(res.headers))

    server_api.mount(f"{URI_PREFIX}", webui_proxy)


def stop_backend():
    global backend_process

    if backend_process:
        try:
            kill(backend_process.pid)
        except:
            pass

    backend_process = None


def uninstall_backend():
    shutil.rmtree(BACKEND_DIR)


def is_installed():
    if not os.path.exists(BACKEND_DIR) or not os.path.exists(SYSTEM_DIR) or not os.path.exists(WEBUI_DIR):
        return True

    env = dict(os.environ)
    env.update(get_env())

    try:
        out = check_output_in_conda(["python", "-m", "pip", "show", "torch"], env=env)
        return "Version" in out.decode()
    except subprocess.CalledProcessError:
        pass

    return False


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


def run_in_conda(cmds: list, *args, **kwargs):
    cmds = [conda, "run", "--no-capture-output", "--prefix", SYSTEM_DIR] + cmds
    return run(cmds, *args, **kwargs)


def check_output_in_conda(cmds: list, cwd=None, env=None):
    cmds = [conda, "run", "--no-capture-output", "--prefix", SYSTEM_DIR] + cmds
    return subprocess.check_output(cmds, cwd=cwd, env=env, stderr=subprocess.PIPE)


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

    model_path_args = get_model_path_args()

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
        "PIP_INSTALLER_LOCATION": [],  # [f"{dir}/python/get-pip.py"],
        "TRANSFORMERS_CACHE": [f"{dir}/transformers-cache"],
        "HF_HUB_DISABLE_SYMLINKS_WARNING": ["true"],
        "COMMANDLINE_ARGS": [f'--api --models-dir "{models_dir}" {model_path_args} --skip-torch-cuda-test'],
        "SKIP_VENV": ["1"],
        "SD_WEBUI_RESTARTING": ["1"],
    }

    if OS_NAME == "Windows":
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
        env_entries["venv_dir"] = ["-"]

    if OS_NAME == "Darwin":
        # based on https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/e26abf87ecd1eefd9ab0a198eee56f9c643e4001/webui-macos-env.sh
        # hack - have to define these here, otherwise webui-macos-env.sh will overwrite COMMANDLINE_ARGS
        env_entries["COMMANDLINE_ARGS"][0] += " --upcast-sampling --no-half-vae --use-cpu interrogate"
        env_entries["PYTORCH_ENABLE_MPS_FALLBACK"] = ["1"]

        cpu_name = str(subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]))
        if "Intel" in cpu_name:
            env_entries["TORCH_COMMAND"] = ["pip install torch==2.1.2 torchvision==0.16.2"]
        else:
            env_entries["TORCH_COMMAND"] = ["pip install torch==2.3.1 torchvision==0.18.1"]
    else:
        vram_usage_level = config.get("vram_usage_level", "balanced")
        if config.get("render_devices", "auto") == "cpu" or not has_discrete_graphics_card():
            env_entries["COMMANDLINE_ARGS"][0] += " --always-cpu"
        elif vram_usage_level == "low":
            env_entries["COMMANDLINE_ARGS"][0] += " --always-low-vram"
        elif vram_usage_level == "high":
            env_entries["COMMANDLINE_ARGS"][0] += " --always-high-vram"

    env = {}
    for key, paths in env_entries.items():
        paths = [p.replace("/", os.path.sep) for p in paths]
        paths = os.pathsep.join(paths)

        env[key] = paths

    return env


def has_discrete_graphics_card():
    system = OS_NAME

    if system == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "path", "win32_videocontroller", "get", "name"], stderr=subprocess.STDOUT
            )
            # Filter for discrete graphics cards (NVIDIA, AMD, etc.)
            discrete_gpus = ["NVIDIA", "AMD", "ATI"]
            return any(gpu in output.decode() for gpu in discrete_gpus)
        except subprocess.CalledProcessError:
            return False

    elif system == "Linux":
        try:
            output = subprocess.check_output(["lspci"], stderr=subprocess.STDOUT)
            # Check for discrete GPUs (NVIDIA, AMD)
            discrete_gpus = ["NVIDIA", "AMD", "Advanced Micro Devices"]
            return any(gpu in line for line in output.decode().splitlines() for gpu in discrete_gpus)
        except subprocess.CalledProcessError:
            return False

    elif system == "Darwin":  # macOS
        try:
            output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], stderr=subprocess.STDOUT)
            # Check for discrete GPU in the output
            return "NVIDIA" in output.decode() or "AMD" in output.decode()
        except subprocess.CalledProcessError:
            return False

    return False


# https://stackoverflow.com/a/25134985
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def get_model_path_args():
    args = []
    for model_type, flag in MODELS_TO_OVERRIDE.items():
        model_dir = get_model_dirs(model_type)[0]
        args.append(f'{flag} "{model_dir}"')

    return " ".join(args)

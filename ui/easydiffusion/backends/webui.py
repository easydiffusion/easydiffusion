import os
import platform
import subprocess
import shutil

from easydiffusion.app import ROOT_DIR, getConfig
from easydiffusion.utils import log
from torchruntime.utils import get_device, get_device_name, get_installed_torch_platform
from sdkit.utils import is_cpu_device

from common import run
import webui_common
from webui_common import (
    ping,
    load_model,
    unload_model,
    flush_model_changes,
    set_options,
    generate_images,
    filter_images,
    get_url,
    stop_rendering,
    refresh_models,
    list_controlnet_filters,
    get_common_cli_args,
    create_context,
    do_start_backend,
    stop_backend,
)


ed_info = {
    "name": "WebUI backend for Easy Diffusion",
    "version": (1, 0, 0),
    "type": "backend",
}

WEBUI_REPO = "https://github.com/easydiffusion/stable-diffusion-webui-forge.git"
WEBUI_COMMIT = "eb44f7b23774d284b767456788489eac51def1f3"

BACKEND_DIR = os.path.abspath(os.path.join(ROOT_DIR, "webui"))
SYSTEM_DIR = os.path.join(BACKEND_DIR, "system")
WEBUI_DIR = os.path.join(BACKEND_DIR, "webui")

OS_NAME = platform.system()

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
    if not os.path.exists(BACKEND_DIR):
        print("Installing the WebUI backend..")

        # create the conda env
        run([conda, "create", "-y", "--prefix", SYSTEM_DIR], cwd=ROOT_DIR)

        print("Installing packages..")

        # install python 3.10 and git in the conda env
        run([conda, "install", "-y", "--prefix", SYSTEM_DIR, "-c", "conda-forge", "python=3.10", "git"], cwd=ROOT_DIR)

    if not os.path.exists(WEBUI_DIR):
        env = dict(os.environ)
        env.update(get_env())

        # print info
        run_in_conda(["git", "--version"], cwd=ROOT_DIR, env=env)
        run_in_conda(["python", "--version"], cwd=ROOT_DIR, env=env)

        # clone webui
        run_in_conda(["git", "clone", WEBUI_REPO, WEBUI_DIR], cwd=ROOT_DIR, env=env)

        # install the appropriate version of torch using torchruntime
        run_in_conda(["python", "-m", "pip", "install", "torchruntime"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["python", "-m", "torchruntime", "install", "torch", "torchvision"], cwd=WEBUI_DIR, env=env)


def start_backend():
    config = getConfig()
    backend_config = config.get("backend_config") or {}

    log.info(f"Backend dir: {BACKEND_DIR}")

    install_backend()  # will do nothing if already installed

    env = dict(os.environ)
    env.update(get_env())

    was_still_installing = not is_installed()

    if backend_config.get("auto_update", True):
        # Ensure the remote origin points to the correct repository
        try:
            current_remote = (
                check_output_in_conda(["git", "remote", "get-url", "origin"], cwd=WEBUI_DIR, env=env)
                .decode("utf-8")
                .strip()
            )
            if current_remote != WEBUI_REPO:
                log.info(f"Updating remote origin from {current_remote} to {WEBUI_REPO}")
                run_in_conda(["git", "remote", "set-url", "origin", WEBUI_REPO], cwd=WEBUI_DIR, env=env)
        except Exception as e:
            log.warning(f"Failed to check/update git remote: {e}")

        run_in_conda(["git", "add", "-A", "."], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "stash"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "reset", "--hard"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "fetch"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "-c", "advice.detachedHead=false", "checkout", WEBUI_COMMIT], cwd=WEBUI_DIR, env=env)

    # workaround for the installations that broke out of conda and used ED's python 3.8 instead of WebUI conda's Py 3.10
    run_in_conda(["python", "-m", "pip", "install", "-q", "--upgrade", "urllib3==2.2.3"], cwd=WEBUI_DIR, env=env)

    webui_common.WEBUI_API_PREFIX = ""

    def run_fn():
        cmd = "webui.bat" if OS_NAME == "Windows" else "./webui.sh"

        log.info(f"starting: {cmd} in {WEBUI_DIR}")
        log.info(f"COMMANDLINE_ARGS: {env['COMMANDLINE_ARGS']}")

        return run_in_conda([cmd], cwd=WEBUI_DIR, env=env, wait=False, output_prefix="[WebUI] ")

    do_start_backend(was_still_installing, run_fn)

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

        res = webui_common.webui_get(uri, headers=req.headers)

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
        res = webui_common.webui_post(uri, data=body, headers=req.headers)

        # Return the same response back to the client
        return Response(content=res.content, status_code=res.status_code, headers=dict(res.headers))

    server_api.mount(f"{URI_PREFIX}", webui_proxy)


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


def run_in_conda(cmds: list, *args, **kwargs):
    cmds = [conda, "run", "--no-capture-output", "--prefix", SYSTEM_DIR] + cmds
    return run(cmds, *args, **kwargs)


def check_output_in_conda(cmds: list, cwd=None, env=None):
    cmds = [conda, "run", "--no-capture-output", "--prefix", SYSTEM_DIR] + cmds
    return subprocess.check_output(cmds, cwd=cwd, env=env, stderr=subprocess.PIPE)


def get_env():
    dir = os.path.abspath(SYSTEM_DIR)

    if not os.path.exists(dir):
        raise RuntimeError("The system folder is missing!")

    config = getConfig()
    backend_config = config.get("backend_config") or {}
    models_dir = config.get("models_dir", os.path.join(ROOT_DIR, "models"))
    models_dir = models_dir.rstrip("/\\")

    common_cli_args = get_common_cli_args()

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
        "COMMANDLINE_ARGS": [
            f'--api --models-dir "{models_dir}" --skip-torch-cuda-test --disable-gpu-warning {common_cli_args}'
        ],
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
        from easydiffusion.device_manager import needs_to_force_full_precision

        torch_platform_name = get_installed_torch_platform()[0]

        vram_usage_level = config.get("vram_usage_level", "balanced")
        if config.get("render_devices", "auto") == "cpu" or is_cpu_device(torch_platform_name):
            env_entries["COMMANDLINE_ARGS"][0] += " --always-cpu"
        elif torch_platform_name == "directml":
            env_entries["COMMANDLINE_ARGS"][0] += " --directml"
        else:
            device = get_device(0)
            if needs_to_force_full_precision(get_device_name(device)):
                env_entries["COMMANDLINE_ARGS"][0] += " --no-half --precision full"

            if vram_usage_level == "low":
                env_entries["COMMANDLINE_ARGS"][0] += " --always-low-vram"
            elif vram_usage_level == "high":
                env_entries["COMMANDLINE_ARGS"][0] += " --always-high-vram"

    cli_args = backend_config.get("COMMANDLINE_ARGS")
    if cli_args:
        env_entries["COMMANDLINE_ARGS"][0] += " " + cli_args

    env = {}
    for key, paths in env_entries.items():
        paths = [p.replace("/", os.path.sep) for p in paths]
        paths = os.pathsep.join(paths)

        env[key] = paths

    return env

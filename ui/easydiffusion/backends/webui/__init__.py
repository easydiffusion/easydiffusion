import os
import platform
import subprocess
import threading
from threading import local
import psutil
import time
import shutil
import atexit

from easydiffusion.app import ROOT_DIR, getConfig
from easydiffusion.model_manager import get_model_dirs
from easydiffusion.utils import log
from torchruntime.utils import get_device, get_device_name, get_installed_torch_platform
from sdkit.utils import is_cpu_device

from . import impl
from .impl import (
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
)


ed_info = {
    "name": "WebUI backend for Easy Diffusion",
    "version": (1, 0, 0),
    "type": "backend",
}

WEBUI_REPO = "https://github.com/lllyasviel/stable-diffusion-webui-forge.git"
WEBUI_COMMIT = "dfdcbab685e57677014f05a3309b48cc87383167"

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
    "text-encoder": "--text-encoder-dir",
}

WEBUI_PATCHES = [
    "forge_exception_leak_patch.patch",
    "forge_model_crash_recovery.patch",
    "forge_api_refresh_text_encoders.patch",
    "forge_loader_force_gc.patch",
    "forge_monitor_parent_process.patch",
    "forge_disable_corrupted_model_renaming.patch",
]

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

    log.info(f"Expected WebUI backend dir: {BACKEND_DIR}")

    if not os.path.exists(BACKEND_DIR):
        install_backend()

    env = dict(os.environ)
    env.update(get_env())

    was_still_installing = not is_installed()

    if backend_config.get("auto_update", True):
        run_in_conda(["git", "status"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "add", "-A", "."], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "stash"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "reset", "--hard"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "fetch"], cwd=WEBUI_DIR, env=env)
        run_in_conda(["git", "-c", "advice.detachedHead=false", "checkout", WEBUI_COMMIT], cwd=WEBUI_DIR, env=env)

        # patch forge for various stability-related fixes
        for patch in WEBUI_PATCHES:
            patch_path = os.path.join(os.path.dirname(__file__), patch)
            log.info(f"Applying WebUI patch: {patch_path}")
            run_in_conda(["git", "apply", patch_path], cwd=WEBUI_DIR, env=env)

    # workaround for the installations that broke out of conda and used ED's python 3.8 instead of WebUI conda's Py 3.10
    run_in_conda(["python", "-m", "pip", "install", "-q", "--upgrade", "urllib3==2.2.3"], cwd=WEBUI_DIR, env=env)

    # hack to prevent webui-macos-env.sh from overwriting the COMMANDLINE_ARGS env variable
    mac_webui_file = os.path.join(WEBUI_DIR, "webui-macos-env.sh")
    if os.path.exists(mac_webui_file):
        os.remove(mac_webui_file)

    impl.WEBUI_HOST = backend_config.get("host", "localhost")
    impl.WEBUI_PORT = backend_config.get("port", "7860")

    def restart_if_webui_dies_after_starting():
        has_started = False

        while True:
            try:
                impl.ping(timeout=30)

                is_first_start = not has_started
                has_started = True

                if was_still_installing and is_first_start:
                    ui = config.get("ui", {})
                    net = config.get("net", {})
                    port = net.get("listen_port", 9000)

                    if ui.get("open_browser_on_start", True):
                        import webbrowser

                        log.info("Opening browser..")

                        webbrowser.open(f"http://localhost:{port}")
            except (TimeoutError, ConnectionError):
                if has_started:  # process probably died
                    print("######################## WebUI probably died. Restarting...")
                    stop_backend()
                    backend_thread = threading.Thread(target=target)
                    backend_thread.start()
                    break
            except Exception:
                import traceback

                log.exception(traceback.format_exc())

            time.sleep(1)

    def target():
        global backend_process

        cmd = "webui.bat" if OS_NAME == "Windows" else "./webui.sh"

        log.info(f"starting: {cmd} in {WEBUI_DIR}")
        log.info(f"COMMANDLINE_ARGS: {env['COMMANDLINE_ARGS']}")

        backend_process = run_in_conda([cmd], cwd=WEBUI_DIR, env=env, wait=False, output_prefix="[WebUI] ")

        # atexit.register isn't 100% reliable, that's why we also use `forge_monitor_parent_process.patch`
        # which causes Forge to kill itself if the parent pid passed to it is no longer valid.
        atexit.register(backend_process.terminate)

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
    print(f"Running command: {' '.join(cmds)} in {cwd}. stream_output={stream_output}, wait={wait}")
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
    context.torch_device = get_device(0)
    context.device = f"{context.torch_device.type}:{context.torch_device.index}"
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
    backend_config = config.get("backend_config") or {}
    models_dir = config.get("models_dir", os.path.join(ROOT_DIR, "models"))
    models_dir = models_dir.rstrip("/\\")

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
        "COMMANDLINE_ARGS": [
            f'--api --models-dir "{models_dir}" {model_path_args} --skip-torch-cuda-test --disable-gpu-warning --port {impl.WEBUI_PORT} --parent-pid {os.getpid()}'
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

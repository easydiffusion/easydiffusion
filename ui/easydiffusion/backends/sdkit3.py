import os
import platform
import shutil
import requests
import hashlib
import concurrent.futures
import tarfile
import tempfile

from easydiffusion.app import ROOT_DIR, getConfig
from easydiffusion.utils import log

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
    "name": "sdkit3 backend for Easy Diffusion",
    "version": (1, 0, 0),
    "type": "backend",
}

BACKENDS_ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "backends"))
SDKIT3_BACKEND_DIR = os.path.join(BACKENDS_ROOT_DIR, "sdkit3")
os.makedirs(BACKENDS_ROOT_DIR, exist_ok=True)


def get_backend_dir():
    target = get_target()
    return os.path.join(SDKIT3_BACKEND_DIR, target)


BACKEND_BINARY_URL_BASE = "https://github.com/easydiffusion/sdkit/releases/download/v3.0.0"

OS_NAME = platform.system()


def install_backend():
    update_backend()


def update_backend():
    target = get_target()
    backend_dir = os.path.join(BACKENDS_ROOT_DIR, "sdkit3", target)

    if os.path.exists(backend_dir):
        print("Updating sdkit3 backend..")
    else:
        print("Installing sdkit3 backend..")

    print("Looking for backend build for target:", target)

    manifest_url = f"{BACKEND_BINARY_URL_BASE}/{target}-manifest.json"

    print(f"Fetching manifest from {manifest_url}")
    response = requests.get(manifest_url)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise ValueError(
                f"Target platform {target} does not exist. Please post a message on our Discord server ( https://discord.com/invite/u9yhsFmEkB ) or create a new issue at https://github.com/easydiffusion/sdkit/issues to request this platform build."
            )
        else:
            raise
    manifest = response.json()
    files = manifest["files"]

    os.makedirs(backend_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(update_or_download_file, filename, info, BACKEND_BINARY_URL_BASE, backend_dir)
            for filename, info in files.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("Backend update complete.")


def update_or_download_file(filename, info, base_url, backend_dir):
    from sdkit.utils import download_file

    filepath = os.path.join(backend_dir, filename)
    expected_sha = info["sha256"]
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if actual_sha == expected_sha:
            print(f"File {filename} is up to date.")
            return

    # download
    uri = info["uri"]
    download_url = f"{base_url}/{uri}"
    print(f"Downloading {filename} from {download_url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        download_file(download_url, tmp_path)

        # extract
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(backend_dir)
        print(f"Extracted {filename}")
    finally:
        os.unlink(tmp_path)


def start_backend():
    config = getConfig()
    backend_config = config.get("backend_config") or {}

    backend_dir = get_backend_dir()
    log.info(f"Backend dir: {backend_dir}")

    was_still_installing = not is_installed()

    if backend_config.get("auto_update", True) or not is_installed():
        update_backend()

    extra_args = ["--log-level", "debug"]

    vram_usage_level = config.get("vram_usage_level", "balanced")
    if vram_usage_level == "low":
        extra_args.append("--control-net-cpu")
        extra_args.append("--clip-on-cpu")
        extra_args.append("--vae-on-cpu")

    if vram_usage_level != "high":
        extra_args.append("--offload-to-cpu")
        extra_args.append("--vae-tiling")

    user_args = backend_config.get("COMMANDLINE_ARGS")
    user_args = user_args.split(" ") if user_args else []

    webui_common.WEBUI_API_PREFIX = "/v1"

    def run_fn():
        exe_name = "sdkit.exe" if OS_NAME == "Windows" else "sdkit"
        common_cli_args = get_common_cli_args(return_string=False)
        cmd = [os.path.join(backend_dir, exe_name)] + common_cli_args + extra_args + user_args

        log.info(f"starting: {cmd}")

        return run(cmd, cwd=backend_dir, wait=False, output_prefix="[sdkit3] ")

    do_start_backend(was_still_installing, run_fn)


def uninstall_backend():
    shutil.rmtree(SDKIT3_BACKEND_DIR)


def is_installed():
    backend_dir = get_backend_dir()
    exe_name = "sdkit.exe" if OS_NAME == "Windows" else "sdkit"
    if os.path.exists(os.path.join(backend_dir, exe_name)):
        return True

    return False


def get_target():
    """Get the target for the build."""
    config = getConfig()
    backend_config = config.get("backend_config") or {}

    def get_os():
        """Get OS name for target."""
        if OS_NAME == "Windows":
            return "win"
        elif OS_NAME == "Darwin":
            return "mac"
        elif OS_NAME == "Linux":
            return "linux"
        else:
            return OS_NAME.lower()

    def get_arch():
        """Get architecture for target."""
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            return "x64"
        elif machine in ("arm64", "aarch64"):
            return "arm64"
        else:
            return machine

    platform_name = backend_config.get("platform", get_platform_name())
    variant_name = backend_config.get("variant", get_variant_name(platform_name))

    target = f"{get_os()}-{get_arch()}-{platform_name}-{variant_name}"

    return target


def get_platform_name():
    if OS_NAME == "Darwin":
        return "metal"

    # use torchruntime to determine if nvidia gpu is present
    from torchruntime.device_db import get_gpus
    from torchruntime.platform_detection import get_torch_platform

    gpus = get_gpus()
    torch_platform = get_torch_platform(gpus)

    if torch_platform == "cpu":
        return "cpu"

    if torch_platform.startswith("cu"):
        return "cuda"

    return "vulkan"


def get_variant_name(platform_name):
    if platform_name == "cuda":
        # deduce the variant from gpu compute capability
        from torchruntime.device_db import get_gpus
        from torchruntime.gpu_db import get_nvidia_arch
        from torchruntime.consts import NVIDIA

        gpus = get_gpus()
        for gpu in gpus:
            if gpu.vendor_id == NVIDIA and gpu.is_discrete:
                arch = get_nvidia_arch([gpu.device_name])  # 7.5, 12 etc
                arch = int(arch * 10)
                return f"sm{arch}"

    return "any"

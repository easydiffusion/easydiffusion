"""
This script checks and installs the required modules.

This script runs inside the legacy "stable-diffusion" folder

TODO - Maybe replace the bulk of this script with a call to `pip install -f requirements.txt`, with
a custom index URL depending on the platform.

"""

import os
from importlib.metadata import version as pkg_version
import platform
import traceback

os_name = platform.system()

modules_to_check = {
    "torch": ("1.11.0", "1.13.1", "2.0.0"),
    "torchvision": ("0.12.0", "0.14.1", "0.15.1"),
    "sdkit": "1.0.177",
    "stable-diffusion-sdkit": "2.1.4",
    "rich": "12.6.0",
    "uvicorn": "0.19.0",
    "fastapi": "0.85.1",
    "pycloudflared": "0.2.0",
    "ruamel.yaml": "0.17.21",
    "sqlalchemy": "2.0.19",
    "python-multipart": "0.0.6",
    # "xformers": "0.0.16",
}
modules_to_log = ["torch", "torchvision", "sdkit", "stable-diffusion-sdkit"]


def version(module_name: str) -> str:
    try:
        return pkg_version(module_name)
    except:
        return None


def install(module_name: str, module_version: str):
    if module_name == "xformers" and (os_name == "Darwin" or is_amd_on_linux()):
        return

    index_url = None
    if module_name in ("torch", "torchvision"):
        module_version, index_url = apply_torch_install_overrides(module_version)

    if is_amd_on_linux():  # hack until AMD works properly on torch 2.0 (avoids black images on some cards)
        if module_name == "torch":
            module_version = "1.13.1+rocm5.2"
        elif module_name == "torchvision":
            module_version = "0.14.1+rocm5.2"
    elif os_name == "Darwin":
        if module_name == "torch":
            module_version = "1.13.1"
        elif module_name == "torchvision":
            module_version = "0.14.1"

    install_cmd = f"python -m pip install --upgrade {module_name}=={module_version}"
    if index_url:
        install_cmd += f" --index-url {index_url}"
    if module_name == "sdkit" and version("sdkit") is not None:
        install_cmd += " -q"

    print(">", install_cmd)
    os.system(install_cmd)


def init():
    for module_name, allowed_versions in modules_to_check.items():
        if os.path.exists(f"../src/{module_name}"):
            print(f"Skipping {module_name} update, since it's in developer/editable mode")
            continue

        allowed_versions, latest_version = get_allowed_versions(module_name, allowed_versions)

        requires_install = False
        if module_name in ("torch", "torchvision"):
            if version(module_name) is None:  # allow any torch version
                requires_install = True
            elif os_name == "Darwin" and (  # force mac to downgrade from torch 2.0
                version("torch").startswith("2.") or version("torchvision").startswith("0.15.")
            ):
                requires_install = True
        elif version(module_name) not in allowed_versions:
            requires_install = True

        if requires_install:
            try:
                install(module_name, latest_version)
            except:
                traceback.print_exc()
                fail(module_name)

        if module_name in modules_to_log:
            print(f"{module_name}: {version(module_name)}")


### utilities


def get_allowed_versions(module_name: str, allowed_versions: tuple):
    allowed_versions = (allowed_versions,) if isinstance(allowed_versions, str) else allowed_versions
    latest_version = allowed_versions[-1]

    if module_name in ("torch", "torchvision"):
        allowed_versions = include_cuda_versions(allowed_versions)

    return allowed_versions, latest_version


def apply_torch_install_overrides(module_version: str):
    index_url = None
    if os_name == "Windows":
        module_version += "+cu117"
        index_url = "https://download.pytorch.org/whl/cu117"
    elif is_amd_on_linux():
        index_url = "https://download.pytorch.org/whl/rocm5.2"

    return module_version, index_url


def include_cuda_versions(module_versions: tuple) -> tuple:
    "Adds CUDA-specific versions to the list of allowed version numbers"

    allowed_versions = tuple(module_versions)
    allowed_versions += tuple(f"{v}+cu116" for v in module_versions)
    allowed_versions += tuple(f"{v}+cu117" for v in module_versions)
    allowed_versions += tuple(f"{v}+rocm5.2" for v in module_versions)
    allowed_versions += tuple(f"{v}+rocm5.4.2" for v in module_versions)

    return allowed_versions


def is_amd_on_linux():
    if os_name == "Linux":
        try:
            with open("/proc/bus/pci/devices", "r") as f:
                device_info = f.read()
                if "amdgpu" in device_info and "nvidia" not in device_info:
                    return True
        except:
            return False

    return False


def fail(module_name):
    print(
        f"""Error installing {module_name}. Sorry about that, please try to:
1. Run this installer again.
2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/easydiffusion/easydiffusion/wiki/Troubleshooting
3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
4. If that doesn't solve the problem, please file an issue at https://github.com/easydiffusion/easydiffusion/issues
Thanks!"""
    )
    exit(1)


### start

init()

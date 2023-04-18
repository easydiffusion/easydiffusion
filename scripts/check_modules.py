"""
This script checks and installs the required modules.
"""

import os
from importlib.metadata import version as pkg_version
import platform

os_name = platform.system()

modules_to_check = {
    "torch": ("1.13.1", "2.0.0"),
    "torchvision": ("0.14.1", "0.15.1"),
    "sdkit": "1.0.72",
    "stable-diffusion-sdkit": "2.1.4",
    "rich": "12.6.0",
    "uvicorn": "0.19.0",
    "fastapi": "0.85.1",
}


def version(module_name: str) -> str:
    try:
        return pkg_version(module_name)
    except:
        return None


def install(module_name: str, module_version: str):
    index_url = None
    if module_name in ("torch", "torchvision"):
        module_version, index_url = apply_torch_install_overrides(module_version)

    install_cmd = f"python -m pip install --upgrade {module_name}=={module_version}"
    if index_url:
        install_cmd += f" --index-url {index_url}"
    if module_name == "sdkit":
        install_cmd += " -q"

    print(">", install_cmd)
    os.system(install_cmd)


def init():
    for module_name, allowed_versions in modules_to_check.items():
        if os.path.exists(f"../src/{module_name}"):
            print(f"Skipping {module_name} update, since it's in developer/editable mode")
            continue

        allowed_versions, latest_version = get_allowed_versions(module_name, allowed_versions)
        if version(module_name) not in allowed_versions:
            try:
                install(module_name, latest_version)
            except:
                fail(module_name)

        print(f"{module_name}: {version(module_name)}")


### utilities


def get_allowed_versions(module_name: str, allowed_versions: tuple[str]):
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
    elif os_name == "Linux":
        with open("/proc/bus/pci/devices", "r") as f:
            device_info = f.read()
            if "amdgpu" in device_info:
                index_url = "https://download.pytorch.org/whl/rocm5.4.2"

    return module_version, index_url


def include_cuda_versions(module_versions: tuple) -> tuple:
    "Adds CUDA-specific versions to the list of allowed version numbers"

    allowed_versions = tuple(module_versions)
    allowed_versions += tuple(f"{v}+cu116" for v in module_versions)
    allowed_versions += tuple(f"{v}+cu117" for v in module_versions)

    return allowed_versions


def fail(module_name):
    print(
        f"""Error installing {module_name}. Sorry about that, please try to:
1. Run this installer again.
2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting
3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues
Thanks!"""
    )
    exit(1)


### start

init()

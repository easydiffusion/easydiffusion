import sys
import os
import platform
from importlib.metadata import version as pkg_version

from sdkit.utils import log

from easydiffusion import app

# future home of scripts/check_modules.py

manifest = {
    "tensorrt": {
        "install": [
            "wheel",
            "nvidia-cudnn-cu11==8.9.4.25",
            "tensorrt==9.0.0.post11.dev1 --pre --extra-index-url=https://pypi.nvidia.com --trusted-host pypi.nvidia.com",
        ],
        "uninstall": ["tensorrt"],
        # TODO also uninstall tensorrt-libs and nvidia-cudnn, but do it upon restarting (avoid 'file in use' error)
    }
}
installing = []

# remove this once TRT releases on pypi
if platform.system() == "Windows":
    trt_dir = os.path.join(app.ROOT_DIR, "tensorrt")
    if os.path.exists(trt_dir) and os.path.isdir(trt_dir) and len(os.listdir(trt_dir)) > 0:
        files = os.listdir(trt_dir)

        packages = manifest["tensorrt"]["install"]
        packages = tuple(p.replace("-", "_") for p in packages)

        wheels = []
        for p in packages:
            p = p.split(" ")[0]
            f = next((f for f in files if f.startswith(p) and f.endswith((".whl", ".tar.gz"))), None)
            if f:
                wheels.append(os.path.join(trt_dir, f))

        manifest["tensorrt"]["install"] = wheels


def get_installed_packages() -> list:
    return {module_name: version(module_name) for module_name in manifest if is_installed(module_name)}


def is_installed(module_name) -> bool:
    return version(module_name) is not None


def install(module_name):
    if is_installed(module_name):
        log.info(f"{module_name} has already been installed!")
        return
    if module_name in installing:
        log.info(f"{module_name} is already installing!")
        return

    if module_name not in manifest:
        raise RuntimeError(f"Can't install unknown package: {module_name}!")

    commands = manifest[module_name]["install"]
    if module_name == "tensorrt":
        commands += [
            "protobuf==3.20.3 polygraphy==0.47.1 onnx==1.14.0 --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com"
        ]
    commands = [f"python -m pip install --upgrade {cmd}" for cmd in commands]

    installing.append(module_name)

    try:
        for cmd in commands:
            print(">", cmd)
            if os.system(cmd) != 0:
                raise RuntimeError(f"Error while running {cmd}. Please check the logs in the command-line.")
    finally:
        installing.remove(module_name)


def uninstall(module_name):
    if not is_installed(module_name):
        log.info(f"{module_name} hasn't been installed!")
        return

    if module_name not in manifest:
        raise RuntimeError(f"Can't uninstall unknown package: {module_name}!")

    commands = manifest[module_name]["uninstall"]
    commands = [f"python -m pip uninstall -y {cmd}" for cmd in commands]

    for cmd in commands:
        print(">", cmd)
        if os.system(cmd) != 0:
            raise RuntimeError(f"Error while running {cmd}. Please check the logs in the command-line.")


def version(module_name: str) -> str:
    try:
        return pkg_version(module_name)
    except:
        return None

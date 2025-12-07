import os
import platform
import re
import traceback

import torch
from easydiffusion.utils import log

from torchruntime.utils import (
    get_installed_torch_platform,
    get_device,
    get_device_count,
    get_device_name,
    SUPPORTED_BACKENDS,
)
from sdkit.utils import mem_get_info, is_cpu_device, has_half_precision_bug

"""
Set `FORCE_FULL_PRECISION` in the environment variables, or in `config.bat`/`config.sh` to set full precision (i.e. float32).
Otherwise the models will load at half-precision (i.e. float16).

Half-precision is fine most of the time. Full precision is only needed for working around GPU bugs (like NVIDIA 16xx GPUs).
"""

COMPARABLE_GPU_PERCENTILE = (
    0.65  # if a GPU's free_mem is within this % of the GPU with the most free_mem, it will be picked
)

mem_free_threshold = 0


def get_device_delta(render_devices, active_devices):
    """
    render_devices: 'auto' or backends listed in `torchruntime.utils.SUPPORTED_BACKENDS`
    active_devices: [backends listed in `torchruntime.utils.SUPPORTED_BACKENDS`]
    """

    render_devices = render_devices or "auto"
    render_devices = [render_devices] if isinstance(render_devices, str) else render_devices

    # check for backend support
    validate_render_devices(render_devices)

    if "auto" in render_devices:
        render_devices = auto_pick_devices(active_devices)
        if "cpu" in render_devices:
            log.warn("WARNING: Could not find a compatible GPU. Using the CPU, but this will be very slow!")

    active_devices = set(active_devices)
    render_devices = set(render_devices)

    devices_to_start = render_devices - active_devices
    devices_to_stop = active_devices - render_devices

    return devices_to_start, devices_to_stop


def validate_render_devices(render_devices):
    supported_backends = ("auto",) + SUPPORTED_BACKENDS
    unsupported_render_devices = [d for d in render_devices if not d.lower().startswith(supported_backends)]

    if unsupported_render_devices:
        raise ValueError(
            f"Invalid render devices in config: {unsupported_render_devices}. Valid render devices: {supported_backends}"
        )


def auto_pick_devices(currently_active_devices):
    global mem_free_threshold

    torch_platform_name = get_installed_torch_platform()[0]

    if is_cpu_device(torch_platform_name):
        return [torch_platform_name]

    device_count = get_device_count()
    log.debug("Autoselecting GPU. Using most free memory.")
    devices = []
    for device_id in range(device_count):
        device_id = f"{torch_platform_name}:{device_id}" if device_count > 1 else torch_platform_name
        device = get_device(device_id)

        mem_free, mem_total = mem_get_info(device)
        mem_free /= float(10**9)
        mem_total /= float(10**9)
        device_name = get_device_name(device)
        log.debug(
            f"{device_id} detected: {device_name} - Memory (free/total): {round(mem_free, 2)}Gb / {round(mem_total, 2)}Gb"
        )
        devices.append({"device": device_id, "device_name": device_name, "mem_free": mem_free})

    devices.sort(key=lambda x: x["mem_free"], reverse=True)
    max_mem_free = devices[0]["mem_free"]
    curr_mem_free_threshold = COMPARABLE_GPU_PERCENTILE * max_mem_free
    mem_free_threshold = max(curr_mem_free_threshold, mem_free_threshold)

    # Auto-pick algorithm:
    # 1. Pick the top 75 percentile of the GPUs, sorted by free_mem.
    # 2. Also include already-running devices (GPU-only), otherwise their free_mem will
    #    always be very low (since their VRAM contains the model).
    #    These already-running devices probably aren't terrible, since they were picked in the past.
    #    Worst case, the user can restart the program and that'll get rid of them.
    devices = [
        x["device"] for x in devices if x["mem_free"] >= mem_free_threshold or x["device"] in currently_active_devices
    ]
    return devices


def device_init(context, device_id):
    context.device = device_id

    if is_cpu_device(context.torch_device):
        context.device_name = get_processor_name()
        context.half_precision = False
    else:
        context.device_name = get_device_name(context.torch_device)

        # Some graphics cards have bugs in their firmware that prevent image generation at half precision
        if needs_to_force_full_precision(context.device_name):
            log.warn(f"forcing full precision on this GPU, to avoid corrupted images. GPU: {context.device_name}")
            context.half_precision = False

    log.info(f'Setting {device_id} as active, with precision: {"half" if context.half_precision else "full"}')


def needs_to_force_full_precision(device_name):
    if "FORCE_FULL_PRECISION" in os.environ:
        return True

    return has_half_precision_bug(device_name.lower())


def get_max_vram_usage_level(device):
    "Expects a torch.device as the argument"

    if is_cpu_device(device):
        return "high"

    _, mem_total = mem_get_info(device)

    if mem_total < 0.001:  # probably a torch platform without a mem_get_info() implementation
        return "high"

    mem_total /= float(10**9)
    if mem_total < 4.5:
        return "low"
    elif mem_total < 6.5:
        return "balanced"

    return "high"


def get_processor_name():
    try:
        import subprocess

        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            if "/usr/sbin" not in os.environ["PATH"].split(os.pathsep):
                os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command, shell=True).decode().strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).strip()
    except:
        log.error(traceback.format_exc())
        return "cpu"

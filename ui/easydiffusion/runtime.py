"""
A runtime that runs on a specific device (in a thread).

It can run various tasks like image generation, image filtering, model merge etc by using that thread-local context.

This creates an `sdkit.Context` that's bound to the device specified while calling the `init()` function.
"""

from easydiffusion import device_manager
from easydiffusion.utils import log
from sdkit import Context
from sdkit.utils import get_device_usage

context = Context()  # thread-local
"""
runtime data (bound locally to this thread), for e.g. device, references to loaded models, optimization flags etc
"""


def init(device):
    """
    Initializes the fields that will be bound to this runtime's context, and sets the current torch device
    """
    context.stop_processing = False
    context.temp_images = {}
    context.partial_x_samples = None
    context.model_load_errors = {}
    context.enable_codeformer = True

    from easydiffusion import app

    app_config = app.getConfig()
    context.test_diffusers = app_config.get("use_v3_engine", True)

    log.info("Device usage during initialization:")
    get_device_usage(device, log_info=True, process_usage_only=False)

    device_manager.device_init(context, device)


def set_vram_optimizations(context: Context):
    from easydiffusion import app

    config = app.getConfig()
    vram_usage_level = config.get("vram_usage_level", "balanced")

    if vram_usage_level != context.vram_usage_level:
        context.vram_usage_level = vram_usage_level
        return True

    return False

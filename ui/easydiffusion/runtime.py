"""
(OUTDATED DOC)
A runtime that runs on a specific device (in a thread).

It can run various tasks like image generation, image filtering, model merge etc by using that thread-local context.

This creates an `sdkit.Context` that's bound to the device specified while calling the `init()` function.
"""

context = None


def init(device):
    """
    Initializes the fields that will be bound to this runtime's context, and sets the current torch device
    """

    global context

    from easydiffusion import device_manager
    from easydiffusion.backend_manager import backend
    from easydiffusion.app import getConfig

    context = backend.create_context()

    context.stop_processing = False
    context.temp_images = {}
    context.partial_x_samples = None
    context.model_load_errors = {}
    context.enable_codeformer = True

    device_manager.device_init(context, device)


def set_vram_optimizations(context):
    from easydiffusion.app import getConfig

    config = getConfig()
    vram_usage_level = config.get("vram_usage_level", "balanced")

    if hasattr(context, "vram_usage_level") and vram_usage_level != context.vram_usage_level:
        context.vram_usage_level = vram_usage_level
        return True

    return False

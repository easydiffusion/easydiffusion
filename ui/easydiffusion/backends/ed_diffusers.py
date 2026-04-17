from sdkit_common import (
    start_backend,
    stop_backend,
    install_backend,
    uninstall_backend,
    is_installed,
    create_sdkit_context,
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
    "name": "Diffusers Backend for Easy Diffusion v3",
    "version": (1, 0, 0),
    "type": "backend",
}


def create_context():
    return create_sdkit_context(use_diffusers=True)

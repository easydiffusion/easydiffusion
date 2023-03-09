import os
import socket
import sys
import json
import traceback
import logging
import shlex
import urllib
from rich.logging import RichHandler

from sdkit.utils import log as sdkit_log  # hack, so we can overwrite the log config

from easydiffusion import task_manager
from easydiffusion.utils import log

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%X",
    handlers=[RichHandler(markup=True, rich_tracebacks=False, show_time=False, show_level=False)],
)

SD_DIR = os.getcwd()

SD_UI_DIR = os.getenv("SD_UI_PATH", None)

CONFIG_DIR = os.path.abspath(os.path.join(SD_UI_DIR, "..", "scripts"))
MODELS_DIR = os.path.abspath(os.path.join(SD_DIR, "..", "models"))

USER_PLUGINS_DIR = os.path.abspath(os.path.join(SD_DIR, "..", "plugins"))
CORE_PLUGINS_DIR = os.path.abspath(os.path.join(SD_UI_DIR, "plugins"))

USER_UI_PLUGINS_DIR = os.path.join(USER_PLUGINS_DIR, "ui")
CORE_UI_PLUGINS_DIR = os.path.join(CORE_PLUGINS_DIR, "ui")
USER_SERVER_PLUGINS_DIR = os.path.join(USER_PLUGINS_DIR, "server")
UI_PLUGINS_SOURCES = ((CORE_UI_PLUGINS_DIR, "core"), (USER_UI_PLUGINS_DIR, "user"))

sys.path.append(os.path.dirname(SD_UI_DIR))
sys.path.append(USER_SERVER_PLUGINS_DIR)

OUTPUT_DIRNAME = "Stable Diffusion UI"  # in the user's home folder
PRESERVE_CONFIG_VARS = ["FORCE_FULL_PRECISION"]
TASK_TTL = 15 * 60  # Discard last session's task timeout
APP_CONFIG_DEFAULTS = {
    # auto: selects the cuda device with the most free memory, cuda: use the currently active cuda device.
    "render_devices": "auto",  # valid entries: 'auto', 'cpu' or 'cuda:N' (where N is a GPU index)
    "update_branch": "main",
    "ui": {
        "open_browser_on_start": True,
    },
}

IMAGE_EXTENSIONS = [".png", ".apng", ".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".jxl", ".gif", ".webp", ".avif", ".svg"]
CUSTOM_MODIFIERS_DIR = os.path.abspath(os.path.join(SD_DIR, "..", "modifiers"))
CUSTOM_MODIFIERS_PORTRAIT_EXTENSIONS=[".portrait", "_portrait", " portrait", "-portrait"]
CUSTOM_MODIFIERS_LANDSCAPE_EXTENSIONS=[".landscape", "_landscape", " landscape", "-landscape"]

def init():
    os.makedirs(USER_UI_PLUGINS_DIR, exist_ok=True)
    os.makedirs(USER_SERVER_PLUGINS_DIR, exist_ok=True)

    load_server_plugins()

    update_render_threads()


def getConfig(default_val=APP_CONFIG_DEFAULTS):
    try:
        config_json_path = os.path.join(CONFIG_DIR, "config.json")
        if not os.path.exists(config_json_path):
            config = default_val
        else:
            with open(config_json_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        if "net" not in config:
            config["net"] = {}
        if os.getenv("SD_UI_BIND_PORT") is not None:
            config["net"]["listen_port"] = int(os.getenv("SD_UI_BIND_PORT"))
        else:
            config["net"]["listen_port"] = 9000
        if os.getenv("SD_UI_BIND_IP") is not None:
            config["net"]["listen_to_network"] = os.getenv("SD_UI_BIND_IP") == "0.0.0.0"
        else:
            config["net"]["listen_to_network"] = True
        return config
    except Exception as e:
        log.warn(traceback.format_exc())
        return default_val


def setConfig(config):
    try:  # config.json
        config_json_path = os.path.join(CONFIG_DIR, "config.json")
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except:
        log.error(traceback.format_exc())

    try:  # config.bat
        config_bat_path = os.path.join(CONFIG_DIR, "config.bat")
        config_bat = []

        if "update_branch" in config:
            config_bat.append(f"@set update_branch={config['update_branch']}")

        config_bat.append(f"@set SD_UI_BIND_PORT={config['net']['listen_port']}")
        bind_ip = "0.0.0.0" if config["net"]["listen_to_network"] else "127.0.0.1"
        config_bat.append(f"@set SD_UI_BIND_IP={bind_ip}")

        # Preserve these variables if they are set
        for var in PRESERVE_CONFIG_VARS:
            if os.getenv(var) is not None:
                config_bat.append(f"@set {var}={os.getenv(var)}")

        if len(config_bat) > 0:
            with open(config_bat_path, "w", encoding="utf-8") as f:
                f.write("\n".join(config_bat))
    except:
        log.error(traceback.format_exc())

    try:  # config.sh
        config_sh_path = os.path.join(CONFIG_DIR, "config.sh")
        config_sh = ["#!/bin/bash"]

        if "update_branch" in config:
            config_sh.append(f"export update_branch={config['update_branch']}")

        config_sh.append(f"export SD_UI_BIND_PORT={config['net']['listen_port']}")
        bind_ip = "0.0.0.0" if config["net"]["listen_to_network"] else "127.0.0.1"
        config_sh.append(f"export SD_UI_BIND_IP={bind_ip}")

        # Preserve these variables if they are set
        for var in PRESERVE_CONFIG_VARS:
            if os.getenv(var) is not None:
                config_bat.append(f'export {var}="{shlex.quote(os.getenv(var))}"')

        if len(config_sh) > 1:
            with open(config_sh_path, "w", encoding="utf-8") as f:
                f.write("\n".join(config_sh))
    except:
        log.error(traceback.format_exc())


def save_to_config(ckpt_model_name, vae_model_name, hypernetwork_model_name, vram_usage_level):
    config = getConfig()
    if "model" not in config:
        config["model"] = {}

    config["model"]["stable-diffusion"] = ckpt_model_name
    config["model"]["vae"] = vae_model_name
    config["model"]["hypernetwork"] = hypernetwork_model_name

    if vae_model_name is None or vae_model_name == "":
        del config["model"]["vae"]
    if hypernetwork_model_name is None or hypernetwork_model_name == "":
        del config["model"]["hypernetwork"]

    config["vram_usage_level"] = vram_usage_level

    setConfig(config)


def update_render_threads():
    config = getConfig()
    render_devices = config.get("render_devices", "auto")
    active_devices = task_manager.get_devices()["active"].keys()

    log.debug(f"requesting for render_devices: {render_devices}")
    task_manager.update_render_threads(render_devices, active_devices)


def getUIPlugins():
    plugins = []

    for plugins_dir, dir_prefix in UI_PLUGINS_SOURCES:
        for file in os.listdir(plugins_dir):
            if file.endswith(".plugin.js"):
                plugins.append(f"/plugins/{dir_prefix}/{file}")

    return plugins


def load_server_plugins():
    if not os.path.exists(USER_SERVER_PLUGINS_DIR):
        return

    import importlib

    def load_plugin(file):
        mod_path = file.replace(".py", "")
        return importlib.import_module(mod_path)

    def apply_plugin(file, plugin):
        if hasattr(plugin, "get_cond_and_uncond"):
            import sdkit.generate.image_generator

            sdkit.generate.image_generator.get_cond_and_uncond = plugin.get_cond_and_uncond
            log.info(f"Overridden get_cond_and_uncond with the one in the server plugin: {file}")

    for file in os.listdir(USER_SERVER_PLUGINS_DIR):
        file_path = os.path.join(USER_SERVER_PLUGINS_DIR, file)
        if (not os.path.isdir(file_path) and not file_path.endswith("_plugin.py")) or (
            os.path.isdir(file_path) and not file_path.endswith("_plugin")
        ):
            continue

        try:
            log.info(f"Loading server plugin: {file}")
            mod = load_plugin(file)

            log.info(f"Applying server plugin: {file}")
            apply_plugin(file, mod)
        except:
            log.warn(f"Error while loading a server plugin")
            log.warn(traceback.format_exc())


def getIPConfig():
    try:
        ips = socket.gethostbyname_ex(socket.gethostname())
        ips[2].append(ips[0])
        return ips[2]
    except Exception as e:
        log.exception(e)
        return []


def open_browser():
    config = getConfig()
    ui = config.get("ui", {})
    net = config.get("net", {"listen_port": 9000})
    port = net.get("listen_port", 9000)
    if ui.get("open_browser_on_start", True):
        import webbrowser

        webbrowser.open(f"http://localhost:{port}")

def get_image_modifiers():
    modifiers_json_path = os.path.join(SD_UI_DIR, "modifiers.json")

    modifier_categories = {}
    original_category_order=[]
    with open(modifiers_json_path, "r", encoding="utf-8") as f:
        modifiers_file = json.load(f)

        # The trailing slash is needed to support symlinks
        if not os.path.isdir(f"{CUSTOM_MODIFIERS_DIR}/"):
            return modifiers_file

        # convert modifiers from a list of objects to a dict of dicts
        for category_item in modifiers_file:
            category_name = category_item['category']
            original_category_order.append(category_name)
            category = {}
            for modifier_item in category_item['modifiers']:
                modifier = {}
                for preview_item in modifier_item['previews']:
                    modifier[preview_item['name']] = preview_item['path']
                category[modifier_item['modifier']] = modifier
            modifier_categories[category_name] = category

    def scan_directory(directory_path: str, category_name="Modifiers"):
        for entry in os.scandir(directory_path):
            if entry.is_file():
                file_extension = list(filter(lambda e: entry.name.endswith(e), IMAGE_EXTENSIONS))
                if len(file_extension) == 0:
                    continue

                modifier_name = entry.name[: -len(file_extension[0])]
                modifier_path = f"custom/{entry.path[len(CUSTOM_MODIFIERS_DIR) + 1:]}"
                # URL encode path segments
                modifier_path = "/".join(map(lambda segment: urllib.parse.quote(segment), modifier_path.split("/")))
                is_portrait = True
                is_landscape = True

                portrait_extension = list(filter(lambda e: modifier_name.lower().endswith(e), CUSTOM_MODIFIERS_PORTRAIT_EXTENSIONS))
                landscape_extension = list(filter(lambda e: modifier_name.lower().endswith(e), CUSTOM_MODIFIERS_LANDSCAPE_EXTENSIONS))

                if len(portrait_extension) > 0:
                    is_landscape = False
                    modifier_name = modifier_name[: -len(portrait_extension[0])]
                elif len(landscape_extension) > 0:
                    is_portrait = False
                    modifier_name = modifier_name[: -len(landscape_extension[0])]
                
                if (category_name not in modifier_categories):
                    modifier_categories[category_name] = {}
                
                category = modifier_categories[category_name]

                if (modifier_name not in category):
                    category[modifier_name] = {}

                if (is_portrait or "portrait" not in category[modifier_name]):
                    category[modifier_name]["portrait"] = modifier_path
                
                if (is_landscape or "landscape" not in category[modifier_name]):
                    category[modifier_name]["landscape"] = modifier_path
            elif entry.is_dir():
                scan_directory(
                    entry.path,
                    entry.name if directory_path==CUSTOM_MODIFIERS_DIR else f"{category_name}/{entry.name}",
                )

    scan_directory(CUSTOM_MODIFIERS_DIR)

    custom_categories = sorted(
        [cn for cn in modifier_categories.keys() if cn not in original_category_order],
        key=str.casefold,
    )

    # convert the modifiers back into a list of objects
    modifier_categories_list = []
    for category_name in [*original_category_order, *custom_categories]:
        category = { 'category': category_name, 'modifiers': [] }
        for modifier_name in sorted(modifier_categories[category_name].keys(), key=str.casefold):
            modifier = { 'modifier': modifier_name, 'previews': [] }
            for preview_name, preview_path in modifier_categories[category_name][modifier_name].items():
                modifier['previews'].append({ 'name': preview_name, 'path': preview_path })
            category['modifiers'].append(modifier)
        modifier_categories_list.append(category)

    return modifier_categories_list

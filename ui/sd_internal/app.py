import os
import socket
import sys
import json
import traceback

from sd_internal import task_manager

SD_DIR = os.getcwd()

SD_UI_DIR = os.getenv('SD_UI_PATH', None)
sys.path.append(os.path.dirname(SD_UI_DIR))

CONFIG_DIR = os.path.abspath(os.path.join(SD_UI_DIR, '..', 'scripts'))
MODELS_DIR = os.path.abspath(os.path.join(SD_DIR, '..', 'models'))

USER_UI_PLUGINS_DIR = os.path.abspath(os.path.join(SD_DIR, '..', 'plugins', 'ui'))
CORE_UI_PLUGINS_DIR = os.path.abspath(os.path.join(SD_UI_DIR, 'plugins', 'ui'))
UI_PLUGINS_SOURCES = ((CORE_UI_PLUGINS_DIR, 'core'), (USER_UI_PLUGINS_DIR, 'user'))

STABLE_DIFFUSION_MODEL_EXTENSIONS = ['.ckpt', '.safetensors']
VAE_MODEL_EXTENSIONS = ['.vae.pt', '.ckpt']
HYPERNETWORK_MODEL_EXTENSIONS = ['.pt']

OUTPUT_DIRNAME = "Stable Diffusion UI" # in the user's home folder
TASK_TTL = 15 * 60 # Discard last session's task timeout
APP_CONFIG_DEFAULTS = {
    # auto: selects the cuda device with the most free memory, cuda: use the currently active cuda device.
    'render_devices': 'auto', # valid entries: 'auto', 'cpu' or 'cuda:N' (where N is a GPU index)
    'update_branch': 'main',
    'ui': {
        'open_browser_on_start': True,
    },
}
DEFAULT_MODELS = [
    # needed to support the legacy installations
    'custom-model', # Check if user has a custom model, use it first.
    'sd-v1-4', # Default fallback.
]

def init():
    os.makedirs(USER_UI_PLUGINS_DIR, exist_ok=True)

    update_render_threads()

def getConfig(default_val=APP_CONFIG_DEFAULTS):
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        if not os.path.exists(config_json_path):
            return default_val
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if 'net' not in config:
                config['net'] = {}
            if os.getenv('SD_UI_BIND_PORT') is not None:
                config['net']['listen_port'] = int(os.getenv('SD_UI_BIND_PORT'))
            if os.getenv('SD_UI_BIND_IP') is not None:
                config['net']['listen_to_network'] = (os.getenv('SD_UI_BIND_IP') == '0.0.0.0')
            return config
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return default_val

def setConfig(config):
    try: # config.json
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    except:
        print(traceback.format_exc())

    try: # config.bat
        config_bat_path = os.path.join(CONFIG_DIR, 'config.bat')
        config_bat = []

        if 'update_branch' in config:
            config_bat.append(f"@set update_branch={config['update_branch']}")

        config_bat.append(f"@set SD_UI_BIND_PORT={config['net']['listen_port']}")
        bind_ip = '0.0.0.0' if config['net']['listen_to_network'] else '127.0.0.1'
        config_bat.append(f"@set SD_UI_BIND_IP={bind_ip}")

        config_bat.append(f"@set test_sd2={'Y' if config.get('test_sd2', False) else 'N'}")

        if len(config_bat) > 0:
            with open(config_bat_path, 'w', encoding='utf-8') as f:
                f.write('\r\n'.join(config_bat))
    except:
        print(traceback.format_exc())

    try: # config.sh
        config_sh_path = os.path.join(CONFIG_DIR, 'config.sh')
        config_sh = ['#!/bin/bash']

        if 'update_branch' in config:
            config_sh.append(f"export update_branch={config['update_branch']}")

        config_sh.append(f"export SD_UI_BIND_PORT={config['net']['listen_port']}")
        bind_ip = '0.0.0.0' if config['net']['listen_to_network'] else '127.0.0.1'
        config_sh.append(f"export SD_UI_BIND_IP={bind_ip}")

        config_sh.append(f"export test_sd2=\"{'Y' if config.get('test_sd2', False) else 'N'}\"")

        if len(config_sh) > 1:
            with open(config_sh_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(config_sh))
    except:
        print(traceback.format_exc())

def save_model_to_config(ckpt_model_name, vae_model_name, hypernetwork_model_name):
    config = getConfig()
    if 'model' not in config:
        config['model'] = {}

    config['model']['stable-diffusion'] = ckpt_model_name
    config['model']['vae'] = vae_model_name
    config['model']['hypernetwork'] = hypernetwork_model_name

    if vae_model_name is None or vae_model_name == "":
        del config['model']['vae']
    if hypernetwork_model_name is None or hypernetwork_model_name == "":
        del config['model']['hypernetwork']

    setConfig(config)

def update_render_threads():
    config = getConfig()
    render_devices = config.get('render_devices', 'auto')
    active_devices = task_manager.get_devices()['active'].keys()

    print('requesting for render_devices', render_devices)
    task_manager.update_render_threads(render_devices, active_devices)

def getUIPlugins():
    plugins = []

    for plugins_dir, dir_prefix in UI_PLUGINS_SOURCES:
        for file in os.listdir(plugins_dir):
            if file.endswith('.plugin.js'):
                plugins.append(f'/plugins/{dir_prefix}/{file}')

    return plugins

def getIPConfig():
    ips = socket.gethostbyname_ex(socket.gethostname())
    ips[2].append(ips[0])
    return ips[2]

def open_browser():
    config = getConfig()
    ui = config.get('ui', {})
    net = config.get('net', {'listen_port':9000})
    port = net.get('listen_port', 9000)
    if ui.get('open_browser_on_start', True):
        import webbrowser; webbrowser.open(f"http://localhost:{port}")

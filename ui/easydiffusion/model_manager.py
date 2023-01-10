import os

from easydiffusion import app, device_manager
from easydiffusion.types import TaskData
from easydiffusion.utils import log

from sdkit import Context
from sdkit.models import load_model, unload_model, get_model_info_from_db, scan_model
from sdkit.utils import hash_file_quick

KNOWN_MODEL_TYPES = ['stable-diffusion', 'vae', 'hypernetwork', 'gfpgan', 'realesrgan']
MODEL_EXTENSIONS = {
    'stable-diffusion': ['.ckpt', '.safetensors'],
    'vae': ['.vae.pt', '.ckpt', '.safetensors'],
    'hypernetwork': ['.pt', '.safetensors'],
    'gfpgan': ['.pth'],
    'realesrgan': ['.pth'],
}
DEFAULT_MODELS = {
    'stable-diffusion': [ # needed to support the legacy installations
        'custom-model', # only one custom model file was supported initially, creatively named 'custom-model'
        'sd-v1-4', # Default fallback.
    ],
    'gfpgan': ['GFPGANv1.3'],
    'realesrgan': ['RealESRGAN_x4plus'],
}
VRAM_USAGE_LEVEL_TO_OPTIMIZATIONS = {
    'balanced': {'KEEP_FS_AND_CS_IN_CPU', 'SET_ATTENTION_STEP_TO_4'},
    'low': {'KEEP_ENTIRE_MODEL_IN_CPU'},
    'high': {},
}
MODELS_TO_LOAD_ON_START = ['stable-diffusion', 'vae', 'hypernetwork']

known_models = {}

def init():
    make_model_folders()
    getModels() # run this once, to cache the picklescan results

def load_default_models(context: Context):
    set_vram_optimizations(context)

    # init default model paths
    for model_type in MODELS_TO_LOAD_ON_START:
        context.model_paths[model_type] = resolve_model_to_use(model_type=model_type)
        set_model_config_path(context, model_type)
        load_model(context, model_type)

def unload_all(context: Context):
    for model_type in KNOWN_MODEL_TYPES:
        unload_model(context, model_type)

def resolve_model_to_use(model_name:str=None, model_type:str=None):
    model_extensions = MODEL_EXTENSIONS.get(model_type, [])
    default_models = DEFAULT_MODELS.get(model_type, [])
    config = app.getConfig()

    model_dirs = [os.path.join(app.MODELS_DIR, model_type), app.SD_DIR]
    if not model_name: # When None try user configured model.
        # config = getConfig()
        if 'model' in config and model_type in config['model']:
            model_name = config['model'][model_type]

    if model_name:
        # Check models directory
        models_dir_path = os.path.join(app.MODELS_DIR, model_type, model_name)
        for model_extension in model_extensions:
            if os.path.exists(models_dir_path + model_extension):
                return models_dir_path + model_extension
            if os.path.exists(model_name + model_extension):
                return os.path.abspath(model_name + model_extension)

    # Default locations
    if model_name in default_models:
        default_model_path = os.path.join(app.SD_DIR, model_name)
        for model_extension in model_extensions:
            if os.path.exists(default_model_path + model_extension):
                return default_model_path + model_extension

    # Can't find requested model, check the default paths.
    for default_model in default_models:
        for model_dir in model_dirs:
            default_model_path = os.path.join(model_dir, default_model)
            for model_extension in model_extensions:
                if os.path.exists(default_model_path + model_extension):
                    if model_name is not None:
                        log.warn(f'Could not find the configured custom model {model_name}{model_extension}. Using the default one: {default_model_path}{model_extension}')
                    return default_model_path + model_extension

    return None

def reload_models_if_necessary(context: Context, task_data: TaskData):
    model_paths_in_req = {
        'stable-diffusion': task_data.use_stable_diffusion_model,
        'vae': task_data.use_vae_model,
        'hypernetwork': task_data.use_hypernetwork_model,
        'gfpgan': task_data.use_face_correction,
        'realesrgan': task_data.use_upscale,
    }
    models_to_reload = {model_type: path for model_type, path in model_paths_in_req.items() if context.model_paths.get(model_type) != path}

    if set_vram_optimizations(context): # reload SD
        models_to_reload['stable-diffusion'] = model_paths_in_req['stable-diffusion']

    for model_type, model_path_in_req in models_to_reload.items():
        context.model_paths[model_type] = model_path_in_req
        set_model_config_path(context, model_type)

        action_fn = unload_model if context.model_paths[model_type] is None else load_model
        action_fn(context, model_type, scan_model=False) # we've scanned them already

def set_model_config_path(context: Context, model_type: str):
    if model_type != 'stable-diffusion':
        return

    context.model_configs['stable-diffusion'] = None # reset this, to avoid loading the last config

    # look for a yaml file next to the model, otherwise let sdkit match it to a known model
    model_path = context.model_paths['stable-diffusion']
    file_path, _ = os.path.splitext(model_path)
    config_path = file_path + '.yaml'
    if os.path.exists(config_path):
        context.model_configs['stable-diffusion'] = config_path

def resolve_model_paths(task_data: TaskData):
    task_data.use_stable_diffusion_model = resolve_model_to_use(task_data.use_stable_diffusion_model, model_type='stable-diffusion')
    task_data.use_vae_model = resolve_model_to_use(task_data.use_vae_model, model_type='vae')
    task_data.use_hypernetwork_model = resolve_model_to_use(task_data.use_hypernetwork_model, model_type='hypernetwork')

    if task_data.use_face_correction: task_data.use_face_correction = resolve_model_to_use(task_data.use_face_correction, 'gfpgan')
    if task_data.use_upscale: task_data.use_upscale = resolve_model_to_use(task_data.use_upscale, 'realesrgan')

def set_vram_optimizations(context: Context):
    config = app.getConfig()

    max_usage_level = device_manager.get_max_vram_usage_level(context.device)
    vram_usage_level = config.get('vram_usage_level', 'balanced')

    v = {'low': 0, 'balanced': 1, 'high': 2}
    if v[vram_usage_level] > v[max_usage_level]:
        log.error(f'Requested GPU Memory Usage level ({vram_usage_level}) is higher than what is ' + \
                  f'possible ({max_usage_level}) on this device ({context.device}). Using "{max_usage_level}" instead')
        vram_usage_level = max_usage_level

    vram_optimizations = VRAM_USAGE_LEVEL_TO_OPTIMIZATIONS[vram_usage_level]

    if vram_optimizations != context.vram_optimizations:
        context.vram_optimizations = vram_optimizations
        return True

    return False

def make_model_folders():
    for model_type in KNOWN_MODEL_TYPES:
        model_dir_path = os.path.join(app.MODELS_DIR, model_type)

        os.makedirs(model_dir_path, exist_ok=True)

        help_file_name = f'Place your {model_type} model files here.txt'
        help_file_contents = f'Supported extensions: {" or ".join(MODEL_EXTENSIONS.get(model_type))}'

        with open(os.path.join(model_dir_path, help_file_name), 'w', encoding='utf-8') as f:
            f.write(help_file_contents)

def is_malicious_model(file_path):
    try:
        scan_result = scan_model(file_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            log.warn(":warning: [bold red]Scan %s: %d scanned, %d issue, %d infected.[/bold red]" % (file_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files))
            return True
        else:
            log.debug("Scan %s: [green]%d scanned, %d issue, %d infected.[/green]" % (file_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files))
            return False
    except Exception as e:
        log.error(f'error while scanning: {file_path}, error: {e}')
    return False

def getModels():
    models = {
        'active': {
            'stable-diffusion': 'sd-v1-4',
            'vae': '',
            'hypernetwork': '',
        },
        'options': {
            'stable-diffusion': ['sd-v1-4'],
            'vae': [],
            'hypernetwork': [],
        },
    }

    models_scanned = 0

    class MaliciousModelException(Exception):
        "Raised when picklescan reports a problem with a model"
        pass

    def scan_directory(directory, suffixes):
        nonlocal models_scanned
        tree = []
        for entry in os.scandir(directory):
            if entry.is_file() and True in [entry.name.endswith(s) for s in suffixes]:
                mtime = entry.stat().st_mtime
                mod_time = known_models[entry.path] if entry.path in known_models else -1
                if mod_time != mtime:
                    models_scanned += 1
                    if is_malicious_model(entry.path):
                        raise MaliciousModelException(entry.path)
                known_models[entry.path] = mtime
                tree.append(entry.name.rsplit('.',1)[0])
            elif entry.is_dir():
                scan=scan_directory(entry.path, suffixes) 
                if len(scan) != 0:
                    tree.append( (entry.name, scan ) )
        return tree

    def listModels(model_type):
        nonlocal models_scanned

        model_extensions = MODEL_EXTENSIONS.get(model_type, [])
        models_dir = os.path.join(app.MODELS_DIR, model_type)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        try:
            models['options'][model_type] = scan_directory(models_dir, model_extensions)
        except MaliciousModelException as e:
            models['scan-error'] = e

    # custom models
    listModels(model_type='stable-diffusion')
    listModels(model_type='vae')
    listModels(model_type='hypernetwork')

    if models_scanned > 0: log.info(f'[green]Scanned {models_scanned} models. Nothing infected[/]')

    # legacy
    custom_weight_path = os.path.join(app.SD_DIR, 'custom-model.ckpt')
    if os.path.exists(custom_weight_path):
        models['options']['stable-diffusion'].append('custom-model')

    return models

import os

from sd_internal import app
import picklescan.scanner
import rich

default_model_to_load = None
default_vae_to_load = None
default_hypernetwork_to_load = None

known_models = {}

def init():
    global default_model_to_load, default_vae_to_load, default_hypernetwork_to_load

    default_model_to_load = resolve_ckpt_to_use()
    default_vae_to_load = resolve_vae_to_use()
    default_hypernetwork_to_load = resolve_hypernetwork_to_use()

    getModels() # run this once, to cache the picklescan results

def resolve_model_to_use(model_name:str, model_type:str, model_dir:str, model_extensions:list, default_models=[]):
    config = app.getConfig()

    model_dirs = [os.path.join(app.MODELS_DIR, model_dir), app.SD_DIR]
    if not model_name: # When None try user configured model.
        # config = getConfig()
        if 'model' in config and model_type in config['model']:
            model_name = config['model'][model_type]

    if model_name:
        is_sd2 = config.get('test_sd2', False)
        if model_name.startswith('sd2_') and not is_sd2: # temp hack, until SD2 is unified with 1.4
            print('ERROR: Cannot use SD 2.0 models with SD 1.0 code. Using the sd-v1-4 model instead!')
            model_name = 'sd-v1-4'

        # Check models directory
        models_dir_path = os.path.join(app.MODELS_DIR, model_dir, model_name)
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
                        print(f'Could not find the configured custom model {model_name}{model_extension}. Using the default one: {default_model_path}{model_extension}')
                    return default_model_path + model_extension

    raise Exception('No valid models found.')

def resolve_ckpt_to_use(model_name:str=None):
    return resolve_model_to_use(model_name, model_type='stable-diffusion', model_dir='stable-diffusion', model_extensions=app.STABLE_DIFFUSION_MODEL_EXTENSIONS, default_models=app.APP_CONFIG_DEFAULT_MODELS)

def resolve_vae_to_use(model_name:str=None):
    try:
        return resolve_model_to_use(model_name, model_type='vae', model_dir='vae', model_extensions=app.VAE_MODEL_EXTENSIONS, default_models=[])
    except:
        return None

def resolve_hypernetwork_to_use(model_name:str=None):
    try:
        return resolve_model_to_use(model_name, model_type='hypernetwork', model_dir='hypernetwork', model_extensions=app.HYPERNETWORK_MODEL_EXTENSIONS, default_models=[])
    except:
        return None

def is_malicious_model(file_path):
    try:
        scan_result = picklescan.scanner.scan_file_path(file_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            rich.print(":warning: [bold red]Scan %s: %d scanned, %d issue, %d infected.[/bold red]" % (file_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files))
            return True
        else:
            rich.print("Scan %s: [green]%d scanned, %d issue, %d infected.[/green]" % (file_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files))
            return False
    except Exception as e:
        print('error while scanning', file_path, 'error:', e)
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

    def listModels(models_dirname, model_type, model_extensions):
        models_dir = os.path.join(app.MODELS_DIR, models_dirname)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for file in os.listdir(models_dir):
            for model_extension in model_extensions:
                if not file.endswith(model_extension):
                    continue

                model_path = os.path.join(models_dir, file)
                mtime = os.path.getmtime(model_path)
                mod_time = known_models[model_path] if model_path in known_models else -1
                if mod_time != mtime:
                    if is_malicious_model(model_path):
                        models['scan-error'] = file
                        return
                known_models[model_path] = mtime

                model_name = file[:-len(model_extension)]
                models['options'][model_type].append(model_name)

        models['options'][model_type] = [*set(models['options'][model_type])] # remove duplicates
        models['options'][model_type].sort()

    # custom models
    listModels(models_dirname='stable-diffusion', model_type='stable-diffusion', model_extensions=app.STABLE_DIFFUSION_MODEL_EXTENSIONS)
    listModels(models_dirname='vae', model_type='vae', model_extensions=app.VAE_MODEL_EXTENSIONS)
    listModels(models_dirname='hypernetwork', model_type='hypernetwork', model_extensions=app.HYPERNETWORK_MODEL_EXTENSIONS)

    # legacy
    custom_weight_path = os.path.join(app.SD_DIR, 'custom-model.ckpt')
    if os.path.exists(custom_weight_path):
        models['options']['stable-diffusion'].append('custom-model')

    return models

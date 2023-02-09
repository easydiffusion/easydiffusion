import os
import torch
import traceback
import re

from easydiffusion.utils import log

'''
Set `FORCE_FULL_PRECISION` in the environment variables, or in `config.bat`/`config.sh` to set full precision (i.e. float32).
Otherwise the models will load at half-precision (i.e. float16).

Half-precision is fine most of the time. Full precision is only needed for working around GPU bugs (like NVIDIA 16xx GPUs).
'''

COMPARABLE_GPU_PERCENTILE = 0.65 # if a GPU's free_mem is within this % of the GPU with the most free_mem, it will be picked

mem_free_threshold = 0

def get_device_delta(render_devices, active_devices):
    '''
    render_devices: 'cpu', or 'auto' or ['cuda:N'...]
    active_devices: ['cpu', 'cuda:N'...]
    '''

    if render_devices in ('cpu', 'auto'):
        render_devices = [render_devices]
    elif render_devices is not None:
        if isinstance(render_devices, str):
            render_devices = [render_devices]
        if isinstance(render_devices, list) and len(render_devices) > 0:
            render_devices = list(filter(lambda x: x.startswith('cuda:'), render_devices))
            if len(render_devices) == 0:
                raise Exception('Invalid render_devices value in config.json. Valid: {"render_devices": ["cuda:0", "cuda:1"...]}, or {"render_devices": "cpu"} or {"render_devices": "auto"}')

            render_devices = list(filter(lambda x: is_device_compatible(x), render_devices))
            if len(render_devices) == 0:
                raise Exception('Sorry, none of the render_devices configured in config.json are compatible with Stable Diffusion')
        else:
            raise Exception('Invalid render_devices value in config.json. Valid: {"render_devices": ["cuda:0", "cuda:1"...]}, or {"render_devices": "cpu"} or {"render_devices": "auto"}')
    else:
        render_devices = ['auto']

    if 'auto' in render_devices:
        render_devices = auto_pick_devices(active_devices)
        if 'cpu' in render_devices:
            log.warn('WARNING: Could not find a compatible GPU. Using the CPU, but this will be very slow!')

    active_devices = set(active_devices)
    render_devices = set(render_devices)

    devices_to_start = render_devices - active_devices
    devices_to_stop = active_devices - render_devices

    return devices_to_start, devices_to_stop

def auto_pick_devices(currently_active_devices):
    global mem_free_threshold

    if not torch.cuda.is_available(): return ['cpu']

    device_count = torch.cuda.device_count()
    if device_count == 1:
        return ['cuda:0'] if is_device_compatible('cuda:0') else ['cpu']

    log.debug('Autoselecting GPU. Using most free memory.')
    devices = []
    for device in range(device_count):
        device = f'cuda:{device}'
        if not is_device_compatible(device):
            continue

        mem_free, mem_total = torch.cuda.mem_get_info(device)
        mem_free /= float(10**9)
        mem_total /= float(10**9)
        device_name = torch.cuda.get_device_name(device)
        log.debug(f'{device} detected: {device_name} - Memory (free/total): {round(mem_free, 2)}Gb / {round(mem_total, 2)}Gb')
        devices.append({'device': device, 'device_name': device_name, 'mem_free': mem_free})

    devices.sort(key=lambda x:x['mem_free'], reverse=True)
    max_mem_free = devices[0]['mem_free']
    curr_mem_free_threshold = COMPARABLE_GPU_PERCENTILE * max_mem_free
    mem_free_threshold = max(curr_mem_free_threshold, mem_free_threshold)

    # Auto-pick algorithm:
    # 1. Pick the top 75 percentile of the GPUs, sorted by free_mem.
    # 2. Also include already-running devices (GPU-only), otherwise their free_mem will
    #    always be very low (since their VRAM contains the model).
    #    These already-running devices probably aren't terrible, since they were picked in the past.
    #    Worst case, the user can restart the program and that'll get rid of them.
    devices = list(filter((lambda x: x['mem_free'] > mem_free_threshold or x['device'] in currently_active_devices), devices))
    devices = list(map(lambda x: x['device'], devices))
    return devices

def device_init(context, device):
    '''
    This function assumes the 'device' has already been verified to be compatible.
    `get_device_delta()` has already filtered out incompatible devices.
    '''

    validate_device_id(device, log_prefix='device_init')

    if device == 'cpu':
        context.device = 'cpu'
        context.device_name = get_processor_name()
        context.half_precision = False
        log.debug(f'Render device CPU available as {context.device_name}')
        return

    context.device_name = torch.cuda.get_device_name(device)
    context.device = device

    # Force full precision on 1660 and 1650 NVIDIA cards to avoid creating green images
    if needs_to_force_full_precision(context):
        log.warn(f'forcing full precision on this GPU, to avoid green images. GPU detected: {context.device_name}')
        # Apply force_full_precision now before models are loaded.
        context.half_precision = False

    log.info(f'Setting {device} as active, with precision: {"half" if context.half_precision else "full"}')
    torch.cuda.device(device)

    return

def needs_to_force_full_precision(context):
    if 'FORCE_FULL_PRECISION' in os.environ:
        return True

    device_name = context.device_name.lower()
    return (('nvidia' in device_name or 'geforce' in device_name or 'quadro' in device_name) and (' 1660' in device_name or ' 1650' in device_name or ' t400' in device_name or ' t500' in device_name or ' t550' in device_name or ' t600' in device_name or ' t1000' in device_name or ' t1200' in device_name or ' t2000' in device_name))

def get_max_vram_usage_level(device):
    if device != 'cpu':
        _, mem_total = torch.cuda.mem_get_info(device)
        mem_total /= float(10**9)

        if mem_total < 4.5:
            return 'low'
        elif mem_total < 6.5:
            return 'balanced'

    return 'high'

def validate_device_id(device, log_prefix=''):
    def is_valid():
        if not isinstance(device, str):
            return False
        if device == 'cpu':
            return True
        if not device.startswith('cuda:') or not device[5:].isnumeric():
            return False
        return True

    if not is_valid():
        raise EnvironmentError(f"{log_prefix}: device id should be 'cpu', or 'cuda:N' (where N is an integer index for the GPU). Got: {device}")

def is_device_compatible(device):
    '''
    Returns True/False, and prints any compatibility errors
    '''
    # static variable "history". 
    is_device_compatible.history = getattr(is_device_compatible, 'history', {})
    try:
        validate_device_id(device, log_prefix='is_device_compatible')
    except:
        log.error(str(e))
        return False

    if device == 'cpu': return True
    # Memory check
    try:
        _, mem_total = torch.cuda.mem_get_info(device)
        mem_total /= float(10**9)
        if mem_total < 3.0:
            if is_device_compatible.history.get(device) == None:
               log.warn(f'GPU {device} with less than 3 GB of VRAM is not compatible with Stable Diffusion')
               is_device_compatible.history[device] = 1
            return False
    except RuntimeError as e:
        log.error(str(e))
        return False
    return True

def get_processor_name():
    try:
        import platform, subprocess
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).strip()
    except:
        log.error(traceback.format_exc())
        return "cpu"

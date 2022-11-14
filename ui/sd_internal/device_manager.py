import os
import torch
import traceback
import re

COMPARABLE_GPU_PERCENTILE = 0.75 # if a GPU's free_mem is within this % of the GPU with the most free_mem, it will be picked

def get_device_delta(render_devices, active_devices):
    '''
    render_devices: 'cpu', or 'auto' or ['cuda:N'...]
    active_devices: ['cpu', 'cuda:N'...]
    '''

    if render_devices is not None:
        if render_devices in ('cpu', 'auto'):
            render_devices = [render_devices]
        elif isinstance(render_devices, list) and len(render_devices) > 0:
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
            print('WARNING: Could not find a compatible GPU. Using the CPU, but this will be very slow!')

    active_devices = set(active_devices)
    render_devices = set(render_devices)

    devices_to_start = render_devices - active_devices
    devices_to_stop = active_devices - render_devices

    return devices_to_start, devices_to_stop

def auto_pick_devices(currently_active_devices):
    if not torch.cuda.is_available(): return ['cpu']

    device_count = torch.cuda.device_count()
    if device_count == 1:
        return ['cuda:0'] if is_device_compatible('cuda:0') else ['cpu']

    print('Autoselecting GPU. Using most free memory.')
    devices = []
    for device in range(device_count):
        device = f'cuda:{device}'
        if not is_device_compatible(device):
            continue

        mem_free, mem_total = torch.cuda.mem_get_info(device)
        mem_free /= float(10**9)
        mem_total /= float(10**9)
        device_name = torch.cuda.get_device_name(device)
        print(f'{device} detected: {device_name} - Memory: {round(mem_total - mem_free, 2)}Gb / {round(mem_total, 2)}Gb')
        devices.append({'device': device, 'device_name': device_name, 'mem_free': mem_free})

    devices.sort(key=lambda x:x['mem_free'], reverse=True)
    max_free_mem = devices[0]['mem_free']
    free_mem_threshold = COMPARABLE_GPU_PERCENTILE * max_free_mem

    # Auto-pick algorithm:
    # 1. Pick the top 75 percentile of the GPUs, sorted by free_mem.
    # 2. Also include already-running devices (GPU-only), otherwise their free_mem will
    #    always be very low (since their VRAM contains the model).
    #    These already-running devices probably aren't terrible, since they were picked in the past.
    #    Worst case, the user can restart the program and that'll get rid of them.
    devices = list(filter((lambda x: x['mem_free'] > free_mem_threshold or x['device'] in currently_active_devices), devices))
    return devices

def device_init(thread_data, device):
    '''
    This function assumes the 'device' has already been verified to be compatible.
    `get_device_delta()` has already filtered out incompatible devices.
    '''

    validate_device_id(device, log_prefix='device_init')

    if device == 'cpu':
        thread_data.device = 'cpu'
        thread_data.device_name = get_processor_name()
        print('Render device CPU available as', thread_data.device_name)
        return

    thread_data.device_name = torch.cuda.get_device_name(device)
    thread_data.device = device

    # Force full precision on 1660 and 1650 NVIDIA cards to avoid creating green images
    device_name = thread_data.device_name.lower()
    thread_data.force_full_precision = ('nvidia' in device_name or 'geforce' in device_name) and (' 1660' in device_name or ' 1650' in device_name)
    if thread_data.force_full_precision:
        print('forcing full precision on NVIDIA 16xx cards, to avoid green images. GPU detected: ', thread_data.device_name)
        # Apply force_full_precision now before models are loaded.
        thread_data.precision = 'full'

    print(f'Setting {device} as active')
    torch.cuda.device(device)

    return

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
    validate_device_id(device, log_prefix='is_device_compatible')

    if device == 'cpu': return True
    # Memory check
    try:
        _, mem_total = torch.cuda.mem_get_info(device)
        mem_total /= float(10**9)
        if mem_total < 3.0:
            print(f'GPU {device} with less than 3 GB of VRAM is not compatible with Stable Diffusion')
            return False
    except RuntimeError as e:
        print(str(e))
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
        print(traceback.format_exc())
        return "cpu"

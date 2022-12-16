"""task_manager.py: manage tasks dispatching and render threads.
Notes:
    render_threads should be the only hard reference held by the manager to the threads.
    Use weak_thread_data to store all other data using weak keys.
    This will allow for garbage collection after the thread dies.
"""
import json
import traceback

TASK_TTL = 15 * 60 # seconds, Discard last session's task timeout

import torch
import queue, threading, time, weakref
from typing import Any, Generator, Hashable, Optional, Union

from pydantic import BaseModel
from sd_internal import Request, Response, runtime, device_manager

THREAD_NAME_PREFIX = 'Runtime-Render/'
ERR_LOCK_FAILED = ' failed to acquire lock within timeout.'
LOCK_TIMEOUT = 15 # Maximum locking time in seconds before failing a task.
# It's better to get an exception than a deadlock... ALWAYS use timeout in critical paths.

DEVICE_START_TIMEOUT = 60 # seconds - Maximum time to wait for a render device to init.

class SymbolClass(type): # Print nicely formatted Symbol names.
    def __repr__(self): return self.__qualname__
    def __str__(self): return self.__name__
class Symbol(metaclass=SymbolClass): pass

class ServerStates:
    class Init(Symbol): pass
    class LoadingModel(Symbol): pass
    class Online(Symbol): pass
    class Rendering(Symbol): pass
    class Unavailable(Symbol): pass

class RenderTask(): # Task with output queue and completion lock.
    def __init__(self, req: Request):
        req.request_id = id(self)
        self.request: Request = req  # Initial Request
        self.response: Any = None # Copy of the last reponse
        self.render_device = None # Select the task affinity. (Not used to change active devices).
        self.temp_images:list = [None] * req.num_outputs * (1 if req.show_only_filtered_image else 2)
        self.error: Exception = None
        self.lock: threading.Lock = threading.Lock() # Locks at task start and unlocks when task is completed
        self.buffer_queue: queue.Queue = queue.Queue() # Queue of JSON string segments
    async def read_buffer_generator(self):
        try:
            while not self.buffer_queue.empty():
                res = self.buffer_queue.get(block=False)
                self.buffer_queue.task_done()
                yield res
        except queue.Empty as e: yield
    @property
    def status(self):
        if self.lock.locked():
            return 'running'
        if isinstance(self.error, StopAsyncIteration):
            return 'stopped'
        if self.error:
            return 'error'
        if not self.buffer_queue.empty():
            return 'buffer'
        if self.response:
            return 'completed'
        return 'pending'
    @property
    def is_pending(self):
        return bool(not self.response and not self.error)

# defaults from https://huggingface.co/blog/stable_diffusion
class ImageRequest(BaseModel):
    session_id: str = "session"
    prompt: str = ""
    negative_prompt: str = ""
    init_image: str = None # base64
    mask: str = None # base64
    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int = 42
    prompt_strength: float = 0.8
    sampler: str = None # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"
    # allow_nsfw: bool = False
    save_to_disk_path: str = None
    turbo: bool = True
    use_cpu: bool = False ##TODO Remove after UI and plugins transition.
    render_device: str = None # Select the task affinity. (Not used to change active devices).
    use_full_precision: bool = False
    use_face_correction: str = None # or "GFPGANv1.3"
    use_upscale: str = None # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B"
    use_stable_diffusion_model: str = "sd-v1-4"
    use_vae_model: str = None
    use_hypernetwork_model: str = None
    hypernetwork_strength: float = None
    show_only_filtered_image: bool = False
    output_format: str = "jpeg" # or "png"
    output_quality: int = 75

    stream_progress_updates: bool = False
    stream_image_progress: bool = False

class FilterRequest(BaseModel):
    session_id: str = "session"
    model: str = None
    name: str = ""
    init_image: str = None # base64
    width: int = 512
    height: int = 512
    save_to_disk_path: str = None
    turbo: bool = True
    render_device: str = None
    use_full_precision: bool = False
    output_format: str = "jpeg" # or "png"
    output_quality: int = 75

# Temporary cache to allow to query tasks results for a short time after they are completed.
class DataCache():
    def __init__(self):
        self._base = dict()
        self._lock: threading.Lock = threading.Lock()
    def _get_ttl_time(self, ttl: int) -> int:
        return int(time.time()) + ttl
    def _is_expired(self, timestamp: int) -> bool:
        return int(time.time()) >= timestamp
    def clean(self) -> None:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('DataCache.clean' + ERR_LOCK_FAILED)
        try:
            # Create a list of expired keys to delete
            to_delete = []
            for key in self._base:
                ttl, _ = self._base[key]
                if self._is_expired(ttl):
                    to_delete.append(key)
            # Remove Items
            for key in to_delete:
                (_, val) = self._base[key]
                if isinstance(val, RenderTask):
                    print(f'RenderTask {key} expired. Data removed.')
                elif isinstance(val, SessionState):
                    print(f'Session {key} expired. Data removed.')
                else:
                    print(f'Key {key} expired. Data removed.')
                del self._base[key]
        finally:
            self._lock.release()
    def clear(self) -> None:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('DataCache.clear' + ERR_LOCK_FAILED)
        try: self._base.clear()
        finally: self._lock.release()
    def delete(self, key: Hashable) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('DataCache.delete' + ERR_LOCK_FAILED)
        try:
            if key not in self._base:
                return False
            del self._base[key]
            return True
        finally:
            self._lock.release()
    def keep(self, key: Hashable, ttl: int) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('DataCache.keep' + ERR_LOCK_FAILED)
        try:
            if key in self._base:
                _, value = self._base.get(key)
                self._base[key] = (self._get_ttl_time(ttl), value)
                return True
            return False
        finally:
            self._lock.release()
    def put(self, key: Hashable, value: Any, ttl: int) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('DataCache.put' + ERR_LOCK_FAILED)
        try:
            self._base[key] = (
                self._get_ttl_time(ttl), value
            )
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            return False
        else:
            return True
        finally:
            self._lock.release()
    def tryGet(self, key: Hashable) -> Any:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('DataCache.tryGet' + ERR_LOCK_FAILED)
        try:
            ttl, value = self._base.get(key, (None, None))
            if ttl is not None and self._is_expired(ttl):
                print(f'Session {key} expired. Discarding data.')
                del self._base[key]
                return None
            return value
        finally:
            self._lock.release()

manager_lock = threading.RLock()
render_threads = []
current_state = ServerStates.Init
current_state_error:Exception = None
current_model_path = None
current_vae_path = None
current_hypernetwork_path = None
tasks_queue = []
session_cache = DataCache()
task_cache = DataCache()
default_model_to_load = None
default_vae_to_load = None
default_hypernetwork_to_load = None
weak_thread_data = weakref.WeakKeyDictionary()
idle_event: threading.Event = threading.Event()

class SessionState():
    def __init__(self, id: str):
        self._id = id
        self._tasks_ids = []
    @property
    def id(self):
        return self._id
    @property
    def tasks(self):
        tasks = []
        for task_id in self._tasks_ids:
            task = task_cache.tryGet(task_id)
            if task:
                tasks.append(task)
        return tasks
    def put(self, task, ttl=TASK_TTL):
        task_id = id(task)
        self._tasks_ids.append(task_id)
        if not task_cache.put(task_id, task, ttl):
            return False
        while len(self._tasks_ids) > len(render_threads) * 2:
            self._tasks_ids.pop(0)
        return True

def preload_model(ckpt_file_path=None, vae_file_path=None, hypernetwork_file_path=None):
    global current_state, current_state_error, current_model_path, current_vae_path, current_hypernetwork_path
    if ckpt_file_path == None:
        ckpt_file_path = default_model_to_load
    if vae_file_path == None:
        vae_file_path = default_vae_to_load
    if hypernetwork_file_path == None:
        hypernetwork_file_path = default_hypernetwork_to_load
    if ckpt_file_path == current_model_path and vae_file_path == current_vae_path:
        return
    current_state = ServerStates.LoadingModel
    try:
        from . import runtime
        runtime.thread_data.hypernetwork_file = hypernetwork_file_path
        runtime.thread_data.ckpt_file = ckpt_file_path
        runtime.thread_data.vae_file = vae_file_path
        runtime.load_model_ckpt()
        runtime.load_hypernetwork()
        current_model_path = ckpt_file_path
        current_vae_path = vae_file_path
        current_hypernetwork_path = hypernetwork_file_path
        current_state_error = None
        current_state = ServerStates.Online
    except Exception as e:
        current_model_path = None
        current_vae_path = None
        current_state_error = e
        current_state = ServerStates.Unavailable
        print(traceback.format_exc())

def thread_get_next_task():
    from . import runtime
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
        print('Render thread on device', runtime.thread_data.device, 'failed to acquire manager lock.')
        return None
    if len(tasks_queue) <= 0:
        manager_lock.release()
        return None
    task = None
    try:  # Select a render task.
        for queued_task in tasks_queue:
            if queued_task.render_device and runtime.thread_data.device != queued_task.render_device:
                # Is asking for a specific render device.
                if is_alive(queued_task.render_device) > 0:
                    continue  # requested device alive, skip current one.
                else:
                    # Requested device is not active, return error to UI.
                    queued_task.error = Exception(queued_task.render_device + ' is not currently active.')
                    task = queued_task
                    break
            if not queued_task.render_device and runtime.thread_data.device == 'cpu' and is_alive() > 1:
                # not asking for any specific devices, cpu want to grab task but other render devices are alive.
                continue  # Skip Tasks, don't run on CPU unless there is nothing else or user asked for it.
            task = queued_task
            break
        if task is not None:
            del tasks_queue[tasks_queue.index(task)]
        return task
    finally:
        manager_lock.release()

def thread_render(device):
    global current_state, current_state_error, current_model_path, current_vae_path, current_hypernetwork_path
    from . import runtime
    try:
        runtime.thread_init(device)
    except Exception as e:
        print(traceback.format_exc())
        weak_thread_data[threading.current_thread()] = {
            'error': e
        }
        return
    weak_thread_data[threading.current_thread()] = {
        'device': runtime.thread_data.device,
        'device_name': runtime.thread_data.device_name,
        'alive': True
    }
    if runtime.thread_data.device != 'cpu' or is_alive() == 1:
        preload_model()
        current_state = ServerStates.Online
    while True:
        session_cache.clean()
        task_cache.clean()
        if not weak_thread_data[threading.current_thread()]['alive']:
            print(f'Shutting down thread for device {runtime.thread_data.device}')
            runtime.unload_models()
            runtime.unload_filters()
            return
        if isinstance(current_state_error, SystemExit):
            current_state = ServerStates.Unavailable
            return
        task = thread_get_next_task()
        if task is None:
            idle_event.clear()
            idle_event.wait(timeout=1)
            continue
        if task.error is not None:
            print(task.error)
            task.response = {"status": 'failed', "detail": str(task.error)}
            task.buffer_queue.put(json.dumps(task.response))
            continue
        if current_state_error:
            task.error = current_state_error
            task.response = {"status": 'failed', "detail": str(task.error)}
            task.buffer_queue.put(json.dumps(task.response))
            continue
        print(f'Session {task.request.session_id} starting task {id(task)} on {runtime.thread_data.device_name}')
        if not task.lock.acquire(blocking=False): raise Exception('Got locked task from queue.')
        try:
            if runtime.is_hypernetwork_reload_necessary(task.request):
                runtime.reload_hypernetwork()
                current_hypernetwork_path = task.request.use_hypernetwork_model
                
            if runtime.is_model_reload_necessary(task.request):
                current_state = ServerStates.LoadingModel
                runtime.reload_model()
                current_model_path = task.request.use_stable_diffusion_model
                current_vae_path = task.request.use_vae_model

            def step_callback():
                global current_state_error

                if isinstance(current_state_error, SystemExit) or isinstance(current_state_error, StopAsyncIteration) or isinstance(task.error, StopAsyncIteration):
                    runtime.thread_data.stop_processing = True
                    if isinstance(current_state_error, StopAsyncIteration):
                        task.error = current_state_error
                        current_state_error = None
                        print(f'Session {task.request.session_id} sent cancel signal for task {id(task)}')

            current_state = ServerStates.Rendering
            task.response = runtime.mk_img(task.request, task.buffer_queue, task.temp_images, step_callback)
            # Before looping back to the generator, mark cache as still alive.
            task_cache.keep(id(task), TASK_TTL)
            session_cache.keep(task.request.session_id, TASK_TTL)
        except Exception as e:
            task.error = e
            task.response = {"status": 'failed', "detail": str(task.error)}
            task.buffer_queue.put(json.dumps(task.response))
            print(traceback.format_exc())
            continue
        finally:
            # Task completed
            task.lock.release()
        task_cache.keep(id(task), TASK_TTL)
        session_cache.keep(task.request.session_id, TASK_TTL)
        if isinstance(task.error, StopAsyncIteration):
            print(f'Session {task.request.session_id} task {id(task)} cancelled!')
        elif task.error is not None:
            print(f'Session {task.request.session_id} task {id(task)} failed!')
        else:
            print(f'Session {task.request.session_id} task {id(task)} completed by {runtime.thread_data.device_name}.')
        current_state = ServerStates.Online

def get_cached_task(task_id:str, update_ttl:bool=False):
    # By calling keep before tryGet, wont discard if was expired.
    if update_ttl and not task_cache.keep(task_id, TASK_TTL):
        # Failed to keep task, already gone.
        return None
    return task_cache.tryGet(task_id)

def get_cached_session(session_id:str, update_ttl:bool=False):
    if update_ttl:
        session_cache.keep(session_id, TASK_TTL)
    session = session_cache.tryGet(session_id)
    if not session:
        session = SessionState(session_id)
        session_cache.put(session_id, session, TASK_TTL)
    return session

def get_devices():
    devices = {
        'all': {},
        'active': {},
    }

    def get_device_info(device):
        if device == 'cpu':
            return {'name': device_manager.get_processor_name()}
            
        mem_free, mem_total = torch.cuda.mem_get_info(device)
        mem_free /= float(10**9)
        mem_total /= float(10**9)

        return {
            'name': torch.cuda.get_device_name(device),
            'mem_free': mem_free,
            'mem_total': mem_total,
        }

    # list the compatible devices
    gpu_count = torch.cuda.device_count()
    for device in range(gpu_count):
        device = f'cuda:{device}'
        if not device_manager.is_device_compatible(device):
            continue

        devices['all'].update({device: get_device_info(device)})

    devices['all'].update({'cpu': get_device_info('cpu')})

    # list the activated devices
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('get_devices' + ERR_LOCK_FAILED)
    try:
        for rthread in render_threads:
            if not rthread.is_alive():
                continue
            weak_data = weak_thread_data.get(rthread)
            if not weak_data or not 'device' in weak_data or not 'device_name' in weak_data:
                continue
            device = weak_data['device']
            devices['active'].update({device: get_device_info(device)})
    finally:
        manager_lock.release()

    return devices

def is_alive(device=None):
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('is_alive' + ERR_LOCK_FAILED)
    nbr_alive = 0
    try:
        for rthread in render_threads:
            if device is not None:
                weak_data = weak_thread_data.get(rthread)
                if weak_data is None or not 'device' in weak_data or weak_data['device'] is None:
                    continue
                thread_device = weak_data['device']
                if thread_device != device:
                    continue
            if rthread.is_alive():
                nbr_alive += 1
        return nbr_alive
    finally:
        manager_lock.release()

def start_render_thread(device):
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('start_render_thread' + ERR_LOCK_FAILED)
    print('Start new Rendering Thread on device', device)
    try:
        rthread = threading.Thread(target=thread_render, kwargs={'device': device})
        rthread.daemon = True
        rthread.name = THREAD_NAME_PREFIX + device
        rthread.start()
        render_threads.append(rthread)
    finally:
        manager_lock.release()
    timeout = DEVICE_START_TIMEOUT
    while not rthread.is_alive() or not rthread in weak_thread_data or not 'device' in weak_thread_data[rthread]:
        if rthread in weak_thread_data and 'error' in weak_thread_data[rthread]:
            print(rthread, device, 'error:', weak_thread_data[rthread]['error'])
            return False
        if timeout <= 0:
            return False
        timeout -= 1
        time.sleep(1)
    return True

def stop_render_thread(device):
    try:
        device_manager.validate_device_id(device, log_prefix='stop_render_thread')
    except:
        print(traceback.format_exc())
        return False

    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('stop_render_thread' + ERR_LOCK_FAILED)
    print('Stopping Rendering Thread on device', device)

    try:
        thread_to_remove = None
        for rthread in render_threads:
            weak_data = weak_thread_data.get(rthread)
            if weak_data is None or not 'device' in weak_data or weak_data['device'] is None:
                continue
            thread_device = weak_data['device']
            if thread_device == device:
                weak_data['alive'] = False
                thread_to_remove = rthread
                break
        if thread_to_remove is not None:
            render_threads.remove(rthread)
            return True
    finally:
        manager_lock.release()

    return False

def update_render_threads(render_devices, active_devices):
    devices_to_start, devices_to_stop = device_manager.get_device_delta(render_devices, active_devices)
    print('devices_to_start', devices_to_start)
    print('devices_to_stop', devices_to_stop)

    for device in devices_to_stop:
        if is_alive(device) <= 0:
            print(device, 'is not alive')
            continue
        if not stop_render_thread(device):
            print(device, 'could not stop render thread')

    for device in devices_to_start:
        if is_alive(device) >= 1:
            print(device, 'already registered.')
            continue
        if not start_render_thread(device):
            print(device, 'failed to start.')

    if is_alive() <= 0: # No running devices, probably invalid user config.
        raise EnvironmentError('ERROR: No active render devices! Please verify the "render_devices" value in config.json')

    print('active devices', get_devices()['active'])

def shutdown_event(): # Signal render thread to close on shutdown
    global current_state_error
    current_state_error = SystemExit('Application shutting down.')

def render(req : ImageRequest):
    current_thread_count = is_alive()
    if current_thread_count <= 0:  # Render thread is dead
        raise ChildProcessError('Rendering thread has died.')

    # Alive, check if task in cache
    session = get_cached_session(req.session_id, update_ttl=True)
    pending_tasks = list(filter(lambda t: t.is_pending, session.tasks))
    if current_thread_count < len(pending_tasks):
        raise ConnectionRefusedError(f'Session {req.session_id} already has {len(pending_tasks)} pending tasks out of {current_thread_count}.')

    from . import runtime
    r = Request()
    r.session_id = req.session_id
    r.prompt = req.prompt
    r.negative_prompt = req.negative_prompt
    r.init_image = req.init_image
    r.mask = req.mask
    r.num_outputs = req.num_outputs
    r.num_inference_steps = req.num_inference_steps
    r.guidance_scale = req.guidance_scale
    r.width = req.width
    r.height = req.height
    r.seed = req.seed
    r.prompt_strength = req.prompt_strength
    r.sampler = req.sampler
    # r.allow_nsfw = req.allow_nsfw
    r.turbo = req.turbo
    r.use_full_precision = req.use_full_precision
    r.save_to_disk_path = req.save_to_disk_path
    r.use_upscale: str = req.use_upscale
    r.use_face_correction = req.use_face_correction
    r.use_stable_diffusion_model = req.use_stable_diffusion_model
    r.use_vae_model = req.use_vae_model
    r.use_hypernetwork_model = req.use_hypernetwork_model
    r.hypernetwork_strength = req.hypernetwork_strength
    r.show_only_filtered_image = req.show_only_filtered_image
    r.output_format = req.output_format
    r.output_quality = req.output_quality

    r.stream_progress_updates = True # the underlying implementation only supports streaming
    r.stream_image_progress = req.stream_image_progress

    if not req.stream_progress_updates:
        r.stream_image_progress = False

    new_task = RenderTask(r)
    if session.put(new_task, TASK_TTL):
        # Use twice the normal timeout for adding user requests.
        # Tries to force session.put to fail before tasks_queue.put would.
        if manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT * 2):
            try:
                tasks_queue.append(new_task)
                idle_event.set()
                return new_task
            finally:
                manager_lock.release()
    raise RuntimeError('Failed to add task to cache.')

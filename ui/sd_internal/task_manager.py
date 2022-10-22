"""task_manager.py: manage tasks dispatching and render threads.
Notes:
    render_threads should be the only hard reference held by the manager to the threads.
    Use weak_thread_data to store all other data using weak keys.
    This will allow for garbage collection after the thread dies.
"""
import json
import traceback

TASK_TTL = 15 * 60 # seconds, Discard last session's task timeout

import queue, threading, time, weakref
from typing import Any, Generator, Hashable, Optional, Union

from pydantic import BaseModel
from sd_internal import Request, Response

THREAD_NAME_PREFIX = 'Runtime-Render/'
ERR_LOCK_FAILED = ' failed to acquire lock within timeout.'
LOCK_TIMEOUT = 15 # Maximum locking time in seconds before failing a task.
# It's better to get an exception than a deadlock... ALWAYS use timeout in critical paths.

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
        self.request: Request = req # Initial Request
        self.response: Any = None # Copy of the last reponse
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
    use_cpu: bool = False
    use_full_precision: bool = False
    use_face_correction: str = None # or "GFPGANv1.3"
    use_upscale: str = None # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B"
    use_stable_diffusion_model: str = "sd-v1-4"
    show_only_filtered_image: bool = False
    output_format: str = "jpeg" # or "png"

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
    use_cpu: bool = False
    use_full_precision: bool = False
    output_format: str = "jpeg" # or "png"

# Temporary cache to allow to query tasks results for a short time after they are completed.
class TaskCache():
    def __init__(self):
        self._base = dict()
        self._lock: threading.Lock = threading.Lock()
    def _get_ttl_time(self, ttl: int) -> int:
        return int(time.time()) + ttl
    def _is_expired(self, timestamp: int) -> bool:
        return int(time.time()) >= timestamp
    def clean(self) -> None:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('TaskCache.clean' + ERR_LOCK_FAILED)
        try:
            # Create a list of expired keys to delete
            to_delete = []
            for key in self._base:
                ttl, _ = self._base[key]
                if self._is_expired(ttl):
                    to_delete.append(key)
            # Remove Items
            for key in to_delete:
                del self._base[key]
                print(f'Session {key} expired. Data removed.')
        finally:
            self._lock.release()
    def clear(self) -> None:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('TaskCache.clear' + ERR_LOCK_FAILED)
        try: self._base.clear()
        finally: self._lock.release()
    def delete(self, key: Hashable) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('TaskCache.delete' + ERR_LOCK_FAILED)
        try:
            if key not in self._base:
                return False
            del self._base[key]
            return True
        finally:
            self._lock.release()
    def keep(self, key: Hashable, ttl: int) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('TaskCache.keep' + ERR_LOCK_FAILED)
        try:
            if key in self._base:
                _, value = self._base.get(key)
                self._base[key] = (self._get_ttl_time(ttl), value)
                return True
            return False
        finally:
            self._lock.release()
    def put(self, key: Hashable, value: Any, ttl: int) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('TaskCache.put' + ERR_LOCK_FAILED)
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
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('TaskCache.tryGet' + ERR_LOCK_FAILED)
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
tasks_queue = []
task_cache = TaskCache()
default_model_to_load = None
weak_thread_data = weakref.WeakKeyDictionary()

def preload_model(file_path=None):
    global current_state, current_state_error, current_model_path
    if file_path == None:
        file_path = default_model_to_load
    if file_path == current_model_path:
        return
    current_state = ServerStates.LoadingModel
    try:
        from . import runtime
        runtime.thread_data.ckpt_file = file_path
        runtime.load_model_ckpt()
        current_model_path = file_path
        current_state_error = None
        current_state = ServerStates.Online
    except Exception as e:
        current_model_path = None
        current_state_error = e
        current_state = ServerStates.Unavailable
        print(traceback.format_exc())

def thread_get_next_task():
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
        print('Render thread on device', runtime.thread_data.device, 'failed to acquire manager lock.')
        return None
    if len(tasks_queue) <= 0:
        manager_lock.release()
        return None
    from . import runtime
    task = None
    try:  # Select a render task.
        for queued_task in tasks_queue:
            if queued_task.request.use_face_correction:  # TODO Remove when fixed - A bug with GFPGANer and facexlib needs to be fixed before use on other devices.
                if is_alive(0) <= 0:  # Allows GFPGANer only on cuda:0.
                    queued_task.error = Exception('cuda:0 is not available with the current config. Remove GFPGANer filter to run task.')
                    task = queued_task
                    break
                if queued_task.request.use_cpu:
                    queued_task.error = Exception('Cpu cannot be used to run this task. Remove GFPGANer filter to run task.')
                    task = queued_task
                    break
                if not runtime.is_first_cuda_device(runtime.thread_data.device):
                    continue  # Wait for cuda:0
            if queued_task.request.use_cpu and runtime.thread_data.device != 'cpu':
                if is_alive('cpu') > 0:
                    continue  # CPU Tasks, Skip GPU device
                else:
                    queued_task.error = Exception('Cpu is not enabled in render_devices.')
                    task = queued_task
                    break
            if not queued_task.request.use_cpu and runtime.thread_data.device == 'cpu':
                if is_alive() > 1:  # cpu is alive, so need more than one.
                    continue  # GPU Tasks, don't run on CPU unless there is nothing else.
                else:
                    queued_task.error = Exception('No active gpu found. Please check the error message in the command-line window at startup.')
                    task = queued_task
                    break
            task = queued_task
            break
        if task is not None:
            del tasks_queue[tasks_queue.index(task)]
        return task
    finally:
        manager_lock.release()

def thread_render(device):
    global current_state, current_state_error, current_model_path
    from . import runtime
    weak_thread_data[threading.current_thread()] = {
        'device': device
    }
    try:
        runtime.device_init(device)
    except:
        print(traceback.format_exc())
        return
    weak_thread_data[threading.current_thread()] = {
        'device': runtime.thread_data.device
    }
    if runtime.thread_data.device != 'cpu':
        preload_model()
    current_state = ServerStates.Online
    while True:
        task_cache.clean()
        if isinstance(current_state_error, SystemExit):
            current_state = ServerStates.Unavailable
            return
        task = thread_get_next_task()
        if task is None:
            time.sleep(1)
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
        print(f'Session {task.request.session_id} starting task {id(task)}')
        if not task.lock.acquire(blocking=False): raise Exception('Got locked task from queue.')
        try:
            # Open data generator.
            res = runtime.mk_img(task.request)
            if current_model_path == task.request.use_stable_diffusion_model:
                current_state = ServerStates.Rendering
            else:
                current_state = ServerStates.LoadingModel
            # Start reading from generator.
            dataQueue = None
            if task.request.stream_progress_updates:
                dataQueue = task.buffer_queue
            for result in res:
                if current_state == ServerStates.LoadingModel:
                    current_state = ServerStates.Rendering
                    current_model_path = task.request.use_stable_diffusion_model
                if isinstance(current_state_error, SystemExit) or isinstance(current_state_error, StopAsyncIteration) or isinstance(task.error, StopAsyncIteration):
                    runtime.thread_data.stop_processing = True
                    if isinstance(current_state_error, StopAsyncIteration):
                        task.error = current_state_error
                        current_state_error = None
                        print(f'Session {task.request.session_id} sent cancel signal for task {id(task)}')
                if dataQueue:
                    dataQueue.put(result)
                if isinstance(result, str):
                    result = json.loads(result)
                task.response = result
                if 'output' in result:
                    for out_obj in result['output']:
                        if 'path' in out_obj:
                            img_id = out_obj['path'][out_obj['path'].rindex('/') + 1:]
                            task.temp_images[int(img_id)] = runtime.thread_data.temp_images[out_obj['path'][11:]]
                        elif 'data' in out_obj:
                            task.temp_images[result['output'].index(out_obj)] = out_obj['data']
                # Before looping back to the generator, mark cache as still alive.
                task_cache.keep(task.request.session_id, TASK_TTL)
        except Exception as e:
            task.error = e
            print(traceback.format_exc())
            continue
        finally:
            # Task completed
            task.lock.release()
        task_cache.keep(task.request.session_id, TASK_TTL)
        if isinstance(task.error, StopAsyncIteration):
            print(f'Session {task.request.session_id} task {id(task)} cancelled!')
        elif task.error is not None:
            print(f'Session {task.request.session_id} task {id(task)} failed!')
        else:
            print(f'Session {task.request.session_id} task {id(task)} completed.')
        current_state = ServerStates.Online

def is_first_cuda_device(device):
    from . import runtime # When calling runtime from outside thread_render DO NOT USE thread specific attributes or functions.
    return runtime.is_first_cuda_device(device)

def is_alive(name=None):
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('is_alive' + ERR_LOCK_FAILED)
    nbr_alive = 0
    try:
        for rthread in render_threads:
            if name is not None:
                weak_data = weak_thread_data.get(rthread)
                if weak_data is None or weak_data['device'] is None:
                    print('The thread', rthread.name, 'is registered but has no data store in the task manager.')
                    continue
                thread_name = str(weak_data['device']).lower()
                if is_first_cuda_device(name):
                    if not is_first_cuda_device(thread_name):
                        continue
                elif thread_name != name:
                    continue
            if rthread.is_alive():
                nbr_alive += 1
        return nbr_alive
    finally:
        manager_lock.release()

def start_render_thread(device='auto'):
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception('start_render_threads' + ERR_LOCK_FAILED)
    print('Start new Rendering Thread on device', device)
    try:
        rthread = threading.Thread(target=thread_render, kwargs={'device': device})
        rthread.daemon = True
        rthread.name = THREAD_NAME_PREFIX + device
        rthread.start()
        timeout = LOCK_TIMEOUT
        while not rthread.is_alive():
            if timeout <= 0: raise Exception('render_thread', rthread.name, 'failed to start before timeout or has crashed.')
            timeout -= 1
            time.sleep(1)
        render_threads.append(rthread)
    finally:
        manager_lock.release()

def shutdown_event(): # Signal render thread to close on shutdown
    global current_state_error
    current_state_error = SystemExit('Application shutting down.')

def render(req : ImageRequest):
    if not is_alive(): # Render thread is dead
        raise ChildProcessError('Rendering thread has died.')
    # Alive, check if task in cache
    task = task_cache.tryGet(req.session_id)
    if task and not task.response and not task.error and not task.lock.locked():
        # Unstarted task pending, deny queueing more than one.
        raise ConnectionRefusedError(f'Session {req.session_id} has an already pending task.')
    #
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
    r.use_cpu = req.use_cpu
    r.use_full_precision = req.use_full_precision
    r.save_to_disk_path = req.save_to_disk_path
    r.use_upscale: str = req.use_upscale
    r.use_face_correction = req.use_face_correction
    r.use_stable_diffusion_model = req.use_stable_diffusion_model
    r.show_only_filtered_image = req.show_only_filtered_image
    r.output_format = req.output_format

    r.stream_progress_updates = True # the underlying implementation only supports streaming
    r.stream_image_progress = req.stream_image_progress

    if not req.stream_progress_updates:
        r.stream_image_progress = False

    new_task = RenderTask(r)
    if task_cache.put(r.session_id, new_task, TASK_TTL):
        # Use twice the normal timeout for adding user requests.
        # Tries to force task_cache.put to fail before tasks_queue.put would. 
        if manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT * 2):
            try:
                tasks_queue.append(new_task)
                return new_task
            finally:
                manager_lock.release()
    raise RuntimeError('Failed to add task to cache.')

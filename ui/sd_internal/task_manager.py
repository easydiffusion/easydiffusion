import json
import traceback

TASK_TTL = 15 * 60 # Discard last session's task timeout

import queue, threading, time
from typing import Any, Generator, Hashable, Optional, Union

from pydantic import BaseModel
from sd_internal import Request, Response

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
        self.temp_images:[] = [None] * req.num_outputs * (1 if req.show_only_filtered_image else 2)
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
        self._lock.acquire()
        try:
            for key in self._base:
                ttl, _ = self._base[key]
                if self._is_expired(ttl):
                    del self._base[key]
        finally:
            self._lock.release()
    def clear(self) -> None:
        self._lock.acquire()
        try: self._base.clear()
        finally: self._lock.release()
    def delete(self, key: Hashable) -> bool:
        self._lock.acquire()
        try:
            if key not in self._base:
                return False
            del self._base[key]
            return True
        finally:
            self._lock.release()
    def keep(self, key: Hashable, ttl: int) -> bool:
        self._lock.acquire()
        try:
            if key in self._base:
                _, value = self._base.get(key)
                self._base[key] = (self._get_ttl_time(ttl), value)
                return True
            return False
        finally:
            self._lock.release()
    def put(self, key: Hashable, value: Any, ttl: int) -> bool:
        self._lock.acquire()
        try:
            self._base[key] = (
                self._get_ttl_time(ttl), value
            )
        except Exception:
            return False
        else:
            return True
        finally:
            self._lock.release()
    def tryGet(self, key: Hashable) -> Any:
        self._lock.acquire()
        try:
            ttl, value = self._base.get(key, (None, None))
            if ttl is not None and self._is_expired(ttl):
                self.delete(key)
                return None
            return value
        finally:
            self._lock.release()

current_state = ServerStates.Init
current_state_error:Exception = None
current_model_path = None
tasks_queue = queue.Queue()
task_cache = TaskCache()
default_model_to_load = None

def preload_model(file_path=None):
    global current_state, current_state_error, current_model_path
    if file_path == None:
        file_path = default_model_to_load
    if file_path == current_model_path:
        return
    current_state = ServerStates.LoadingModel
    try:
        from . import runtime
        runtime.load_model_ckpt(ckpt_to_use=file_path)
        current_model_path = file_path
        current_state_error = None
        current_state = ServerStates.Online
    except Exception as e:
        current_model_path = None
        current_state_error = e
        current_state = ServerStates.Unavailable
        print(traceback.format_exc())

def thread_render():
    global current_state, current_state_error, current_model_path
    from . import runtime
    current_state = ServerStates.Online
    preload_model()
    while True:
        task_cache.clean()
        if isinstance(current_state_error, SystemExit):
            current_state = ServerStates.Unavailable
            return
        task = None
        try:
            task = tasks_queue.get(timeout=1)
        except queue.Empty as e:
            if isinstance(current_state_error, SystemExit):
                current_state = ServerStates.Unavailable
                return
            else: continue
        #if current_model_path != task.request.use_stable_diffusion_model:
        #    preload_model(task.request.use_stable_diffusion_model)
        if current_state_error:
            task.error = current_state_error
            continue
        print(f'Session {task.request.session_id} starting task {id(task)}')
        try:
            task.lock.acquire(blocking=False)
            res = runtime.mk_img(task.request)
            if current_model_path == task.request.use_stable_diffusion_model:
                current_state = ServerStates.Rendering
            else:
                current_state = ServerStates.LoadingModel
        except Exception as e:
            task.error = e
            task.lock.release()
            tasks_queue.task_done()
            print(traceback.format_exc())
            continue
        dataQueue = None
        if task.request.stream_progress_updates:
            dataQueue = task.buffer_queue
        for result in res:
            if current_state == ServerStates.LoadingModel:
                current_state = ServerStates.Rendering
                current_model_path = task.request.use_stable_diffusion_model
            if isinstance(current_state_error, SystemExit) or isinstance(current_state_error, StopAsyncIteration) or isinstance(task.error, StopAsyncIteration):
                runtime.stop_processing = True
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
                        task.temp_images[int(img_id)] = runtime.temp_images[out_obj['path'][11:]]
                    elif 'data' in out_obj:
                        task.temp_images[result['output'].index(out_obj)] = out_obj['data']
            task_cache.keep(task.request.session_id, TASK_TTL)
        # Task completed
        task.lock.release()
        tasks_queue.task_done()
        task_cache.keep(task.request.session_id, TASK_TTL)
        if isinstance(task.error, StopAsyncIteration):
            print(f'Session {task.request.session_id} task {id(task)} cancelled!')
        elif task.error is not None:
            print(f'Session {task.request.session_id} task {id(task)} failed!')
        else:
            print(f'Session {task.request.session_id} task {id(task)} completed.')
        current_state = ServerStates.Online

render_thread = threading.Thread(target=thread_render)

def start_render_thread():
    # Start Rendering Thread
    render_thread.daemon = True
    render_thread.start()

def shutdown_event(): # Signal render thread to close on shutdown
    global current_state_error
    current_state_error = SystemExit('Application shutting down.')

def render(req : ImageRequest):
    if not render_thread.is_alive(): # Render thread is dead
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
    r.show_only_filtered_image = req.show_only_filtered_image
    r.output_format = req.output_format

    r.stream_progress_updates = True # the underlying implementation only supports streaming
    r.stream_image_progress = req.stream_image_progress

    if not req.stream_progress_updates:
        r.stream_image_progress = False

    new_task = RenderTask(r)
    task_cache.put(r.session_id, new_task, TASK_TTL)
    tasks_queue.put(new_task)
    return new_task

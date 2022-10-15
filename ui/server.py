import json
import traceback

import sys
import os

SD_DIR = os.getcwd()
print('started in ', SD_DIR)

SD_UI_DIR = os.getenv('SD_UI_PATH', None)
sys.path.append(os.path.dirname(SD_UI_DIR))

CONFIG_DIR = os.path.abspath(os.path.join(SD_UI_DIR, '..', 'scripts'))
MODELS_DIR = os.path.abspath(os.path.join(SD_DIR, '..', 'models'))

OUTPUT_DIRNAME = "Stable Diffusion UI" # in the user's home folder
TASK_TTL = 15 * 60 # Discard last session's task timeout

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
import queue, threading, time
from typing import Any, Generator, Hashable, Optional, Union

from sd_internal import Request, Response

app = FastAPI()

modifiers_cache = None
outpath = os.path.join(os.path.expanduser("~"), OUTPUT_DIRNAME)

# don't show access log entries for URLs that start with the given prefix
ACCESS_LOG_SUPPRESS_PATH_PREFIXES = ['/ping', '/image', '/modifier-thumbnails']

NOCACHE_HEADERS={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
app.mount('/media', StaticFiles(directory=os.path.join(SD_UI_DIR, 'media/')), name="media")

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

current_state = ServerStates.Init
current_state_error:Exception = None
current_model_path = None
tasks_queue = queue.Queue()

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
    def _get_ttl_time(self, ttl: int) -> int:
        return int(time.time()) + ttl
    def _is_expired(self, timestamp: int) -> bool:
        return int(time.time()) >= timestamp
    def clean(self) -> None:
        for key in self._base:
            ttl, _ = self._base[key]
            if self._is_expired(ttl):
                del self._base[key]
    def clear(self) -> None:
        self._base.clear()
    def delete(self, key: Hashable) -> bool:
        if key not in self._base:
            return False
        del self._base[key]
        return True
    def keep(self, key: Hashable, ttl: int) -> bool:
        if key in self._base:
            _, value = self._base.get(key)
            self._base[key] = (self._get_ttl_time(ttl), value)
            return True
        return False
    def put(self, key: Hashable, value: Any, ttl: int) -> bool:
        try:
            self._base[key] = (
                self._get_ttl_time(ttl), value
            )
        except Exception:
            return False
        return True
    def tryGet(self, key: Hashable) -> Any:
        ttl, value = self._base.get(key, (None, None))
        if ttl is not None and self._is_expired(ttl):
            self.delete(key)
            return None
        return value

task_cache = TaskCache()

class SetAppConfigRequest(BaseModel):
    update_branch: str = "main"

@app.get('/')
def read_root():
    return FileResponse(os.path.join(SD_UI_DIR, 'index.html'), headers=NOCACHE_HEADERS)

def preload_model(file_path=None):
    global current_state, current_state_error, current_model_path
    if file_path == None:
        file_path = get_initial_model_to_load()
    if file_path == current_model_path:
        return
    current_state = ServerStates.LoadingModel
    try:
        from sd_internal import runtime
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
    from sd_internal import runtime
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
# Start Rendering Thread
render_thread = threading.Thread(target=thread_render)
render_thread.daemon = True
render_thread.start()

@app.on_event("shutdown")
def shutdown_event(): # Signal render thread to close on shutdown
    global current_state_error
    current_state_error = SystemExit('Application shutting down.')

# needs to support the legacy installations
def get_initial_model_to_load():
    custom_weight_path = os.path.join(SD_DIR, 'custom-model.ckpt')
    ckpt_to_use = "sd-v1-4" if not os.path.exists(custom_weight_path) else "custom-model"

    ckpt_to_use = os.path.join(SD_DIR, ckpt_to_use)

    config = getConfig()
    if 'model' in config and 'stable-diffusion' in config['model']:
        model_name = config['model']['stable-diffusion']
        model_path = resolve_model_to_use(model_name)

        if os.path.exists(model_path + '.ckpt'):
            ckpt_to_use = model_path
        else:
            print('Could not find the configured custom model at:', model_path + '.ckpt', '. Using the default one:', ckpt_to_use + '.ckpt')

    return ckpt_to_use

def resolve_model_to_use(model_name):
    if model_name in ('sd-v1-4', 'custom-model'):
        model_path = os.path.join(MODELS_DIR, 'stable-diffusion', model_name)

        legacy_model_path = os.path.join(SD_DIR, model_name)
        if not os.path.exists(model_path + '.ckpt') and os.path.exists(legacy_model_path + '.ckpt'):
            model_path = legacy_model_path
    else:
        model_path = os.path.join(MODELS_DIR, 'stable-diffusion', model_name)
    return model_path

def save_model_to_config(model_name):
    config = getConfig()
    if 'model' not in config:
        config['model'] = {}

    config['model']['stable-diffusion'] = model_name
    setConfig(config)

@app.get('/ping') # Get server and optionally session status.
def ping(session_id:str=None):
    if not render_thread.is_alive(): # Render thread is dead.
        if current_state_error: return HTTPException(status_code=500, detail=str(current_state_error))
        return HTTPException(status_code=500, detail='Render thread is dead.')
    if current_state_error and not isinstance(current_state_error, StopAsyncIteration): return HTTPException(status_code=500, detail=str(current_state_error))
    # Alive
    response = {'status': str(current_state)}
    if session_id:
        task = task_cache.tryGet(session_id)
        if task:
            response['task'] = id(task)
            if task.lock.locked():
                response['session'] = 'running'
            elif isinstance(task.error, StopAsyncIteration):
                response['session'] = 'stopped'
            elif task.error:
                response['session'] = 'error'
            elif not task.buffer_queue.empty():
                response['session'] = 'buffer'
            elif task.response:
                response['session'] = 'completed'
            else:
                response['session'] = 'pending'
    return JSONResponse(response, headers=NOCACHE_HEADERS)

@app.post('/render')
def render(req : ImageRequest):
    if not render_thread.is_alive(): # Render thread is dead
        return HTTPException(status_code=500, detail=f'Rendering thread has died.') # HTTP500 Internal Server Error
    # Alive, check if task in cache
    task = task_cache.tryGet(req.session_id)
    if task and not task.response and not task.error and not task.lock.locked(): # Unstarted task pending, deny queueing more than one.
        return HTTPException(status_code=503, detail=f'Session {req.session_id} has an already pending task.') # HTTP503 Service Unavailable
    #
    from sd_internal import runtime
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

    r.use_stable_diffusion_model = resolve_model_to_use(req.use_stable_diffusion_model)

    save_model_to_config(req.use_stable_diffusion_model)

    if not req.stream_progress_updates:
        r.stream_image_progress = False

    new_task = RenderTask(r)
    task_cache.put(r.session_id, new_task, TASK_TTL)
    tasks_queue.put(new_task)

    response = {
        'status': str(current_state), 
        'queue': tasks_queue.qsize(),
        'stream': f'/image/stream/{req.session_id}/{id(new_task)}',
        'task': id(new_task)
    }
    return JSONResponse(response, headers=NOCACHE_HEADERS)

async def read_data_generator(data:queue.Queue, lock:threading.Lock):
    try:
        while not data.empty():
            res = data.get(block=False)
            data.task_done()
            yield res
    except queue.Empty as e: yield

@app.get('/image/stream/{session_id:str}/{task_id:int}')
def stream(session_id:str, task_id:int):
    #TODO Move to WebSockets ??
    task = task_cache.tryGet(session_id)
    if not task: return HTTPException(status_code=410, detail='No request received.') # HTTP410 Gone
    if (id(task) != task_id): return HTTPException(status_code=409, detail=f'Wrong task id received. Expected:{id(task)}, Received:{task_id}') # HTTP409 Conflict
    if task.buffer_queue.empty() and not task.lock.locked():
        if task.response:
            #print(f'Session {session_id} sending cached response')
            return JSONResponse(task.response, headers=NOCACHE_HEADERS)
        return HTTPException(status_code=425, detail='Too Early, task not started yet.') # HTTP425 Too Early
    #print(f'Session {session_id} opened live render stream {id(task.buffer_queue)}')
    return StreamingResponse(read_data_generator(task.buffer_queue, task.lock), media_type='application/json')

@app.get('/image/stop')
def stop(session_id:str=None):
    if not session_id:
        if current_state == ServerStates.Online or current_state == ServerStates.Unavailable:
            return HTTPException(status_code=409, detail='Not currently running any tasks.') # HTTP409 Conflict
        global current_state_error
        current_state_error = StopAsyncIteration()
        return {'OK'}
    task = task_cache.tryGet(session_id)
    if not task: return HTTPException(status_code=404, detail=f'Session {session_id} has no active task.') # HTTP404 Not Found
    if isinstance(task.error, StopAsyncIteration): return HTTPException(status_code=409, detail=f'Session {session_id} task is already stopped.') # HTTP409 Conflict
    task.error = StopAsyncIteration('')
    return {'OK'}

@app.get('/image/tmp/{session_id}/{img_id:int}')
def get_image(session_id, img_id):
    task = task_cache.tryGet(session_id)
    if not task: return HTTPException(status_code=410, detail=f'Session {session_id} has not submitted a task.') # HTTP410 Gone
    if not task.temp_images[img_id]: return HTTPException(status_code=425, detail='Too Early, task data is not available yet.') # HTTP425 Too Early
    try:
        img_data = task.temp_images[img_id]
        if isinstance(img_data, str):
            return img_data
        img_data.seek(0)
        return StreamingResponse(img_data, media_type='image/jpeg')
    except KeyError as e:
        return HTTPException(status_code=500, detail=str(e))

@app.post('/app_config')
async def setAppConfig(req : SetAppConfigRequest):
    try:
        config = {
            'update_branch': req.update_branch
        }

        config_json_str = json.dumps(config)
        config_bat_str = f'@set update_branch={req.update_branch}'
        config_sh_str = f'export update_branch={req.update_branch}'

        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        config_bat_path = os.path.join(CONFIG_DIR, 'config.bat')
        config_sh_path = os.path.join(CONFIG_DIR, 'config.sh')

        with open(config_json_path, 'w') as f:
            f.write(config_json_str)

        with open(config_bat_path, 'w') as f:
            f.write(config_bat_str)

        with open(config_sh_path, 'w') as f:
            f.write(config_sh_str)

        return {'OK'}
    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))

def getConfig(default_val={}):
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        if not os.path.exists(config_json_path):
            return default_val
        with open(config_json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return default_val

def setConfig(config):
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        with open(config_json_path, 'w') as f:
            return json.dump(config, f)
    except:
        print(str(e))
        print(traceback.format_exc())

def getModels():
    models = {
        'active': {
            'stable-diffusion': 'sd-v1-4',
        },
        'options': {
            'stable-diffusion': ['sd-v1-4'],
        },
    }

    # custom models
    sd_models_dir = os.path.join(MODELS_DIR, 'stable-diffusion')
    for file in os.listdir(sd_models_dir):
        if file.endswith('.ckpt'):
            model_name = os.path.splitext(file)[0]
            models['options']['stable-diffusion'].append(model_name)

    # legacy
    custom_weight_path = os.path.join(SD_DIR, 'custom-model.ckpt')
    if os.path.exists(custom_weight_path):
        models['active']['stable-diffusion'] = 'custom-model'
        models['options']['stable-diffusion'].append('custom-model')

    config = getConfig()
    if 'model' in config and 'stable-diffusion' in config['model']:
        models['active']['stable-diffusion'] = config['model']['stable-diffusion']

    return models

@app.get('/get')
def read_web_data(key:str=None):
    if key is None: # /get without parameters, stable-diffusion easter egg.
        return HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!") # HTTP418 I'm a teapot
    elif key == 'app_config':
        config = getConfig(default_val=None)
        if config is None:
            return HTTPException(status_code=500, detail="Config file is missing or unreadable")
        return config
    elif key == 'models':
        return getModels()
    elif key == 'modifiers': return FileResponse(os.path.join(SD_UI_DIR, 'modifiers.json'), headers=NOCACHE_HEADERS)
    elif key == 'output_dir': return {outpath}
    else:
        return HTTPException(status_code=404, detail=f'Request for unknown {key}') # HTTP404 Not Found

# don't log certain requests
class LogSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        path = record.getMessage()
        for prefix in ACCESS_LOG_SUPPRESS_PATH_PREFIXES:
            if path.find(prefix) != -1:
                return False
        return True
logging.getLogger('uvicorn.access').addFilter(LogSuppressFilter())

# start the browser ui
import webbrowser; webbrowser.open('http://localhost:9000')
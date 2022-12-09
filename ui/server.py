"""server.py: FastAPI SD-UI Web Host.
Notes:
    async endpoints always run on the main thread. Without they run on the thread pool.
"""
import os
import traceback
import logging
from typing import List, Union

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from sd_internal import app, model_manager, task_manager

print('started in ', app.SD_DIR)

server_api = FastAPI()

# don't show access log entries for URLs that start with the given prefix
ACCESS_LOG_SUPPRESS_PATH_PREFIXES = ['/ping', '/image', '/modifier-thumbnails']
NOCACHE_HEADERS={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}

class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        if 'content-type' in response_headers and ('javascript' in response_headers['content-type'] or 'css' in response_headers['content-type']):
            response_headers.update(NOCACHE_HEADERS)
            return False

        return super().is_not_modified(response_headers, request_headers)

class SetAppConfigRequest(BaseModel):
    update_branch: str = None
    render_devices: Union[List[str], List[int], str, int] = None
    model_vae: str = None
    ui_open_browser_on_start: bool = None
    listen_to_network: bool = None
    listen_port: int = None
    test_sd2: bool = None

class LogSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        path = record.getMessage()
        for prefix in ACCESS_LOG_SUPPRESS_PATH_PREFIXES:
            if path.find(prefix) != -1:
                return False
        return True

# don't log certain requests
logging.getLogger('uvicorn.access').addFilter(LogSuppressFilter())

server_api.mount('/media', NoCacheStaticFiles(directory=os.path.join(app.SD_UI_DIR, 'media')), name="media")

for plugins_dir, dir_prefix in app.UI_PLUGINS_SOURCES:
    app.mount(f'/plugins/{dir_prefix}', NoCacheStaticFiles(directory=plugins_dir), name=f"plugins-{dir_prefix}")

@server_api.post('/app_config')
async def setAppConfig(req : SetAppConfigRequest):
    config = app.getConfig()
    if req.update_branch is not None:
        config['update_branch'] = req.update_branch
    if req.render_devices is not None:
        update_render_devices_in_config(config, req.render_devices)
    if req.ui_open_browser_on_start is not None:
        if 'ui' not in config:
            config['ui'] = {}
        config['ui']['open_browser_on_start'] = req.ui_open_browser_on_start
    if req.listen_to_network is not None:
       if 'net' not in config:
           config['net'] = {}
       config['net']['listen_to_network'] = bool(req.listen_to_network)
    if req.listen_port is not None:
       if 'net' not in config:
           config['net'] = {}
       config['net']['listen_port'] = int(req.listen_port)
    if req.test_sd2 is not None:
        config['test_sd2'] = req.test_sd2
    try:
        app.setConfig(config)

        if req.render_devices:
            app.update_render_threads()

        return JSONResponse({'status': 'OK'}, headers=NOCACHE_HEADERS)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def update_render_devices_in_config(config, render_devices):
    if render_devices not in ('cpu', 'auto') and not render_devices.startswith('cuda:'):
        raise HTTPException(status_code=400, detail=f'Invalid render device requested: {render_devices}')

    if render_devices.startswith('cuda:'):
        render_devices = render_devices.split(',')

    config['render_devices'] = render_devices

@server_api.get('/get/{key:path}')
def read_web_data(key:str=None):
    if not key: # /get without parameters, stable-diffusion easter egg.
        raise HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!") # HTTP418 I'm a teapot
    elif key == 'app_config':
        return JSONResponse(app.getConfig(), headers=NOCACHE_HEADERS)
    elif key == 'system_info':
        config = app.getConfig()
        system_info = {
            'devices': task_manager.get_devices(),
            'hosts': app.getIPConfig(),
            'default_output_dir': os.path.join(os.path.expanduser("~"), app.OUTPUT_DIRNAME),
        }
        system_info['devices']['config'] = config.get('render_devices', "auto")
        return JSONResponse(system_info, headers=NOCACHE_HEADERS)
    elif key == 'models':
        return JSONResponse(model_manager.getModels(), headers=NOCACHE_HEADERS)
    elif key == 'modifiers': return FileResponse(os.path.join(app.SD_UI_DIR, 'modifiers.json'), headers=NOCACHE_HEADERS)
    elif key == 'ui_plugins': return JSONResponse(app.getUIPlugins(), headers=NOCACHE_HEADERS)
    else:
        raise HTTPException(status_code=404, detail=f'Request for unknown {key}') # HTTP404 Not Found

@server_api.get('/ping') # Get server and optionally session status.
def ping(session_id:str=None):
    if task_manager.is_alive() <= 0: # Check that render threads are alive.
        if task_manager.current_state_error: raise HTTPException(status_code=500, detail=str(task_manager.current_state_error))
        raise HTTPException(status_code=500, detail='Render thread is dead.')
    if task_manager.current_state_error and not isinstance(task_manager.current_state_error, StopAsyncIteration): raise HTTPException(status_code=500, detail=str(task_manager.current_state_error))
    # Alive
    response = {'status': str(task_manager.current_state)}
    if session_id:
        session = task_manager.get_cached_session(session_id, update_ttl=True)
        response['tasks'] = {id(t): t.status for t in session.tasks}
    response['devices'] = task_manager.get_devices()
    return JSONResponse(response, headers=NOCACHE_HEADERS)

@server_api.post('/render')
def render(req : task_manager.ImageRequest):
    try:
        app.save_model_to_config(req.use_stable_diffusion_model, req.use_vae_model, req.use_hypernetwork_model)
        req.use_stable_diffusion_model = model_manager.resolve_sd_model_to_use(req.use_stable_diffusion_model)
        req.use_vae_model = model_manager.resolve_vae_model_to_use(req.use_vae_model)
        req.use_hypernetwork_model = model_manager.resolve_hypernetwork_model_to_use(req.use_hypernetwork_model)

        new_task = task_manager.render(req)
        response = {
            'status': str(task_manager.current_state), 
            'queue': len(task_manager.tasks_queue),
            'stream': f'/image/stream/{id(new_task)}',
            'task': id(new_task)
        }
        return JSONResponse(response, headers=NOCACHE_HEADERS)
    except ChildProcessError as e: # Render thread is dead
        raise HTTPException(status_code=500, detail=f'Rendering thread has died.') # HTTP500 Internal Server Error
    except ConnectionRefusedError as e: # Unstarted task pending limit reached, deny queueing too many.
        raise HTTPException(status_code=503, detail=str(e)) # HTTP503 Service Unavailable
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@server_api.get('/image/stream/{task_id:int}')
def stream(task_id:int):
    #TODO Move to WebSockets ??
    task = task_manager.get_cached_task(task_id, update_ttl=True)
    if not task: raise HTTPException(status_code=404, detail=f'Request {task_id} not found.') # HTTP404 NotFound
    #if (id(task) != task_id): raise HTTPException(status_code=409, detail=f'Wrong task id received. Expected:{id(task)}, Received:{task_id}') # HTTP409 Conflict
    if task.buffer_queue.empty() and not task.lock.locked():
        if task.response:
            #print(f'Session {session_id} sending cached response')
            return JSONResponse(task.response, headers=NOCACHE_HEADERS)
        raise HTTPException(status_code=425, detail='Too Early, task not started yet.') # HTTP425 Too Early
    #print(f'Session {session_id} opened live render stream {id(task.buffer_queue)}')
    return StreamingResponse(task.read_buffer_generator(), media_type='application/json')

@server_api.get('/image/stop')
def stop(task: int):
    if not task:
        if task_manager.current_state == task_manager.ServerStates.Online or task_manager.current_state == task_manager.ServerStates.Unavailable:
            raise HTTPException(status_code=409, detail='Not currently running any tasks.') # HTTP409 Conflict
        task_manager.current_state_error = StopAsyncIteration('')
        return {'OK'}
    task_id = task
    task = task_manager.get_cached_task(task_id, update_ttl=False)
    if not task: raise HTTPException(status_code=404, detail=f'Task {task_id} was not found.') # HTTP404 Not Found
    if isinstance(task.error, StopAsyncIteration): raise HTTPException(status_code=409, detail=f'Task {task_id} is already stopped.') # HTTP409 Conflict
    task.error = StopAsyncIteration(f'Task {task_id} stop requested.')
    return {'OK'}

@server_api.get('/image/tmp/{task_id:int}/{img_id:int}')
def get_image(task_id: int, img_id: int):
    task = task_manager.get_cached_task(task_id, update_ttl=True)
    if not task: raise HTTPException(status_code=410, detail=f'Task {task_id} could not be found.') # HTTP404 NotFound
    if not task.temp_images[img_id]: raise HTTPException(status_code=425, detail='Too Early, task data is not available yet.') # HTTP425 Too Early
    try:
        img_data = task.temp_images[img_id]
        img_data.seek(0)
        return StreamingResponse(img_data, media_type='image/jpeg')
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@server_api.get('/')
def read_root():
    return FileResponse(os.path.join(app.SD_UI_DIR, 'index.html'), headers=NOCACHE_HEADERS)

@server_api.on_event("shutdown")
def shutdown_event(): # Signal render thread to close on shutdown
    task_manager.current_state_error = SystemExit('Application shutting down.')

# Init the app
model_manager.init()
app.init()

# start the browser ui
app.open_browser()

"""
FastAPI server for EasyDiffusion.

Provides REST API endpoints for configuration management and task submission.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from typing import Optional

from easydiffusion.types import Task, ConfigUpdate, RenderRequest

# HTTP Headers for no-cache responses
NOCACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

# Create the FastAPI application
server_api = FastAPI(title="EasyDiffusion API", version="1.0.0")


def init():
    """Bind HTTP routes to internal handler functions.

    This mirrors the old server.py pattern so the API surface is easy to scan.
    """
    server_api.add_api_route("/", root, methods=["GET"], response_class=HTMLResponse)
    server_api.add_api_route("/ping", ping, methods=["GET"])
    server_api.add_api_route("/app_config", set_app_config, methods=["POST"])
    server_api.add_api_route("/get/{key:path}", read_web_data, methods=["GET"])
    server_api.add_api_route("/render", create_render_task, methods=["POST"])
    server_api.add_api_route("/image/stream/{task_id:int}", stream_task, methods=["GET"])
    server_api.add_api_route("/image/tmp/{task_id:int}/{img_id:int}", get_temp_image, methods=["GET"])
    server_api.add_api_route("/image/stop", stop_task, methods=["GET"])


async def root():
    """
    Serve a simple HTML page.

    Returns:
        Static HTML content
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EasyDiffusion</title>
    </head>
    <body>
        <h1>EasyDiffusion Server</h1>
        <p>Server is running.</p>
        <ul>
            <li><a href="/ping">Ping</a></li>
            <li><a href="/get/app_config">Configuration</a></li>
            <li><a href="/get/system_info">System Info</a></li>
            <li><a href="/get/models">Models</a></li>
            <li><a href="/get/modifiers">Modifiers</a></li>
            <li><a href="/get/ui_plugins">UI Plugins</a></li>
        </ul>
    </body>
    </html>
    """


async def ping():
    """
    Health check endpoint.

    Returns:
        Simple OK response
    """
    return {"status": "OK"}


async def set_app_config(req: ConfigUpdate):
    """
    Update application configuration (old API: /app_config).

    Args:
        req: Configuration updates

    Returns:
        Status OK response
    """
    try:
        config_manager = server_api.state.config_manager
        config = config_manager.get_all()

        # Track if render_devices changed
        devices_changed = False
        old_devices = config.get("render_devices", "auto")
        backend_changed = False

        # Update configuration fields
        if req.update_branch is not None:
            config["update_branch"] = req.update_branch
        if req.backend is not None:
            if config.get("backend") != req.backend:
                backend_changed = True
            config["backend"] = req.backend
            config["use_v3_engine"] = req.backend == "ed_diffusers"
        if req.render_devices is not None:
            if old_devices != req.render_devices:
                devices_changed = True
            config["render_devices"] = req.render_devices
        if req.model_vae is not None:
            config["model_vae"] = req.model_vae
        if req.ui_open_browser_on_start is not None:
            if "ui" not in config:
                config["ui"] = {}
            config["ui"]["open_browser_on_start"] = req.ui_open_browser_on_start
        if req.listen_to_network is not None:
            if "net" not in config:
                config["net"] = {}
            config["net"]["listen_to_network"] = bool(req.listen_to_network)
        if req.listen_port is not None:
            if "net" not in config:
                config["net"] = {}
            config["net"]["listen_port"] = int(req.listen_port)
        if req.use_v3_engine is not None:
            config["use_v3_engine"] = req.use_v3_engine
        if req.models_dir is not None:
            config["models_dir"] = req.models_dir
        if req.vram_usage_level is not None:
            config["vram_usage_level"] = req.vram_usage_level

        # Handle additional fields (extra="allow")
        req_dict = req.model_dump()
        defined_fields = set(ConfigUpdate.model_fields.keys())
        for key, value in req_dict.items():
            if value is not None and key not in defined_fields:
                config[key] = value

        config_manager.update(config)

        # Update workers if devices or backend changed
        if backend_changed:
            # Backend change - update backend and recreate workers
            server_api.state.worker_manager.set_backend(config["backend"], config.get("render_devices", "auto"))
        elif devices_changed:
            # Just update workers for device changes
            server_api.state.worker_manager.update_workers(config.get("render_devices", "auto"))

        return JSONResponse({"status": "OK"}, headers=NOCACHE_HEADERS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def read_web_data(key: str):
    """
    Get configuration and system data (old API: /get/{key}).

    Supported keys:
    - app_config: Application configuration
    - system_info: System information (placeholder)
    - models: Available models (placeholder)
    - modifiers: Image modifiers (placeholder)
    - ui_plugins: UI plugins (placeholder)

    Args:
        key: Data key to retrieve

    Returns:
        Requested data
    """
    if not key:
        # Easter egg from old implementation
        raise HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!")

    if key == "app_config":
        try:
            config_manager = server_api.state.config_manager
            config = config_manager.get_all()
            return JSONResponse(config, headers=NOCACHE_HEADERS)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif key == "system_info":
        # Placeholder - would need actual device/system info
        system_info = {
            "devices": {"all": {}, "active": {}, "config": "auto"},
            "hosts": [],
            "default_output_dir": ".",
            "enforce_output_dir": False,
            "enforce_output_metadata": False,
        }
        return JSONResponse(system_info, headers=NOCACHE_HEADERS)
    elif key == "models":
        # Placeholder - would need actual model list
        return JSONResponse({"models": {}}, headers=NOCACHE_HEADERS)
    elif key == "modifiers":
        # Placeholder - would need actual modifiers
        return JSONResponse([], headers=NOCACHE_HEADERS)
    elif key == "ui_plugins":
        # Placeholder - would need actual UI plugins
        return JSONResponse([], headers=NOCACHE_HEADERS)
    else:
        raise HTTPException(status_code=404, detail=f"Request for unknown {key}")


async def create_render_task(render_request: RenderRequest):
    """
    Create a new render task.

    Args:
        render_request: Render task parameters

    Returns:
        Task ID, status, queue position, and stream URL
    """
    try:
        task_queue = server_api.state.task_queue

        # Create a task with a unique ID
        task = Task(**render_request.model_dump())

        # Add to task queue
        task_queue.add_task(task)

        # Store task in cache for later retrieval
        if not hasattr(server_api.state, "task_cache"):
            server_api.state.task_cache = {}
        server_api.state.task_cache[task.id] = task

        # Return API-consistent response
        return {
            "task": task.id,  # Integer ID for compatibility
            "status": "queued",
            "queue": task_queue._queue.qsize(),
            "stream": f"/image/stream/{task.id}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_task(task_id: int):
    """
    Stream intermediate results from a task.

    Args:
        task_id: Integer task ID

    Returns:
        Streaming response with JSON progress updates
    """
    # Get task from cache
    if not hasattr(server_api.state, "task_cache"):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = server_api.state.task_cache.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # If task is complete and buffer is empty, return cached response
    if task.buffer_queue.empty() and not task.lock.locked():
        if task.response:
            return JSONResponse(task.response)
        raise HTTPException(status_code=425, detail="Too Early, task not started yet.")

    # Stream the buffer queue
    return StreamingResponse(task.read_buffer_generator(), media_type="application/json")


async def get_temp_image(task_id: int, img_id: int):
    """
    Get a temporary intermediate image from a task.

    Args:
        task_id: Integer task ID
        img_id: Image index in temp_images list

    Returns:
        Streaming response with image data
    """
    # Get task from cache
    if not hasattr(server_api.state, "task_cache"):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = server_api.state.task_cache.get(task_id)
    if not task:
        raise HTTPException(status_code=410, detail=f"Task {task_id} could not be found.")

    # Check if image exists
    if img_id >= len(task.temp_images) or not task.temp_images[img_id]:
        raise HTTPException(status_code=425, detail="Too Early, task data is not available yet.")

    try:
        img_data = task.temp_images[img_id]
        img_data.seek(0)
        return StreamingResponse(img_data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stop_task(task: Optional[int] = None):
    """
    Stop a running task or all tasks.

    Args:
        task: Integer task ID to stop, or None to stop all

    Returns:
        Success response
    """
    if not hasattr(server_api.state, "task_cache"):
        return {"OK": True}

    if task is None:
        # Stop all tasks - would need global state management
        return {"OK": True}

    # Stop specific task
    task_obj = server_api.state.task_cache.get(task)
    if not task_obj:
        raise HTTPException(status_code=404, detail=f"Task {task} was not found.")

    if isinstance(task_obj.error, StopAsyncIteration):
        raise HTTPException(status_code=409, detail=f"Task {task} is already stopped.")

    task_obj.error = StopAsyncIteration(f"Task {task} stop requested.")
    return {"OK": True}


# Register routes at import time so the server API is ready.
init()

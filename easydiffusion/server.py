"""
FastAPI server for EasyDiffusion.

Provides the REST API for image generation and configuration management.
"""

from fastapi import FastAPI, APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pathlib import Path
import os
import mimetypes
from typing import Optional

from easydiffusion.types import (
    HealthResponse,
    ConfigResponse,
    UserConfigResponse,
    UserSettingsConfig,
    DevicesResponse,
    DeviceInfo,
    ModelsResponse,
    ModelInfo,
    GenerateRequest,
    FilterRequest,
    TaskQueuedResponse,
    TasksResponse,
    TaskInfo,
    TaskDetail,
    StatusResponse,
    UserListResponse,
    CreateUserRequest,
    CreateUserResponse,
    DeleteUserResponse,
)
from easydiffusion.tasks import GenerateTask, FilterTask

# Create the FastAPI application
server_api = FastAPI(title="EasyDiffusion API", version="1.0.0")

v1_router = APIRouter(prefix="/v1")

NOCACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoCacheStaticFiles(StaticFiles):
    def __init__(self, directory: str):
        # follow_symlink is only available on fastapi >= 0.92.0
        if os.path.islink(directory):
            super().__init__(directory=os.path.realpath(directory))
        else:
            super().__init__(directory=directory)

    def is_not_modified(self, response_headers, request_headers) -> bool:
        if "content-type" in response_headers and (
            "javascript" in response_headers["content-type"] or "css" in response_headers["content-type"]
        ):
            response_headers.update(NOCACHE_HEADERS)
            return False

        return super().is_not_modified(response_headers, request_headers)


def init():
    """Initialize the server and bind all v1 API routes."""
    mimetypes.init()
    mimetypes.add_type("text/css", ".css")

    ui_dir = Path(__file__).parent.parent / "ui"

    server_api.mount(
        "/media",
        NoCacheStaticFiles(directory=str(ui_dir / "media")),
        name="media",
    )

    v1_router.add_api_route("/health", get_health, methods=["GET"])
    v1_router.add_api_route("/config", get_config, methods=["GET"], response_model=ConfigResponse)
    v1_router.add_api_route("/config", update_config, methods=["PUT"], response_model=StatusResponse)
    v1_router.add_api_route("/devices", get_devices, methods=["GET"], response_model=DevicesResponse)
    v1_router.add_api_route("/models", get_models, methods=["GET"], response_model=ModelsResponse)
    v1_router.add_api_route(
        "/generate", create_generate_task, methods=["POST"], response_model=TaskQueuedResponse, status_code=202
    )
    v1_router.add_api_route(
        "/filter", create_filter_task, methods=["POST"], response_model=TaskQueuedResponse, status_code=202
    )
    v1_router.add_api_route("/tasks", get_tasks, methods=["GET"], response_model=TasksResponse)
    v1_router.add_api_route("/tasks/{task_id}", get_task, methods=["GET"], response_model=TaskDetail)
    v1_router.add_api_route("/tasks/{task_id}/outputs/{output_id}", get_task_output, methods=["GET"])
    v1_router.add_api_route("/tasks/{task_id}", stop_task, methods=["DELETE"], response_model=StatusResponse)
    v1_router.add_api_route("/tasks", stop_all_tasks, methods=["DELETE"], response_model=StatusResponse)
    v1_router.add_api_route("/users", get_users, methods=["GET"], response_model=UserListResponse)
    v1_router.add_api_route("/users", create_user, methods=["POST"], response_model=CreateUserResponse, status_code=201)
    v1_router.add_api_route(
        "/users/{username}/config", get_user_config, methods=["GET"], response_model=UserConfigResponse
    )
    v1_router.add_api_route(
        "/users/{username}/config", update_user_config, methods=["PUT"], response_model=StatusResponse
    )
    v1_router.add_api_route("/users/{username}", delete_user, methods=["DELETE"], response_model=DeleteUserResponse)

    server_api.include_router(v1_router)

    @server_api.get("/")
    def read_root():
        ui_dir = Path(__file__).parent.parent / "ui"
        return FileResponse(str(ui_dir / "index.html"), headers=NOCACHE_HEADERS)


async def get_health():
    """Check server health."""
    content = HealthResponse(status="healthy", version="1.0.0").model_dump_json()
    return Response(content=content, media_type="application/json", headers=NOCACHE_HEADERS)


def _detect_media_type_from_bytes(data: bytes) -> Optional[str]:
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"BM", "image/bmp"),
        (b"II*\x00", "image/tiff"),
        (b"MM\x00*", "image/tiff"),
    )

    for prefix, media_type in signatures:
        if data.startswith(prefix):
            return media_type

    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"

    return None


def _read_output_payload(output_artifact: bytes) -> tuple[bytes, str]:
    """Return task output bytes and the inferred response MIME type."""
    if not isinstance(output_artifact, (bytes, bytearray)):
        raise HTTPException(status_code=500, detail="Invalid output data")

    data = bytes(output_artifact)
    return data, _detect_media_type_from_bytes(data) or "application/octet-stream"


async def get_config():
    """Retrieve the public system-wide configuration."""
    try:
        config_manager = server_api.state.config_manager
        response_data = config_manager.get_system_config()

        return JSONResponse(
            ConfigResponse(**response_data).model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def update_config(req: ConfigResponse):
    """Update the public system-wide configuration."""
    try:
        config_manager = server_api.state.config_manager
        config_manager.update(req.model_dump(exclude_none=True))

        return JSONResponse(
            StatusResponse(status="updated").model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_user_config(username: str):
    """Retrieve the effective user-specific configuration."""
    try:
        config_manager = server_api.state.config_manager

        return JSONResponse(
            UserConfigResponse(**config_manager.get_user_config(username)).model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def update_user_config(req: UserSettingsConfig, username: str):
    """Update user-specific configuration overrides."""
    try:
        config_manager = server_api.state.config_manager
        config_manager.update_user_config(username, req.model_dump(exclude_none=True))

        return JSONResponse(
            StatusResponse(status="updated").model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_devices():
    """List available devices."""
    try:
        from torchruntime.device_db import get_gpus

        devices = []
        devices.append(DeviceInfo(id="cpu", name="CPU", available=True))

        gpus = get_gpus()
        for idx, gpu in enumerate(gpus):
            devices.append(
                DeviceInfo(
                    id=str(idx),
                    name=gpu.device_name or f"GPU {idx}",
                    available=True,
                    vram_free=None,
                )
            )

        return JSONResponse(
            DevicesResponse(devices=devices).model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_models():
    """List available models."""
    try:
        from easydiffusion.local_models import enumerate_all_models

        config_manager = server_api.state.config_manager
        config = config_manager.get_system_config()
        models_dir = config.get("models", {}).get("models_dir", "models")

        all_models = enumerate_all_models(models_dir)
        models = [
            ModelInfo(
                model=m["model"],
                name=m["name"],
                tags=m["tags"],
            )
            for m in all_models
        ]

        return JSONResponse(
            ModelsResponse(models=models).model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def create_generate_task(req: GenerateRequest):
    """Enqueue image generation."""
    try:
        task_queue = server_api.state.task_queue

        task = GenerateTask(**req.model_dump())
        task.request_data = req.model_dump()

        task_queue.add_task(task)

        if not hasattr(server_api.state, "task_cache"):
            server_api.state.task_cache = {}
        server_api.state.task_cache[task.task_id] = task

        return JSONResponse(
            TaskQueuedResponse(
                task_id=task.task_id,
                status="queued",
                queue_position=task_queue.qsize(),
            ).model_dump(),
            headers=NOCACHE_HEADERS,
            status_code=202,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def create_filter_task(req: FilterRequest):
    """Enqueue image filtering."""
    try:
        task_queue = server_api.state.task_queue

        task = FilterTask(**req.model_dump())
        task.request_data = req.model_dump()

        task_queue.add_task(task)

        if not hasattr(server_api.state, "task_cache"):
            server_api.state.task_cache = {}
        server_api.state.task_cache[task.task_id] = task

        return JSONResponse(
            TaskQueuedResponse(
                task_id=task.task_id,
                status="queued",
                queue_position=task_queue.qsize(),
            ).model_dump(),
            headers=NOCACHE_HEADERS,
            status_code=202,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_tasks(username: Optional[str] = Query(None, description="Username to filter tasks by user")):
    """List all tasks. If username is provided, returns tasks for that user only; otherwise, returns tasks for all users."""
    try:
        if not hasattr(server_api.state, "task_cache"):
            server_api.state.task_cache = {}

        tasks = []
        for task in server_api.state.task_cache.values():
            if username is None or task.username == username:
                tasks.append(
                    TaskInfo(
                        task_id=task.task_id,
                        username=task.username,
                        status=task.status,
                        progress=task.progress,
                        type=task.task_type,
                    )
                )

        return JSONResponse(
            TasksResponse(tasks=tasks).model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_task(task_id: str, include_request: bool = Query(False)):
    """Get task status/results."""
    try:
        if not hasattr(server_api.state, "task_cache"):
            raise HTTPException(status_code=404, detail="Task not found")

        task = server_api.state.task_cache.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        outputs = [f"/v1/tasks/{task_id}/outputs/{i}" for i in range(len(task.outputs))]

        request_data = None
        if include_request and task.request_data:
            request_data = task.request_data

        return JSONResponse(
            TaskDetail(
                task_id=task.task_id,
                username=task.username,
                status=task.status,
                progress=task.progress,
                outputs=outputs,
                request=request_data,
            ).model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_task_output(task_id: str, output_id: int):
    """Retrieve an output artifact from a task."""
    try:
        if not hasattr(server_api.state, "task_cache"):
            raise HTTPException(status_code=404, detail="Task not found")

        task = server_api.state.task_cache.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if output_id >= len(task.outputs):
            raise HTTPException(status_code=404, detail="Output not found")

        output_data, media_type = _read_output_payload(task.outputs[output_id])
        return Response(content=output_data, media_type=media_type, headers=NOCACHE_HEADERS)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stop_task(task_id: str):
    """Stop a task."""
    try:
        if not hasattr(server_api.state, "task_cache"):
            raise HTTPException(status_code=404, detail="Task not found")

        task = server_api.state.task_cache.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status == "stopped":
            raise HTTPException(status_code=409, detail="Task already stopped")

        task.error = StopAsyncIteration(f"Task {task_id} stop requested")

        return JSONResponse(
            StatusResponse(status="stopped").model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stop_all_tasks(username: str = Query(..., description="Username for user-specific tasks")):
    """Stop all tasks for the specified user."""
    try:
        if not hasattr(server_api.state, "task_cache"):
            return JSONResponse(
                StatusResponse(status="all_stopped").model_dump(),
                headers=NOCACHE_HEADERS,
            )

        for task in server_api.state.task_cache.values():
            if task.username == username and task.status not in ["completed", "stopped", "error"]:
                task.error = StopAsyncIteration("All tasks stop requested")

        return JSONResponse(
            StatusResponse(status="all_stopped").model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_users():
    """List all configured users."""
    try:
        config_manager = server_api.state.config_manager
        users = config_manager.get_users()
        return JSONResponse(
            UserListResponse(users=users).model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def create_user(req: CreateUserRequest):
    """Create a new user."""
    try:
        if req.username.lower() == "default":
            raise HTTPException(status_code=400, detail="Cannot create user 'default'")

        config_manager = server_api.state.config_manager
        config_manager.add_user(req.username)
        return JSONResponse(
            CreateUserResponse(status="created", username=req.username).model_dump(),
            headers=NOCACHE_HEADERS,
            status_code=201,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def delete_user(username: str):
    """Delete a user and their settings."""
    try:
        if username.lower() == "default":
            raise HTTPException(status_code=400, detail="Cannot delete user 'default'")

        config_manager = server_api.state.config_manager
        config_manager.delete_user(username)
        return JSONResponse(
            DeleteUserResponse(status="deleted").model_dump(),
            headers=NOCACHE_HEADERS,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


init()

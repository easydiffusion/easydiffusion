"""Shared data types for the EasyDiffusion API."""

from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
import uuid
import queue
import threading


class Task:
    """Represents a render task with intermediate results storage."""

    def __init__(
        self, task_id: Optional[str] = None, session_id: Optional[str] = None, task_type: str = "generate", **kwargs
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.id = id(self)
        self.session_id = session_id or "default"
        self.task_type = task_type
        self.params = kwargs

        self.buffer_queue: queue.Queue = queue.Queue()
        self.temp_images: list = []
        self.lock: threading.Lock = threading.Lock()
        self.response: Optional[Any] = None
        self.error: Optional[Exception] = None
        self.render_device: Optional[str] = None
        self.progress: int = 0
        self.output_images: List[str] = []
        self.request_data: Optional[Dict[str, Any]] = None

    async def read_buffer_generator(self):
        try:
            while not self.buffer_queue.empty():
                try:
                    res = self.buffer_queue.get(block=False)
                    self.buffer_queue.task_done()
                    yield res
                except queue.Empty:
                    break
        except Exception:
            pass

    @property
    def status(self) -> str:
        if self.lock.locked():
            return "running"
        if isinstance(self.error, StopAsyncIteration):
            return "stopped"
        if self.error:
            return "error"
        if not self.buffer_queue.empty():
            return "buffer"
        if self.response:
            return "completed"
        return "pending"

    @property
    def is_pending(self) -> bool:
        return not self.response and not self.error

    def __repr__(self):
        return f"Task(task_id={self.task_id}, id={self.id}, status={self.status})"


# V1 API Models


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigResponse(BaseModel):
    render_devices: List[str]
    models_dir: str
    vram_usage_level: str
    backend: str


class ConfigUpdate(BaseModel):
    render_devices: Optional[List[str]] = None
    models_dir: Optional[str] = None
    vram_usage_level: Optional[str] = None


class DeviceInfo(BaseModel):
    id: str
    name: str
    available: bool
    vram_free: Optional[str] = None


class DevicesResponse(BaseModel):
    devices: List[DeviceInfo]


class ModelInfo(BaseModel):
    name: str
    type: str
    path: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: int = 42
    width: int = 512
    height: int = 512
    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    model: str
    output_format: str = "jpeg"
    save_path: Optional[str] = None


class FilterRequest(BaseModel):
    image: str
    filter: str
    filter_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    model: Optional[str] = None
    output_format: str = "jpeg"
    save_path: Optional[str] = None


class TaskQueuedResponse(BaseModel):
    task_id: str
    status: str
    queue_position: int


class TaskInfo(BaseModel):
    task_id: str
    status: str
    progress: int
    type: str


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]


class TaskDetail(BaseModel):
    task_id: str
    status: str
    progress: int
    images: List[str]
    request: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str

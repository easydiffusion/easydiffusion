"""Shared data types for the EasyDiffusion API."""

from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
from abc import abstractmethod
import uuid
import queue
import threading


class Task:
    """Represents a render task with intermediate results storage."""

    def __init__(
        self,
        username: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.id = id(self)
        self.session_id = session_id or "default"
        self.username = username
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

    @abstractmethod
    def run(self, backend: Any) -> Any:
        """
        Run the task using the provided backend.

        This method should be implemented by subclasses to define how the task
        interacts with the backend to perform its work.

        Args:
            backend: The backend instance to use for processing the task

        Returns:
            The result of processing the task
        """
        pass


# V1 API Models


class HealthResponse(BaseModel):
    status: str
    version: str


class NetworkConfig(BaseModel):
    port: int
    external_access: bool


class UpdatesConfig(BaseModel):
    branch: str


class BackendConfig(BaseModel):
    commandline_args: str
    force_full_precision: bool


class RenderingConfig(BaseModel):
    devices: str
    models_dir: str
    vram_usage_level: str
    block_nsfw: bool
    backend: str
    backend_config: BackendConfig


class SaveConfig(BaseModel):
    auto_save_images: Optional[bool] = None
    save_path: Optional[str] = None
    metadata_format: Optional[str] = None
    filename_format: Optional[str] = None


class UiConfig(BaseModel):
    theme: Optional[str] = None
    open_browser_on_start: Optional[bool] = None
    sound_enabled: Optional[bool] = None
    process_newest_first: Optional[bool] = None
    confirm_dangerous_actions: Optional[bool] = None
    auto_save_image_settings: Optional[bool] = None
    image_settings_auto_save_overrides: Optional[Dict[str, bool]] = None


class UserSettingsConfig(BaseModel):
    save: Optional[SaveConfig] = None
    ui: Optional[UiConfig] = None


class ConfigResponse(BaseModel):
    network: NetworkConfig
    updates: UpdatesConfig
    rendering: RenderingConfig
    user_settings: UserSettingsConfig


class ConfigUpdate(UserSettingsConfig):
    pass


class DeviceInfo(BaseModel):
    id: str
    name: str
    available: bool
    vram_free: Optional[str] = None


class DevicesResponse(BaseModel):
    devices: List[DeviceInfo]


class ModelInfo(BaseModel):
    model: str
    name: str
    tags: List[str]


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class GenerateRequest(BaseModel):
    username: str
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
    username: str
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
    username: str
    status: str
    progress: int
    type: str


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]


class TaskDetail(BaseModel):
    task_id: str
    username: str
    status: str
    progress: int
    images: List[str]
    request: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str


class UserListResponse(BaseModel):
    users: List[str]


class CreateUserRequest(BaseModel):
    username: str


class CreateUserResponse(BaseModel):
    status: str
    username: str


class DeleteUserResponse(BaseModel):
    status: str

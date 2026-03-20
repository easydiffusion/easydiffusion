"""Shared data types for the EasyDiffusion API."""

from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict, Union
from abc import abstractmethod
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "error"
    STOPPED = "stopped"


class Task:
    """Represents a task with output storage."""

    task_type = "task"

    def __init__(
        self,
        username: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.session_id = session_id or "default"
        self.username = username

        self._status = TaskStatus.PENDING
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.outputs: List[bytes] = []
        self.input: Optional[Dict[str, Any]] = input

    @property
    def status(self) -> str:
        return self._status.value

    @property
    def is_terminal(self) -> bool:
        return self._status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED}

    def mark_running(self) -> None:
        if not self.is_terminal:
            self._status = TaskStatus.RUNNING

    def update_progress(self, progress: float) -> None:
        self.progress = min(1.0, max(0.0, float(progress)))
        if self._status == TaskStatus.PENDING:
            self._status = TaskStatus.RUNNING

    def mark_completed(self) -> None:
        if self._status not in {TaskStatus.FAILED, TaskStatus.STOPPED}:
            if self.progress < 1.0:
                self.progress = 1.0
            self._status = TaskStatus.COMPLETED

    def mark_failed(self, error: Any) -> None:
        self.error = str(error)
        self._status = TaskStatus.FAILED

    def request_stop(self, reason: str = "") -> None:
        if reason:
            self.error = reason
        self._status = TaskStatus.STOPPED

    @property
    def is_pending(self) -> bool:
        return self._status in {TaskStatus.PENDING, TaskStatus.RUNNING}

    def __repr__(self):
        return f"Task(task_id={self.task_id}, status={self.status})"

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
    port: Optional[int] = None
    external_access: Optional[bool] = None


class UpdatesConfig(BaseModel):
    branch: Optional[str] = None


class BackendOverridesConfig(BaseModel):
    commandline_args: Optional[str] = None
    force_full_precision: Optional[bool] = None


class ModelsConfig(BaseModel):
    models_dir: Optional[str] = None


class BackendSettingsConfig(BaseModel):
    devices: Optional[Union[str, List[str]]] = None
    vram_usage_level: Optional[str] = None
    backend_name: Optional[str] = None
    backend_config: Optional[BackendOverridesConfig] = None


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
    block_nsfw: Optional[bool] = None


class UserSettingsConfig(BaseModel):
    save: Optional[SaveConfig] = None
    ui: Optional[UiConfig] = None


class ConfigResponse(BaseModel):
    network: Optional[NetworkConfig] = None
    updates: Optional[UpdatesConfig] = None
    models: Optional[ModelsConfig] = None
    backend: Optional[BackendSettingsConfig] = None


class UserConfigResponse(UserSettingsConfig):
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


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: int = 42
    width: int = 512
    height: int = 512
    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    distilled_guidance_scale: float = 3.5
    init_image: Any = None
    init_image_mask: Any = None
    ref_images: Any = None
    control_image: Any = None
    control_alpha: Union[float, List[float], None] = None
    controlnet_filter: Optional[str] = None
    prompt_strength: float = 0.8
    preserve_init_image_color_profile: bool = False
    strict_mask_border: bool = False
    sampler_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    hypernetwork_strength: float = 0
    lora_alpha: Union[float, List[float]] = 0.0
    tiling: Optional[str] = None
    filters: List[str] = Field(default_factory=list)
    filter_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class FilterImageRequest(BaseModel):
    image: Any = None
    filter: Union[str, List[str], None] = None
    filter_params: Dict[str, Any] = Field(default_factory=dict)


class ModelsData(BaseModel):
    model_paths: Optional[Dict[str, Union[str, None, List[str]]]] = None
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class OutputFormatData(BaseModel):
    output_format: str = "jpeg"
    output_quality: int = 75
    output_lossless: bool = False


class SaveToDiskData(BaseModel):
    save_to_disk_path: Optional[str] = None
    metadata_output_format: str = "txt"


class TaskData(BaseModel):
    task_id: Optional[str] = None
    session_id: str = "default"
    username: Optional[str] = None


class RenderTaskData(TaskData):
    vram_usage_level: str = "balanced"
    enable_vae_tiling: bool = True
    show_only_filtered_image: bool = False
    block_nsfw: bool = False
    stream_image_progress: bool = False
    stream_image_progress_interval: int = 5
    clip_skip: bool = False


class GenerateTaskRequest(
    GenerateImageRequest,
    ModelsData,
    OutputFormatData,
    SaveToDiskData,
    RenderTaskData,
):
    pass


class FilterTaskRequest(FilterImageRequest, ModelsData, OutputFormatData, SaveToDiskData, TaskData):
    pass


class GenerateTaskInput(BaseModel):
    request: GenerateImageRequest
    models: ModelsData
    output: OutputFormatData
    save: SaveToDiskData
    task: RenderTaskData

    @classmethod
    def from_flat_request(cls, payload: GenerateTaskRequest, task_id: str, username: str) -> "GenerateTaskInput":
        data = payload.model_dump()
        task_data = dict(data)
        task_data["task_id"] = task_id
        task_data["username"] = username

        return cls(
            request=GenerateImageRequest(**data),
            models=ModelsData(**data),
            output=OutputFormatData(**data),
            save=SaveToDiskData(**data),
            task=RenderTaskData(**task_data),
        )


class FilterTaskInput(BaseModel):
    request: FilterImageRequest
    models: ModelsData
    output: OutputFormatData
    save: SaveToDiskData
    task: TaskData

    @classmethod
    def from_flat_request(cls, payload: FilterTaskRequest, task_id: str, username: str) -> "FilterTaskInput":
        data = payload.model_dump()
        task_data = dict(data)
        task_data["task_id"] = task_id
        task_data["username"] = username

        return cls(
            request=FilterImageRequest(**data),
            models=ModelsData(**data),
            output=OutputFormatData(**data),
            save=SaveToDiskData(**data),
            task=TaskData(**task_data),
        )


class TaskQueuedResponse(BaseModel):
    task_id: str
    status: str
    queue_position: int


class TaskInfo(BaseModel):
    task_id: str
    username: str
    status: str
    progress: float
    type: str


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]


class TaskInfoDetailed(TaskInfo):
    outputs: List[str]
    detail: Optional[str] = None
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

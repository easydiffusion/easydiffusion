"""Shared data types for the EasyDiffusion API."""

from pydantic import BaseModel
from typing import Any, Optional
import uuid
import queue
import threading


class Task:
    """Represents a render task with intermediate results storage."""

    def __init__(self, task_id: Optional[str] = None, session_id: Optional[str] = None, **kwargs):
        self.task_id = task_id or str(uuid.uuid4())
        self.id = id(self)
        self.session_id = session_id or "default"
        self.params = kwargs

        self.buffer_queue: queue.Queue = queue.Queue()
        self.temp_images: list = []
        self.lock: threading.Lock = threading.Lock()
        self.response: Optional[Any] = None
        self.error: Optional[Exception] = None
        self.render_device: Optional[str] = None

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


class ConfigUpdate(BaseModel):
    """Model for configuration updates."""

    model_config = {"extra": "allow"}

    update_branch: Optional[str] = None
    backend: Optional[str] = None
    render_devices: Optional[Any] = None
    model_vae: Optional[str] = None
    ui_open_browser_on_start: Optional[bool] = None
    listen_to_network: Optional[bool] = None
    listen_port: Optional[int] = None
    use_v3_engine: Optional[bool] = None
    models_dir: Optional[str] = None
    vram_usage_level: Optional[str] = None


class RenderRequest(BaseModel):
    """Model for render requests. Extend as needed."""

    pass

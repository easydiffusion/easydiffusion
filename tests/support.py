"""Shared test helpers for EasyDiffusion unit tests."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from torchruntime.device_db import GPU

from easydiffusion.backends import BACKEND_REGISTRY, Backend
from easydiffusion.types import Task


class DummyBackend(Backend):
    """Lightweight backend used by unit tests."""

    instances = []
    mock_generate_outputs = []
    mock_filter_outputs = []
    last_generate_input = None
    last_filter_input = None

    def __init__(self, device: GPU):
        super().__init__(device)
        self.initialized = True
        self.stop_called = False
        self.start_called = False
        self.start_thread_id = None
        self.tasks_processed = []
        self.lock = threading.Lock()
        type(self).instances.append(self)

    @classmethod
    def reset_mock_state(cls) -> None:
        cls.instances = []
        cls.mock_generate_outputs = []
        cls.mock_filter_outputs = []
        cls.last_generate_input = None
        cls.last_filter_input = None

    @classmethod
    def set_generate_outputs(cls, outputs) -> None:
        cls.mock_generate_outputs = list(outputs)

    @classmethod
    def set_filter_outputs(cls, outputs) -> None:
        cls.mock_filter_outputs = list(outputs)

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def is_installed(self) -> bool:
        return self.initialized

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict) -> None:
        return None

    def start(self) -> None:
        self.start_called = True
        self.start_thread_id = threading.current_thread().ident

    def stop(self) -> None:
        self.stop_called = True

    def ping(self, timeout: float = 1.0) -> bool:
        return True

    def process(self, data):
        with self.lock:
            self.tasks_processed.append(data)
        return f"Processed: {data}"

    def generate_images(self, task_input: dict) -> list[bytes]:
        type(self).last_generate_input = task_input
        with self.lock:
            self.tasks_processed.append(task_input)
        return list(type(self).mock_generate_outputs)

    def filter_images(self, task_input: dict) -> list[bytes]:
        type(self).last_filter_input = task_input
        with self.lock:
            self.tasks_processed.append(task_input)
        return list(type(self).mock_filter_outputs)

    def get_progress(self, task: Task) -> float:
        return task.progress

    def stop_task(self, task: Task) -> None:
        task.request_stop("Stopped by dummy backend")

    def render_image(self, context=None, **kwargs):
        return None

    @classmethod
    def list_controlnet_filters(cls) -> list[str]:
        return ["canny", "depth", "openpose", "scribble"]


class DummyBackendTask(Task):
    """Task that records which backend processed it."""

    def __init__(self, data, callback: Optional[Callable[["DummyBackendTask"], None]] = None):
        super().__init__(username="test-user")
        self.data = data
        self.callback = callback
        self.result = None
        self.backend_used = None

    def run(self, backend):
        self.backend_used = backend
        self.result = backend.process(self.data)
        if self.callback:
            self.callback(self)
        return self.result


def register_dummy_backend() -> tuple[str, type[DummyBackend], type[Backend] | None]:
    backend_name = "dummy"
    previous = BACKEND_REGISTRY.get(backend_name)
    DummyBackend.reset_mock_state()
    BACKEND_REGISTRY[backend_name] = DummyBackend
    return backend_name, DummyBackend, previous


def unregister_dummy_backend(backend_name: str, previous: type[Backend] | None) -> None:
    if previous is None:
        BACKEND_REGISTRY.pop(backend_name, None)
    else:
        BACKEND_REGISTRY[backend_name] = previous

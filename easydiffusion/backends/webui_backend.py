from .base import Backend
from torchruntime.device_db import GPU


class WebUIBackend(Backend):
    def __init__(self, device: GPU):
        super().__init__(device)
        self.context = None
        self._started = False

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def is_installed(self) -> bool:
        return self._started

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict) -> None:
        return None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def ping(self, timeout=1.0):
        pass

    def generate_images(self, task_input: dict) -> list[bytes]:
        return []

    def filter_images(self, task_input: dict) -> list[bytes]:
        return []

    def get_progress(self, task) -> float:
        return 0.0

    def stop_task(self, task) -> None:
        return None

    def render_image(self):
        pass

from .base import Backend
from torchruntime.device_db import GPU


class EDClassicBackend(Backend):
    def __init__(self, device: GPU, config=None):
        super().__init__(device, config=config)
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

    def generate(self, input: dict) -> list[bytes]:
        return []

    def filter(self, input: dict) -> list[bytes]:
        return []

    def get_progress(self) -> float:
        return 0.0

    def stop_task(self) -> None:
        return None

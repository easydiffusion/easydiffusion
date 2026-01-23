from .base import Backend
from torchruntime.device_db import GPU


class Sdkit3Backend(Backend):
    def __init__(self, device: GPU):
        super().__init__(device)
        self.context = None
        self._started = False

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def ping(self, timeout=1.0):
        pass

    def render_image(self):
        pass

from __future__ import annotations

import struct
import threading
import time
import zlib
from copy import deepcopy
from collections.abc import Callable
from typing import Any

from torchruntime.device_db import GPU

from .base import Backend


class TestBackend(Backend):
    """Configurable backend for browser integration and automated tests."""

    __test__ = False

    DEFAULT_IMAGE_SIZE = 64
    DEFAULT_GENERATE_STEP_DELAY_SECONDS = 0.2
    GENERATE_STEP_DELAY_SECONDS = DEFAULT_GENERATE_STEP_DELAY_SECONDS
    GENERATE_STEP_CALLBACK: Callable[[float], None] | None = None
    CONTROLNET_FILTERS = ["canny", "depth", "openpose", "scribble"]

    instances: list["TestBackend"] = []

    def __init__(self, device: GPU, config: dict[str, Any] | None = None):
        super().__init__(device, config=config)
        self.runtime_config: dict[str, Any] = {}
        self._progress = 0.0
        self._outputs: list[bytes] = []
        self._stop_requested = False
        self._started = False
        self.stop_called = False
        self.start_called = False
        self.start_thread_id = None
        self.tasks_processed: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        TestBackend.instances.append(self)

    @classmethod
    def reset_mock_state(cls) -> None:
        cls.instances = []

    @classmethod
    def list_controlnet_filters(cls) -> list[str]:
        return list(cls.CONTROLNET_FILTERS)

    def install(self) -> None:
        return None

    def uninstall(self) -> None:
        return None

    def is_installed(self) -> bool:
        return True

    def get_config(self) -> dict[str, Any]:
        return deepcopy(self.runtime_config)

    def set_config(self, config: dict[str, Any]) -> None:
        self.runtime_config = deepcopy(config or {})

    def start(self) -> None:
        self._started = True
        self.start_called = True
        self.start_thread_id = threading.current_thread().ident

    def stop(self) -> None:
        self._stop_requested = True
        self._started = False
        self.stop_called = True

    def ping(self, timeout: float = 1.0) -> bool:
        return True

    def generate(self, input: dict[str, Any]) -> list[bytes]:
        width = input["request"]["width"]
        height = input["request"]["height"]
        num_outputs = input["request"]["num_outputs"]
        num_inference_steps = int(input["request"]["num_inference_steps"])

        with self.lock:
            self._progress = 0.0
            self._outputs = []
            self._stop_requested = False

        for step_index in range(1, num_inference_steps):
            progress = step_index / num_inference_steps
            outputs = [
                self._build_gradient_png(
                    width,
                    height,
                    "generate",
                    image_index,
                    phase=step_index,
                )
                for image_index in range(num_outputs)
            ]

            with self.lock:
                if self._stop_requested:
                    return []
                self._progress = progress
                self._outputs = outputs

            if TestBackend.GENERATE_STEP_CALLBACK is not None:
                TestBackend.GENERATE_STEP_CALLBACK()

            time.sleep(TestBackend.GENERATE_STEP_DELAY_SECONDS)

        with self.lock:
            if self._stop_requested:
                return []
            self._progress = 1.0
            self._outputs = [
                self._build_gradient_png(
                    width,
                    height,
                    "generate",
                    image_index,
                    phase=num_inference_steps,
                )
                for image_index in range(num_outputs)
            ]

        return self._outputs

    def filter(self, input: dict[str, Any]) -> list[bytes]:
        with self.lock:
            self._progress = 0.0
            self._outputs = []
            self._stop_requested = False

        outputs = [self._build_gradient_png(512, 512, "filter", 0)]

        with self.lock:
            if self._stop_requested:
                return []
            self._progress = 1.0
            self._outputs = outputs

        return self._outputs

    def get_progress(self) -> float:
        with self.lock:
            return self._progress

    def get_progress_outputs(self) -> list[bytes]:
        with self.lock:
            return self._outputs

    def stop_task(self) -> None:
        with self.lock:
            self._stop_requested = True

    @classmethod
    def _build_gradient_png(
        cls,
        width: int,
        height: int,
        variant: str,
        image_index: int,
        phase: int = 0,
    ) -> bytes:
        width = min(width, cls.DEFAULT_IMAGE_SIZE)
        height = min(height, cls.DEFAULT_IMAGE_SIZE)

        def pixel(x: int, y: int) -> tuple[int, int, int]:
            x_ratio = x / max(1, width - 1)
            y_ratio = y / max(1, height - 1)
            offset = ((image_index + 1) * 29 + phase * 47) % 255

            if variant == "filter":
                red = int(255 * (1.0 - y_ratio))
                green = int(90 + 120 * x_ratio)
                blue = int((255 * y_ratio + offset) % 256)
            else:
                red = int((255 * x_ratio + offset) % 256)
                green = int(255 * y_ratio)
                blue = int(255 * (1.0 - x_ratio * 0.6 - y_ratio * 0.4))

            return red, green, blue

        return cls._encode_png(width, height, pixel)

    @staticmethod
    def _encode_png(width: int, height: int, pixel_fn) -> bytes:
        rows = bytearray()
        for y in range(height):
            rows.append(0)
            for x in range(width):
                rows.extend(pixel_fn(x, y))

        compressed = zlib.compress(bytes(rows), level=6)
        header = b"\x89PNG\r\n\x1a\n"
        ihdr = TestBackend._png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        idat = TestBackend._png_chunk(b"IDAT", compressed)
        iend = TestBackend._png_chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    @staticmethod
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

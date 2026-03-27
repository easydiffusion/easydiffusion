from __future__ import annotations

import struct
import threading
import time
import zlib
from copy import deepcopy
from typing import Any

from torchruntime.device_db import GPU

from .base import Backend


class TestBackend(Backend):
    """Configurable backend for browser integration and automated tests."""

    __test__ = False

    DEFAULT_IMAGE_SIZE = 512
    DEFAULT_PROGRESS_INTERVAL_SECONDS = 0.1
    DEFAULT_PROGRESS_STEPS = 8
    CONTROLNET_FILTERS = ["canny", "depth", "openpose", "scribble"]

    instances: list["TestBackend"] = []
    mock_generate_outputs: list[bytes] = []
    mock_filter_outputs: list[bytes] = []
    mock_progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS
    mock_progress_steps: int = DEFAULT_PROGRESS_STEPS
    last_generate_input: dict[str, Any] | None = None
    last_filter_input: dict[str, Any] | None = None

    def __init__(self, device: GPU, config: dict[str, Any] | None = None):
        super().__init__(device, config=config)
        self.runtime_config: dict[str, Any] = {}
        self._progress = 0.0
        self._stop_requested = False
        self._started = False
        self.stop_called = False
        self.start_called = False
        self.start_thread_id = None
        self.tasks_processed: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        type(self).instances.append(self)

    @classmethod
    def reset_mock_state(cls) -> None:
        cls.instances = []
        cls.mock_generate_outputs = []
        cls.mock_filter_outputs = []
        cls.mock_progress_interval_seconds = cls.DEFAULT_PROGRESS_INTERVAL_SECONDS
        cls.mock_progress_steps = cls.DEFAULT_PROGRESS_STEPS
        cls.last_generate_input = None
        cls.last_filter_input = None

    @classmethod
    def configure_mock_behavior(
        cls,
        *,
        progress_interval_seconds: float | None = None,
        progress_steps: int | None = None,
    ) -> None:
        if progress_interval_seconds is not None:
            cls.mock_progress_interval_seconds = max(0.0, float(progress_interval_seconds))
        if progress_steps is not None:
            cls.mock_progress_steps = max(1, int(progress_steps))

    @classmethod
    def set_generate_outputs(cls, outputs: list[bytes]) -> None:
        cls.mock_generate_outputs = [bytes(output) for output in outputs]

    @classmethod
    def set_filter_outputs(cls, outputs: list[bytes]) -> None:
        cls.mock_filter_outputs = [bytes(output) for output in outputs]

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
        type(self).last_generate_input = deepcopy(input)
        self._record_task(input)

        def build_outputs() -> list[bytes]:
            if type(self).mock_generate_outputs:
                return [bytes(output) for output in type(self).mock_generate_outputs]

            request = dict(input.get("request") or {})
            width = self._sanitize_dimension(request.get("width"), self.config.get("image_width"))
            height = self._sanitize_dimension(request.get("height"), self.config.get("image_height"))
            num_outputs = max(1, int(request.get("num_outputs") or 1))
            return [self._build_gradient_png(width, height, "generate", index) for index in range(num_outputs)]

        return self._run_operation(build_outputs)

    def filter(self, input: dict[str, Any]) -> list[bytes]:
        type(self).last_filter_input = deepcopy(input)
        self._record_task(input)

        def build_outputs() -> list[bytes]:
            if type(self).mock_filter_outputs:
                return [bytes(output) for output in type(self).mock_filter_outputs]

            width = self._sanitize_dimension(None, self.config.get("image_width"))
            height = self._sanitize_dimension(None, self.config.get("image_height"))
            return [self._build_gradient_png(width, height, "filter", 0)]

        return self._run_operation(build_outputs)

    def progress(self) -> float:
        with self.lock:
            return self._progress

    def stop_task(self) -> None:
        with self.lock:
            self._stop_requested = True

    def process(self, data: Any) -> str:
        with self.lock:
            self.tasks_processed.append({"data": data})
        return f"Processed: {data}"

    def _record_task(self, task_input: dict[str, Any]) -> None:
        with self.lock:
            self.tasks_processed.append(deepcopy(task_input))

    def _run_operation(self, output_factory) -> list[bytes]:
        with self.lock:
            self._progress = 0.0
            self._stop_requested = False

        for step in range(type(self).mock_progress_steps):
            with self.lock:
                if self._stop_requested:
                    return []
                self._progress = (step + 1) / type(self).mock_progress_steps

            time.sleep(type(self).mock_progress_interval_seconds)

        with self.lock:
            if self._stop_requested:
                return []

        outputs = output_factory()
        with self.lock:
            if self._stop_requested:
                return []
            self._progress = 1.0

        return outputs

    @classmethod
    def _sanitize_dimension(cls, requested: Any, fallback: Any) -> int:
        value = requested if requested is not None else fallback
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = cls.DEFAULT_IMAGE_SIZE
        return max(64, min(parsed, 2048))

    @classmethod
    def _build_gradient_png(cls, width: int, height: int, variant: str, image_index: int) -> bytes:
        def pixel(x: int, y: int) -> tuple[int, int, int]:
            x_ratio = x / max(1, width - 1)
            y_ratio = y / max(1, height - 1)
            offset = (image_index * 29) % 255

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

"""Shared test helpers for EasyDiffusion unit tests."""

from __future__ import annotations

import time

from easydiffusion.backends import BACKEND_REGISTRY, Backend
from easydiffusion.backends.test_backend import TestBackend


def register_dummy_backend() -> tuple[str, type[TestBackend], type[Backend] | None]:
    backend_name = "dummy"
    previous = BACKEND_REGISTRY.get(backend_name)
    TestBackend.reset_mock_state()
    BACKEND_REGISTRY[backend_name] = TestBackend
    return backend_name, TestBackend, previous


def unregister_dummy_backend(backend_name: str, previous: type[Backend] | None) -> None:
    if previous is None:
        BACKEND_REGISTRY.pop(backend_name, None)
    else:
        BACKEND_REGISTRY[backend_name] = previous


def wait_for_status(client, expected_status, timeout=1.0):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            response = client.get("/v1/status")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == expected_status:
                    return data
        except Exception:
            pass
        time.sleep(0.01)

    raise TimeoutError(f"Expected status '{expected_status}' not received within {timeout} seconds.")

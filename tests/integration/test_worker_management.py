"""
Integration tests for worker management.

Tests the complete workflow of:
1. Starting workers on application startup
2. Dynamically changing workers when configuration changes via API
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from easydiffusion.config import ConfigManager, create_default_config
from easydiffusion.server import server_api
from easydiffusion.workers import Workers
from torchruntime.device_db import GPU
from support import TestBackend, register_dummy_backend, unregister_dummy_backend


def mock_resolve_devices(devices):
    """Mock implementation of resolve_devices that returns predictable GPU objects."""
    if isinstance(devices, str):
        if devices == "auto":
            return [
                GPU(vendor_id="cpu", vendor_name="CPU", device_id="cpu", device_name="cpu", is_discrete=False),
                GPU(vendor_id="nvidia", vendor_name="NVIDIA", device_id="0", device_name="0", is_discrete=True),
            ]
        elif devices == "cpu":
            return [GPU(vendor_id="cpu", vendor_name="CPU", device_id="cpu", device_name="cpu", is_discrete=False)]
        elif devices == "cuda:0":
            return [GPU(vendor_id="nvidia", vendor_name="NVIDIA", device_id="0", device_name="0", is_discrete=True)]
        else:
            # For other single devices, assume they resolve to their name
            return [GPU(vendor_id="mock", vendor_name="Mock", device_id=devices, device_name=devices, is_discrete=True)]

    if isinstance(devices, list):
        result = []
        for device in devices:
            if device == "cpu":
                result.append(
                    GPU(vendor_id="cpu", vendor_name="CPU", device_id="cpu", device_name="cpu", is_discrete=False)
                )
            elif device == "cuda:0":
                result.append(
                    GPU(vendor_id="nvidia", vendor_name="NVIDIA", device_id="0", device_name="0", is_discrete=True)
                )
            elif device == "auto":
                result.extend(
                    [
                        GPU(vendor_id="cpu", vendor_name="CPU", device_id="cpu", device_name="cpu", is_discrete=False),
                        GPU(vendor_id="nvidia", vendor_name="NVIDIA", device_id="0", device_name="0", is_discrete=True),
                    ]
                )
            else:
                result.append(
                    GPU(vendor_id="mock", vendor_name="Mock", device_id=device, device_name=device, is_discrete=True)
                )
        return result

    return [GPU(vendor_id="cpu", vendor_name="CPU", device_id="cpu", device_name="cpu", is_discrete=False)]


@pytest.fixture
def temp_config_path():
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def app_state(temp_config_path):
    """Set up application state with mock components."""
    with patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices):
        backend_name, backend_class, previous = register_dummy_backend()

        # Create config
        create_default_config(temp_config_path)
        config_manager = ConfigManager(temp_config_path)
        config = config_manager.load()
        config.setdefault("backend", {})
        config["backend"]["backend_name"] = backend_name
        config["backend"]["devices"] = "cpu"
        config_manager.save(config)
        config_manager.load()

        workers = Workers(backend_class, backend_name=backend_name)
        workers.update_devices("cpu")

        # Set up server state for API tests
        server_api.state.config_manager = config_manager
        server_api.state.workers = workers
        server_api.state.task_cache = {}
        client = TestClient(server_api)

        yield {
            "config_manager": config_manager,
            "workers": workers,
            "client": client,
        }

        # Cleanup
        workers.shutdown()
        server_api.state.task_cache = {}
        unregister_dummy_backend(backend_name, previous)


class TestWorkerManagement:
    """Integration tests for worker management."""

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_start_on_initialization(self, mock_resolve, temp_config_path):
        """Test that workers are started when the application initializes."""
        backend_name, backend_class, previous = register_dummy_backend()

        # Create config with specific device
        create_default_config(temp_config_path)
        config_manager = ConfigManager(temp_config_path)
        config = config_manager.load()
        config.setdefault("backend", {})
        config["backend"]["backend_name"] = backend_name
        config["backend"]["devices"] = ["cpu"]
        config_manager.save(config)

        workers = Workers(backend_class, backend_name=backend_name)

        # Start workers as would happen in main.py
        render_devices = config.get("backend", {}).get("devices", "auto")
        workers.update_devices(render_devices)
        active_devices = workers.get_active_devices()
        assert "cpu" in active_devices
        assert len(active_devices) == 1

        # Cleanup
        workers.shutdown()
        unregister_dummy_backend(backend_name, previous)

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_update_on_config_change(self, mock_resolve, app_state):
        """Test that workers are updated when configuration changes via API."""
        workers = app_state["workers"]
        client = app_state["client"]

        assert workers.get_active_devices() == ["cpu"]

        response = client.put("/v1/config", json={"backend": {"devices": ["cpu", "cuda:0"]}})
        assert response.status_code == 200
        assert response.json() == {"status": "updated"}

        active_devices = workers.get_active_devices()
        assert "cpu" in active_devices
        assert "0" in active_devices
        assert len(active_devices) == 2

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_removed_on_config_change(self, mock_resolve, app_state):
        """Test that workers are removed when devices are removed from config."""
        workers = app_state["workers"]
        client = app_state["client"]

        response = client.put("/v1/config", json={"backend": {"devices": ["cpu", "cuda:0"]}})
        assert response.status_code == 200
        assert len(workers.get_active_devices()) == 2

        response = client.put("/v1/config", json={"backend": {"devices": ["cpu"]}})
        assert response.status_code == 200

        active_devices = workers.get_active_devices()
        assert "cpu" in active_devices
        assert "0" not in active_devices
        assert len(active_devices) == 1

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_api_endpoint_updates_workers(self, mock_resolve, app_state):
        """Test that the config API updates worker devices without changing backend."""
        workers = app_state["workers"]
        client = app_state["client"]
        initial_thread = workers.workers["cpu"][0]

        response = client.put("/v1/config", json={"backend": {"devices": ["cpu", "cuda:0"]}})

        assert response.status_code == 200
        assert response.json() == {"status": "updated"}
        assert workers.backend_name == "dummy"
        assert workers.workers["cpu"][0] is initial_thread
        assert set(workers.get_active_devices()) == {"cpu", "0"}

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_backend_change_recreates_workers(self, mock_resolve, app_state):
        """Test that changing backend recreates all workers."""
        from easydiffusion.backends import BACKEND_REGISTRY

        class AltMockBackend(TestBackend):
            pass

        original_registry = BACKEND_REGISTRY.copy()
        BACKEND_REGISTRY["alt_mock"] = AltMockBackend

        try:
            workers = app_state["workers"]
            client = app_state["client"]
            original_thread = workers.workers["cpu"][0]
            old_instances = list(TestBackend.instances)

            response = client.put("/v1/config", json={"backend": {"backend_name": "alt_mock"}})

            assert response.status_code == 200
            assert response.json() == {"status": "updated"}
            assert workers.backend_name == "alt_mock"
            assert workers.backend_class is AltMockBackend
            assert workers.get_active_devices() == ["cpu"]
            assert workers.workers["cpu"][0] is not original_thread
            assert all(instance.stop_called for instance in old_instances)
        finally:
            BACKEND_REGISTRY.clear()
            BACKEND_REGISTRY.update(original_registry)

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_backend_change_only_when_different(self, mock_resolve, app_state):
        """Test that workers are not recreated when backend doesn't change."""
        workers = app_state["workers"]
        client = app_state["client"]

        initial_workers = workers.get_active_devices()
        assert len(initial_workers) == 1

        initial_thread = workers.workers["cpu"][0]

        response = client.put("/v1/config", json={"backend": {"backend_name": "dummy", "devices": ["cpu"]}})

        assert response.status_code == 200
        current_thread = workers.workers["cpu"][0]
        assert current_thread is initial_thread
        assert workers.backend_name == "dummy"

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_handle_auto_device_selection(self, mock_resolve, app_state):
        """Test that 'auto' device selection is handled properly."""
        workers = app_state["workers"]
        client = app_state["client"]

        response = client.put("/v1/config", json={"backend": {"devices": "auto"}})
        assert response.status_code == 200

        active_devices = workers.get_active_devices()
        assert len(active_devices) >= 1
        assert "auto" not in active_devices
        assert "cpu" in active_devices or any(d.isdigit() for d in active_devices)

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_concurrent_config_updates(self, mock_resolve, app_state):
        """Test that concurrent config updates are handled safely."""
        import threading

        workers = app_state["workers"]
        client = app_state["client"]
        results = []

        def update_config(devices):
            try:
                response = client.put("/v1/config", json={"backend": {"devices": devices}})
                results.append(response.status_code == 200)
            except Exception:
                results.append(False)

        # Start multiple threads updating config
        threads = []
        for i in range(5):
            devices = [f"cuda:{i}"]
            t = threading.Thread(target=update_config, args=(devices,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All updates should succeed (thread-safety)
        assert all(results)

        # Should have exactly one device active (the last one to update)
        active_devices = workers.get_active_devices()
        assert len(active_devices) == 1

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_worker_task_processing_after_update(self, mock_resolve, app_state):
        """Test that workers can process tasks after device updates."""
        workers = app_state["workers"]
        client = app_state["client"]

        response = client.put("/v1/config", json={"backend": {"devices": ["cpu", "cuda:0"]}})
        assert response.status_code == 200

        active_devices = workers.get_active_devices()
        assert len(active_devices) == 2
        assert "cpu" in active_devices
        assert "0" in active_devices

        response = client.post(
            "/v1/tasks",
            json={
                "username": "test-user",
                "prompt": "Test",
                "model_paths": {"stable-diffusion": "test-model"},
                "num_inference_steps": 4,
            },
        )

        assert response.status_code == 202
        assert workers.wait(timeout=2.0)

        task_id = response.json()["task_id"]
        task = server_api.state.task_cache[task_id]
        assert task.status == "completed"
        assert len(task.outputs) == 1

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_progress_outputs_change_while_task_runs(self, mock_resolve, app_state):
        """Test that the API exposes changing in-progress outputs at the backend's real step cadence."""
        client = app_state["client"]

        response = client.post(
            "/v1/tasks",
            json={
                "username": "test-user",
                "prompt": "Test",
                "model_paths": {"stable-diffusion": "test-model"},
                "num_inference_steps": 4,
            },
        )

        assert response.status_code == 202
        task_id = response.json()["task_id"]

        observed_progresses = []
        observed_outputs = []

        for _ in range(20):
            detail_response = client.get(f"/v1/tasks/{task_id}")
            assert detail_response.status_code == 200

            payload = detail_response.json()
            progress = payload["progress"]

            if 0.0 < progress < 1.0 and payload["outputs"]:
                output_response = client.get(f"/v1/tasks/{task_id}/outputs/0")
                assert output_response.status_code == 200

                if not observed_progresses or progress != observed_progresses[-1]:
                    observed_progresses.append(progress)
                    observed_outputs.append(output_response.content)

            if progress == pytest.approx(1.0):
                break

            time.sleep(TestBackend.GENERATE_STEP_DELAY_SECONDS)

        assert observed_progresses, "No in-progress values were observed"
        assert all(a < b for a, b in zip(observed_progresses, observed_progresses[1:]))
        assert len(observed_progresses) == len(set(observed_progresses))
        assert len(observed_outputs) == len(observed_progresses)
        assert len(set(observed_outputs)) == len(observed_outputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

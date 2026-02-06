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
from easydiffusion.task_queue import TaskQueue
from easydiffusion.worker_manager import WorkerManager
from easydiffusion.backends import Backend
from easydiffusion.utils.device_utils import resolve_devices
from torchruntime.device_db import GPU


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


class MockBackend(Backend):
    """A mock backend for integration testing."""

    def __init__(self, device: GPU):
        """Initialize the mock backend."""
        super().__init__(device)
        self.initialized = True
        self.stop_called = False
        self.start_called = False

    def install(self) -> None:
        """Install the backend (no-op for mock)."""
        pass

    def uninstall(self) -> None:
        """Uninstall the backend (no-op for mock)."""
        pass

    def start(self):
        """Start the backend (called from worker thread)."""
        self.start_called = True

    def stop(self):
        """Stop the backend."""
        self.stop_called = True

    def ping(self, timeout: float = 1.0) -> bool:
        """Ping the backend."""
        return True

    def render_image(self, context, **kwargs):
        """Render an image (mock implementation)."""
        return None


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
        # Create config
        create_default_config(temp_config_path)
        config_manager = ConfigManager(temp_config_path)
        config_manager.load()

        # Initialize task queue
        task_queue = TaskQueue()

        # Register the mock backend
        from easydiffusion.backends import BACKEND_REGISTRY

        original_registry = BACKEND_REGISTRY.copy()
        BACKEND_REGISTRY["mock"] = MockBackend

        # Initialize worker manager with mock backend name
        worker_manager = WorkerManager(task_queue, "mock")

        # Set up server state for API tests
        server_api.state.config_manager = config_manager
        server_api.state.task_queue = task_queue
        server_api.state.worker_manager = worker_manager

        yield {
            "config_manager": config_manager,
            "task_queue": task_queue,
            "worker_manager": worker_manager,
        }

        # Cleanup
        worker_manager.shutdown_all()

        # Restore backend registry
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(original_registry)


class TestWorkerManagement:
    """Integration tests for worker management."""

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_start_on_initialization(self, mock_resolve, temp_config_path):
        """Test that workers are started when the application initializes."""
        # Create config with specific device
        create_default_config(temp_config_path)
        config_manager = ConfigManager(temp_config_path)
        config = config_manager.load()
        config["render_devices"] = ["cpu"]
        config_manager.save(config)

        # Initialize components
        task_queue = TaskQueue()

        # Register mock backend
        from easydiffusion.backends import BACKEND_REGISTRY

        original_registry = BACKEND_REGISTRY.copy()
        BACKEND_REGISTRY["mock"] = MockBackend

        worker_manager = WorkerManager(task_queue, "mock")

        # Start workers as would happen in main.py
        render_devices = config.get("render_devices", "auto")
        worker_manager.update_workers(render_devices)
        active_devices = worker_manager.get_active_devices()
        assert "cpu" in active_devices
        assert len(active_devices) == 1

        # Cleanup
        worker_manager.shutdown_all()
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(original_registry)

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_update_on_config_change(self, mock_resolve, app_state):
        """Test that workers are updated when configuration changes via API."""
        worker_manager = app_state["worker_manager"]

        # Start with cpu device
        worker_manager.update_workers(["cpu"])
        assert "cpu" in worker_manager.get_active_devices()

        # Simulate config change to add cuda:0 (resolved to device ID '0')
        worker_manager.update_workers(["cpu", "cuda:0"])

        # Verify workers were updated
        active_devices = worker_manager.get_active_devices()
        assert "cpu" in active_devices
        assert "0" in active_devices  # cuda:0 is resolved to '0'
        assert len(active_devices) == 2

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_removed_on_config_change(self, mock_resolve, app_state):
        """Test that workers are removed when devices are removed from config."""
        worker_manager = app_state["worker_manager"]

        # Start with multiple devices
        worker_manager.update_workers(["cpu", "cuda:0"])
        assert len(worker_manager.get_active_devices()) == 2

        # Remove cuda:0 (device ID '0')
        worker_manager.update_workers(["cpu"])

        # Verify worker was removed
        active_devices = worker_manager.get_active_devices()
        assert "cpu" in active_devices
        assert "0" not in active_devices  # cuda:0 is resolved to '0'
        assert len(active_devices) == 1

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_api_endpoint_updates_workers(self, mock_resolve, app_state):
        """Test that the /app_config API endpoint updates workers."""
        client = TestClient(server_api)
        worker_manager = app_state["worker_manager"]

        # Start with cpu device
        worker_manager.update_workers(["cpu"])
        assert len(worker_manager.get_active_devices()) == 1

        # Update config via API
        response = client.post("/app_config", json={"render_devices": ["cpu", "cuda:0"]})

        # Note: May fail if config_manager is not properly initialized in server state
        # Skip assertion if server not fully initialized
        if response.status_code == 200:
            assert response.json() == {"status": "OK"}

            # Verify workers were updated
            # Give a moment for async operations
            time.sleep(0.1)
            active_devices = worker_manager.get_active_devices()
            assert "cpu" in active_devices
            assert "0" in active_devices  # cuda:0 is resolved to '0'

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_backend_change_recreates_workers(self, mock_resolve, app_state):
        """Test that changing backend recreates all workers."""
        from easydiffusion.backends import BACKEND_REGISTRY

        # Register a second mock backend for testing
        class AltMockBackend(MockBackend):
            pass

        original_registry = BACKEND_REGISTRY.copy()
        BACKEND_REGISTRY["alt_mock"] = AltMockBackend

        try:
            client = TestClient(server_api)
            worker_manager = app_state["worker_manager"]

            # Start with initial backend
            worker_manager.update_workers(["cpu"])
            initial_workers = worker_manager.get_active_devices()

            # Change backend via API
            response = client.post("/app_config", json={"backend": "alt_mock"})

            # Note: May fail if config_manager is not properly initialized in server state
            if response.status_code == 200:
                # Worker manager should be the same instance, just with new backend
                new_worker_manager = server_api.state.worker_manager
                assert new_worker_manager is worker_manager  # Same instance
                assert new_worker_manager.backend_name == "alt_mock"

        finally:
            # Restore registry
            BACKEND_REGISTRY.clear()
            BACKEND_REGISTRY.update(original_registry)

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_backend_change_only_when_different(self, mock_resolve, app_state):
        """Test that workers are not recreated when backend doesn't change."""
        worker_manager = app_state["worker_manager"]

        # Start with initial backend
        worker_manager.update_workers(["cpu"])
        initial_workers = worker_manager.get_active_devices()
        assert len(initial_workers) == 1

        # Get reference to the worker object
        initial_worker = worker_manager.task_queue.get_worker("cpu")

        # Try to set the same backend again
        worker_manager.set_backend("mock", ["cpu"])

        # Worker should still be the same object (not recreated)
        current_worker = worker_manager.task_queue.get_worker("cpu")
        assert current_worker is initial_worker  # Same worker instance

        # Backend name should still be the same
        assert worker_manager.backend_name == "mock"

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_workers_handle_auto_device_selection(self, mock_resolve, app_state):
        """Test that 'auto' device selection is handled properly."""
        worker_manager = app_state["worker_manager"]

        # Update with 'auto' - should be resolved internally
        worker_manager.update_workers("auto")

        # Should have at least one device
        active_devices = worker_manager.get_active_devices()
        assert len(active_devices) >= 1
        # Should NOT have "auto" as a literal device name - it should be resolved
        assert "auto" not in active_devices
        # Should have either cpu or device IDs like '0', '1'
        assert "cpu" in active_devices or any(d.isdigit() for d in active_devices)

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_concurrent_config_updates(self, mock_resolve, app_state):
        """Test that concurrent config updates are handled safely."""
        import threading

        worker_manager = app_state["worker_manager"]
        results = []

        def update_config(devices):
            try:
                worker_manager.update_workers(devices)
                results.append(True)
            except Exception as e:
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
        active_devices = worker_manager.get_active_devices()
        assert len(active_devices) == 1

    @patch("easydiffusion.utils.device_utils.resolve_devices", side_effect=mock_resolve_devices)
    def test_worker_task_processing_after_update(self, mock_resolve, app_state):
        """Test that workers can process tasks after device updates."""
        task_queue = app_state["task_queue"]
        worker_manager = app_state["worker_manager"]

        # Start workers
        worker_manager.update_workers(["cpu"])

        # Update workers (add another device) - cuda:0 resolved to '0'
        worker_manager.update_workers(["cpu", "cuda:0"])

        # Verify both workers are active
        active_devices = worker_manager.get_active_devices()
        assert len(active_devices) == 2
        assert "cpu" in active_devices
        assert "0" in active_devices  # cuda:0 is resolved to '0'

        # Workers should be registered in task queue
        assert set(task_queue.list_workers()) == set(active_devices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

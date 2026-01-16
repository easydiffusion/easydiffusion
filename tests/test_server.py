"""
Tests for FastAPI server endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

from easydiffusion.config import ConfigManager, create_default_config
from easydiffusion.server import server_api, Task
from easydiffusion.task_queue import TaskQueue
from easydiffusion.worker_manager import WorkerManager
from easydiffusion.backends import BACKEND_REGISTRY, Backend


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def config_manager(temp_dir):
    """Create a test config manager."""
    config_path = temp_dir / "config.yaml"
    create_default_config(config_path)
    manager = ConfigManager(config_path)
    manager.load()
    return manager


@pytest.fixture
def client(config_manager):
    """Create a test client."""
    task_queue = TaskQueue()

    # Register a test backend if not already registered
    # We'll use sdkit3 which is already registered
    worker_manager = WorkerManager(task_queue, "sdkit3")

    # Set up app state
    server_api.state.config_manager = config_manager
    server_api.state.task_queue = task_queue
    server_api.state.worker_manager = worker_manager

    client = TestClient(server_api)

    yield client

    # Cleanup
    worker_manager.shutdown_all(timeout=1.0)


def test_root_endpoint(client):
    """Test the root endpoint returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "EasyDiffusion" in response.text


def test_ping_endpoint(client):
    """Test the ping endpoint."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_get_config(client):
    """Test getting configuration via /get/app_config endpoint."""
    response = client.get("/get/app_config")
    assert response.status_code == 200

    config = response.json()
    assert "update_branch" in config
    assert "backend" in config
    assert "render_devices" in config

    # Check for no-cache headers
    assert "cache-control" in response.headers
    assert "no-cache" in response.headers["cache-control"]


def test_update_config_single_field(client, config_manager):
    """Test updating a single config field via /app_config endpoint."""
    response = client.post("/app_config", json={"update_branch": "develop"})
    assert response.status_code == 200

    # Old API returns {"status": "OK"}
    assert response.json() == {"status": "OK"}

    # Verify config was updated
    config = config_manager.get_all()
    assert config["update_branch"] == "develop"
    assert config["backend"] == "sdkit3"  # unchanged


def test_update_config_multiple_fields(client):
    """Test updating multiple config fields."""
    response = client.post("/app_config", json={"update_branch": "feature", "backend": "webui"})
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_update_config_render_devices(client):
    """Test updating render_devices with different types."""
    # Test string value
    response = client.post("/app_config", json={"render_devices": "cpu"})
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    # Test list value
    response = client.post("/app_config", json={"render_devices": ["cuda:0", "cuda:1"]})
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_update_config_nested_fields(client, config_manager):
    """Test updating nested config fields like ui and net settings."""
    response = client.post(
        "/app_config", json={"ui_open_browser_on_start": True, "listen_to_network": True, "listen_port": 9000}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    config = config_manager.get_all()
    assert config.get("ui", {}).get("open_browser_on_start") is True
    assert config.get("net", {}).get("listen_to_network") is True
    assert config.get("net", {}).get("listen_port") == 9000


def test_get_endpoint_easter_egg(client):
    """Test the easter egg when accessing /get without a key."""
    response = client.get("/get/")
    assert response.status_code == 418  # I'm a teapot
    assert "teapot" in response.json()["detail"].lower()


def test_get_system_info(client):
    """Test getting system info via /get/system_info."""
    response = client.get("/get/system_info")
    assert response.status_code == 200

    info = response.json()
    assert "devices" in info
    assert "hosts" in info
    assert "default_output_dir" in info


def test_get_models(client):
    """Test getting models via /get/models."""
    response = client.get("/get/models")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data


def test_get_modifiers(client):
    """Test getting modifiers via /get/modifiers."""
    response = client.get("/get/modifiers")
    assert response.status_code == 200

    # Returns a list of modifiers
    assert isinstance(response.json(), list)


def test_get_ui_plugins(client):
    """Test getting UI plugins via /get/ui_plugins."""
    response = client.get("/get/ui_plugins")
    assert response.status_code == 200

    # Returns a list of plugins
    assert isinstance(response.json(), list)


def test_get_unknown_key(client):
    """Test accessing unknown key returns 404."""
    response = client.get("/get/unknown_key")
    assert response.status_code == 404
    assert "unknown" in response.json()["detail"].lower()


def test_update_config_empty(client):
    """Test updating config with no fields still works."""
    response = client.post("/app_config", json={})
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_create_render_task(client):
    """Test creating a render task."""
    response = client.post("/render", json={})
    assert response.status_code == 200

    data = response.json()
    # Check for old API compatibility
    assert "task" in data  # Integer task ID
    assert "status" in data
    assert "queue" in data
    assert "stream" in data
    assert data["status"] == "queued"

    # Verify task is an integer
    task_id = data["task"]
    assert isinstance(task_id, int)

    # Verify stream URL format
    assert data["stream"] == f"/image/stream/{task_id}"


def test_task_id_uniqueness(client):
    """Test that each task gets a unique ID."""
    response1 = client.post("/render", json={})
    response2 = client.post("/render", json={})

    task_id1 = response1.json()["task"]
    task_id2 = response2.json()["task"]

    assert task_id1 != task_id2


def test_task_creation():
    """Test Task class."""
    task1 = Task()
    assert task1.task_id is not None
    assert isinstance(task1.task_id, str)
    assert isinstance(task1.id, int)  # Integer ID for old API
    assert task1.buffer_queue is not None
    assert task1.temp_images == []
    assert task1.lock is not None
    assert task1.response is None
    assert task1.error is None

    task2 = Task(task_id="custom-id", session_id="test-session", param1="value1")
    assert task2.task_id == "custom-id"
    assert task2.session_id == "test-session"
    assert task2.params["param1"] == "value1"

    # Ensure unique integer IDs
    task3 = Task()
    assert task1.id != task3.id


def test_task_status(client):
    """Test task status property."""
    task = Task()

    # Initially pending
    assert task.status == "pending"
    assert task.is_pending is True

    # When response is set, becomes completed
    task.response = {"result": "done"}
    assert task.status == "completed"
    assert task.is_pending is False

    # Test error status
    task2 = Task()
    task2.error = Exception("test error")
    assert task2.status == "error"

    # Test stopped status
    task3 = Task()
    task3.error = StopAsyncIteration("stopped")
    assert task3.status == "stopped"


def test_stream_endpoint_not_found(client):
    """Test streaming from non-existent task."""
    response = client.get("/image/stream/99999")
    assert response.status_code == 404


def test_temp_image_endpoint_not_found(client):
    """Test getting temp image from non-existent task."""
    response = client.get("/image/tmp/99999/0")
    assert response.status_code == 410  # Old API returns 410 Gone for missing tasks


def test_stop_task_not_found(client):
    """Test stopping non-existent task."""
    response = client.get("/image/stop?task=99999")
    assert response.status_code == 404


def test_buffer_queue_streaming():
    """Test buffer queue for streaming intermediate results."""
    import asyncio

    task = Task()

    # Add some progress data to the buffer queue
    task.buffer_queue.put('{"step": 1, "total_steps": 10}')
    task.buffer_queue.put('{"step": 2, "total_steps": 10}')
    task.buffer_queue.put('{"step": 3, "total_steps": 10}')

    # Collect all data from the generator
    async def collect_data():
        results = []
        async for data in task.read_buffer_generator():
            results.append(data)
        return results

    results = asyncio.run(collect_data())
    assert len(results) == 3
    assert '{"step": 1, "total_steps": 10}' in results
    assert '{"step": 2, "total_steps": 10}' in results
    assert '{"step": 3, "total_steps": 10}' in results


def test_temp_images_storage():
    """Test temporary image storage."""
    from io import BytesIO

    task = Task()

    # Initialize temp_images with some buffers
    task.temp_images = [BytesIO(b"image1data"), BytesIO(b"image2data"), BytesIO(b"image3data")]

    assert len(task.temp_images) == 3

    # Verify we can read from the buffers
    task.temp_images[0].seek(0)
    assert task.temp_images[0].read() == b"image1data"


def test_task_lock_status():
    """Test that task status reflects lock state."""
    task = Task()

    # Initially not locked, should be pending
    assert not task.lock.locked()
    assert task.status == "pending"

    # When locked, should show as running
    task.lock.acquire()
    assert task.status == "running"
    task.lock.release()

    # After release without response, back to pending
    assert task.status == "pending"


def test_stream_with_response(client):
    """Test streaming endpoint returns cached response when complete."""
    # Create a task
    response = client.post("/render", json={})
    task_id = response.json()["task"]

    # Get the task and set a response
    task = server_api.state.task_cache[task_id]
    task.response = {"status": "completed", "images": []}

    # Stream should return the cached response
    stream_response = client.get(f"/image/stream/{task_id}")
    assert stream_response.status_code == 200
    data = stream_response.json()
    assert data["status"] == "completed"


def test_stream_too_early(client):
    """Test streaming endpoint returns 425 when task hasn't started."""
    # Create a task
    response = client.post("/render", json={})
    task_id = response.json()["task"]

    # Stream should return 425 Too Early
    stream_response = client.get(f"/image/stream/{task_id}")
    assert stream_response.status_code == 425
    assert "Too Early" in stream_response.json()["detail"]


def test_stop_already_stopped_task(client):
    """Test stopping an already stopped task returns 409."""
    # Create a task
    response = client.post("/render", json={})
    task_id = response.json()["task"]

    # Stop it once
    stop_response = client.get(f"/image/stop?task={task_id}")
    assert stop_response.status_code == 200

    # Try to stop it again
    stop_response2 = client.get(f"/image/stop?task={task_id}")
    assert stop_response2.status_code == 409
    assert "already stopped" in stop_response2.json()["detail"]

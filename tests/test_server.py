"""
Tests for the v1 API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil
import io

from easydiffusion.config import ConfigManager, create_default_config
from easydiffusion.server import server_api
from easydiffusion.task_queue import TaskQueue
from easydiffusion.worker_manager import WorkerManager


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
    """Create a test client with v1 API."""
    task_queue = TaskQueue()
    worker_manager = WorkerManager(task_queue, "sdkit3")

    server_api.state.config_manager = config_manager
    server_api.state.task_queue = task_queue
    server_api.state.worker_manager = worker_manager
    server_api.state.task_cache = {}

    client = TestClient(server_api)

    yield client

    worker_manager.shutdown_all(timeout=1.0)
    server_api.state.task_cache = {}


class TestHealthEndpoint:
    """Tests for /v1/health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns correct response."""
        response = client.get("/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestConfigEndpoints:
    """Tests for /v1/config endpoints."""

    def test_get_config(self, client):
        """Test getting configuration."""
        response = client.get("/v1/config")
        assert response.status_code == 200

        data = response.json()
        assert "render_devices" in data
        assert "models_dir" in data
        assert "vram_usage_level" in data
        assert "backend" in data

        assert isinstance(data["render_devices"], list)
        assert data["backend"] == "sdkit3"

    def test_update_config(self, client, config_manager):
        """Test updating configuration."""
        response = client.put(
            "/v1/config",
            json={
                "render_devices": ["cpu"],
                "vram_usage_level": "high",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "updated"

        config = config_manager.get_all()
        assert config["render_devices"] == ["cpu"]
        assert config["vram_usage_level"] == "high"

    def test_update_config_partial(self, client, config_manager):
        """Test partial configuration update."""
        response = client.put(
            "/v1/config",
            json={"models_dir": "/custom/models"},
        )
        assert response.status_code == 200

        config = config_manager.get_all()
        assert config["models_dir"] == "/custom/models"
        assert config["backend"] == "sdkit3"


class TestDevicesEndpoint:
    """Tests for /v1/devices endpoint."""

    def test_get_devices(self, client):
        """Test listing devices."""
        response = client.get("/v1/devices")
        assert response.status_code == 200

        data = response.json()
        assert "devices" in data
        assert isinstance(data["devices"], list)
        assert len(data["devices"]) > 0

        cpu_device = next((d for d in data["devices"] if d["id"] == "cpu"), None)
        assert cpu_device is not None
        assert cpu_device["name"] == "CPU"
        assert cpu_device["available"] is True


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    def test_get_models_empty(self, client, temp_dir, config_manager):
        """Test listing models when models directory is empty."""
        models_dir = temp_dir / "models"
        models_dir.mkdir()

        config = config_manager.get_all()
        config["models_dir"] = str(models_dir)
        config_manager.update(config)

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 0

    def test_get_models_with_files(self, client, temp_dir, config_manager):
        """Test listing models with actual model files."""
        models_dir = temp_dir / "models"
        models_dir.mkdir()

        (models_dir / "model1.safetensors").touch()
        (models_dir / "model2.ckpt").touch()

        config = config_manager.get_all()
        config["models_dir"] = str(models_dir)
        config_manager.update(config)

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert len(data["models"]) == 2

        model_names = [m["name"] for m in data["models"]]
        assert "model1" in model_names
        assert "model2" in model_names


class TestGenerateEndpoint:
    """Tests for /v1/generate endpoint."""

    def test_create_generate_task(self, client):
        """Test creating a generation task."""
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "A beautiful landscape",
                "model": "test-model",
                "width": 512,
                "height": 512,
            },
        )
        assert response.status_code == 202

        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert "queue_position" in data

    def test_create_generate_task_with_defaults(self, client):
        """Test creating task with minimal parameters."""
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "A cat",
                "model": "test-model",
            },
        )
        assert response.status_code == 202

        data = response.json()
        assert "task_id" in data

    def test_create_generate_task_full_params(self, client):
        """Test creating task with all parameters."""
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "A dog",
                "negative_prompt": "blurry",
                "seed": 123,
                "width": 768,
                "height": 768,
                "num_outputs": 2,
                "num_inference_steps": 30,
                "guidance_scale": 8.5,
                "model": "test-model",
                "output_format": "png",
                "save_path": "/tmp/output",
            },
        )
        assert response.status_code == 202


class TestFilterEndpoint:
    """Tests for /v1/filter endpoint."""

    def test_create_filter_task(self, client):
        """Test creating a filter task."""
        response = client.post(
            "/v1/filter",
            json={
                "image": "base64_encoded_image_data",
                "filter": "blur",
                "filter_params": {"strength": 0.5},
            },
        )
        assert response.status_code == 202

        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    def test_create_filter_task_minimal(self, client):
        """Test creating filter task with minimal parameters."""
        response = client.post(
            "/v1/filter",
            json={
                "image": "image_data",
                "filter": "sharpen",
            },
        )
        assert response.status_code == 202


class TestTasksEndpoints:
    """Tests for task management endpoints."""

    def test_get_tasks_empty(self, client):
        """Test getting tasks when no tasks exist."""
        response = client.get("/v1/tasks")
        assert response.status_code == 200

        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 0

    def test_get_tasks_with_tasks(self, client):
        """Test getting tasks list."""
        response1 = client.post(
            "/v1/generate",
            json={"prompt": "Test 1", "model": "test"},
        )
        response2 = client.post(
            "/v1/generate",
            json={"prompt": "Test 2", "model": "test"},
        )

        response = client.get("/v1/tasks")
        assert response.status_code == 200

        data = response.json()
        assert len(data["tasks"]) == 2

    def test_get_task_detail(self, client):
        """Test getting task details."""
        create_response = client.post(
            "/v1/generate",
            json={"prompt": "Test", "model": "test"},
        )
        task_id = create_response.json()["task_id"]

        response = client.get(f"/v1/tasks/{task_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
        assert "images" in data

    def test_get_task_detail_with_request(self, client):
        """Test getting task details with request data."""
        create_response = client.post(
            "/v1/generate",
            json={"prompt": "Test prompt", "model": "test", "seed": 42},
        )
        task_id = create_response.json()["task_id"]

        response = client.get(f"/v1/tasks/{task_id}?include_request=true")
        assert response.status_code == 200

        data = response.json()
        assert data["request"] is not None
        assert data["request"]["prompt"] == "Test prompt"
        assert data["request"]["seed"] == 42

    def test_get_task_not_found(self, client):
        """Test getting non-existent task."""
        response = client.get("/v1/tasks/nonexistent-task-id")
        assert response.status_code == 404

    def test_stop_task(self, client):
        """Test stopping a task."""
        create_response = client.post(
            "/v1/generate",
            json={"prompt": "Test", "model": "test"},
        )
        task_id = create_response.json()["task_id"]

        response = client.delete(f"/v1/tasks/{task_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stopped"

    def test_stop_task_twice(self, client):
        """Test stopping an already stopped task."""
        create_response = client.post(
            "/v1/generate",
            json={"prompt": "Test", "model": "test"},
        )
        task_id = create_response.json()["task_id"]

        client.delete(f"/v1/tasks/{task_id}")

        response = client.delete(f"/v1/tasks/{task_id}")
        assert response.status_code == 409

    def test_stop_task_not_found(self, client):
        """Test stopping non-existent task."""
        response = client.delete("/v1/tasks/nonexistent-task-id")
        assert response.status_code == 404

    def test_stop_all_tasks(self, client):
        """Test stopping all tasks."""
        client.post("/v1/generate", json={"prompt": "Test 1", "model": "test"})
        client.post("/v1/generate", json={"prompt": "Test 2", "model": "test"})

        response = client.delete("/v1/tasks")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "all_stopped"

    def test_stop_all_tasks_empty(self, client):
        """Test stopping all tasks when none exist."""
        response = client.delete("/v1/tasks")
        assert response.status_code == 200


class TestImageEndpoint:
    """Tests for image retrieval endpoint."""

    def test_get_task_image(self, client):
        """Test retrieving a task image."""
        create_response = client.post(
            "/v1/generate",
            json={"prompt": "Test", "model": "test"},
        )
        task_id = create_response.json()["task_id"]

        task = server_api.state.task_cache[task_id]

        test_image = io.BytesIO(b"fake_image_data")
        task.output_images.append(test_image)

        response = client.get(f"/v1/tasks/{task_id}/images/0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_get_task_image_not_found(self, client):
        """Test retrieving non-existent image."""
        create_response = client.post(
            "/v1/generate",
            json={"prompt": "Test", "model": "test"},
        )
        task_id = create_response.json()["task_id"]

        response = client.get(f"/v1/tasks/{task_id}/images/0")
        assert response.status_code == 404

    def test_get_task_image_invalid_task(self, client):
        """Test retrieving image from non-existent task."""
        response = client.get("/v1/tasks/invalid-task/images/0")
        assert response.status_code == 404


class TestStaticEndpoints:
    """Tests for static file serving endpoints."""

    def test_get_root_index_html(self, client):
        """Test GET / returns index.html."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "cache-control" in response.headers
        assert "no-cache" in response.headers["cache-control"]
        # Check that it has content
        assert len(response.text) > 0

    def test_get_media_file(self, client):
        """Test GET /media/... serves files from media directory."""
        response = client.get("/media/css/main.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        assert "cache-control" in response.headers
        assert "no-cache" in response.headers["cache-control"]
        # Check that it has content
        assert len(response.text) > 0


class TestCacheHeaders:
    """Tests for cache control headers."""

    def test_health_no_cache(self, client):
        """Test health endpoint has no-cache headers."""
        response = client.get("/v1/health")
        assert "cache-control" in response.headers
        assert "no-cache" in response.headers["cache-control"]

    def test_config_no_cache(self, client):
        """Test config endpoint has no-cache headers."""
        response = client.get("/v1/config")
        assert "cache-control" in response.headers
        assert "no-cache" in response.headers["cache-control"]

    def test_devices_no_cache(self, client):
        """Test devices endpoint has no-cache headers."""
        response = client.get("/v1/devices")
        assert "cache-control" in response.headers
        assert "no-cache" in response.headers["cache-control"]

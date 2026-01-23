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
        response = client.get("/v1/config?username=easydiffusion")
        assert response.status_code == 200

        data = response.json()
        assert "network" in data
        assert "updates" in data
        assert "rendering" in data
        assert "users" not in data  # Ensure users are not returned
        assert "security" not in data  # Ensure security is not returned
        assert "user_settings" in data
        assert "save" in data["user_settings"]
        assert "ui" in data["user_settings"]
        assert data["rendering"]["backend"] == "sdkit3"

    def test_update_config(self, client, config_manager):
        """Test updating configuration."""
        response = client.put(
            "/v1/config?username=default",
            json={"save": {"save_path": "/new/path"}, "ui": {"theme": "theme-dark"}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "updated"

        user_config = config_manager.get_user_config("easydiffusion")
        assert user_config["save"]["save_path"] == "/new/path"
        assert user_config["ui"]["theme"] == "theme-dark"

    def test_update_config_forbidden_keys(self, client, config_manager):
        """Test that forbidden keys like 'users' and 'security' are not set via config update."""
        # Try to update with forbidden keys
        response = client.put(
            "/v1/config?username=default",
            json={"save": {"save_path": "/new/path"}, "users": ["new_user"], "security": {"foo": "bar"}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "updated"

        user_config = config_manager.get_user_config("default")
        assert user_config["save"]["save_path"] == "/new/path"
        # Ensure forbidden keys are not set
        assert "users" not in user_config
        assert "security" not in user_config


class TestUserEndpoints:
    """Tests for user management endpoints."""

    def test_get_users(self, client, config_manager):
        """Test listing users."""
        response = client.get("/v1/users")
        assert response.status_code == 200

        data = response.json()
        assert "users" in data
        assert "easydiffusion" in data["users"]

    def test_create_user(self, client, config_manager):
        """Test creating a new user."""
        response = client.post(
            "/v1/users",
            json={"username": "alice"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["status"] == "created"
        assert data["username"] == "alice"

        users = config_manager.get_users()
        assert "alice" in users

    def test_create_default_user_fails(self, client):
        """Test creating 'default' user fails."""
        response = client.post(
            "/v1/users",
            json={"username": "default"},
        )
        assert response.status_code == 400

    def test_delete_user(self, client, config_manager):
        """Test deleting a user."""
        # First create a user
        config_manager.add_user("bob")

        response = client.delete("/v1/users/bob")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "deleted"

        users = config_manager.get_users()
        assert "bob" not in users

    def test_delete_default_user_fails(self, client):
        """Test deleting 'default' user fails."""
        response = client.delete("/v1/users/default")
        assert response.status_code == 400


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
        config["rendering"]["models_dir"] = str(models_dir)
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

        # Create subdirs and files for different types
        sd_dir = models_dir / "stable-diffusion"
        sd_dir.mkdir()
        (sd_dir / "model1.safetensors").touch()
        (sd_dir / "model2.ckpt").touch()

        vae_dir = models_dir / "vae"
        vae_dir.mkdir()
        (vae_dir / "vae1.safetensors").touch()

        config = config_manager.get_all()
        config["rendering"]["models_dir"] = str(models_dir)
        config_manager.update(config)

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert len(data["models"]) == 3

        # Check model names
        model_ids = [m["model"] for m in data["models"]]
        assert "model1" in model_ids
        assert "model2" in model_ids
        assert "vae1" in model_ids

        # Check names and tags
        models_by_model = {m["model"]: m for m in data["models"]}
        assert models_by_model["model1"]["name"] == "model1"
        assert models_by_model["model1"]["tags"] == ["stable-diffusion"]
        assert models_by_model["model2"]["name"] == "model2"
        assert models_by_model["model2"]["tags"] == ["stable-diffusion"]
        assert models_by_model["vae1"]["name"] == "vae1"
        assert models_by_model["vae1"]["tags"] == ["vae"]


class TestGenerateEndpoint:
    """Tests for /v1/generate endpoint."""

    def test_create_generate_task(self, client):
        """Test creating a generation task."""
        response = client.post(
            "/v1/generate",
            json={
                "username": "easydiffusion",
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
                "username": "easydiffusion",
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
                "username": "easydiffusion",
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
                "username": "easydiffusion",
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
                "username": "easydiffusion",
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
            json={"username": "easydiffusion", "prompt": "Test 1", "model": "test"},
        )
        response2 = client.post(
            "/v1/generate",
            json={"username": "easydiffusion", "prompt": "Test 2", "model": "test"},
        )

        response = client.get("/v1/tasks")
        assert response.status_code == 200

        data = response.json()
        assert len(data["tasks"]) == 2
        for task in data["tasks"]:
            assert "username" in task
            assert task["username"] == "easydiffusion"

    def test_get_task_detail(self, client):
        """Test getting task details."""
        create_response = client.post(
            "/v1/generate",
            json={"username": "easydiffusion", "prompt": "Test", "model": "test"},
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
            json={"username": "easydiffusion", "prompt": "Test prompt", "model": "test", "seed": 42},
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
            json={"username": "easydiffusion", "prompt": "Test", "model": "test"},
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
            json={"username": "easydiffusion", "prompt": "Test", "model": "test"},
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
        client.post("/v1/generate", json={"username": "easydiffusion", "prompt": "Test 1", "model": "test"})
        client.post("/v1/generate", json={"username": "easydiffusion", "prompt": "Test 2", "model": "test"})

        response = client.delete("/v1/tasks?username=easydiffusion")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "all_stopped"

    def test_stop_all_tasks_empty(self, client):
        """Test stopping all tasks when none exist."""
        response = client.delete("/v1/tasks?username=easydiffusion")
        assert response.status_code == 200


class TestImageEndpoint:
    """Tests for image retrieval endpoint."""

    def test_get_task_image(self, client):
        """Test retrieving a task image."""
        create_response = client.post(
            "/v1/generate",
            json={"username": "easydiffusion", "prompt": "Test", "model": "test"},
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
            json={"username": "easydiffusion", "prompt": "Test", "model": "test"},
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
        response = client.get("/v1/config?username=easydiffusion")
        assert "cache-control" in response.headers
        assert "no-cache" in response.headers["cache-control"]

    def test_devices_no_cache(self, client):
        """Test devices endpoint has no-cache headers."""
        response = client.get("/v1/devices")
        assert "cache-control" in response.headers
        assert "no-cache" in response.headers["cache-control"]

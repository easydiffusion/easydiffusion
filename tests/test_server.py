"""
Tests for the v1 API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

from easydiffusion.config import ConfigManager, create_default_config
from easydiffusion.server import server_api
from easydiffusion.tasks import Task
from easydiffusion.workers import Workers
from easydiffusion.backends.test_backend import TestBackend


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
def client(config_manager, dummy_backend_registry):
    """Create a test client with v1 API."""
    backend_name, backend_class = dummy_backend_registry
    workers = Workers(backend_class, backend_name=backend_name)

    config = config_manager.get_all()
    config["backend"]["backend_name"] = backend_name
    config["backend"]["devices"] = "cpu"
    config_manager.save(config)
    config_manager.load()
    backend_class.reset_mock_state()

    server_api.state.config_manager = config_manager
    server_api.state.workers = workers
    server_api.state.task_cache = {}

    workers.update_devices("cpu")

    client = TestClient(server_api)

    yield client

    workers.shutdown()
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


class TestSystemConfigEndpoints:
    """Tests for /v1/config endpoints."""

    def test_get_config(self, client):
        """Test getting configuration."""
        response = client.get("/v1/config")
        assert response.status_code == 200

        data = response.json()
        assert "network" in data
        assert "updates" in data
        assert "models" in data
        assert "backend" in data
        assert "users" not in data  # Ensure users are not returned
        assert "security" not in data  # Ensure security is not returned
        assert "user_settings" not in data
        assert data["backend"]["backend_name"] == "dummy"

    def test_update_config(self, client, config_manager):
        """Test updating configuration."""
        response = client.put(
            "/v1/config",
            json={"models": {"models_dir": "/new/models"}, "backend": {"devices": "cpu"}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "updated"

        system_config = config_manager.get_system_config()
        assert system_config["models"]["models_dir"] == "/new/models"
        assert system_config["backend"]["devices"] == "cpu"
        assert system_config["backend"]["backend_name"] == "dummy"

    def test_update_config_forbidden_keys(self, client, config_manager):
        """Test that forbidden keys like 'users' and 'security' are not set via config update."""
        # Try to update with forbidden keys
        response = client.put(
            "/v1/config",
            json={"models": {"models_dir": "/new/path"}, "users": ["new_user"], "security": {"foo": "bar"}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "updated"

        system_config = config_manager.get_system_config()
        assert system_config["models"]["models_dir"] == "/new/path"
        # Ensure forbidden keys are not set
        assert config_manager.get_all()["users"] == ["easydiffusion"]
        assert "foo" not in config_manager.get_all().get("security", {})


class TestUserConfigEndpoints:
    """Tests for /v1/users/{username}/config endpoints."""

    def test_get_user_config(self, client):
        """Test getting the effective user configuration."""
        response = client.get("/v1/users/easydiffusion/config")
        assert response.status_code == 200

        data = response.json()
        assert "save" in data
        assert "ui" in data
        assert data["ui"]["block_nsfw"] is False
        assert "network" not in data

    def test_update_user_config(self, client, config_manager):
        """Test updating user-specific overrides."""
        response = client.put(
            "/v1/users/easydiffusion/config",
            json={"save": {"save_path": "/new/path"}, "ui": {"theme": "theme-dark"}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "updated"

        user_config = config_manager.get_user_config("easydiffusion")
        assert user_config["save"]["save_path"] == "/new/path"
        assert user_config["save"]["auto_save_images"] is False
        assert user_config["ui"]["theme"] == "theme-dark"
        assert user_config["ui"]["open_browser_on_start"] is True

    def test_get_user_config_honors_force_block_nsfw(self, client, config_manager):
        """Test security.force_block_nsfw overrides the user config response."""
        config = config_manager.get_all()
        config["security"]["force_block_nsfw"] = True
        config_manager.save(config)
        config_manager.load()

        response = client.get("/v1/users/easydiffusion/config")
        assert response.status_code == 200
        assert response.json()["ui"]["block_nsfw"] is True


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
        config["models"]["models_dir"] = str(models_dir)
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
        config["models"]["models_dir"] = str(models_dir)
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
    """Tests for generation payloads posted to /v1/tasks."""

    def test_create_generate_task(self, client):
        """Test creating a generation task."""
        response = client.post(
            "/v1/tasks",
            json={
                "username": "easydiffusion",
                "prompt": "A beautiful landscape",
                "model_paths": {"stable-diffusion": "test-model"},
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
            "/v1/tasks",
            json={
                "username": "easydiffusion",
                "prompt": "A cat",
                "model_paths": {"stable-diffusion": "test-model"},
            },
        )
        assert response.status_code == 202

        data = response.json()
        assert "task_id" in data

    def test_create_generate_task_full_params(self, client):
        """Test creating task with all parameters."""
        response = client.post(
            "/v1/tasks",
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
                "distilled_guidance_scale": 4.0,
                "model_paths": {"stable-diffusion": "test-model", "vae": "test-vae"},
                "model_params": {"stable-diffusion": {"clip_skip": True}},
                "output_format": "png",
                "output_quality": 90,
                "output_lossless": True,
                "save_to_disk_path": "/tmp/output",
                "metadata_output_format": "json",
                "session_id": "browser-session",
                "task_id": "client-supplied-id",
                "filters": ["nsfw_checker", "realesrgan"],
                "filter_params": {"realesrgan": {"scale": 4, "upscaler": "4x"}},
                "stream_image_progress": True,
                "stream_image_progress_interval": 2,
                "show_only_filtered_image": True,
                "block_nsfw": True,
                "clip_skip": True,
            },
        )
        assert response.status_code == 202

        task_id = response.json()["task_id"]
        task = server_api.state.task_cache[task_id]

        assert task.username == "easydiffusion"
        assert task.input["request"]["prompt"] == "A dog"
        assert task.input["request"]["filters"] == ["nsfw_checker", "realesrgan"]
        assert task.input["request"]["filter_params"]["realesrgan"]["scale"] == 4
        assert task.input["models"]["model_paths"]["stable-diffusion"] == "test-model"
        assert task.input["save"]["save_to_disk_path"] == "/tmp/output"
        assert task.input["task"]["task_id"] == task_id
        assert task.input["task"]["session_id"] == "browser-session"
        assert task.input["task"]["username"] == "easydiffusion"
        assert task.input["task"]["block_nsfw"] is True

    def test_create_generate_task_requires_body_username(self, client):
        """Test username is required in the request body."""
        response = client.post(
            "/v1/tasks",
            json={
                "prompt": "A fox",
                "model_paths": {"stable-diffusion": "test-model"},
            },
        )

        assert response.status_code == 422

    def test_create_generate_task_accepts_body_username(self, client):
        """Test username can be supplied in the body and is stored with the queued task."""
        response = client.post(
            "/v1/tasks",
            json={
                "username": "body-user",
                "prompt": "A fox",
                "model_paths": {"stable-diffusion": "test-model"},
            },
        )

        assert response.status_code == 202

        task_id = response.json()["task_id"]
        task = server_api.state.task_cache[task_id]

        assert task.username == "body-user"
        assert task.input["task"]["username"] == "body-user"


class TestFilterEndpoint:
    """Tests for filter payloads posted to /v1/tasks."""

    def test_create_filter_task(self, client):
        """Test creating a filter task."""
        response = client.post(
            "/v1/tasks",
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
            "/v1/tasks",
            json={
                "username": "easydiffusion",
                "image": "image_data",
                "filter": "sharpen",
            },
        )
        assert response.status_code == 202

    def test_create_filter_task_captures_split_input(self, client):
        """Test filter input is split into task.input sections."""
        response = client.post(
            "/v1/tasks",
            json={
                "username": "easydiffusion",
                "image": ["image_data"],
                "filter": ["sharpen", "contrast"],
                "filter_params": {"sharpen": {"amount": 1.2}},
                "model_paths": {"realesrgan": "4x"},
                "output_format": "webp",
                "output_quality": 80,
                "save_to_disk_path": "/tmp/filtered",
                "session_id": "filter-session",
                "task_id": "ignore-me",
            },
        )

        assert response.status_code == 202

        task_id = response.json()["task_id"]
        task = server_api.state.task_cache[task_id]

        assert task.input["request"]["filter"] == ["sharpen", "contrast"]
        assert task.input["models"]["model_paths"]["realesrgan"] == "4x"
        assert task.input["output"]["output_format"] == "webp"
        assert task.input["save"]["save_to_disk_path"] == "/tmp/filtered"
        assert task.input["task"]["task_id"] == task_id
        assert task.input["task"]["session_id"] == "filter-session"
        assert task.input["task"]["username"] == "easydiffusion"

    def test_create_filter_task_requires_body_username(self, client):
        """Test filter tasks require username in the request body."""
        response = client.post(
            "/v1/tasks",
            json={
                "image": "image_data",
                "filter": "sharpen",
            },
        )

        assert response.status_code == 422

    def test_create_filter_task_accepts_body_username(self, client):
        """Test filter tasks also accept username in the request body."""
        response = client.post(
            "/v1/tasks",
            json={
                "username": "body-user",
                "image": "image_data",
                "filter": "sharpen",
            },
        )

        assert response.status_code == 202

        task_id = response.json()["task_id"]
        task = server_api.state.task_cache[task_id]

        assert task.username == "body-user"
        assert task.input["task"]["username"] == "body-user"


class TestLegacyTaskEndpoints:
    """Tests for legacy /render and /filter endpoint compatibility."""

    def test_legacy_render_path_returns_legacy_response_and_maps_request(self, client):
        """Test /render keeps the old response body while queueing translated task input."""
        response = client.post(
            "/render",
            json={
                "prompt": "A lighthouse",
                "negative_prompt": "fog",
                "use_stable_diffusion_model": "legacy-model",
                "use_face_correction": "codeformer",
                "use_upscale": "realesrgan",
                "upscale_amount": 2,
                "session_id": "legacy-session",
                "request_id": "legacy-client-id",
                "mask": "mask-data",
                "control_image": "control-image-data",
                "control_filter_to_apply": "controlnet_canny",
                "block_nsfw": True,
            },
        )

        assert response.status_code == 202

        data = response.json()
        assert set(data) == {"task_id", "status", "queue_position"}
        assert data["status"] == "queued"
        assert data["queue_position"] == 1

        task = next(iter(server_api.state.task_cache.values()))

        assert task.username == "default"
        assert task.input["request"]["prompt"] == "A lighthouse"
        assert task.input["request"]["filters"] == ["nsfw_checker", "codeformer", "realesrgan"]
        assert task.input["request"]["filter_params"]["realesrgan"]["scale"] == 2
        assert task.input["request"]["filter_params"]["codeformer"]["codeformer_fidelity"] == 0.5
        assert task.input["request"]["init_image_mask"] == "mask-data"
        assert task.input["request"]["controlnet_filter"] == "controlnet_canny"
        assert task.input["models"]["model_paths"]["stable-diffusion"] == "legacy-model"
        assert task.input["models"]["model_paths"]["nsfw_checker"] == "nsfw_checker"
        assert task.input["task"]["session_id"] == "legacy-session"
        assert task.input["task"]["task_id"] == task.task_id
        assert task.input["task"]["username"] == "default"

    def test_legacy_filter_path_returns_legacy_response_and_maps_request(self, client):
        """Test /filter keeps the old response body while queueing translated task input."""
        response = client.post(
            "/filter",
            json={
                "image": "image-data",
                "filter": "realesrgan",
                "request_id": "legacy-filter-id",
                "model_paths": {"realesrgan": "4x-ultrasharp"},
                "session_id": "legacy-filter-session",
            },
        )

        assert response.status_code == 202

        data = response.json()
        assert set(data) == {"task_id", "status", "queue_position"}
        assert data["status"] == "queued"
        assert data["queue_position"] == 1

        task = next(iter(server_api.state.task_cache.values()))

        assert task.username == "default"
        assert task.input["request"]["filter"] == "realesrgan"
        assert task.input["models"]["model_paths"]["realesrgan"] == "4x-ultrasharp"
        assert task.input["request"]["filter_params"]["realesrgan"]["upscaler"] == "4x-ultrasharp"
        assert task.input["task"]["session_id"] == "legacy-filter-session"
        assert task.input["task"]["task_id"] == task.task_id
        assert task.input["task"]["username"] == "default"


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
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test 1", "model_paths": {"stable-diffusion": "test"}},
        )
        response2 = client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test 2", "model_paths": {"stable-diffusion": "test"}},
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
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test", "model_paths": {"stable-diffusion": "test"}},
        )
        task_id = create_response.json()["task_id"]

        response = client.get(f"/v1/tasks/{task_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
        assert "outputs" in data

    def test_get_task_detail_progress_updates_while_running(self, client):
        """Test task detail surfaces unique in-progress values and outputs while a task is running."""

        observed_progresses = []
        observed_payloads = []
        callback_errors = []
        task_id = None

        def capture_progress_update():
            try:
                task = next(iter(server_api.state.task_cache.values()))
                backend = TestBackend.instances[0]
                task._refresh_progress(backend)  # hack: ensure progress is updated before fetching details

                detail_response = client.get(f"/v1/tasks/{task_id}")
                output_response = client.get(f"/v1/tasks/{task_id}/outputs/0")

                if detail_response.status_code != 200:
                    return

                observed_progresses.append(detail_response.json()["progress"])
                observed_payloads.append(output_response.content)
            except Exception as error:
                callback_errors.append(error)

        TestBackend.GENERATE_STEP_CALLBACK = capture_progress_update

        create_response = client.post(
            "/v1/tasks",
            json={
                "username": "easydiffusion",
                "prompt": "Test",
                "model_paths": {"stable-diffusion": "test"},
                "num_inference_steps": 4,
            },
        )
        task_id = create_response.json()["task_id"]

        assert server_api.state.workers.wait(timeout=2.0)
        assert not callback_errors

        assert observed_progresses, "No in-progress values were observed"
        assert all(0.0 < progress < 1.0 for progress in observed_progresses)
        assert all(a < b for a, b in zip(observed_progresses, observed_progresses[1:]))
        assert len(observed_progresses) == len(set(observed_progresses))

        assert observed_payloads, "No in-progress outputs were observed"
        assert len(observed_payloads) == len(observed_progresses)
        assert len(set(observed_payloads)) == len(observed_payloads)

        final_detail_response = client.get(f"/v1/tasks/{task_id}")
        assert final_detail_response.status_code == 200
        assert final_detail_response.json()["outputs"] == [f"/v1/tasks/{task_id}/outputs/0"]

        final_output_response = client.get(f"/v1/tasks/{task_id}/outputs/0")
        assert final_output_response.status_code == 200
        assert final_output_response.content not in observed_payloads

    def test_get_task_detail_with_request(self, client):
        """Test getting task details with request data."""
        create_response = client.post(
            "/v1/tasks",
            json={
                "username": "easydiffusion",
                "prompt": "Test prompt",
                "model_paths": {"stable-diffusion": "test"},
                "seed": 42,
            },
        )
        task_id = create_response.json()["task_id"]

        response = client.get(f"/v1/tasks/{task_id}?include_request=true")
        assert response.status_code == 200

        data = response.json()
        assert data["request"] is not None
        assert data["request"]["request"]["prompt"] == "Test prompt"
        assert data["request"]["request"]["seed"] == 42
        assert data["request"]["task"]["task_id"] == task_id

    def test_get_task_not_found(self, client):
        """Test getting non-existent task."""
        response = client.get("/v1/tasks/nonexistent-task-id")
        assert response.status_code == 404

    def test_stop_task(self, client):
        """Test stopping a task."""
        create_response = client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test", "model_paths": {"stable-diffusion": "test"}},
        )
        task_id = create_response.json()["task_id"]

        response = client.delete(f"/v1/tasks/{task_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stopped"

    def test_stop_task_twice(self, client):
        """Test stopping an already stopped task."""
        create_response = client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test", "model_paths": {"stable-diffusion": "test"}},
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
        client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test 1", "model_paths": {"stable-diffusion": "test"}},
        )
        client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test 2", "model_paths": {"stable-diffusion": "test"}},
        )

        response = client.delete("/v1/tasks?username=easydiffusion")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "all_stopped"

    def test_stop_all_tasks_empty(self, client):
        """Test stopping all tasks when none exist."""
        response = client.delete("/v1/tasks?username=easydiffusion")
        assert response.status_code == 200


class TestOutputEndpoint:
    """Tests for task output retrieval endpoint."""

    def test_get_task_output(self, client):
        """Test retrieving a task output."""

        create_response = client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test", "model_paths": {"stable-diffusion": "test"}},
        )
        task_id = create_response.json()["task_id"]

        assert server_api.state.workers.wait(timeout=2.0)

        response = client.get(f"/v1/tasks/{task_id}/outputs/0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_get_filter_task_output(self, client):
        """Test retrieving a filtered task output."""

        create_response = client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "image": "image-data", "filter": "blur"},
        )
        task_id = create_response.json()["task_id"]

        assert server_api.state.workers.wait(timeout=2.0)

        response = client.get(f"/v1/tasks/{task_id}/outputs/0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_get_task_output_not_found(self, client):
        """Test retrieving an out-of-range output index."""
        create_response = client.post(
            "/v1/tasks",
            json={"username": "easydiffusion", "prompt": "Test", "model_paths": {"stable-diffusion": "test"}},
        )
        task_id = create_response.json()["task_id"]

        assert server_api.state.workers.wait(timeout=2.0)

        response = client.get(f"/v1/tasks/{task_id}/outputs/1")
        assert response.status_code == 404

    def test_get_task_output_invalid_task(self, client):
        """Test retrieving output from non-existent task."""
        response = client.get("/v1/tasks/invalid-task/outputs/0")
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

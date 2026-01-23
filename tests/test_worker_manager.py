"""
Tests for the worker manager system.
"""

import time
import threading
from unittest.mock import patch
from easydiffusion.task_queue import TaskQueue
from easydiffusion.worker_manager import BackendWorker, WorkerManager
from easydiffusion.backends import Backend, BACKEND_REGISTRY
from torchruntime.device_db import GPU


class MockBackend(Backend):
    """A mock backend for testing."""

    def __init__(self, device: GPU):
        """Initialize the mock backend."""
        super().__init__(device)
        self.initialized = True
        self.stop_called = False
        self.start_called = False
        self.start_thread_id = None
        self.tasks_processed = []
        self.lock = threading.Lock()

    def install(self) -> None:
        """Install the backend (no-op for mock)."""
        pass

    def uninstall(self) -> None:
        """Uninstall the backend (no-op for mock)."""
        pass

    def start(self):
        """Start the backend (called from worker thread)."""
        self.start_called = True
        self.start_thread_id = threading.current_thread().ident

    def stop(self):
        """Stop the backend."""
        self.stop_called = True

    def ping(self, timeout: float = 1.0) -> bool:
        """Ping the backend."""
        return True

    def render_image(self, context, **kwargs):
        """Render an image (mock implementation)."""
        return None

    def process(self, data):
        """Process some data."""
        with self.lock:
            self.tasks_processed.append(data)
        return f"Processed: {data}"


class MockTask:
    """A mock task for testing."""

    def __init__(self, data, callback=None):
        """Initialize the task."""
        self.data = data
        self.callback = callback
        self.result = None
        self.backend_used = None

    def run(self, backend):
        """Run the task with the given backend."""
        self.backend_used = backend
        self.result = backend.process(self.data)
        if self.callback:
            self.callback(self)
        return self.result


class TestBackendWorker:
    """Test the BackendWorker class."""

    def test_initialization(self):
        """Test that a BackendWorker can be initialized."""
        gpu = GPU("10de", "NVIDIA", "2684", "RTX 4090", True)
        backend = MockBackend(device=gpu)
        worker = BackendWorker(gpu.device_name, backend)
        assert worker.name == gpu.device_name
        assert worker.backend is not None
        assert worker.backend is backend

    def test_task_execution(self):
        """Test that tasks are executed with the backend."""
        gpu = GPU("10de", "NVIDIA", "2684", "RTX 4090", True)
        backend = MockBackend(device=gpu)
        worker = BackendWorker(gpu.device_name, backend)
        task = MockTask("test-data")

        result = worker.run(task)

        assert result == "Processed: test-data"
        assert task.result == "Processed: test-data"
        assert task.backend_used is not None
        assert task.backend_used is backend

    def test_multiple_tasks_same_backend(self):
        """Test that multiple tasks use the same backend instance."""
        gpu = GPU("10de", "NVIDIA", "2684", "RTX 4090", True)
        backend = MockBackend(device=gpu)
        worker = BackendWorker(gpu.device_name, backend)

        task1 = MockTask("data-1")
        task2 = MockTask("data-2")

        worker.run(task1)
        worker.run(task2)

        # Both tasks should use the same backend instance
        assert task1.backend_used is task2.backend_used
        assert len(backend.tasks_processed) == 2

    def test_shutdown(self):
        """Test that shutdown cleans up the backend."""
        gpu = GPU("10de", "NVIDIA", "2684", "RTX 4090", True)
        backend = MockBackend(device=gpu)
        worker = BackendWorker(gpu.device_name, backend)
        task = MockTask("test-data")
        worker.run(task)

        assert not backend.stop_called

        worker.shutdown()

        assert backend.stop_called
        assert worker.backend is None

    def test_backend_start_called_in_worker_thread(self):
        """Test that backend.start() is called in the worker thread."""
        task_queue = TaskQueue()
        gpu = GPU("10de", "NVIDIA", "2684", "RTX 4090", True)
        backend = MockBackend(device=gpu)
        worker = BackendWorker(gpu.device_name, backend)

        # Start the worker
        task_queue.add_worker(worker)

        # Give the worker thread time to start and call backend.start()
        time.sleep(0.2)

        # Verify backend.start() was called
        assert backend.start_called

        # Verify it was called from a different thread (not main thread)
        main_thread_id = threading.current_thread().ident
        assert backend.start_thread_id is not None
        assert backend.start_thread_id != main_thread_id

        # Cleanup
        task_queue.shutdown(timeout=2.0)


class TestWorkerManager:
    """Test the WorkerManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Register the mock backend
        BACKEND_REGISTRY["mock"] = MockBackend

        self.task_queue = TaskQueue()
        self.worker_manager = WorkerManager(self.task_queue, "mock")

    def teardown_method(self):
        """Clean up after tests."""
        self.worker_manager.shutdown_all(timeout=2.0)

        # Unregister the mock backend
        if "mock" in BACKEND_REGISTRY:
            del BACKEND_REGISTRY["mock"]

    def test_initialization(self):
        """Test that a WorkerManager can be initialized."""
        assert len(self.task_queue) == 0
        assert len(self.worker_manager.get_active_devices()) == 0

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_update_workers_add_devices(self, mock_get_gpus):
        """Test adding workers for new devices."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
            GPU("10de", "NVIDIA", "2704", "1", True),
        ]
        devices_to_add = ["cuda:0", "cuda:1", "cpu"]
        self.worker_manager.update_workers(devices_to_add)

        # Give workers time to start
        time.sleep(0.2)

        active_devices = self.worker_manager.get_active_devices()
        assert len(active_devices) == 3
        # cuda:0 and cuda:1 are resolved to '0' and '1'
        assert "0" in active_devices
        assert "1" in active_devices
        assert "cpu" in active_devices

        # Verify workers are BackendWorker instances
        for device_id in ["0", "1", "cpu"]:
            worker = self.task_queue.get_worker(device_id)
            assert isinstance(worker, BackendWorker)
            # Verify backend.start() was called
            assert worker.backend.start_called

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_update_workers_remove_devices(self, mock_get_gpus):
        """Test removing workers for devices."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
            GPU("10de", "NVIDIA", "2704", "1", True),
        ]
        # Add some workers first
        devices_to_add = ["cuda:0", "cuda:1", "cpu"]
        self.worker_manager.update_workers(devices_to_add)
        assert len(self.worker_manager.get_active_devices()) == 3

        # Remove some workers (keeping only cuda:0 which resolves to '0')
        self.worker_manager.update_workers(["cuda:0"])

        active_devices = self.worker_manager.get_active_devices()
        assert len(active_devices) == 1
        assert "0" in active_devices
        assert "1" not in active_devices
        assert "cpu" not in active_devices

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_update_workers_add_and_remove(self, mock_get_gpus):
        """Test adding and removing workers simultaneously."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
            GPU("10de", "NVIDIA", "2704", "1", True),
            GPU("10de", "NVIDIA", "2705", "2", True),
        ]
        # Add initial workers
        self.worker_manager.update_workers(["cuda:0", "cuda:1"])
        assert len(self.worker_manager.get_active_devices()) == 2

        # Add new and remove old in same call
        self.worker_manager.update_workers(["cuda:1", "cpu", "cuda:2"])

        active_devices = self.worker_manager.get_active_devices()
        assert len(active_devices) == 3
        assert "1" in active_devices  # cuda:1 resolved to '1'
        assert "cpu" in active_devices
        assert "2" in active_devices  # cuda:2 resolved to '2'
        assert "0" not in active_devices  # cuda:0 was removed

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_update_workers_duplicate_add(self, mock_get_gpus):
        """Test that adding a device twice doesn't create duplicate workers."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
        ]
        self.worker_manager.update_workers(["cuda:0"])
        assert len(self.worker_manager.get_active_devices()) == 1

        # Try to add the same device again
        self.worker_manager.update_workers(["cuda:0"])
        assert len(self.worker_manager.get_active_devices()) == 1

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_update_workers_remove_nonexistent(self, mock_get_gpus):
        """Test that removing a nonexistent device doesn't raise an error."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
        ]
        self.worker_manager.update_workers(["cuda:0"])

        # Try to remove a device that doesn't exist - should not raise (just update to same state)
        self.worker_manager.update_workers(["cuda:0"])

        # Original device should still be there (cuda:0 resolved to '0')
        active_devices = self.worker_manager.get_active_devices()
        assert len(active_devices) == 1
        assert "0" in active_devices

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_workers_process_tasks(self, mock_get_gpus):
        """Test that workers actually process tasks from the queue."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
            GPU("10de", "NVIDIA", "2704", "1", True),
        ]
        # Add workers
        self.worker_manager.update_workers(["cuda:0", "cuda:1"])

        # Give workers time to start
        time.sleep(0.1)

        # Add tasks
        completed_tasks = []
        lock = threading.Lock()

        def on_complete(task):
            with lock:
                completed_tasks.append(task)

        tasks = []
        for i in range(10):
            task = MockTask(f"data-{i}", callback=on_complete)
            tasks.append(task)
            self.task_queue.add_task(task)

        # Wait for tasks to complete
        self.task_queue.wait_completion(timeout=5.0)

        # All tasks should be completed
        assert len(completed_tasks) == 10

        # Each task should have been processed
        for task in tasks:
            assert task.result is not None
            assert task.backend_used is not None
            assert task.data in task.backend_used.tasks_processed

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_shutdown_all(self, mock_get_gpus):
        """Test shutting down all workers."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
            GPU("10de", "NVIDIA", "2704", "1", True),
        ]
        # Add workers
        self.worker_manager.update_workers(["cuda:0", "cuda:1"])
        time.sleep(0.1)

        # Get backends (cuda:0 and cuda:1 resolve to '0' and '1')
        worker1 = self.task_queue.get_worker("0")
        worker2 = self.task_queue.get_worker("1")
        backend1 = worker1.backend
        backend2 = worker2.backend

        # Shutdown all
        self.worker_manager.shutdown_all(timeout=2.0)

        # All workers should be stopped
        assert len(self.worker_manager.get_active_devices()) == 0

        # Backends should be shutdown
        assert backend1.stop_called
        assert backend2.stop_called

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_backend_shutdown_on_worker_removal(self, mock_get_gpus):
        """Test that backend is shutdown when worker is removed."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
        ]
        self.worker_manager.update_workers(["cuda:0"])
        time.sleep(0.1)

        # cuda:0 resolves to '0'
        worker = self.task_queue.get_worker("0")
        backend = worker.backend
        assert not backend.stop_called

        # Remove the worker
        self.worker_manager.update_workers([])

        # Backend should be shutdown
        assert backend.stop_called

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_worker_manager_with_backend_args(self, mock_get_gpus):
        """Test that WorkerManager creates backends correctly."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
        ]
        # Use the mock backend registered in setup_method
        worker_manager = WorkerManager(self.task_queue, "mock")

        worker_manager.update_workers(["cuda:0"])
        time.sleep(0.1)

        # cuda:0 resolves to '0'
        worker = self.task_queue.get_worker("0")
        # Backend should be created with GPU object with device_name '0'
        assert worker.backend.device.device_name == "0"

        worker_manager.shutdown_all(timeout=2.0)


class TestIntegration:
    """Integration tests for WorkerManager with TaskQueue."""

    def setup_method(self):
        """Set up test fixtures."""
        # Register the mock backend
        BACKEND_REGISTRY["mock"] = MockBackend

    def teardown_method(self):
        """Clean up after tests."""
        # Unregister the mock backend
        if "mock" in BACKEND_REGISTRY:
            del BACKEND_REGISTRY["mock"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_concurrent_task_processing(self, mock_get_gpus):
        """Test processing multiple tasks concurrently on multiple devices."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
            GPU("10de", "NVIDIA", "2704", "1", True),
        ]
        task_queue = TaskQueue()
        worker_manager = WorkerManager(task_queue, "mock")

        # Add multiple workers
        worker_manager.update_workers(["cuda:0", "cuda:1", "cpu"])
        time.sleep(0.1)

        # Add many tasks
        num_tasks = 30
        completed_tasks = []
        lock = threading.Lock()

        def on_complete(task):
            with lock:
                completed_tasks.append(task)

        for i in range(num_tasks):
            task = MockTask(f"data-{i}", callback=on_complete)
            task_queue.add_task(task)

        # Wait for all tasks to complete
        success = task_queue.wait_completion(timeout=10.0)
        assert success
        assert len(completed_tasks) == num_tasks

        # Cleanup
        worker_manager.shutdown_all(timeout=2.0)

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_dynamic_worker_scaling(self, mock_get_gpus):
        """Test dynamically adding and removing workers while processing tasks."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "0", True),
            GPU("10de", "NVIDIA", "2704", "1", True),
        ]
        task_queue = TaskQueue()
        worker_manager = WorkerManager(task_queue, "mock")

        # Start with one worker
        worker_manager.update_workers(["cuda:0"])
        time.sleep(0.1)

        # Add tasks
        num_tasks = 20
        for i in range(num_tasks):
            task_queue.add_task(MockTask(f"data-{i}"))

        # Scale up
        time.sleep(0.2)
        worker_manager.update_workers(["cuda:0", "cuda:1", "cpu"])

        # Add more tasks
        for i in range(num_tasks, num_tasks * 2):
            task_queue.add_task(MockTask(f"data-{i}"))

        # Scale down
        time.sleep(0.2)
        worker_manager.update_workers(["cuda:1", "cpu"])

        # Wait for completion
        success = task_queue.wait_completion(timeout=10.0)
        assert success

        # Cleanup
        worker_manager.shutdown_all(timeout=2.0)

import threading
import time

from torchruntime.device_db import GPU

from easydiffusion.backends import BACKEND_REGISTRY
from easydiffusion.types import Task
from easydiffusion.workers import Workers
from support import DummyBackend, DummyBackendTask


class ExplodingTask(Task):
    def __init__(self):
        super().__init__(username="test-user")

    def run(self, backend):
        raise RuntimeError("boom")


class StopTask(Task):
    def __init__(self):
        super().__init__(username="test-user")

    def run(self, backend):
        raise StopAsyncIteration("stop requested")


class SleepTask(Task):
    def __init__(self, duration):
        super().__init__(username="test-user")
        self.duration = duration

    def run(self, backend):
        time.sleep(self.duration)


class TestWorkers:
    def setup_method(self):
        BACKEND_REGISTRY["dummy"] = DummyBackend
        DummyBackend.reset_mock_state()
        self.gpu0 = GPU("10de", "NVIDIA", "2684", "0", True)
        self.gpu1 = GPU("10de", "NVIDIA", "2704", "1", True)
        self.cpu = GPU("cpu", "CPU", "cpu", "cpu", False)
        self.workers = Workers(DummyBackend, backend_name="dummy")

    def teardown_method(self):
        self.workers.shutdown()
        BACKEND_REGISTRY.pop("dummy", None)

    def test_initialization(self):
        assert self.workers.backend_class is DummyBackend
        assert self.workers.backend_name == "dummy"
        assert self.workers.get_active_devices() == []
        assert self.workers.qsize() == 0

    def test_update_devices_adds_and_removes_workers(self):
        self.workers.update_devices([self.gpu0, self.cpu])

        active_devices = self.workers.get_active_devices()
        assert set(active_devices) == {"0", "cpu"}

        self.workers.update_devices([self.gpu1])

        assert self.workers.get_active_devices() == ["1"]

    def test_update_devices_does_not_duplicate_workers(self):
        self.workers.update_devices([self.gpu0])
        first_thread = self.workers.workers["0"][0]

        self.workers.update_devices([self.gpu0])

        assert self.workers.get_active_devices() == ["0"]
        assert self.workers.workers["0"][0] is first_thread

    def test_submit_processes_tasks(self):
        self.workers.update_devices([self.gpu0])
        tasks = [DummyBackendTask(f"data-{index}") for index in range(3)]

        for task in tasks:
            self.workers.submit(task)

        assert self.workers.wait(timeout=2.0)
        assert [task.status for task in tasks] == ["completed", "completed", "completed"]
        assert all(task.backend_used is not None for task in tasks)
        assert tasks[0].backend_used is tasks[1].backend_used

    def test_wait_times_out(self):
        self.workers.update_devices([self.gpu0])
        self.workers.submit(SleepTask(0.5))

        assert not self.workers.wait(timeout=0.05)
        assert self.workers.wait(timeout=2.0)

    def test_stop_async_iteration_marks_task_stopped(self):
        self.workers.update_devices([self.gpu0])
        task = StopTask()

        self.workers.submit(task)

        assert self.workers.wait(timeout=2.0)
        assert task.status == "stopped"
        assert task.error == "stop requested"

    def test_exception_marks_task_failed(self):
        self.workers.update_devices([self.gpu0])
        task = ExplodingTask()

        self.workers.submit(task)

        assert self.workers.wait(timeout=2.0)
        assert task.status == "error"
        assert task.error == "boom"

    def test_backend_start_runs_on_worker_thread(self):
        self.workers.update_devices([self.gpu0])
        deadline = time.time() + 2.0
        while time.time() < deadline and not DummyBackend.instances:
            time.sleep(0.01)

        assert DummyBackend.instances
        backend = DummyBackend.instances[0]
        assert backend.start_called
        assert backend.start_thread_id is not None
        assert backend.start_thread_id != threading.current_thread().ident

    def test_shutdown_stops_backends(self):
        self.workers.update_devices([self.gpu0, self.gpu1])
        deadline = time.time() + 2.0
        while time.time() < deadline and len(DummyBackend.instances) < 2:
            time.sleep(0.01)

        instances = list(DummyBackend.instances)
        self.workers.shutdown()

        assert self.workers.get_active_devices() == []
        assert all(instance.stop_called for instance in instances)

    def test_set_backend_reuses_workers_when_name_unchanged(self):
        self.workers.update_devices([self.cpu])
        first_thread = self.workers.workers["cpu"][0]

        self.workers.set_backend("dummy", [self.cpu])

        assert self.workers.workers["cpu"][0] is first_thread

    def test_set_backend_recreates_workers_when_name_changes(self):
        class OtherDummyBackend(DummyBackend):
            pass

        BACKEND_REGISTRY["other"] = OtherDummyBackend
        self.workers.update_devices([self.cpu])
        old_instances = list(DummyBackend.instances)

        self.workers.set_backend("other", [self.cpu])
        task = DummyBackendTask("new-backend")
        self.workers.submit(task)

        assert self.workers.wait(timeout=2.0)
        assert isinstance(task.backend_used, OtherDummyBackend)
        assert all(instance.stop_called for instance in old_instances)

        BACKEND_REGISTRY.pop("other", None)
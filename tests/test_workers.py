import time
import queue

from torchruntime.device_db import GPU

from easydiffusion.backends import BACKEND_REGISTRY
from easydiffusion.workers import Task, Worker, Workers
from .support import TestBackend


def wait_until(condition, timeout=1.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)

    return condition()


class SuccessfulTask(Task):
    def __init__(self):
        super().__init__(username="test-user")

    def run(self, backend):
        pass


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


class TestWorkers:
    def setup_method(self):
        TestBackend.reset_mock_state()
        self.gpu0 = GPU("10de", "NVIDIA", "2684", "0", True)
        self.gpu1 = GPU("10de", "NVIDIA", "2704", "1", True)
        self.cpu = GPU("cpu", "CPU", "cpu", "cpu", False)
        self.workers = Workers(TestBackend, backend_name="dummy")

    def teardown_method(self):
        self.workers.shutdown()

    def test_initialization(self):
        assert self.workers.backend_class is TestBackend
        assert self.workers.backend_name == "dummy"
        assert self.workers.get_active_devices() == []
        assert self.workers.queue_size() == 0

    def test_update_devices_adds_and_removes_workers(self):
        self.workers.update_devices([self.gpu0, self.cpu])

        active_devices = self.workers.get_active_devices()
        assert set(active_devices) == {"0", "cpu"}

        self.workers.update_devices([self.gpu1])

        assert self.workers.get_active_devices() == ["1"]

    def test_update_devices_does_not_duplicate_workers(self):
        self.workers.update_devices([self.gpu0])
        first_worker = self.workers.workers["0"]

        self.workers.update_devices([self.gpu0])

        assert self.workers.get_active_devices() == ["0"]
        assert self.workers.workers["0"] is first_worker

    def test_submit_processes_tasks(self):
        self.workers.update_devices([self.gpu0])
        tasks = [SuccessfulTask() for _ in range(3)]

        for task in tasks:
            self.workers.submit(task)

        assert self.workers.wait(timeout=2.0)
        assert [task.status for task in tasks] == ["completed", "completed", "completed"]

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

    def test_shutdown_stops_backends(self):
        self.workers.update_devices([self.gpu0, self.gpu1])
        self.workers.shutdown()

        instances = list(TestBackend.instances)
        assert len(instances) == 2
        assert self.workers.get_active_devices() == []
        assert all(instance.stop_called for instance in instances)

    def test_is_any_backend_ready_returns_true_when_one_backend_is_ready(self):
        class MixedReadyBackend(TestBackend):
            def ping(self, timeout=0.1):
                return self.device.device_name == "1"

        workers = Workers(MixedReadyBackend, backend_name="mixed")

        try:
            workers.update_devices([self.gpu0, self.gpu1])
            assert wait_until(workers.any_ready)
        finally:
            workers.shutdown()

    def test_is_any_backend_ready_returns_false_when_no_backend_is_ready(self):
        self.workers.update_devices([self.gpu0])

        try:
            TestBackend.PING_RESPONSE = False
            assert wait_until(lambda: self.workers.any_ready() is False)
        finally:
            TestBackend.PING_RESPONSE = True

    def test_worker_does_not_pick_up_tasks_until_backend_is_ready(self):
        TestBackend.PING_RESPONSE = False
        self.workers.update_devices([self.gpu0])
        task = SuccessfulTask()

        try:
            self.workers.submit(task)

            assert self.workers.wait(timeout=0.1) is False
            assert task.status == "pending"
            assert self.workers.queue_size() == 1

            TestBackend.PING_RESPONSE = True

            assert wait_until(self.workers.any_ready)
            assert self.workers.wait(timeout=1.0)
            assert task.status == "completed"
        finally:
            TestBackend.PING_RESPONSE = True

    def test_worker_is_ready_uses_cached_backend_status(self):
        class CountingBackend(TestBackend):
            ping_calls = 0

            def ping(self, timeout=0.1):
                type(self).ping_calls += 1
                return True

        original_interval = Worker.BACKEND_PING_CHECK_INTERVAL
        Worker.BACKEND_PING_CHECK_INTERVAL = 1.0
        worker = Worker(CountingBackend, self.gpu0, queue.Queue())

        try:
            worker.start()

            assert wait_until(lambda: CountingBackend.ping_calls > 0)
            ping_calls = CountingBackend.ping_calls

            assert worker.is_ready()
            assert worker.is_ready()
            assert worker.is_ready()
            assert CountingBackend.ping_calls == ping_calls
        finally:
            worker.stop()
            worker.join()
            Worker.BACKEND_PING_CHECK_INTERVAL = original_interval

    def test_set_backend_reuses_workers_when_name_unchanged(self):
        self.workers.update_devices([self.cpu])
        first_worker = self.workers.workers["cpu"]

        self.workers.set_backend("dummy", [self.cpu])

        assert self.workers.workers["cpu"] is first_worker

    def test_set_backend_recreates_workers_when_name_changes(self):
        class OtherDummyBackend(TestBackend):
            pass

        BACKEND_REGISTRY["other"] = OtherDummyBackend
        self.workers.update_devices([self.cpu])
        first_worker = self.workers.workers["cpu"]

        self.workers.set_backend("other", [self.cpu])

        assert isinstance(self.workers.workers["cpu"], Worker)
        assert self.workers.workers["cpu"] is not first_worker

        BACKEND_REGISTRY.pop("other", None)

    def test_dummy_backend_stores_runtime_config(self):
        backend = TestBackend(self.cpu, config={"foo": True})
        config = {"custom_value": "ok"}

        backend.set_config(config)
        config["custom_value"] = "changed"

        assert backend.config == {"foo": True}
        stored = backend.get_config()
        assert stored == {"custom_value": "ok"}

        stored["custom_value"] = "other"
        assert backend.get_config()["custom_value"] == "ok"

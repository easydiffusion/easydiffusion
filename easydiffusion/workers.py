import queue
import threading
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from .backends import get_backend_class


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "error"
    STOPPED = "stopped"


class Task:
    PROGRESS_UPDATE_INTERVAL = 0.3  # seconds

    def __init__(
        self,
        username: str,
        task_type: str = "task",
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.session_id = session_id or "default"
        self.username = username
        self.task_type = task_type

        self._status = TaskStatus.PENDING
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.outputs: list[bytes] = []
        self.input: Optional[Dict[str, Any]] = input

    @property
    def status(self) -> str:
        return self._status.value

    @property
    def is_terminal(self) -> bool:
        return self._status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED}

    def mark_running(self) -> None:
        if not self.is_terminal:
            self._status = TaskStatus.RUNNING

    def update_progress(self, progress: float, outputs: list[bytes]) -> None:
        self.progress = progress
        self.outputs = outputs
        if self._status == TaskStatus.PENDING:
            self._status = TaskStatus.RUNNING

    def mark_completed(self) -> None:
        if self._status not in {TaskStatus.FAILED, TaskStatus.STOPPED}:
            if self.progress < 1.0:
                self.progress = 1.0
            self._status = TaskStatus.COMPLETED

    def mark_failed(self, error: Any) -> None:
        self.error = str(error)
        self._status = TaskStatus.FAILED

    def stop(self, reason: str = "") -> None:
        if reason:
            self.error = reason
        self._status = TaskStatus.STOPPED

    @property
    def is_pending(self) -> bool:
        return self._status in {TaskStatus.PENDING, TaskStatus.RUNNING}

    def __repr__(self):
        return f"Task(task_id={self.task_id}, status={self.status}, type={self.task_type})"

    def run(self, backend: Any) -> Any:
        stop_progress = threading.Event()
        progress_worker = threading.Thread(
            target=self._track_progress,
            args=(backend, stop_progress),
            daemon=True,
            name=f"task-progress-{self.task_id}",
        )
        progress_worker.start()

        try:
            if self.task_type == "generate":
                result = backend.generate(self.input)
            elif self.task_type == "filter":
                result = backend.filter(self.input)
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        finally:
            stop_progress.set()
            progress_worker.join()

        self.outputs = result or []
        return self.outputs

    def _track_progress(self, backend: Any, stop_event: threading.Event) -> None:
        stop_requested = False

        while True:
            self._refresh_progress(backend)

            if self.status == TaskStatus.STOPPED.value and not stop_requested:
                backend.stop_task()
                stop_requested = True

            if stop_event.wait(timeout=self.PROGRESS_UPDATE_INTERVAL):
                break

        self._refresh_progress(backend)

    def _refresh_progress(self, backend: Any) -> None:
        try:
            progress = backend.get_progress()
            outputs = backend.get_progress_outputs()
            self.update_progress(progress, outputs)
        except Exception as error:
            print(f"Error refreshing progress for task: {error}")
            time.sleep(0.01)


class Worker:
    BACKEND_PING_CHECK_INTERVAL = 1.0

    def __init__(self, backend_class, device, task_queue, backend_config=None):
        self.device = device
        self.backend_class = backend_class
        self.backend_config = dict(backend_config or {})
        self.task_queue = task_queue
        self.stop_flag = threading.Event()
        self.backend = None
        self._backend_status = False
        self._next_ping_time = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True, name=device.device_name)

    def _refresh_backend_status(self):
        if self.backend is None:
            self._backend_status = False
            return

        try:
            self._backend_status = self.backend.ping(timeout=0.1)
        except Exception:
            self._backend_status = False
        finally:
            self._next_ping_time = time.monotonic() + self.BACKEND_PING_CHECK_INTERVAL

    def _run(self):
        self.backend = self.backend_class(self.device, config=self.backend_config)
        self.backend.start()

        while not self.stop_flag.is_set():
            if time.monotonic() >= self._next_ping_time:
                self._refresh_backend_status()

            if not self._backend_status:
                self.stop_flag.wait(timeout=0.1)
                continue

            try:
                task: Task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if task.status == "stopped":
                    continue

                task.mark_running()
                task.run(self.backend)

                if task.status not in ("completed", "error", "stopped"):
                    task.mark_completed()
            except StopAsyncIteration as error:
                task.stop(str(error))
            except Exception as error:
                print("task error", task, error)
                task.mark_failed(error)
            finally:
                self.task_queue.task_done()

        self.backend.stop()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag.set()

    def join(self):
        self.thread.join()

    def is_ready(self, timeout=0.1):
        return self.backend is not None and self._backend_status


class Workers:
    def __init__(self, backend_class, backend_name=None, backend_config=None, maxsize=0):
        self.backend_class = backend_class
        self.backend_name = backend_name or getattr(backend_class, "__name__", "backend")
        self.backend_config = dict(backend_config or {})
        self.q = queue.Queue(maxsize=maxsize)
        self.workers = {}
        self.lock = threading.Lock()

    def update_devices(self, devices):
        from .utils.device_utils import resolve_devices

        if isinstance(devices, str):
            devices = resolve_devices(devices)
        else:
            devices = list(devices)
            if devices and isinstance(devices[0], str):
                devices = resolve_devices(devices)

        names = set(device.device_name for device in devices)

        with self.lock:
            current = set(self.workers.keys())

            for name in current - names:
                worker = self.workers.pop(name)
                worker.stop()
                worker.join()

            for device in devices:
                if device.device_name in self.workers:
                    continue

                worker = Worker(self.backend_class, device, self.q, backend_config=self.backend_config)
                self.workers[device.device_name] = worker
                worker.start()

    def set_backend(self, backend_name, devices, backend_config=None):
        if backend_name == self.backend_name:
            self.backend_config = dict(backend_config or self.backend_config)
            self.update_devices(devices)
            return

        self.shutdown()
        self.backend_name = backend_name
        self.backend_class = get_backend_class(backend_name)
        self.backend_config = dict(backend_config or {})
        self.update_devices(devices)

    def get_active_devices(self):
        with self.lock:
            return list(self.workers.keys())

    def any_ready(self, timeout=0.1):
        with self.lock:
            workers = list(self.workers.values())

        return any(worker.is_ready(timeout=timeout) for worker in workers)

    def submit(self, task):
        self.q.put(task)

    def wait(self, timeout=None):
        if timeout is None:
            self.q.join()
            return True

        # Use a temporary thread to call the blocking join()
        t = threading.Thread(target=self.q.join, daemon=True)
        t.start()
        t.join(timeout)

        # If the thread is still alive, join() timed out
        return not t.is_alive()

    def queue_size(self):
        return self.q.qsize()

    def shutdown(self):
        with self.lock:
            items = list(self.workers.items())
            self.workers.clear()

        for _, worker in items:
            worker.stop()
            worker.join()

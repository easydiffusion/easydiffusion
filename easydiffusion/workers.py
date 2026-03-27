import queue
import threading
import time

from .backends import get_backend_class
from .tasks import Task


class Workers:
    def __init__(self, backend_class, backend_name=None, backend_config=None, maxsize=0):
        self.backend_class = backend_class
        self.backend_name = backend_name or getattr(backend_class, "__name__", "backend")
        self.backend_config = dict(backend_config or {})
        self.q = queue.Queue(maxsize=maxsize)
        self.workers = {}
        self.lock = threading.Lock()

    def _worker(self, name, device, stop_flag):
        backend = self.backend_class(device, config=self.backend_config)
        backend.start()

        while not stop_flag.is_set():
            try:
                task: Task = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if task.status == "stopped":
                    continue

                task.mark_running()

                task.run(backend)

                if task.status not in ("completed", "error", "stopped"):
                    task.mark_completed()

            except StopAsyncIteration as error:
                task.stop(str(error))
            except Exception as error:
                print("task error", task, error)
                task.mark_failed(error)
            finally:
                self.q.task_done()

        backend.stop()

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
                thread, stop_flag = self.workers.pop(name)
                stop_flag.set()
                thread.join()

            for device in devices:
                if device.device_name in self.workers:
                    continue

                stop_flag = threading.Event()
                thread = threading.Thread(
                    target=self._worker,
                    args=(device.device_name, device, stop_flag),
                    daemon=True,
                    name=device.device_name,
                )
                self.workers[device.device_name] = (thread, stop_flag)
                thread.start()

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

        for _, (thread, stop_flag) in items:
            stop_flag.set()
            thread.join()

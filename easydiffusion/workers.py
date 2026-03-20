import queue
import threading
import time

from .backends import get_backend_class


class Workers:
    def __init__(self, backend_class, backend_name=None, maxsize=0):
        self.backend_class = backend_class
        self.backend_name = backend_name or getattr(backend_class, "__name__", "backend")
        self.q = queue.Queue(maxsize=maxsize)
        self.workers = {}
        self.lock = threading.Lock()

    def _worker(self, name, device, stop_flag):
        backend = self.backend_class(device)
        backend.start()

        while not stop_flag.is_set():
            try:
                task = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if getattr(task, "status", None) == "stopped":
                    continue

                if hasattr(task, "mark_running"):
                    task.mark_running()

                task.run(backend)

                if hasattr(task, "mark_completed") and getattr(task, "status", None) not in (
                    "completed",
                    "error",
                    "stopped",
                ):
                    task.mark_completed()

            except StopAsyncIteration as error:
                if hasattr(task, "request_stop"):
                    task.request_stop(str(error))
                else:
                    raise

            except Exception as error:
                if hasattr(task, "mark_failed"):
                    task.mark_failed(error)
                else:
                    raise

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

    def set_backend(self, backend_name, devices):
        if backend_name == self.backend_name:
            self.update_devices(devices)
            return

        self.shutdown()
        self.backend_name = backend_name
        self.backend_class = get_backend_class(backend_name)
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

        start = time.time()
        while self.q.unfinished_tasks > 0:
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)
        return True

    def qsize(self):
        return self.q.qsize()

    def shutdown(self):
        with self.lock:
            items = list(self.workers.items())
            self.workers.clear()

        for _, (thread, stop_flag) in items:
            stop_flag.set()
            thread.join()

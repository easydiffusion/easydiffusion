"""
Worker manager for backend devices.

This module provides a worker manager that dynamically creates and removes
BackendWorker instances based on device names. Each worker wraps a backend
instance and processes tasks from the task queue.
"""

import queue
import threading
from typing import Type, Any, List, Union
from .task_queue import Worker, TaskQueue
from .backends import Backend


class BackendWorker(Worker):
    """
    A worker that wraps a backend instance.

    This worker simply calls task.run(backend) for each task it processes.
    """

    def __init__(self, name: str, backend: Backend):
        """
        Initialize the BackendWorker.

        Args:
            name: The name of the worker (typically the device name)
            backend: The backend instance to use for processing tasks
        """
        super().__init__(name)
        self.backend = backend
        self._backend_started = False

    def _worker_loop(self, task_queue: queue.Queue):
        """
        Internal worker loop that processes tasks from the queue.

        Calls backend.start() before processing tasks, ensuring it runs
        in the worker thread context.

        Args:
            task_queue: The queue to get tasks from
        """
        # Call backend.start() in the worker thread
        self.backend.start()
        self._backend_started = True

        # Run the standard worker loop
        super()._worker_loop(task_queue)

    def run(self, task: Any) -> Any:
        """
        Process a task by calling task.run(backend).

        Args:
            task: The task to process (must have a run method)

        Returns:
            The result of task.run(backend)
        """
        return task.run(self.backend)

    def shutdown(self):
        """Shutdown and cleanup the backend instance."""
        # Call stop method
        self.backend.stop()
        self.backend = None


class WorkerManager:
    """
    Manages a pool of BackendWorker instances.

    The WorkerManager dynamically creates and removes workers based on device names,
    creating one worker per device. Each worker wraps a backend instance.
    """

    def __init__(self, task_queue: TaskQueue, backend_name: str):
        """
        Initialize the WorkerManager.

        Args:
            task_queue: The TaskQueue instance to manage workers for
            backend_name: The name of the backend to use (e.g., 'webui', 'sdkit3')
        """
        from .backends import get_backend_class

        self.task_queue = task_queue
        self.backend_name = backend_name
        self.backend_class = get_backend_class(backend_name)
        self._lock = threading.Lock()

    def update_workers(self, desired_devices: Union[str, List[str]]):
        """
        Update workers to match the desired device configuration.

        This method resolves 'auto' device selection, calculates which workers
        to add/remove, and updates the worker pool accordingly.

        Args:
            desired_devices: Device specification ('auto', single device, or list of devices)
        """
        from .utils.device_utils import resolve_devices

        # Resolve 'auto' to actual devices
        new_devices = resolve_devices(desired_devices)  # List[GPU]

        with self._lock:
            # Get current active device names
            active_device_names = self.get_active_devices()  # List[str]

            # Desired device names
            desired_device_names = set(gpu.device_name for gpu in new_devices)

            # Calculate delta
            devices_to_add_names = desired_device_names - set(active_device_names)
            devices_to_remove_names = set(active_device_names) - desired_device_names

            # Remove workers first
            for device_name in devices_to_remove_names:
                try:
                    worker = self.task_queue.get_worker(device_name)
                    if isinstance(worker, BackendWorker):
                        worker.shutdown()
                    self.task_queue.remove_worker(device_name)
                except KeyError:
                    # Worker doesn't exist, skip
                    pass

            # Add new workers
            for gpu in new_devices:
                if gpu.device_name in devices_to_add_names:
                    # Check if worker already exists
                    if self.task_queue.get_worker(gpu.device_name) is not None:
                        continue

                    # Create backend instance and worker
                    backend = self.backend_class(gpu)
                    worker = BackendWorker(gpu.device_name, backend)
                    self.task_queue.add_worker(worker)

    def get_active_devices(self) -> List[str]:
        """
        Get a list of currently active device names.

        Returns:
            List of device names that have active workers
        """
        return self.task_queue.list_workers()

    def set_backend(self, backend_name: str, desired_devices: Union[str, List[str]]):
        """
        Change the backend and recreate all workers if the backend has changed.

        This shuts down all existing workers and creates new ones with the new backend
        only if the backend_name is different from the current one.

        Args:
            backend_name: The name of the backend to use (e.g., 'webui', 'sdkit3')
            desired_devices: Device specification to start workers for
        """
        from .backends import get_backend_class

        # Only recreate workers if backend has actually changed
        if backend_name == self.backend_name:
            # Backend hasn't changed, just update workers if needed
            self.update_workers(desired_devices)
            return

        # Backend has changed - shutdown all existing workers
        self.shutdown_all()

        # Update backend
        self.backend_name = backend_name
        self.backend_class = get_backend_class(backend_name)

        # Start workers with new backend
        self.update_workers(desired_devices)

    def shutdown_all(self, timeout: float = 5.0):
        """
        Shutdown all workers and cleanup backends.

        Args:
            timeout: Maximum time to wait for each worker to stop (in seconds)
        """
        with self._lock:
            worker_names = self.task_queue.list_workers()
            for name in worker_names:
                worker = self.task_queue.get_worker(name)
                if isinstance(worker, BackendWorker):
                    worker.shutdown()

            self.task_queue.shutdown(timeout=timeout)

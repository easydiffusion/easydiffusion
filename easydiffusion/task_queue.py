"""
Simple task queue system with named workers.

This module provides a lightweight task queue implementation with support for
named worker threads that can be dynamically added or removed.
"""

import threading
import queue
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class Worker(ABC):
    """
    Abstract base class for workers.

    Subclasses must implement the run() method to define task processing logic.
    """

    def __init__(self, name: str):
        """
        Initialize the worker.

        Args:
            name: The name of the worker
        """
        self.name = name
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @abstractmethod
    def run(self, task: Any) -> Any:
        """
        Process a task. Must be implemented by subclasses.

        Args:
            task: The task to process

        Returns:
            The result of processing the task
        """
        pass

    def _worker_loop(self, task_queue: queue.Queue):
        """
        Internal worker loop that processes tasks from the queue.

        Args:
            task_queue: The queue to get tasks from
        """
        while not self._stop_event.is_set():
            try:
                # Use timeout to periodically check stop event
                task = task_queue.get(timeout=0.1)
                try:
                    self.run(task)
                finally:
                    task_queue.task_done()
            except queue.Empty:
                continue

    def start(self, task_queue: queue.Queue):
        """
        Start the worker thread.

        Args:
            task_queue: The queue to get tasks from
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"Worker {self.name} is already running")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, args=(task_queue,), name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None):
        """
        Stop the worker thread.

        Args:
            timeout: Maximum time to wait for the thread to stop (in seconds)
        """
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Check if the worker thread is running."""
        return self._thread is not None and self._thread.is_alive()


class TaskQueue:
    """
    A simple task queue system with named workers.

    This class manages a pool of named worker threads that process tasks from
    a shared queue. Workers can be dynamically added or removed by name.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize the task queue.

        Args:
            maxsize: Maximum size of the queue (0 for unlimited)
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._workers: Dict[str, Worker] = {}
        self._lock = threading.Lock()

    def add_worker(self, worker: Worker):
        """
        Add a new worker to the queue.

        Args:
            worker: The worker instance to add

        Raises:
            ValueError: If a worker with the same name already exists
        """
        with self._lock:
            if worker.name in self._workers:
                raise ValueError(f"Worker with name '{worker.name}' already exists")

            self._workers[worker.name] = worker
            worker.start(self._queue)

    def remove_worker(self, name: str, timeout: Optional[float] = 5.0):
        """
        Remove a worker by name.

        Args:
            name: The name of the worker to remove
            timeout: Maximum time to wait for the worker to stop (in seconds)

        Raises:
            KeyError: If no worker with the given name exists
        """
        with self._lock:
            if name not in self._workers:
                raise KeyError(f"No worker with name '{name}' exists")

            worker = self._workers.pop(name)

        worker.stop(timeout=timeout)

    def add_task(self, task: Any, block: bool = True, timeout: Optional[float] = None):
        """
        Add a task to the queue.

        Args:
            task: The task to add
            block: Whether to block if the queue is full
            timeout: Maximum time to wait if blocking (in seconds)

        Raises:
            queue.Full: If the queue is full and block is False or timeout expires
        """
        self._queue.put(task, block=block, timeout=timeout)

    def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks in the queue to be completed.

        Args:
            timeout: Maximum time to wait (in seconds)

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        if timeout is None:
            self._queue.join()
            return True
        else:
            # Queue.join() doesn't support timeout, so we poll
            import time

            start_time = time.time()
            while self._queue.unfinished_tasks > 0:
                if time.time() - start_time > timeout:
                    return False
                time.sleep(0.01)
            return True

    def get_worker(self, name: str) -> Optional[Worker]:
        """
        Get a worker by name.

        Args:
            name: The name of the worker

        Returns:
            The worker instance, or None if not found
        """
        with self._lock:
            return self._workers.get(name)

    def list_workers(self) -> list[str]:
        """
        Get a list of all worker names.

        Returns:
            List of worker names
        """
        with self._lock:
            return list(self._workers.keys())

    def shutdown(self, timeout: Optional[float] = 5.0):
        """
        Shutdown all workers.

        Args:
            timeout: Maximum time to wait for each worker to stop (in seconds)
        """
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()

        for worker in workers:
            worker.stop(timeout=timeout)

    def qsize(self) -> int:
        """Get the approximate size of the queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.empty()

    def __len__(self) -> int:
        """Get the number of workers."""
        with self._lock:
            return len(self._workers)

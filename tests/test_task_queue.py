"""
Tests for the task queue system.
"""

import time
import threading
from easydiffusion.task_queue import Worker, TaskQueue
from easydiffusion.types import Task


class QueueTask(Task):
    def __init__(self, value, callback=None):
        super().__init__(username="test-user")
        self.value = value
        self.callback = callback

    def run(self, backend):
        if self.callback:
            self.callback()
        return self.value


class SimpleWorker(Worker):
    """A simple worker for testing that processes callable tasks."""

    def __init__(self, name: str):
        super().__init__(name)
        self.processed_tasks = []

    def run(self, task):
        """Execute the task and record it."""
        self.processed_tasks.append(task)
        if task.callback:
            task.callback()


class CounterWorker(Worker):
    """A worker that counts processed tasks."""

    def __init__(self, name: str):
        super().__init__(name)
        self.count = 0
        self.lock = threading.Lock()

    def run(self, task):
        """Increment counter for each task."""
        with self.lock:
            self.count += 1
        if task.callback:
            task.callback()


class TestWorker:
    """Test the Worker base class."""

    def test_worker_initialization(self):
        """Test that a worker can be initialized with a name."""
        worker = SimpleWorker("test-worker")
        assert worker.name == "test-worker"
        assert not worker.is_alive()

    def test_worker_must_implement_run(self):
        """Test that Worker.run() is abstract."""
        import pytest

        with pytest.raises(TypeError):
            Worker("abstract-worker")

    def test_worker_start_stop(self):
        """Test starting and stopping a worker."""
        task_queue = TaskQueue()
        worker = SimpleWorker("test-worker")

        # Start the worker
        worker.start(task_queue._queue)
        assert worker.is_alive()

        # Stop the worker
        worker.stop(timeout=1.0)
        assert not worker.is_alive()

    def test_worker_cannot_start_twice(self):
        """Test that a worker cannot be started twice."""
        task_queue = TaskQueue()
        worker = SimpleWorker("test-worker")

        worker.start(task_queue._queue)

        import pytest

        with pytest.raises(RuntimeError):
            worker.start(task_queue._queue)

        worker.stop(timeout=1.0)


class TestTaskQueue:
    """Test the TaskQueue class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.task_queue = TaskQueue()

    def teardown_method(self):
        """Clean up after tests."""
        self.task_queue.shutdown(timeout=2.0)

    def test_initialization(self):
        """Test that a task queue can be initialized."""
        assert len(self.task_queue) == 0
        assert self.task_queue.empty()
        assert self.task_queue.qsize() == 0

    def test_add_worker(self):
        """Test adding a worker to the queue."""
        worker = SimpleWorker("worker-1")
        self.task_queue.add_worker(worker)

        assert len(self.task_queue) == 1
        assert "worker-1" in self.task_queue.list_workers()
        assert worker.is_alive()

    def test_add_multiple_workers(self):
        """Test adding multiple workers."""
        for i in range(3):
            worker = SimpleWorker(f"worker-{i}")
            self.task_queue.add_worker(worker)

        assert len(self.task_queue) == 3
        workers = self.task_queue.list_workers()
        assert "worker-0" in workers
        assert "worker-1" in workers
        assert "worker-2" in workers

    def test_add_duplicate_worker_raises_error(self):
        """Test that adding a worker with a duplicate name raises an error."""
        worker1 = SimpleWorker("worker")
        worker2 = SimpleWorker("worker")

        self.task_queue.add_worker(worker1)

        import pytest

        with pytest.raises(ValueError) as ctx:
            self.task_queue.add_worker(worker2)

        assert "already exists" in str(ctx.value)
        worker2.stop()

    def test_remove_worker(self):
        """Test removing a worker from the queue."""
        worker = SimpleWorker("worker-1")
        self.task_queue.add_worker(worker)

        assert len(self.task_queue) == 1

        self.task_queue.remove_worker("worker-1")

        assert len(self.task_queue) == 0
        assert "worker-1" not in self.task_queue.list_workers()
        assert not worker.is_alive()

    def test_remove_nonexistent_worker_raises_error(self):
        """Test that removing a nonexistent worker raises an error."""
        import pytest

        with pytest.raises(KeyError) as ctx:
            self.task_queue.remove_worker("nonexistent")

        assert "No worker" in str(ctx.value)

    def test_get_worker(self):
        """Test getting a worker by name."""
        worker = SimpleWorker("worker-1")
        self.task_queue.add_worker(worker)

        retrieved = self.task_queue.get_worker("worker-1")
        assert retrieved is worker

        nonexistent = self.task_queue.get_worker("nonexistent")
        assert nonexistent is None

    def test_add_task(self):
        """Test adding a task to the queue."""
        worker = SimpleWorker("worker-1")
        self.task_queue.add_worker(worker)

        task = QueueTask("task-1")
        self.task_queue.add_task(task)
        self.task_queue.wait_completion(timeout=1.0)

        assert task in worker.processed_tasks

    def test_multiple_tasks(self):
        """Test processing multiple tasks."""
        worker = CounterWorker("worker-1")
        self.task_queue.add_worker(worker)

        num_tasks = 10
        for i in range(num_tasks):
            self.task_queue.add_task(QueueTask(f"task-{i}"))

        self.task_queue.wait_completion(timeout=2.0)

        assert worker.count == num_tasks

    def test_multiple_workers_process_tasks(self):
        """Test that multiple workers process tasks from the same queue."""
        workers = []
        for i in range(3):
            worker = CounterWorker(f"worker-{i}")
            workers.append(worker)
            self.task_queue.add_worker(worker)

        num_tasks = 30
        for i in range(num_tasks):
            self.task_queue.add_task(QueueTask(f"task-{i}"))

        self.task_queue.wait_completion(timeout=2.0)

        total_processed = sum(w.count for w in workers)
        assert total_processed == num_tasks

    def test_shutdown(self):
        """Test shutting down all workers."""
        workers = []
        for i in range(3):
            worker = SimpleWorker(f"worker-{i}")
            workers.append(worker)
            self.task_queue.add_worker(worker)

        assert len(self.task_queue) == 3

        self.task_queue.shutdown(timeout=1.0)

        assert len(self.task_queue) == 0
        for worker in workers:
            assert not worker.is_alive()

    def test_task_execution_order(self):
        """Test that tasks are executed in FIFO order by a single worker."""
        worker = SimpleWorker("worker-1")
        self.task_queue.add_worker(worker)

        results = []
        lock = threading.Lock()

        def make_task(value):
            def callback():
                with lock:
                    results.append(value)

            return QueueTask(value, callback=callback)

        for i in range(5):
            self.task_queue.add_task(make_task(i))

        self.task_queue.wait_completion(timeout=2.0)

        assert results == [0, 1, 2, 3, 4]

    def test_dynamic_worker_management(self):
        """Test adding and removing workers dynamically."""
        # Start with one worker
        worker1 = CounterWorker("worker-1")
        self.task_queue.add_worker(worker1)

        # Add some tasks
        for i in range(5):
            self.task_queue.add_task(QueueTask(f"task-{i}"))

        # Add another worker mid-processing
        worker2 = CounterWorker("worker-2")
        self.task_queue.add_worker(worker2)

        # Add more tasks
        for i in range(5, 10):
            self.task_queue.add_task(QueueTask(f"task-{i}"))

        self.task_queue.wait_completion(timeout=2.0)

        # Both workers should have processed some tasks
        total = worker1.count + worker2.count
        assert total == 10

        # Remove worker1
        self.task_queue.remove_worker("worker-1")
        assert len(self.task_queue) == 1

        # Add more tasks (only worker2 should process them)
        worker2_count_before = worker2.count
        for i in range(5):
            self.task_queue.add_task(QueueTask(f"task-extra-{i}"))

        self.task_queue.wait_completion(timeout=2.0)

        assert worker2.count - worker2_count_before == 5

    def test_wait_completion_timeout(self):
        """Test that wait_completion respects timeout."""
        worker = SimpleWorker("worker-1")
        self.task_queue.add_worker(worker)

        # Add a task that takes a long time
        def slow_task():
            time.sleep(2.0)

        self.task_queue.add_task(QueueTask("slow", callback=slow_task))

        # Wait with a short timeout
        result = self.task_queue.wait_completion(timeout=0.1)

        # Should timeout
        assert not result

    def test_custom_worker_subclass(self):
        """Test using a custom worker subclass."""

        class MultiplyWorker(Worker):
            def __init__(self, name, multiplier):
                super().__init__(name)
                self.multiplier = multiplier
                self.results = []
                self.lock = threading.Lock()

            def run(self, task):
                result = task.value * self.multiplier
                with self.lock:
                    self.results.append(result)

        worker = MultiplyWorker("multiply-worker", 2)
        self.task_queue.add_worker(worker)

        for i in range(5):
            self.task_queue.add_task(QueueTask(i))

        self.task_queue.wait_completion(timeout=1.0)

        assert sorted(worker.results) == [0, 2, 4, 6, 8]

    def test_queue_maxsize(self):
        """Test queue with maximum size."""
        import queue as queue_module

        limited_queue = TaskQueue(maxsize=2)
        worker = SimpleWorker("worker-1")
        limited_queue.add_worker(worker)

        # Add tasks that will fill the queue
        def slow_task():
            time.sleep(0.5)

        # Fill the queue
        limited_queue.add_task(QueueTask("slow-1", callback=slow_task))
        limited_queue.add_task(QueueTask("slow-2", callback=slow_task))

        # Give worker time to start processing first task
        time.sleep(0.1)

        # Queue should still have room (one was taken by worker)
        limited_queue.add_task(QueueTask("quick"), block=False)

        # Now try to add another - should fail as queue is full
        import pytest

        with pytest.raises(queue_module.Full):
            limited_queue.add_task(QueueTask("will-fail"), block=False)

        limited_queue.shutdown(timeout=2.0)

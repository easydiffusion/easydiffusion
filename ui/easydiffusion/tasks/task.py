from threading import Lock
from queue import Queue, Empty as EmptyQueueException
from typing import Any


class Task:
    "Task with output queue and completion lock"

    def __init__(self, session_id):
        self.id = id(self)
        self.session_id = session_id
        self.render_device = None  # Select the task affinity. (Not used to change active devices).
        self.error: Exception = None
        self.lock: Lock = Lock()  # Locks at task start and unlocks when task is completed
        self.buffer_queue: Queue = Queue()  # Queue of JSON string segments
        self.response: Any = None  # Copy of the last reponse

    async def read_buffer_generator(self):
        try:
            while not self.buffer_queue.empty():
                res = self.buffer_queue.get(block=False)
                self.buffer_queue.task_done()
                yield res
        except EmptyQueueException as e:
            yield

    @property
    def status(self):
        if self.lock.locked():
            return "running"
        if isinstance(self.error, StopAsyncIteration):
            return "stopped"
        if self.error:
            return "error"
        if not self.buffer_queue.empty():
            return "buffer"
        if self.response:
            return "completed"
        return "pending"

    @property
    def is_pending(self):
        return bool(not self.response and not self.error)

    def run(self):
        "Override this to implement the task's behavior"
        pass

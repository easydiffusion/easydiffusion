from __future__ import annotations

import threading
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "error"
    STOPPED = "stopped"


class Task:
    """Represents a queued backend task with output storage."""

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

    def update_progress(self, progress: float) -> None:
        self.progress = min(1.0, max(0.0, float(progress)))
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

        self.outputs = list(result or [])
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
            self.update_progress(backend.get_progress())
        except Exception as e:
            print(f"Error refreshing progress for task: {e}")
            time.sleep(0.01)

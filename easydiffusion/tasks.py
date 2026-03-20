from typing import Any, Dict, Optional

from .types import Task


class GenerateTask(Task):
    """Represents a generation task."""

    task_type = "generate"

    def __init__(self, username: str, session_id: Optional[str] = None, input: Optional[Dict[str, Any]] = None):
        super().__init__(username=username, session_id=session_id, input=input)

    def run(self, backend):
        """Run the generation task using the provided backend."""
        self.outputs = list(backend.generate_images(self.input))
        return self.outputs


class FilterTask(Task):
    """Represents a filter task."""

    task_type = "filter"

    def __init__(self, username: str, session_id: Optional[str] = None, input: Optional[Dict[str, Any]] = None):
        super().__init__(username=username, session_id=session_id, input=input)

    def run(self, backend):
        """Run the filter task using the provided backend."""
        self.outputs = list(backend.filter_images(self.input))
        return self.outputs

from .types import Task


class GenerateTask(Task):
    """Represents a generation task."""

    task_type = "generate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, backend):
        """Run the generation task using the provided backend."""
        # This method should be implemented by specific task types
        raise NotImplementedError("Subclasses must implement the run method.")


class FilterTask(Task):
    """Represents a filter task."""

    task_type = "filter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, backend):
        """Run the filter task using the provided backend."""
        # This method should be implemented by specific task types
        raise NotImplementedError("Subclasses must implement the run method.")

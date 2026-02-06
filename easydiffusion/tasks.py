from .types import Task


class RenderTask(Task):
    """Represents a render task with additional attributes specific to rendering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, backend):
        """Run the render task using the provided backend."""
        # This method should be implemented by specific task types
        raise NotImplementedError("Subclasses must implement the run method.")


class FilterTask(Task):
    """Represents a filter task with additional attributes specific to filtering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, backend):
        """Run the filter task using the provided backend."""
        # This method should be implemented by specific task types
        raise NotImplementedError("Subclasses must implement the run method.")

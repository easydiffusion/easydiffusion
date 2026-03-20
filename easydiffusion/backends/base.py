"""
Abstract base class for backend implementations.

This module defines the Backend interface that all backend implementations
must implement. This ensures consistent behavior across different backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from torchruntime.device_db import GPU


class Backend(ABC):
    """
    Abstract base class for all backend implementations.

    All backends must implement the core methods defined here to ensure
    consistent behavior with the worker manager.
    """

    def __init__(self, device: GPU):
        """
        Initialize the backend.

        Args:
            device: The GPU object to use
        """
        self.device = device

    @abstractmethod
    def is_installed(self) -> bool:
        """Return whether the backend is installed and ready to start."""
        pass

    @classmethod
    def list_controlnet_filters(cls) -> list[str]:
        """Return backend-supported legacy ControlNet filter names."""
        return []

    @abstractmethod
    def install(self) -> None:
        """
        Install the backend and its dependencies.

        This should download/install any required files, packages, or
        dependencies needed for the backend to function.
        """
        pass

    @abstractmethod
    def uninstall(self) -> None:
        """
        Uninstall the backend and clean up resources.

        This should remove any files, packages, or dependencies that were
        installed by the install() method.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return backend-specific runtime configuration."""
        pass

    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update backend-specific runtime configuration."""
        pass

    @abstractmethod
    def start(self) -> None:
        """
        Start the backend.

        This is called once when the worker thread starts. It should initialize
        any resources needed for processing tasks (e.g., load models, start
        processes, etc.).

        This method runs in the worker thread context.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the backend and cleanup resources.

        This should gracefully shutdown the backend, clean up any running
        processes, release GPU memory, etc.
        """
        pass

    @abstractmethod
    def ping(self, timeout: float = 1.0) -> bool:
        """
        Check if the backend is responsive.

        Args:
            timeout: Maximum time to wait for a response (in seconds)

        Returns:
            True if the backend is responsive, False otherwise
        """
        pass

    @abstractmethod
    def generate_images(self, task_input: Dict[str, Any]) -> list[bytes]:
        """Generate output images for a queued generation task."""
        pass

    @abstractmethod
    def filter_images(self, task_input: Dict[str, Any]) -> list[bytes]:
        """Generate output images for a queued filter task."""
        pass

    @abstractmethod
    def get_progress(self, task: Any) -> float:
        """Return normalized progress in the range [0, 1] for the given task."""
        pass

    @abstractmethod
    def stop_task(self, task: Any) -> None:
        """Request backend-specific cancellation for the given task."""
        pass

    def render_image(self, context: Any, **kwargs) -> Any:
        """Legacy compatibility shim."""
        pass

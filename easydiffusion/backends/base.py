"""
Abstract base class for backend implementations.

This module defines the Backend interface that all backend implementations
must implement. This ensures consistent behavior across different backends.
"""

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """
    Abstract base class for all backend implementations.

    All backends must implement the core methods defined here to ensure
    consistent behavior with the worker manager.
    """

    def __init__(self, device: str):
        """
        Initialize the backend.

        Args:
            device: The device name to use (e.g., "cuda:0", "cpu")
        """
        self.device = device

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
    def render_image(self, context: Any, **kwargs) -> Any:
        """
        Render an image using the backend.

        Args:
            context: The context object for this render operation
            **kwargs: Additional rendering parameters

        Returns:
            The rendered image(s) or result
        """
        pass

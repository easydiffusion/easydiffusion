"""
Backend registry and management module.

This module provides access to all available backend implementations and
a registry that maps backend names to their implementation classes.
"""

from typing import Dict, Type
from .base import Backend

# Import backend implementations
from .webui_backend import WebUIBackend
from .ed_classic_backend import EDClassicBackend
from .sdkit3_backend import Sdkit3Backend

# Backend registry mapping names to implementation classes
BACKEND_REGISTRY: Dict[str, Type[Backend]] = {
    "webui": WebUIBackend,
    "ed_classic": EDClassicBackend,
    "sdkit3": Sdkit3Backend,
}


def get_backend_class(name: str) -> Type[Backend]:
    """
    Get a backend class by name.

    Args:
        name: The backend name (e.g., "webui", "ed_classic", "sdkit3")

    Returns:
        The backend class

    Raises:
        KeyError: If the backend name is not found in the registry
    """
    if name not in BACKEND_REGISTRY:
        available = ", ".join(BACKEND_REGISTRY.keys())
        raise KeyError(f"Backend '{name}' not found. Available backends: {available}")
    return BACKEND_REGISTRY[name]


def list_backends() -> list:
    """
    Get a list of all available backend names.

    Returns:
        List of backend names
    """
    return list(BACKEND_REGISTRY.keys())


__all__ = [
    "Backend",
    "BACKEND_REGISTRY",
    "get_backend_class",
    "list_backends",
]

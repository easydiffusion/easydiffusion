"""
Device management utilities for translating device names to device IDs.

This module provides functionality to resolve device names (like 'auto', 'cpu',
device arrays, etc.) to actual device IDs that can be used by the system.
"""

from typing import Union, List
from torchruntime.device_db import get_gpus, GPU

# CPU GPU object
CPU = GPU(vendor_id="cpu", vendor_name="CPU", device_id="cpu", device_name="cpu", is_discrete=False)


def _get_discrete_gpus(gpus: List[GPU]) -> List[GPU]:
    """
    Filter GPUs to return only discrete GPUs.

    Args:
        gpus: List of GPU objects from torchruntime

    Returns:
        List of discrete GPU objects
    """
    return [gpu for gpu in gpus if gpu.is_discrete]


def _get_integrated_gpus(gpus: List[GPU]) -> List[GPU]:
    """
    Filter GPUs to return only integrated GPUs.

    Args:
        gpus: List of GPU objects from torchruntime

    Returns:
        List of integrated GPU objects
    """
    return [gpu for gpu in gpus if not gpu.is_discrete]


def _resolve_auto_devices() -> List[GPU]:
    """
    Resolve 'auto' to the appropriate list of GPU objects.

    Logic:
    1. If discrete GPUs are available, use all of them (ignore integrated)
    2. If only integrated GPUs are available, use them
    3. Otherwise, fall back to CPU

    Returns:
        List of GPU objects
    """
    gpus = get_gpus()

    # Get discrete GPUs
    discrete_gpus = _get_discrete_gpus(gpus)
    if discrete_gpus:
        return discrete_gpus

    # Get integrated GPUs
    integrated_gpus = _get_integrated_gpus(gpus)
    if integrated_gpus:
        return integrated_gpus

    # No GPUs found, use CPU
    return [CPU]


def _parse_device_string(device_str: str) -> GPU:
    """
    Parse a device string and return the corresponding GPU object.

    Handles formats like:
    - 'cpu' -> CPU_GPU
    - 'cuda:0' -> GPU at index 0
    - '0' -> GPU at index 0

    Args:
        device_str: Device string to parse

    Returns:
        GPU object

    Raises:
        ValueError: If the device ID is out of bounds or invalid
    """
    if device_str == "cpu":
        return CPU

    # Handle formats like 'cuda:0', 'dml:1', etc.
    if ":" in device_str:
        _, device_id = device_str.split(":", 1)
    else:
        device_id = device_str

    # Parse device_id as integer index
    try:
        idx = int(device_id)
    except ValueError:
        raise ValueError(f"Invalid device ID: {device_id}")

    gpus = get_gpus()
    if idx < 0 or idx >= len(gpus):
        raise ValueError(f"Device ID {idx} is out of bounds for available GPUs (0-{len(gpus)-1})")

    return gpus[idx]


def resolve_devices(devices: Union[str, List[str]]) -> List[GPU]:
    """
    Resolve device names to a list of GPU objects.

    This function translates various device specifications into a standardized
    list of GPU objects. It handles:
    - 'auto': Automatically selects devices based on available hardware
    - 'cpu': Maps to [CPU_GPU]
    - Named devices like 'cuda:0', 'dml:1': Extracts device IDs and gets GPU objects
    - Arrays of devices: Processes each device in the array
    - Direct device IDs: Gets the corresponding GPU object

    Args:
        devices: Either a string ('auto', 'cpu', 'cuda:0', etc.) or a list
                of device strings

    Returns:
        List of GPU objects

    Examples:
        >>> resolve_devices('auto')
        [GPU(...), GPU(...)]  # If two discrete GPUs are available

        >>> resolve_devices('cpu')
        [CPU_GPU]

        >>> resolve_devices('cuda:0')
        [GPU at index 0]

        >>> resolve_devices(['cuda:0', 'cuda:1'])
        [GPU at index 0, GPU at index 1]

        >>> resolve_devices(['dml:0', 'cpu'])
        [GPU at index 0, CPU_GPU]
    """
    # Handle string input
    if isinstance(devices, str):
        if devices == "auto":
            return _resolve_auto_devices()
        return [_parse_device_string(devices)]

    # Handle list input
    if isinstance(devices, list):
        result = []
        for device in devices:
            if device == "auto":
                # If 'auto' is in a list, resolve it and extend the result
                result.extend(_resolve_auto_devices())
            else:
                result.append(_parse_device_string(device))
        return result

    # Fallback to CPU for unexpected input
    return [CPU]

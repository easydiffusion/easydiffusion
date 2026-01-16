"""
Device management utilities for translating device names to device IDs.

This module provides functionality to resolve device names (like 'auto', 'cpu',
device arrays, etc.) to actual device IDs that can be used by the system.
"""

from typing import Union, List
from torchruntime.device_db import get_gpus, GPU


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


def _resolve_auto_devices() -> List[str]:
    """
    Resolve 'auto' to the appropriate list of device IDs.

    Logic:
    1. If discrete GPUs are available, use all of them (ignore integrated)
    2. If only integrated GPUs are available, use them
    3. Otherwise, fall back to CPU

    Returns:
        List of device ID strings (e.g., ['0', '1'] or ['cpu'])
    """
    gpus = get_gpus()

    # Get discrete GPUs
    discrete_gpus = _get_discrete_gpus(gpus)
    if discrete_gpus:
        # Use all discrete GPUs, identified by their index in the list
        return [str(i) for i in range(len(discrete_gpus))]

    # Get integrated GPUs
    integrated_gpus = _get_integrated_gpus(gpus)
    if integrated_gpus:
        # Use all integrated GPUs
        return [str(i) for i in range(len(integrated_gpus))]

    # No GPUs found, use CPU
    return ["cpu"]


def _parse_device_string(device_str: str) -> str:
    """
    Parse a device string and extract the device ID.

    Handles formats like:
    - 'cpu' -> 'cpu'
    - 'cuda:0' -> '0'
    - 'cuda:1' -> '1'
    - 'dml:0' -> '0'
    - '0' -> '0'

    Args:
        device_str: Device string to parse

    Returns:
        Device ID string
    """
    if device_str == "cpu":
        return "cpu"

    # Handle formats like 'cuda:0', 'dml:1', etc.
    if ":" in device_str:
        _, device_id = device_str.split(":", 1)
        return device_id

    # Already a device ID
    return device_str


def resolve_devices(devices: Union[str, List[str]]) -> List[str]:
    """
    Resolve device names to a list of device IDs.

    This function translates various device specifications into a standardized
    list of device IDs. It handles:
    - 'auto': Automatically selects devices based on available hardware
    - 'cpu': Maps to ['cpu']
    - Named devices like 'cuda:0', 'dml:1': Extracts device IDs
    - Arrays of devices: Processes each device in the array
    - Direct device IDs: Uses them as-is

    The goal is to identify GPU devices without specifying the graphics API,
    allowing the system to work with device IDs like '0', '1', etc. instead
    of 'cuda:0', 'dml:0', etc.

    Args:
        devices: Either a string ('auto', 'cpu', 'cuda:0', etc.) or a list
                of device strings

    Returns:
        List of device ID strings (e.g., ['0', '1'] or ['cpu'])

    Examples:
        >>> resolve_devices('auto')
        ['0', '1']  # If two discrete GPUs are available

        >>> resolve_devices('cpu')
        ['cpu']

        >>> resolve_devices('cuda:0')
        ['0']

        >>> resolve_devices(['cuda:0', 'cuda:1'])
        ['0', '1']

        >>> resolve_devices(['dml:0', 'cpu'])
        ['0', 'cpu']
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
    return ["cpu"]

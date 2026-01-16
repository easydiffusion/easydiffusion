"""
Configuration management system using ruamel.yaml.

This module provides functionality to read, write, and update configuration
from a YAML file with support for preserving comments and formatting.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from ruamel.yaml import YAML
import threading


class ConfigManager:
    """
    Manages application configuration using YAML files.

    Supports reading, writing, and partial updates while preserving
    YAML formatting and comments.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False
        self._config: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from the YAML file.

        Returns:
            The loaded configuration dictionary

        Raises:
            FileNotFoundError: If the config file doesn't exist
        """
        with self._lock:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            with open(self.config_path, "r") as f:
                self._config = self.yaml.load(f) or {}

            return dict(self._config)

    def save(self, config: Optional[Dict[str, Any]] = None):
        """
        Save configuration to the YAML file.

        Args:
            config: Configuration to save. If None, saves the current config.
        """
        with self._lock:
            if config is not None:
                self._config = config

            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, "w") as f:
                self.yaml.dump(self._config, f)

    def get(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key. If None, returns the entire config.
            default: Default value if key doesn't exist

        Returns:
            The configuration value or default
        """
        with self._lock:
            if key is None:
                return dict(self._config)
            return self._config.get(key, default)

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values and save to file.

        Only updates the provided keys, leaving others untouched.

        Args:
            updates: Dictionary of configuration updates
        """
        with self._lock:
            self._config.update(updates)

        self.save()

    def set(self, key: str, value: Any):
        """
        Set a single configuration value and save to file.

        Args:
            key: The configuration key
            value: The value to set
        """
        self.update({key: value})

    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file.

        Returns:
            The reloaded configuration dictionary
        """
        return self.load()

    def get_all(self) -> Dict[str, Any]:
        """
        Get a copy of the entire configuration.

        Returns:
            Dictionary containing all configuration values
        """
        with self._lock:
            return dict(self._config)


# Default configuration values
DEFAULT_CONFIG = {"update_branch": "main", "backend": "sdkit3", "render_devices": "auto"}


def create_default_config(config_path: Union[str, Path]):
    """
    Create a configuration file with default values.

    Args:
        config_path: Path where the config file should be created
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False

    with open(config_path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f)

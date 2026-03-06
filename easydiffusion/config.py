"""
Configuration management system using ruamel.yaml.

This module provides functionality to read, write, and update configuration
from a YAML file with support for preserving comments and formatting.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from collections.abc import Mapping
from ruamel.yaml import YAML
import threading
import shutil
from copy import deepcopy


SYSTEM_CONFIG_KEYS = ("network", "updates", "models", "backend")


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
            filtered_updates = {key: value for key, value in updates.items() if key in SYSTEM_CONFIG_KEYS}
            self._deep_merge(self._config, filtered_updates)

        self.save()

    def get_all(self) -> Dict[str, Any]:
        """
        Get a copy of the entire configuration.

        Returns:
            Dictionary containing all configuration values
        """
        with self._lock:
            return dict(self._config)

    def get_user_config(self, username: str) -> Dict[str, Any]:
        """
        Get user-specific configuration, merged with defaults.

        Args:
            username: The username

        Returns:
            Merged user settings
        """
        with self._lock:
            user_settings = self._config.get("user_settings", {})
            default_settings = deepcopy(user_settings.get("default", {}))
            user_specific = deepcopy(user_settings.get(username, {}))
            merged = default_settings
            self._deep_merge(merged, user_specific)

            # ensure that certain keys always exist in the merged config
            merged.setdefault("save", {})
            merged.setdefault("ui", {})
            merged["ui"].setdefault("block_nsfw", False)

            # enforce global NSFW blocking if configured
            if self._config.get("security", {}).get("force_block_nsfw", False):
                merged["ui"]["block_nsfw"] = True

            return merged

    def update_user_config(self, username: str, updates: Dict[str, Any]):
        """
        Update user-specific configuration.

        Args:
            username: The username
            updates: The updates to apply
        """
        with self._lock:
            if "user_settings" not in self._config:
                self._config["user_settings"] = {}
            if username not in self._config["user_settings"]:
                self._config["user_settings"][username] = {}

            filtered_updates = {key: value for key, value in updates.items() if key in ("save", "ui")}
            self._deep_merge(self._config["user_settings"][username], filtered_updates)
        self.save()

    def get_system_config(self) -> Dict[str, Any]:
        """Get the public system-wide configuration."""
        with self._lock:
            return {key: deepcopy(self._config.get(key, {})) for key in SYSTEM_CONFIG_KEYS}

    def get_users(self) -> List[str]:
        """
        Get list of users.

        Returns:
            List of usernames
        """
        with self._lock:
            users = self._config.get("users", [])
            return users

    def add_user(self, username: str):
        """
        Add a new user.

        Args:
            username: The username to add
        """
        with self._lock:
            if "users" not in self._config:
                self._config["users"] = []
            if username not in self._config["users"] and username.lower() != "default":
                self._config["users"].append(username)
        self.save()

    def delete_user(self, username: str):
        """
        Delete a user.

        Args:
            username: The username to delete
        """
        with self._lock:
            if "users" in self._config and username in self._config["users"] and username.lower() != "default":
                self._config["users"].remove(username)
                if "user_settings" in self._config and username in self._config["user_settings"]:
                    del self._config["user_settings"][username]
        self.save()

    def _deep_merge(self, current: Dict[str, Any], updates: Dict[str, Any]):
        """
        Recursively merge updates into the current configuration.

        dict.update() doesn't merge nested dictionaries but replaces them, so we need
        to do this recursively to allow partial updates.
        """
        for key, value in updates.items():
            if isinstance(value, Mapping) and isinstance(current.get(key), Mapping):
                self._deep_merge(current[key], value)
            else:
                current[key] = value


def create_default_config(config_path: Union[str, Path]):
    """
    Create a configuration file with default values.

    Args:
        config_path: Path where the config file should be created
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    sample_path = Path(__file__).parent / "config.yaml.sample"
    shutil.copy(sample_path, config_path)

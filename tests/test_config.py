"""
Tests for configuration management system.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from ruamel.yaml import YAML

from easydiffusion.config import ConfigManager, create_default_config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def config_file(temp_dir):
    """Create a test config file."""
    config_path = temp_dir / "config.yaml"
    create_default_config(config_path)
    return config_path


def test_create_default_config(temp_dir):
    """Test creating a default configuration file."""
    config_path = temp_dir / "config.yaml"
    create_default_config(config_path)

    assert config_path.exists()

    # Load and verify
    manager = ConfigManager(config_path)
    config = manager.load()

    # Load expected config from sample
    yaml = YAML()
    sample_path = Path(__file__).parent.parent / "easydiffusion" / "config.yaml.sample"
    with open(sample_path, "r") as f:
        expected_config = yaml.load(f)

    assert config == expected_config


def test_load_config(config_file):
    """Test loading configuration."""
    manager = ConfigManager(config_file)
    config = manager.load()

    assert isinstance(config, dict)
    assert "models" in config
    assert "updates" in config
    assert "backend" in config
    assert "network" in config


def test_get_config_value(config_file):
    """Test getting individual config values."""
    manager = ConfigManager(config_file)
    manager.load()

    assert manager.get("updates", {}).get("branch") == "main"
    assert manager.get("models", {}).get("models_dir") == "models"
    assert manager.get("backend", {}).get("backend_name") == "sdkit3"
    assert manager.get("backend", {}).get("devices") == "auto"
    assert manager.get("nonexistent", "default") == "default"


def test_update_config(config_file):
    """Test updating configuration."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.update({"updates": {"branch": "beta"}})

    assert manager.get("updates", {}).get("branch") == "beta"
    assert manager.get("backend", {}).get("backend_name") == "sdkit3"  # unchanged


def test_set_config_value(config_file):
    """Test setting a single config value."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.update({"backend": {"backend_name": "custom"}})

    assert manager.get("backend", {}).get("backend_name") == "custom"


def test_partial_update(config_file):
    """Test partial config update."""
    manager = ConfigManager(config_file)
    manager.load()

    original_backend = manager.get("backend")

    manager.update({"backend": {"devices": "cpu"}})

    assert manager.get("backend", {}).get("devices") == "cpu"
    assert manager.get("backend", {}).get("backend_name") == original_backend.get("backend_name")


def test_get_all(config_file):
    """Test getting all configuration."""
    manager = ConfigManager(config_file)
    manager.load()

    all_config = manager.get_all()

    assert isinstance(all_config, dict)
    assert "models" in all_config
    assert "updates" in all_config
    assert "backend" in all_config
    assert "network" in all_config


def test_devices_types(config_file):
    """Test different backend device configurations."""
    manager = ConfigManager(config_file)
    manager.load()

    # Test auto
    manager.update({"backend": {"devices": "auto"}})
    assert manager.get("backend", {}).get("devices") == "auto"

    # Test cpu
    manager.update({"backend": {"devices": "cpu"}})
    assert manager.get("backend", {}).get("devices") == "cpu"

    # Test list
    manager.update({"backend": {"devices": ["cuda:0", "cuda:1"]}})
    assert manager.get("backend", {}).get("devices") == ["cuda:0", "cuda:1"]


def test_get_user_config_applies_nested_fallbacks(config_file):
    """Test user config fallback logic for nested sections."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.update_user_config("easydiffusion", {"save": {"save_path": "/tmp/out"}, "ui": {"theme": "theme-dark"}})

    user_config = manager.get_user_config("easydiffusion")
    assert user_config["save"]["save_path"] == "/tmp/out"
    assert user_config["save"]["auto_save_images"] is False
    assert user_config["ui"]["theme"] == "theme-dark"
    assert user_config["ui"]["open_browser_on_start"] is True
    assert user_config["ui"]["block_nsfw"] is False


def test_force_block_nsfw_overrides_user_setting(config_file):
    """Test security.force_block_nsfw overrides user config."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.update_user_config("easydiffusion", {"ui": {"block_nsfw": False}})
    manager.save({**manager.get_all(), "security": {"force_block_nsfw": True}})
    manager.load()

    user_config = manager.get_user_config("easydiffusion")
    assert user_config["ui"]["block_nsfw"] is True

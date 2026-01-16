"""
Tests for configuration management system.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from easydiffusion.config import ConfigManager, create_default_config, DEFAULT_CONFIG


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

    assert config["update_branch"] == DEFAULT_CONFIG["update_branch"]
    assert config["backend"] == DEFAULT_CONFIG["backend"]
    assert config["render_devices"] == DEFAULT_CONFIG["render_devices"]


def test_load_config(config_file):
    """Test loading configuration."""
    manager = ConfigManager(config_file)
    config = manager.load()

    assert isinstance(config, dict)
    assert "update_branch" in config
    assert "backend" in config
    assert "render_devices" in config


def test_get_config_value(config_file):
    """Test getting individual config values."""
    manager = ConfigManager(config_file)
    manager.load()

    assert manager.get("update_branch") == "main"
    assert manager.get("backend") == "sdkit3"
    assert manager.get("render_devices") == "auto"
    assert manager.get("nonexistent", "default") == "default"


def test_update_config(config_file):
    """Test updating configuration."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.update({"update_branch": "develop"})

    assert manager.get("update_branch") == "develop"
    assert manager.get("backend") == "sdkit3"  # unchanged

    # Reload and verify persistence
    manager.reload()
    assert manager.get("update_branch") == "develop"


def test_set_config_value(config_file):
    """Test setting a single config value."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.set("backend", "custom")

    assert manager.get("backend") == "custom"

    # Reload and verify
    manager.reload()
    assert manager.get("backend") == "custom"


def test_partial_update(config_file):
    """Test partial config update."""
    manager = ConfigManager(config_file)
    manager.load()

    original_backend = manager.get("backend")

    manager.update({"update_branch": "feature-x"})

    assert manager.get("update_branch") == "feature-x"
    assert manager.get("backend") == original_backend


def test_get_all(config_file):
    """Test getting all configuration."""
    manager = ConfigManager(config_file)
    manager.load()

    all_config = manager.get_all()

    assert isinstance(all_config, dict)
    assert "update_branch" in all_config
    assert "backend" in all_config
    assert "render_devices" in all_config


def test_render_devices_types(config_file):
    """Test different render_devices configurations."""
    manager = ConfigManager(config_file)
    manager.load()

    # Test auto
    manager.set("render_devices", "auto")
    assert manager.get("render_devices") == "auto"

    # Test cpu
    manager.set("render_devices", "cpu")
    assert manager.get("render_devices") == "cpu"

    # Test list
    manager.set("render_devices", ["cuda:0", "cuda:1"])
    assert manager.get("render_devices") == ["cuda:0", "cuda:1"]

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
    assert "updates" in config
    assert "rendering" in config
    assert "network" in config


def test_get_config_value(config_file):
    """Test getting individual config values."""
    manager = ConfigManager(config_file)
    manager.load()

    assert manager.get("updates", {}).get("branch") == "main"
    assert manager.get("rendering", {}).get("backend") == "sdkit3"
    assert manager.get("rendering", {}).get("devices") == "auto"
    assert manager.get("nonexistent", "default") == "default"


def test_update_config(config_file):
    """Test updating configuration."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.update({"updates": {"branch": "beta"}})

    assert manager.get("updates", {}).get("branch") == "beta"
    assert manager.get("rendering", {}).get("backend") == "sdkit3"  # unchanged


def test_set_config_value(config_file):
    """Test setting a single config value."""
    manager = ConfigManager(config_file)
    manager.load()

    manager.update({"rendering": {"backend": "custom"}})

    assert manager.get("rendering", {}).get("backend") == "custom"


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
    assert "updates" in all_config
    assert "rendering" in all_config
    assert "network" in all_config


def test_render_devices_types(config_file):
    """Test different render_devices configurations."""
    manager = ConfigManager(config_file)
    manager.load()

    # Test auto
    manager.update({"rendering": {"devices": "auto"}})
    assert manager.get("rendering", {}).get("devices") == "auto"

    # Test cpu
    manager.update({"rendering": {"devices": "cpu"}})
    assert manager.get("rendering", {}).get("devices") == "cpu"

    # Test list
    manager.update({"rendering": {"devices": ["cuda:0", "cuda:1"]}})
    assert manager.get("rendering", {}).get("devices") == ["cuda:0", "cuda:1"]

"""
Tests for device utility functions.
"""

from unittest.mock import patch
from easydiffusion.utils.device_utils import (
    resolve_devices,
    _get_discrete_gpus,
    _get_integrated_gpus,
    _resolve_auto_devices,
    _parse_device_string,
)
from torchruntime.device_db import GPU


class TestDeviceUtilsHelpers:
    """Test helper functions for device utilities."""

    def test_get_discrete_gpus(self):
        """Test filtering for discrete GPUs."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        discrete = _get_discrete_gpus(gpus)
        assert len(discrete) == 2
        assert all(gpu.is_discrete for gpu in discrete)

    def test_get_integrated_gpus(self):
        """Test filtering for integrated GPUs."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
            GPU("1002", "AMD", "15bf", "Radeon Graphics", False),
        ]
        integrated = _get_integrated_gpus(gpus)
        assert len(integrated) == 2
        assert not any(gpu.is_discrete for gpu in integrated)

    def test_parse_device_string_cpu(self):
        """Test parsing 'cpu' device string."""
        assert _parse_device_string("cpu") == "cpu"

    def test_parse_device_string_cuda(self):
        """Test parsing CUDA device strings."""
        assert _parse_device_string("cuda:0") == "0"
        assert _parse_device_string("cuda:1") == "1"
        assert _parse_device_string("cuda:2") == "2"

    def test_parse_device_string_dml(self):
        """Test parsing DirectML device strings."""
        assert _parse_device_string("dml:0") == "0"
        assert _parse_device_string("dml:1") == "1"

    def test_parse_device_string_other_apis(self):
        """Test parsing other API device strings."""
        assert _parse_device_string("xpu:0") == "0"
        assert _parse_device_string("mps:0") == "0"

    def test_parse_device_string_direct_id(self):
        """Test parsing direct device ID strings."""
        assert _parse_device_string("0") == "0"
        assert _parse_device_string("1") == "1"


class TestResolveAutoDevices:
    """Test the auto device resolution logic."""

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_discrete_gpus(self, mock_get_gpus):
        """Test auto resolution with discrete GPUs available."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        result = _resolve_auto_devices()
        assert result == ["0", "1"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_discrete_and_integrated(self, mock_get_gpus):
        """Test auto resolution ignores integrated when discrete is available."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
        ]
        result = _resolve_auto_devices()
        # Should only return discrete GPU
        assert result == ["0"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_only_integrated(self, mock_get_gpus):
        """Test auto resolution with only integrated GPUs."""
        mock_get_gpus.return_value = [
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
        ]
        result = _resolve_auto_devices()
        assert result == ["0"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_multiple_integrated(self, mock_get_gpus):
        """Test auto resolution with multiple integrated GPUs."""
        mock_get_gpus.return_value = [
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
            GPU("1002", "AMD", "15bf", "Radeon Graphics", False),
        ]
        result = _resolve_auto_devices()
        assert result == ["0", "1"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_no_gpus(self, mock_get_gpus):
        """Test auto resolution falls back to CPU when no GPUs."""
        mock_get_gpus.return_value = []
        result = _resolve_auto_devices()
        assert result == ["cpu"]


class TestResolveDevices:
    """Test the main resolve_devices function."""

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_auto_string(self, mock_get_gpus):
        """Test resolving 'auto' as a string."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]
        result = resolve_devices("auto")
        assert result == ["0"]

    def test_resolve_cpu_string(self):
        """Test resolving 'cpu' as a string."""
        result = resolve_devices("cpu")
        assert result == ["cpu"]

    def test_resolve_cuda_string(self):
        """Test resolving CUDA device strings."""
        result = resolve_devices("cuda:0")
        assert result == ["0"]

        result = resolve_devices("cuda:1")
        assert result == ["1"]

    def test_resolve_dml_string(self):
        """Test resolving DirectML device strings."""
        result = resolve_devices("dml:0")
        assert result == ["0"]

    def test_resolve_direct_id_string(self):
        """Test resolving direct device ID."""
        result = resolve_devices("0")
        assert result == ["0"]

    def test_resolve_list_of_cuda_devices(self):
        """Test resolving a list of CUDA devices."""
        result = resolve_devices(["cuda:0", "cuda:1"])
        assert result == ["0", "1"]

    def test_resolve_list_of_mixed_devices(self):
        """Test resolving a list of mixed device types."""
        result = resolve_devices(["cuda:0", "dml:1", "cpu"])
        assert result == ["0", "1", "cpu"]

    def test_resolve_list_with_direct_ids(self):
        """Test resolving a list with direct device IDs."""
        result = resolve_devices(["0", "1", "cpu"])
        assert result == ["0", "1", "cpu"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_list_with_auto(self, mock_get_gpus):
        """Test resolving a list containing 'auto'."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        result = resolve_devices(["auto", "cpu"])
        # 'auto' should expand to ['0', '1'], then 'cpu' is added
        assert result == ["0", "1", "cpu"]

    def test_resolve_empty_list(self):
        """Test resolving an empty list falls back to CPU."""
        result = resolve_devices([])
        assert result == []

    def test_resolve_unexpected_input(self):
        """Test resolving unexpected input types."""
        # None should fall back to CPU
        result = resolve_devices(None)
        assert result == ["cpu"]

        # Numeric input should fall back to CPU
        result = resolve_devices(0)
        assert result == ["cpu"]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_multi_gpu_workstation(self, mock_get_gpus):
        """Test scenario: Multi-GPU workstation with 2 discrete GPUs."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]

        # Auto should select both GPUs
        result = resolve_devices("auto")
        assert result == ["0", "1"]

        # User can also explicitly select one GPU
        result = resolve_devices("cuda:0")
        assert result == ["0"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_laptop_with_integrated_gpu(self, mock_get_gpus):
        """Test scenario: Laptop with only integrated GPU."""
        mock_get_gpus.return_value = [
            GPU("8086", "Intel", "46a6", "Iris Xe Graphics", False),
        ]

        # Auto should select the integrated GPU
        result = resolve_devices("auto")
        assert result == ["0"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_laptop_with_discrete_and_integrated(self, mock_get_gpus):
        """Test scenario: Gaming laptop with discrete + integrated GPU."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "28e0", "RTX 4060 Laptop", True),
            GPU("8086", "Intel", "46a6", "Iris Xe Graphics", False),
        ]

        # Auto should select only the discrete GPU
        result = resolve_devices("auto")
        assert result == ["0"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_no_gpu_cpu_fallback(self, mock_get_gpus):
        """Test scenario: No GPU available, should use CPU."""
        mock_get_gpus.return_value = []

        result = resolve_devices("auto")
        assert result == ["cpu"]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_user_override_to_cpu(self, mock_get_gpus):
        """Test scenario: User forces CPU despite GPUs being available."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]

        # User can override and use CPU
        result = resolve_devices("cpu")
        assert result == ["cpu"]

    def test_mixed_api_specification(self):
        """Test scenario: User specifies devices with different API prefixes."""
        # This should work since we extract device IDs
        result = resolve_devices(["cuda:0", "dml:1"])
        assert result == ["0", "1"]

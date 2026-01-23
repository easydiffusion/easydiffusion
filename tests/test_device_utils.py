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
    CPU,
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
        assert _parse_device_string("cpu") == CPU

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_parse_device_string_cuda(self, mock_get_gpus):
        """Test parsing CUDA device strings."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        assert _parse_device_string("cuda:0") == mock_get_gpus.return_value[0]
        assert _parse_device_string("cuda:1") == mock_get_gpus.return_value[1]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_parse_device_string_dml(self, mock_get_gpus):
        """Test parsing DirectML device strings."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        assert _parse_device_string("dml:0") == mock_get_gpus.return_value[0]
        assert _parse_device_string("dml:1") == mock_get_gpus.return_value[1]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_parse_device_string_other_apis(self, mock_get_gpus):
        """Test parsing other API device strings."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]
        assert _parse_device_string("xpu:0") == mock_get_gpus.return_value[0]
        assert _parse_device_string("mps:0") == mock_get_gpus.return_value[0]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_parse_device_string_direct_id(self, mock_get_gpus):
        """Test parsing direct device ID strings."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        assert _parse_device_string("0") == mock_get_gpus.return_value[0]
        assert _parse_device_string("1") == mock_get_gpus.return_value[1]


class TestResolveAutoDevices:
    """Test the auto device resolution logic."""

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_discrete_gpus(self, mock_get_gpus):
        """Test auto resolution with discrete GPUs available."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        mock_get_gpus.return_value = gpus
        result = _resolve_auto_devices()
        assert result == gpus

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_discrete_and_integrated(self, mock_get_gpus):
        """Test auto resolution ignores integrated when discrete is available."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
        ]
        mock_get_gpus.return_value = gpus
        result = _resolve_auto_devices()
        # Should only return discrete GPU
        assert result == [gpus[0]]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_only_integrated(self, mock_get_gpus):
        """Test auto resolution with only integrated GPUs."""
        gpus = [
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
        ]
        mock_get_gpus.return_value = gpus
        result = _resolve_auto_devices()
        assert result == gpus

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_multiple_integrated(self, mock_get_gpus):
        """Test auto resolution with multiple integrated GPUs."""
        gpus = [
            GPU("8086", "Intel", "46a6", "UHD Graphics", False),
            GPU("1002", "AMD", "15bf", "Radeon Graphics", False),
        ]
        mock_get_gpus.return_value = gpus
        result = _resolve_auto_devices()
        assert result == gpus

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_auto_with_no_gpus(self, mock_get_gpus):
        """Test auto resolution falls back to CPU when no GPUs."""
        mock_get_gpus.return_value = []
        result = _resolve_auto_devices()
        assert result == [CPU]


class TestResolveDevices:
    """Test the main resolve_devices function."""

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_auto_string(self, mock_get_gpus):
        """Test resolving 'auto' as a string."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices("auto")
        assert result == gpus

    def test_resolve_cpu_string(self):
        """Test resolving 'cpu' as a string."""
        result = resolve_devices("cpu")
        assert result == [CPU]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_cuda_string(self, mock_get_gpus):
        """Test resolving CUDA device strings."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices("cuda:0")
        assert result == [gpus[0]]

        result = resolve_devices("cuda:1")
        assert result == [gpus[1]]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_dml_string(self, mock_get_gpus):
        """Test resolving DirectML device strings."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices("dml:0")
        assert result == [gpus[0]]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_direct_id_string(self, mock_get_gpus):
        """Test resolving direct device ID."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices("0")
        assert result == [gpus[0]]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_list_of_cuda_devices(self, mock_get_gpus):
        """Test resolving a list of CUDA devices."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices(["cuda:0", "cuda:1"])
        assert result == [gpus[0], gpus[1]]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_list_of_mixed_devices(self, mock_get_gpus):
        """Test resolving a list of mixed device types."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices(["cuda:0", "dml:1", "cpu"])
        assert result == [gpus[0], gpus[1], CPU]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_list_with_direct_ids(self, mock_get_gpus):
        """Test resolving a list with direct device IDs."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices(["0", "1", "cpu"])
        assert result == [gpus[0], gpus[1], CPU]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_resolve_list_with_auto(self, mock_get_gpus):
        """Test resolving a list containing 'auto'."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        mock_get_gpus.return_value = gpus
        result = resolve_devices(["auto", "cpu"])
        # 'auto' should expand to the GPUs, then 'cpu' is added
        assert result == [gpus[0], gpus[1], CPU]

    def test_resolve_empty_list(self):
        """Test resolving an empty list falls back to CPU."""
        result = resolve_devices([])
        assert result == []

    def test_resolve_unexpected_input(self):
        """Test resolving unexpected input types."""
        # None should fall back to CPU
        result = resolve_devices(None)
        assert result == [CPU]

        # Numeric input should fall back to CPU
        result = resolve_devices(0)
        assert result == [CPU]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_multi_gpu_workstation(self, mock_get_gpus):
        """Test scenario: Multi-GPU workstation with 2 discrete GPUs."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]
        mock_get_gpus.return_value = gpus

        # Auto should select both GPUs
        result = resolve_devices("auto")
        assert result == gpus

        # User can also explicitly select one GPU
        result = resolve_devices("cuda:0")
        assert result == [gpus[0]]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_laptop_with_integrated_gpu(self, mock_get_gpus):
        """Test scenario: Laptop with only integrated GPU."""
        gpus = [
            GPU("8086", "Intel", "46a6", "Iris Xe Graphics", False),
        ]
        mock_get_gpus.return_value = gpus

        # Auto should select the integrated GPU
        result = resolve_devices("auto")
        assert result == gpus

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_laptop_with_discrete_and_integrated(self, mock_get_gpus):
        """Test scenario: Gaming laptop with discrete + integrated GPU."""
        gpus = [
            GPU("10de", "NVIDIA", "28e0", "RTX 4060 Laptop", True),
            GPU("8086", "Intel", "46a6", "Iris Xe Graphics", False),
        ]
        mock_get_gpus.return_value = gpus

        # Auto should select only the discrete GPU
        result = resolve_devices("auto")
        assert result == [gpus[0]]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_no_gpu_cpu_fallback(self, mock_get_gpus):
        """Test scenario: No GPU available, should use CPU."""
        mock_get_gpus.return_value = []

        result = resolve_devices("auto")
        assert result == [CPU]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_user_override_to_cpu(self, mock_get_gpus):
        """Test scenario: User forces CPU despite GPUs being available."""
        mock_get_gpus.return_value = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
        ]

        # User can override and use CPU
        result = resolve_devices("cpu")
        assert result == [CPU]

    @patch("easydiffusion.utils.device_utils.get_gpus")
    def test_mixed_api_specification(self, mock_get_gpus):
        """Test scenario: User specifies devices with different API prefixes."""
        gpus = [
            GPU("10de", "NVIDIA", "2684", "RTX 4090", True),
            GPU("10de", "NVIDIA", "2704", "RTX 4080", True),
        ]
        mock_get_gpus.return_value = gpus
        # This should work since we extract device IDs
        result = resolve_devices(["cuda:0", "dml:1"])
        assert result == [gpus[0], gpus[1]]

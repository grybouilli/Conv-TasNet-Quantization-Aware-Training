"""
Tests for the quantization framework.

This module tests the basic functionality of the quantization system.
"""

import torch
import torch.nn as nn
import pytest

from hatorch.quantizers.base import IdentityQuantizer, BaseQuantizer
from hatorch.transforms.config import QuantizerSpec, LayerQuantConfig, uniform_8bit_config, ModelQuantConfig
from hatorch.quantizers.lsq import LsqQuantizer
from hatorch.observers.min_max_observer import MinMaxObserver
from hatorch.layers.quantized_conv2d import QuantConv2d
from hatorch.layers.quantized_linear import QuantLinear
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.transforms import utils as transforms_utils


class TestObservers:
    """Test observer functionality."""

    def test_minmax_observer(self):
        """Test MinMaxObserver statistics collection."""
        observer = MinMaxObserver()

        # Observe some data
        x1 = torch.randn(10, 10) * 2 + 1
        x2 = torch.randn(10, 10) * 3 - 2

        observer.observe(x1)
        observer.observe(x2)

        # Check statistics were collected
        assert observer.count == 2
        assert observer.min_val is not None
        assert observer.max_val is not None

        # Get quantization parameters
        qparams = observer.calculate_qparams()
        assert 'min_val' in qparams
        assert 'max_val' in qparams


class TestQuantizers:
    """Test quantizer implementations."""

    def test_identity_quantizer(self):
        """Test IdentityQuantizer passes through unchanged."""
        quantizer = IdentityQuantizer()
        x = torch.randn(10, 10)
        y = quantizer(x)

        # y is now a QuantTensor, extract value
        assert isinstance(y, QuantTensor)
        assert torch.allclose(x, y.value)
        assert y.scale == 1.0

    def test_lsq_quantizer_forward(self):
        """Test LSQ quantizer forward pass."""
        quantizer = LsqQuantizer(
            bit_width=8,
            signed=True,
            symmetric=True,
            observer=MinMaxObserver(),
        )

        # Initialize manually
        with torch.no_grad():
            quantizer.scale.data.fill_(0.1)
            quantizer._is_init.fill_(True)

        x = torch.randn(10, 10)
        y = quantizer(x)

        # y is now a QuantTensor
        assert isinstance(y, QuantTensor)
        assert y.value.shape == x.shape
        # Values should be different (quantized)
        assert not torch.allclose(x, y.value)

    def test_lsq_quantizer_calibration(self):
        """Test LSQ quantizer calibration mode."""
        observer = MinMaxObserver()
        quantizer = LsqQuantizer(
            bit_width=8,
            signed=True,
            symmetric=True,
            observer=observer,
        )

        # Set to calibration mode
        quantizer.calibration_mode_(True)

        x = torch.randn(10, 10)
        y = quantizer(x)

        # In calibration mode, output should be unchanged (but wrapped in QuantTensor)
        assert isinstance(y, QuantTensor)
        assert torch.allclose(x, y.value)
        # Observer should have collected statistics
        assert observer.count > 0

    def test_lsq_quantizer_initialization(self):
        """Test LSQ quantizer observer-based initialization."""
        observer = MinMaxObserver()
        quantizer = LsqQuantizer(
            bit_width=8,
            signed=True,
            symmetric=True,
            observer=observer,
        )

        # Collect statistics
        quantizer.calibration_mode_(True)
        for _ in range(10):
            x = torch.randn(10, 10)
            quantizer(x)

        # Initialize from observer
        quantizer.calibration_mode_(False)
        quantizer.initialize_from_observer()

        # Scale should be initialized
        assert quantizer._is_init.item() == True
        assert quantizer.scale.item() > 0


class TestQuantizedLayers:
    """Test quantized layer implementations."""

    def test_quant_conv2d_creation(self):
        """Test QuantConv2d creation."""
        layer = QuantConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            weight_quantizer=IdentityQuantizer(),
            activation_quantizer=IdentityQuantizer(),
        )

        assert layer.in_channels == 3
        assert layer.out_channels == 64
        assert layer.weight.shape == (64, 3, 3, 3)

    def test_quant_conv2d_forward(self):
        """Test QuantConv2d forward pass."""
        layer = QuantConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            weight_quantizer=IdentityQuantizer(),
            activation_quantizer=IdentityQuantizer(),
        )

        x = torch.randn(1, 3, 32, 32)
        y = layer(x)

        assert y.shape == (1, 64, 32, 32)

    def test_quant_conv2d_from_conv2d(self):
        """Test creating QuantConv2d from Conv2d."""
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        quant_conv = QuantConv2d.from_conv2d(
            conv,
            weight_quantizer=IdentityQuantizer(),
            activation_quantizer=IdentityQuantizer(),
        )

        # Weights should be copied
        assert torch.allclose(conv.weight, quant_conv.weight)
        if conv.bias is not None:
            assert torch.allclose(conv.bias, quant_conv.bias)

    def test_quant_linear_forward(self):
        """Test QuantLinear forward pass."""
        layer = QuantLinear(
            in_features=128,
            out_features=10,
            weight_quantizer=IdentityQuantizer(),
            activation_quantizer=IdentityQuantizer(),
        )

        x = torch.randn(4, 128)
        y = layer(x)

        assert y.shape == (4, 10)


class TestConfiguration:
    """Test configuration system."""

    def test_quantizer_spec(self):
        """Test QuantizerSpec creation."""
        spec = QuantizerSpec(
            quantizer_class=LsqQuantizer,
            kwargs={'bit_width': 8, 'signed': True}
        )

        quantizer = spec.create_quantizer()
        assert isinstance(quantizer, LsqQuantizer)
        assert quantizer.bit_width == 8

    def test_layer_quant_config(self):
        """Test LayerQuantConfig creation."""
        weight_spec = QuantizerSpec(
            quantizer_class=LsqQuantizer,
            kwargs={'bit_width': 8}
        )

        config = LayerQuantConfig(
            weight_quantizer=weight_spec,
            activation_quantizer=None,
            bias_quantizer=None,
        )

        assert config.weight_quantizer is not None
        assert config.activation_quantizer is None

    def test_uniform_8bit_config(self):
        """Test uniform_8bit_config helper."""
        config = uniform_8bit_config(
            quantize_weights=True,
            quantize_activations=True,
            quantize_bias=False,
        )

        assert config.weight_quantizer is not None
        assert config.activation_quantizer is not None
        assert config.bias_quantizer is None


class TestModelTransformation:
    """Test model transformation utilities."""

    def test_quantize_simple_model(self):
        """Test quantizing a simple model."""
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
        )

        # Create configuration
        layer_config = uniform_8bit_config(
            quantize_weights=True,
            quantize_activations=True,
        )
        model_config = ModelQuantConfig(default_config=layer_config)

        # Quantize
        quant_model = transforms_utils.quantize_model(model, model_config, inplace=False)

        # Check that Conv2d layers were replaced
        conv_count = sum(1 for m in quant_model.modules() if isinstance(m, QuantConv2d))
        assert conv_count == 2

    def test_quantize_model_with_linear(self):
        """Test quantizing a model with Linear layers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, kernel_size=3)
                self.fc = nn.Linear(64, 10)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleModel()

        layer_config = uniform_8bit_config()
        model_config = ModelQuantConfig(default_config=layer_config)

        quant_model = transforms_utils.quantize_model(model, model_config, inplace=False)

        # Check replacements
        assert isinstance(quant_model.conv, QuantConv2d)
        assert isinstance(quant_model.fc, QuantLinear)

    def test_set_quantizers_to_mode(self):
        """Test setting quantizer modes."""
        quantizer = LsqQuantizer(bit_width=8)

        model = nn.Sequential(quantizer)

        # Test calibration mode
        transforms_utils.set_quantizers_to_mode(model, 'calibration')
        assert quantizer.calibration_mode == True
        assert quantizer.enabled == True

        # Test training mode
        transforms_utils.set_quantizers_to_mode(model, 'training')
        assert quantizer.calibration_mode == False
        assert quantizer.enabled == True

        # Test disable
        transforms_utils.set_quantizers_to_mode(model, 'disable')
        assert quantizer.enabled == False


class TestCalibration:
    """Test calibration functionality."""

    def test_calibrate_model(self):
        """Test model calibration."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = SimpleModel()

        # Quantize
        layer_config = uniform_8bit_config()
        model_config = ModelQuantConfig(default_config=layer_config, calibration_batches=5)
        quant_model = transforms_utils.quantize_model(model, model_config, inplace=False)

        # Create calibration data
        calibration_data = [torch.randn(2, 3, 32, 32) for _ in range(10)]
        calibration_loader = torch.utils.data.DataLoader(
            calibration_data,
            batch_size=1,
        )

        # Calibrate
        transforms_utils.calibrate_model(quant_model, calibration_loader, num_batches=5, device='cpu')

        # Check that quantizers were initialized
        for module in quant_model.modules():
            if isinstance(module, BaseQuantizer) and not isinstance(module, IdentityQuantizer):
                if hasattr(module, '_is_init'):
                    assert module._is_init.item() == True


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self):
        """Test complete quantization workflow."""
        # 1. Create model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32, 10)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleModel()

        # 2. Quantize
        layer_config = uniform_8bit_config()
        model_config = ModelQuantConfig(default_config=layer_config)
        quant_model = transforms_utils.quantize_model(model, model_config, inplace=False)

        # 3. Calibrate
        calibration_data = [torch.randn(2, 3, 32, 32) for _ in range(20)]
        calibration_loader = torch.utils.data.DataLoader(calibration_data, batch_size=1)
        transforms_utils.calibrate_model(quant_model, calibration_loader, num_batches=10, device='cpu')
        
        # 4. Test forward pass
        test_input = torch.randn(1, 3, 32, 32)
        output = quant_model(test_input)
        
        assert output.shape == (1, 10)
        
        # 5. Test training step
        quant_model.train()
        optimizer = torch.optim.Adam(quant_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        batch = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 10, (2,))
        
        optimizer.zero_grad()
        output = quant_model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

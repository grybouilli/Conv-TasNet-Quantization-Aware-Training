import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from typing import Dict, Any

from hatorch.utils.model_quantizer import ModelQuantizer
from hatorch.utils.quantize_config import QuantizeConfig
from hatorch.utils.quantized_linear import QuantizedLinear
from hatorch.utils.quantized_conv2d import QuantizedConv2d


class MockFunction(torch.autograd.Function):
    """Mock quantization function for testing."""
    @staticmethod
    def forward(ctx, x, param):
        return x
    
    @staticmethod  
    def backward(ctx, grad_output):
        return grad_output, None


class MockQuantizer:
    """Mock quantizer for testing."""
    @property
    def quantizer(self):
        return MockFunction
    
    def initialize_params(self, weight, bias, channels, features):
        return {'param': torch.tensor(1.0)}
    
    def learnable_params(self):
        return {'param': False}


class MockModel:
    """Mock model class that behaves like the actual Model class."""
    def __init__(self, model):
        self.model = model


# Test model architectures
class SimpleLinearModel(MockModel):
    def __init__(self):
        super().__init__(nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ))


class SimpleConvModel(MockModel):
    def __init__(self):
        super().__init__(nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3)
        ))


class MixedModel(MockModel):
    def __init__(self):
        super().__init__(nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.Linear(64, 10)
        ))


class BatchNormModel(MockModel):
    def __init__(self):
        super().__init__(nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3)
        ))


class NestedModel(MockModel):
    def __init__(self):
        super().__init__(nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64)
            ),
            nn.Conv2d(64, 128, 3)
        ))


class EmptyModel(MockModel):
    def __init__(self):
        super().__init__(nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5)
        ))


# Fixtures
@pytest.fixture
def mock_quantizer_config():
    return QuantizeConfig(weight=MockQuantizer())


@pytest.fixture
def different_quantizer_config():
    return QuantizeConfig(weight=MockQuantizer())


class TestModelQuantizerInitialization:
    """Test ModelQuantizer initialization."""
    
    def test_init_with_all_params(self, mock_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            conv_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config,
            fold_batchnorm=False
        )
        
        assert quantizer._model == model
        assert quantizer._linear_quantizers == mock_quantizer_config
        assert quantizer._conv_quantizers == mock_quantizer_config
        assert quantizer._first_layer_quantizers == mock_quantizer_config
        assert quantizer._last_layer_quantizers == mock_quantizer_config
        assert quantizer._fold_batchnorm == False
    
    def test_init_with_minimal_params(self):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model)
        
        assert quantizer._model == model
        assert quantizer._linear_quantizers is None
        assert quantizer._conv_quantizers is None
        assert quantizer._first_layer_quantizers is None
        assert quantizer._last_layer_quantizers is None
        assert quantizer._fold_batchnorm == False  # Default value
    
    def test_init_with_partial_params(self, mock_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            fold_batchnorm=False
        )
        
        assert quantizer._linear_quantizers == mock_quantizer_config
        assert quantizer._conv_quantizers is None
        assert quantizer._fold_batchnorm == False


class TestLayerCollection:
    """Test layer collection functionality."""
    
    def test_collect_linear_layers(self):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert len(layers) == 2
        assert layers[0][1] == 'linear'  # layer_type
        assert layers[1][1] == 'linear'
        assert isinstance(layers[0][2], nn.Linear)  # actual layer
        assert isinstance(layers[1][2], nn.Linear)
    
    def test_collect_conv_layers(self):
        model = SimpleConvModel()
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert len(layers) == 3
        for i in range(3):
            assert layers[i][1] == 'conv'
            assert isinstance(layers[i][2], nn.Conv2d)
    
    def test_collect_mixed_layers(self):
        model = MixedModel()
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert len(layers) == 4
        assert layers[0][1] == 'conv'    # Conv2d
        assert layers[1][1] == 'linear'  # Linear  
        assert layers[2][1] == 'conv'    # Conv2d
        assert layers[3][1] == 'linear'  # Linear
    
    def test_collect_nested_layers(self):
        model = NestedModel()
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert len(layers) == 4
        assert layers[0][1] == 'conv'    # First Conv2d
        assert layers[1][1] == 'linear'  # First nested Linear
        assert layers[2][1] == 'linear'  # Second nested Linear
        assert layers[3][1] == 'conv'    # Last Conv2d
    
    def test_collect_no_quantizable_layers(self):
        model = EmptyModel()
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert len(layers) == 0


class TestLayerPositionDetection:
    """Test first/last layer detection."""
    
    def test_first_layer_detection(self):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert quantizer._is_first_layer_in_network(layers[0][0], layers) == True
        assert quantizer._is_first_layer_in_network(layers[1][0], layers) == False
    
    def test_last_layer_detection(self):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert quantizer._is_last_layer_in_network(layers[0][0], layers) == False
        assert quantizer._is_last_layer_in_network(layers[1][0], layers) == True
    
    def test_single_layer_is_both_first_and_last(self):
        model = MockModel(nn.Sequential(
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU()
        ))
        quantizer = ModelQuantizer(model)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        assert len(layers) == 1
        assert quantizer._is_first_layer_in_network(layers[0][0], layers) == True
        assert quantizer._is_last_layer_in_network(layers[0][0], layers) == True


class TestQuantizerConfigSelection:
    """Test quantizer config selection logic."""
    
    def test_first_layer_with_special_quantizer(self, mock_quantizer_config, different_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            first_layer_quantizers=different_quantizer_config
        )
        
        # First layer should use first_layer_quantizers
        config = quantizer._get_quantizer_config('linear', True, False)
        assert config == different_quantizer_config
    
    def test_first_layer_without_special_quantizer(self, mock_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model, linear_quantizers=mock_quantizer_config)
        
        # First layer should not be quantized if no first_layer_quantizers
        config = quantizer._get_quantizer_config('linear', True, False)
        assert config is None
    
    def test_last_layer_with_special_quantizer(self, mock_quantizer_config, different_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            last_layer_quantizers=different_quantizer_config
        )
        
        # Last layer should use last_layer_quantizers
        config = quantizer._get_quantizer_config('linear', False, True)
        assert config == different_quantizer_config
    
    def test_last_layer_without_special_quantizer(self, mock_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model, linear_quantizers=mock_quantizer_config)
        
        # Last layer should not be quantized if no last_layer_quantizers
        config = quantizer._get_quantizer_config('linear', False, True)
        assert config is None
    
    def test_middle_linear_layer(self, mock_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model, linear_quantizers=mock_quantizer_config)
        
        # Middle layer should use regular linear_quantizers
        config = quantizer._get_quantizer_config('linear', False, False)
        assert config == mock_quantizer_config
    
    def test_middle_conv_layer(self, mock_quantizer_config):
        model = SimpleConvModel()
        quantizer = ModelQuantizer(model, conv_quantizers=mock_quantizer_config)
        
        # Middle layer should use regular conv_quantizers
        config = quantizer._get_quantizer_config('conv', False, False)
        assert config == mock_quantizer_config
    
    def test_no_applicable_quantizer(self):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(model)  # No quantizers provided
        
        config = quantizer._get_quantizer_config('linear', False, False)
        assert config is None


class TestLastQuantizedLayerDetection:
    """Test detection of which layer will be the last to be quantized."""
    
    def test_all_layers_quantized(self, mock_quantizer_config):
        model = SimpleLinearModel()
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        layers = quantizer._collect_quantizable_layers(model.model)
        
        # First layer should not be last quantized
        assert quantizer._will_be_last_quantized_layer(layers[0][0], layers) == False
        # Second layer should be last quantized
        assert quantizer._will_be_last_quantized_layer(layers[1][0], layers) == True
    
    def test_only_middle_layers_quantized(self, mock_quantizer_config):
        model = MockModel(nn.Sequential(
            nn.Linear(10, 20),  # Not quantized (first layer, no special quantizer)
            nn.Linear(20, 30),  # Quantized (middle)
            nn.Linear(30, 40),  # Quantized (middle)  
            nn.Linear(40, 5)    # Not quantized (last layer, no special quantizer)
        ))
        quantizer = ModelQuantizer(model, linear_quantizers=mock_quantizer_config)
        layers = quantizer._collect_quantizable_layers(model.model)
        
        # Second layer (index 1) should not be last quantized
        assert quantizer._will_be_last_quantized_layer(layers[1][0], layers) == False
        # Third layer (index 2) should be last quantized
        assert quantizer._will_be_last_quantized_layer(layers[2][0], layers) == True


class TestBatchNormFolding:
    """Test BatchNorm folding functionality."""
    
    def test_find_following_batchnorm_found(self):
        model = BatchNormModel()
        quantizer = ModelQuantizer(model)
        
        # Get module children list
        module_children = list(model.model.named_children())
        
        # First Conv2d (index 0) should find BatchNorm at index 1
        bn_layer, bn_name = quantizer._find_following_batchnorm(module_children, 0)
        assert isinstance(bn_layer, nn.BatchNorm2d)
        assert bn_name == '1'
        
        # Second Conv2d (index 3) should find BatchNorm at index 4
        bn_layer, bn_name = quantizer._find_following_batchnorm(module_children, 3)
        assert isinstance(bn_layer, nn.BatchNorm2d)
        assert bn_name == '4'
    
    def test_find_following_batchnorm_not_found(self):
        model = SimpleConvModel()  # No BatchNorm layers
        quantizer = ModelQuantizer(model)
        
        module_children = list(model.model.named_children())
        
        # No BatchNorm should be found
        bn_layer, bn_name = quantizer._find_following_batchnorm(module_children, 0)
        assert bn_layer is None
        assert bn_name is None
    
    def test_find_following_batchnorm_at_end(self):
        model = BatchNormModel()
        quantizer = ModelQuantizer(model)
        
        module_children = list(model.model.named_children())
        
        # Last Conv2d (index 5) has no following layer
        bn_layer, bn_name = quantizer._find_following_batchnorm(module_children, 5)
        assert bn_layer is None
        assert bn_name is None


class TestModelQuantization:
    """Test complete model quantization scenarios."""
    
    def test_quantize_all_linear_layers(self, mock_quantizer_config):
        model = SimpleLinearModel()
        original_layers = list(model.model.children())
        
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        quantizer.quantize_model()
        
        # Check that Linear layers are replaced with QuantizedLinear
        quantized_layers = list(model.model.children())
        assert isinstance(quantized_layers[0], QuantizedLinear)  # First layer
        assert isinstance(quantized_layers[2], QuantizedLinear)  # Last layer
        
        # ReLU should remain unchanged
        assert isinstance(quantized_layers[1], nn.ReLU)
    
    def test_quantize_all_conv_layers(self, mock_quantizer_config):
        model = SimpleConvModel()
        
        quantizer = ModelQuantizer(
            model,
            conv_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        quantizer.quantize_model()
        
        # Check that Conv2d layers are replaced with QuantizedConv2d
        quantized_layers = list(model.model.children())
        assert isinstance(quantized_layers[0], QuantizedConv2d)  # First layer
        assert isinstance(quantized_layers[2], QuantizedConv2d)  # Middle layer
        assert isinstance(quantized_layers[4], QuantizedConv2d)  # Last layer
        
        # ReLU layers should remain unchanged
        assert isinstance(quantized_layers[1], nn.ReLU)
        assert isinstance(quantized_layers[3], nn.ReLU)
    
    def test_quantize_mixed_model(self, mock_quantizer_config):
        model = MixedModel()
        
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            conv_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        quantizer.quantize_model()
        
        quantized_layers = list(model.model.children())
        assert isinstance(quantized_layers[0], QuantizedConv2d)  # First Conv2d
        assert isinstance(quantized_layers[2], QuantizedLinear)  # Linear
        assert isinstance(quantized_layers[4], QuantizedConv2d) # Conv2d  
        assert isinstance(quantized_layers[5], QuantizedLinear) # Last Linear
    
    def test_quantize_no_layers_due_to_missing_configs(self):
        model = SimpleLinearModel()
        original_layers = list(model.model.children())
        
        quantizer = ModelQuantizer(model)  # No quantizer configs provided
        quantizer.quantize_model()
        
        # All layers should remain unchanged
        quantized_layers = list(model.model.children())
        assert len(quantized_layers) == len(original_layers)
        for orig, quant in zip(original_layers, quantized_layers):
            assert orig is quant  # Same object reference
    
    def test_quantize_only_middle_layers(self, mock_quantizer_config):
        model = SimpleLinearModel()
        
        quantizer = ModelQuantizer(model, linear_quantizers=mock_quantizer_config)
        # No first_layer_quantizers or last_layer_quantizers provided
        quantizer.quantize_model()
        
        # Only middle layers should be quantized (none in this case since we only have 2 layers)
        quantized_layers = list(model.model.children())
        assert isinstance(quantized_layers[0], nn.Linear)      # First layer unchanged
        assert isinstance(quantized_layers[2], nn.Linear)      # Last layer unchanged
    
    def test_empty_model_quantization(self):
        model = EmptyModel()
        
        quantizer = ModelQuantizer(model)
        quantizer.quantize_model()  # Should not crash
        
        # Model should remain unchanged
        layers = list(model.model.children())
        assert isinstance(layers[0], nn.ReLU)
        assert isinstance(layers[1], nn.Dropout)


class TestBatchNormFoldingBehavior:
    """Test BatchNorm folding behavior in detail."""
    
    def test_batchnorm_folding_enabled(self, mock_quantizer_config):
        model = BatchNormModel()
        
        quantizer = ModelQuantizer(
            model,
            conv_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config,
            fold_batchnorm=True
        )
        quantizer.quantize_model()
        
        # BatchNorm layers should be replaced with Identity
        layers = list(model.model.children())
        assert isinstance(layers[0], QuantizedConv2d)  # Conv2d quantized
        assert isinstance(layers[1], nn.Identity)      # BatchNorm -> Identity
        assert isinstance(layers[3], QuantizedConv2d)  # Conv2d quantized
        assert isinstance(layers[4], nn.Identity)      # BatchNorm -> Identity
        assert isinstance(layers[6], QuantizedConv2d)  # Last Conv2d quantized
    
    def test_batchnorm_folding_disabled(self, mock_quantizer_config):
        model = BatchNormModel()
        
        quantizer = ModelQuantizer(
            model,
            conv_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config,
            fold_batchnorm=False
        )
        quantizer.quantize_model()
        
        # BatchNorm layers should remain unchanged
        layers = list(model.model.children())
        assert isinstance(layers[0], QuantizedConv2d)   # Conv2d quantized
        assert isinstance(layers[1], nn.BatchNorm2d)    # BatchNorm unchanged
        assert isinstance(layers[3], QuantizedConv2d)   # Conv2d quantized  
        assert isinstance(layers[4], nn.BatchNorm2d)    # BatchNorm unchanged
        assert isinstance(layers[6], QuantizedConv2d)   # Last Conv2d quantized
    
    def test_batchnorm_folding_default_disabled(self, mock_quantizer_config):
        model = BatchNormModel()
        
        quantizer = ModelQuantizer(
            model,
            conv_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
            # fold_batchnorm not specified - should default to False
        )
        quantizer.quantize_model()
        
        # BatchNorm layers should remain unchanged (default behavior)
        layers = list(model.model.children())
        assert isinstance(layers[1], nn.BatchNorm2d)  # BatchNorm preserved
        assert isinstance(layers[4], nn.BatchNorm2d)  # BatchNorm preserved


class TestQuantizedLayerFlags:
    """Test that quantized layers receive correct flags."""
    
    def test_linear_layer_flags(self, mock_quantizer_config):
        model = MockModel(nn.Sequential(
            nn.Linear(10, 20),  # First layer
            nn.Linear(20, 30),  # Middle layer
            nn.Linear(30, 5)    # Last layer  
        ))
        
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        quantizer.quantize_model()
        
        layers = list(model.model.children())
        
        # First layer
        first_layer = layers[0]
        assert isinstance(first_layer, QuantizedLinear)
        assert first_layer._is_first_layer == True
        assert first_layer._is_first_quantized_layer == True
        assert first_layer._is_last_layer == False
        assert first_layer._is_last_quantized_layer == False
        
        # Middle layer
        middle_layer = layers[1]
        assert isinstance(middle_layer, QuantizedLinear)
        assert middle_layer._is_first_layer == False
        assert middle_layer._is_first_quantized_layer == False
        assert middle_layer._is_last_layer == False
        assert middle_layer._is_last_quantized_layer == False
        
        # Last layer
        last_layer = layers[2]
        assert isinstance(last_layer, QuantizedLinear)
        assert last_layer._is_first_layer == False
        assert last_layer._is_first_quantized_layer == False
        assert last_layer._is_last_layer == True
        assert last_layer._is_last_quantized_layer == True
    
    def test_conv_layer_flags_with_batchnorm(self, mock_quantizer_config):
        model = BatchNormModel()
        
        quantizer = ModelQuantizer(
            model,
            conv_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config,
            fold_batchnorm=True
        )
        quantizer.quantize_model()
        
        layers = list(model.model.children())
        
        # First Conv2d layer
        first_conv = layers[0]
        assert isinstance(first_conv, QuantizedConv2d)
        assert first_conv._is_first_layer == True
        assert first_conv._is_first_quantized_layer == True
        assert first_conv._is_last_layer == False
        assert first_conv._is_last_quantized_layer == False
        
        # Last Conv2d layer  
        last_conv = layers[6]
        assert isinstance(last_conv, QuantizedConv2d)
        assert last_conv._is_first_layer == False
        assert last_conv._is_first_quantized_layer == False
        assert last_conv._is_last_layer == True
        assert last_conv._is_last_quantized_layer == True
    
    def test_mixed_layer_quantization_flags(self, mock_quantizer_config):
        # Model where only some layers get quantized
        model = MockModel(nn.Sequential(
            nn.Linear(10, 20),  # First layer - won't be quantized (no first_layer_quantizers)
            nn.Linear(20, 30),  # Middle layer - will be quantized
            nn.Linear(30, 40),  # Middle layer - will be quantized  
            nn.Linear(40, 5)    # Last layer - won't be quantized (no last_layer_quantizers)
        ))
        
        quantizer = ModelQuantizer(model, linear_quantizers=mock_quantizer_config)
        quantizer.quantize_model()
        
        layers = list(model.model.children())
        
        # First layer should remain unchanged
        assert isinstance(layers[0], nn.Linear)
        
        # Second layer should be first quantized layer
        second_layer = layers[1]
        assert isinstance(second_layer, QuantizedLinear)
        assert second_layer._is_first_layer == False
        assert second_layer._is_first_quantized_layer == True
        assert second_layer._is_last_layer == False
        assert second_layer._is_last_quantized_layer == False
        
        # Third layer should be last quantized layer
        third_layer = layers[2]
        assert isinstance(third_layer, QuantizedLinear)
        assert third_layer._is_first_layer == False
        assert third_layer._is_first_quantized_layer == False
        assert third_layer._is_last_layer == False
        assert third_layer._is_last_quantized_layer == True
        
        # Fourth layer should remain unchanged
        assert isinstance(layers[3], nn.Linear)
    
    def test_single_quantized_layer_flags(self, mock_quantizer_config):
        # Model with single quantizable layer
        model = MockModel(nn.Sequential(
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU()
        ))
        
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        quantizer.quantize_model()
        
        # The single layer should have both first and last flags
        linear_layer = list(model.model.children())[1]
        assert isinstance(linear_layer, QuantizedLinear)
        assert linear_layer._is_first_layer == True
        assert linear_layer._is_first_quantized_layer == True
        assert linear_layer._is_last_layer == True
        assert linear_layer._is_last_quantized_layer == True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_model_with_no_quantizable_layers(self):
        model = EmptyModel()
        
        quantizer = ModelQuantizer(model)
        
        # Should not raise any exceptions
        quantizer.quantize_model()
        
        # Model should remain unchanged
        original_structure = str(model.model)
        assert 'ReLU' in original_structure
        assert 'Dropout' in original_structure
    
    def test_deeply_nested_model(self, mock_quantizer_config):
        # Create a deeply nested model
        model = MockModel(nn.Sequential(
            nn.Sequential(
                nn.Sequential(
                    nn.Linear(10, 20)
                ),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(20, 5)
            )
        ))
        
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        quantizer.quantize_model()
        
        # Should properly find and quantize nested layers
        def find_quantized_layers(module):
            quantized = []
            for child in module.children():
                if isinstance(child, QuantizedLinear):
                    quantized.append(child)
                else:
                    quantized.extend(find_quantized_layers(child))
            return quantized
        
        quantized_layers = find_quantized_layers(model.model)
        assert len(quantized_layers) == 2
    
    def test_model_with_different_quantizer_types(self, mock_quantizer_config, different_quantizer_config):
        model = MixedModel()
        
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            conv_quantizers=different_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=different_quantizer_config
        )
        quantizer.quantize_model()
        
        # Should handle different quantizer configs without issues
        layers = list(model.model.children())
        assert isinstance(layers[0], QuantizedConv2d)  # First layer (Conv2d)
        assert isinstance(layers[2], QuantizedLinear)  # Linear
        assert isinstance(layers[4], QuantizedConv2d)  # Conv2d
        assert isinstance(layers[5], QuantizedLinear)  # Last layer (Linear)
    
    @patch('hatorch.utils.model_quantizer.logger')
    def test_logging_behavior(self, mock_logger, mock_quantizer_config):
        model = SimpleLinearModel()
        
        quantizer = ModelQuantizer(
            model,
            linear_quantizers=mock_quantizer_config,
            first_layer_quantizers=mock_quantizer_config,
            last_layer_quantizers=mock_quantizer_config
        )
        quantizer.quantize_model()
        
        # Verify that logging was called
        mock_logger.info.assert_called()
        
        # Check that the final summary log was made
        call_args_list = [call.args[0] for call in mock_logger.info.call_args_list]
        summary_logs = [log for log in call_args_list if "Quantized" in log and "layers of" in log]
        assert len(summary_logs) == 1
        assert "Quantized 2 Linear and 0 Conv2d layers" in summary_logs[0]


if __name__ == "__main__":
    pytest.main([__file__])
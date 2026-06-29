"""Test script to verify first/last layer configuration works correctly"""

import torch
import torch.nn as nn
from hatorch.transforms.config import create_model_config_with_edge_layers
from hatorch.transforms import utils as transforms_utils


class SimpleModel(nn.Module):
    """Simple test model with multiple conv layers"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # First layer
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # Middle layer
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # Last layer
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def test_first_last_layer_config():
    """Test that first and last layer configurations are applied correctly"""
    
    # Create a simple model
    model = SimpleModel()
    print("Original model:")
    print(model)
    print("\n" + "="*80 + "\n")
    
    # Create configuration with different bit widths for first/last layers
    config = create_model_config_with_edge_layers(
        default_weight_bits=4,           # 4-bit weights for middle layers
        default_activation_bits=4,        # 4-bit activations for middle layers
        first_layer_weight_bits=8,       # 8-bit weights for first layer
        first_layer_activation_bits=8,   # 8-bit activations for first layer
        last_layer_weight_bits=8,        # 8-bit weights for last layer
        last_layer_activation_bits=8,    # 8-bit activations for last layer
        bias_bits=32,
        per_channel_weights=True,
    )
    
    # Quantize the model
    quantized_model = transforms_utils.quantize_model(
        model,
        config,
        inplace=False,
        add_quantize=True,
        add_dequantize=True,
    )
    
    print("Quantized model:")
    print(quantized_model)
    print("\n" + "="*80 + "\n")
    
    # Verify quantizers have correct bit widths
    print("Verifying bit widths:")
    
    quantized_layers = []
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            quantized_layers.append((name, module))
    
    print(f"\nFound {len(quantized_layers)} quantized layers")
    
    for i, (name, module) in enumerate(quantized_layers):
        is_first = (i == 0)
        is_last = (i == len(quantized_layers) - 1)
        
        weight_bits = module.weight_quantizer.bit_width if hasattr(module.weight_quantizer, 'bit_width') else 'N/A'
        act_bits = module.activation_quantizer.bit_width if hasattr(module.activation_quantizer, 'bit_width') else 'N/A'
        
        layer_type = "FIRST" if is_first else ("LAST" if is_last else "MIDDLE")
        print(f"\n{layer_type} Layer: {name}")
        print(f"  Weight bits: {weight_bits}")
        print(f"  Activation bits: {act_bits}")
        
        # Check if bit widths match expectations
        if is_first:
            expected_weight = 8
            expected_act = 8
        elif is_last:
            expected_weight = 8
            expected_act = 8
        else:
            expected_weight = 4
            expected_act = 4
        
        if isinstance(weight_bits, int):
            assert weight_bits == expected_weight, f"Expected weight bits {expected_weight}, got {weight_bits}"
        if isinstance(act_bits, int):
            assert act_bits == expected_act, f"Expected activation bits {expected_act}, got {act_bits}"
    
    print("\n" + "="*80)
    print("✓ All bit width checks passed!")
    print("="*80)


if __name__ == "__main__":
    test_first_last_layer_config()

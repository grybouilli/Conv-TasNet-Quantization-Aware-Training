"""
Quantized ResUnit for ResNet architectures.

This module provides a quantized version of ResUnit that fuses Conv+BN+ReLU blocks
and handles skip connections with proper quantization.
"""

from typing import Optional
import torch
import torch.nn as nn

from hatorch.layers.quantized_conv2d_bn_relu import QuantConvBn2d, QuantConvBnReLU2d
from hatorch.layers.quantized_relu import QuantReLU
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.base import BaseQuantizer
from hatorch.quantizers.common.scale_approximation import ScaleApproximation


__all__ = ["QuantResUnit"]


class QuantResUnit(nn.Module):
    """
    Quantized ResUnit for ResNet architectures.
    
    This module replaces a standard ResUnit which typically has the structure:
    
    ResUnit(
        body: ResBlock(
            conv1: ConvBlock(Conv2d + BatchNorm2d + ReLU)
            conv2: ConvBlock(Conv2d + BatchNorm2d)
        )
        activ: ReLU
        [optional] identity_conv: for skip connection with different dimensions
    )
    
    The forward pass is:
        identity = x (or identity_conv(x) if dimensions change)
        x = body(x)
        x = x + identity  # Skip connection
        x = activ(x)      # Final ReLU
    
    In the quantized version:
    - conv1 (Conv+BN+ReLU) -> QuantConvBnReLU2d
    - conv2 (Conv+BN) -> QuantConvBn2d
    - Final ReLU -> QuantReLU
    - Skip connection addition: TODO - implement rescaling logic for mismatched scales
    
    Attributes:
        conv1: First quantized convolution block (Conv+BN+ReLU)
        conv2: Second quantized convolution block (Conv+BN)
        activ: Final quantized ReLU activation
        identity_conv: Optional quantized convolution for skip connection dimension matching
        resize_identity: Whether to resize identity with convolution
    """
    
    def __init__(
        self,
        # Conv1 parameters (Conv+BN+ReLU)
        conv1_in_channels: int,
        conv1_out_channels: int,
        conv1_kernel_size: int = 3,
        conv1_stride: int = 1,
        conv1_padding: int = 1,
        # Conv2 parameters (Conv+BN)
        conv2_out_channels: int = None,  # If None, same as conv1_out_channels
        conv2_kernel_size: int = 3,
        conv2_stride: int = 1,
        conv2_padding: int = 1,
        # Identity/skip connection
        resize_identity: bool = False,
        identity_conv: Optional[nn.Module] = None,
        # Quantizers for conv1
        conv1_weight_quantizer: Optional[BaseQuantizer] = None,
        conv1_activation_quantizer: Optional[BaseQuantizer] = None,
        conv1_bias_quantizer: Optional[BaseQuantizer] = None,
        # Quantizers for conv2
        conv2_weight_quantizer: Optional[BaseQuantizer] = None,
        conv2_activation_quantizer: Optional[BaseQuantizer] = None,
        conv2_bias_quantizer: Optional[BaseQuantizer] = None,
        # BatchNorm parameters
        bn_eps: float = 1e-05,
        bn_momentum: float = 0.1,
        freeze_bn: bool = False,
        # Skip rescale approximation
        scale_approximation: ScaleApproximation = ScaleApproximation.NONE,
        fixed_point_bits: int = 8,
        fold_batch_norm: bool = True,
    ):
        super().__init__()
        
        if conv2_out_channels is None:
            conv2_out_channels = conv1_out_channels
        
        # First conv block: Conv + BN + ReLU
        self.conv1 = QuantConvBnReLU2d(
            in_channels=conv1_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            bias=False,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            weight_quantizer=conv1_weight_quantizer,
            activation_quantizer=conv1_activation_quantizer,
            bias_quantizer=conv1_bias_quantizer,
            freeze_bn=freeze_bn,
            fold_batch_norm=fold_batch_norm,
        )
        
        # Second conv block: Conv + BN (no ReLU)
        self.conv2 = QuantConvBn2d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel_size,
            stride=conv2_stride,
            padding=conv2_padding,
            bias=False,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            weight_quantizer=conv2_weight_quantizer,
            activation_quantizer=conv2_activation_quantizer,
            bias_quantizer=conv2_bias_quantizer,
            freeze_bn=freeze_bn,
            fold_batch_norm=fold_batch_norm,
        )
        
        # Final ReLU activation (applied after skip connection)
        self.activ = QuantReLU(inplace=True)
        
        # Skip connection handling
        self.resize_identity = resize_identity
        self.identity_conv = identity_conv
        self.scale_approximation = scale_approximation
        self.fixed_point_bits = fixed_point_bits
    
    @classmethod
    def from_resunit(
        cls,
        resunit: nn.Module,
        conv1_weight_quantizer: Optional[BaseQuantizer] = None,
        conv1_activation_quantizer: Optional[BaseQuantizer] = None,
        conv1_bias_quantizer: Optional[BaseQuantizer] = None,
        conv2_weight_quantizer: Optional[BaseQuantizer] = None,
        conv2_activation_quantizer: Optional[BaseQuantizer] = None,
        conv2_bias_quantizer: Optional[BaseQuantizer] = None,
        freeze_bn: bool = False,
        identity_config: Optional[object] = None,  # LayerQuantConfig for identity_conv
        scale_approximation: ScaleApproximation = ScaleApproximation.NONE,
        fixed_point_bits: int = 8,
        fold_batch_norm: bool = True,
    ) -> 'QuantResUnit':
        """
        Create a QuantResUnit from a standard ResUnit.
        
        Args:
            resunit: Original ResUnit module
            conv1_weight_quantizer: Weight quantizer for first conv
            conv1_activation_quantizer: Activation quantizer for first conv
            conv1_bias_quantizer: Bias quantizer for first conv
            conv2_weight_quantizer: Weight quantizer for second conv
            conv2_activation_quantizer: Activation quantizer for second conv
            conv2_bias_quantizer: Bias quantizer for second conv
            freeze_bn: Whether to freeze BN statistics
            
        Returns:
            QuantResUnit with copied parameters from original ResUnit
        """
        # Extract conv blocks from resunit.body.conv1 and resunit.body.conv2
        body = resunit.body
        conv1_block = body.conv1
        conv2_block = body.conv2
        
        # Extract Conv2d and BatchNorm2d from ConvBlocks
        conv1_conv = conv1_block.conv
        conv1_bn = conv1_block.bn
        conv2_conv = conv2_block.conv
        conv2_bn = conv2_block.bn
        
        # Create quantized conv blocks using factory methods
        quant_conv1 = QuantConvBnReLU2d.from_conv_bn(
            conv1_conv,
            conv1_bn,
            weight_quantizer=conv1_weight_quantizer,
            activation_quantizer=conv1_activation_quantizer,
            bias_quantizer=conv1_bias_quantizer,
            freeze_bn=freeze_bn,
            fold_batch_norm=fold_batch_norm,
        )
        
        quant_conv2 = QuantConvBn2d.from_conv_bn(
            conv2_conv,
            conv2_bn,
            weight_quantizer=conv2_weight_quantizer,
            activation_quantizer=conv2_activation_quantizer,
            bias_quantizer=conv2_bias_quantizer,
            freeze_bn=freeze_bn,
            fold_batch_norm=fold_batch_norm,
        )
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        
        # Assign the created modules
        instance.conv1 = quant_conv1
        instance.conv2 = quant_conv2
        instance.activ = QuantReLU(inplace=True)
        
        # Handle identity/skip connection
        instance.resize_identity = hasattr(resunit, 'resize_identity') and resunit.resize_identity
        instance.identity_conv = None
        instance.scale_approximation = scale_approximation
        instance.fixed_point_bits = fixed_point_bits
        
        if instance.resize_identity and hasattr(resunit, 'identity_conv'):
            # Quantize identity_conv if it exists
            # Identity conv is typically a ConvBlock (Conv+BN) for dimension matching
            identity_conv_module = resunit.identity_conv
            
            if hasattr(identity_conv_module, 'conv') and hasattr(identity_conv_module, 'bn'):
                # It's a ConvBlock - create quantized version
                identity_out_channels = identity_conv_module.conv.out_channels
                
                # Create quantizers from identity_config if provided
                identity_weight_quantizer = None
                identity_activation_quantizer = None
                identity_bias_quantizer = None
                
                if identity_config is not None:
                    if identity_config.weight_quantizer is not None:
                        identity_weight_quantizer = identity_config.weight_quantizer.create_quantizer(identity_out_channels)
                    if identity_config.activation_quantizer is not None:
                        identity_activation_quantizer = identity_config.activation_quantizer.create_quantizer()
                    if identity_config.bias_quantizer is not None:
                        identity_bias_quantizer = identity_config.bias_quantizer
                else:
                    # Fallback to conv2 quantizers if no identity_config provided
                    identity_weight_quantizer = conv2_weight_quantizer
                    identity_activation_quantizer = conv2_activation_quantizer
                    identity_bias_quantizer = conv2_bias_quantizer
                
                instance.identity_conv = QuantConvBn2d.from_conv_bn(
                    identity_conv_module.conv,
                    identity_conv_module.bn,
                    weight_quantizer=identity_weight_quantizer,
                    activation_quantizer=identity_activation_quantizer,
                    bias_quantizer=identity_bias_quantizer,
                    freeze_bn=freeze_bn,
                    fold_batch_norm=fold_batch_norm,
                )
            else:
                # Not a ConvBlock - keep as-is (fallback)
                instance.identity_conv = identity_conv_module
        
        return instance
    
    def forward(self, x: QuantTensor) -> QuantTensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Input QuantTensor
            
        Returns:
            Output QuantTensor after residual connection and final activation
        """
        # Store identity for skip connection
        if self.resize_identity and self.identity_conv is not None:
            # Use quantized identity_conv
            identity = self.identity_conv(x)
        else:
            identity = x
        
        # Main path through conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Rescale the skip connection to the main path before addition

        # Compute scale ratio (convert identity to main-path scale)
        scale_ratio = identity.scale / x.scale
        
        # Reshape scale_ratio for proper broadcasting with activation tensors
        # Activations are [B, C, H, W], scales are typically [C] for per-channel or scalar
        if isinstance(scale_ratio, torch.Tensor):
            # Reshape to [1, C, 1, 1] for broadcasting
            if scale_ratio.dim() == 1:
                scale_ratio = scale_ratio.view(1, -1, 1, 1)
            elif scale_ratio.dim() == 0:
                # Scalar tensor - no reshaping needed
                pass
        
        # Check spatial dimensions match before rescaling
        if x.value.shape[2:] != identity.value.shape[2:]:
            raise RuntimeError(
                f"Spatial dimension mismatch in ResUnit skip connection: "
                f"main path spatial size {x.value.shape[2:]} != identity path spatial size {identity.value.shape[2:]}. "
                f"Main path shape: {x.value.shape}, Identity path shape: {identity.value.shape}. "
                f"resize_identity={self.resize_identity}, has_identity_conv={self.identity_conv is not None}"
            )

        if self.scale_approximation == ScaleApproximation.FIXED_POINT and self.fixed_point_bits is not None:
            exact_rescaled = identity.value * scale_ratio
            # Fixed-point approximation like LsqUniformQuantizer
            scale_factor = float(1 << self.fixed_point_bits)
            int_scale = torch.round(scale_ratio * scale_factor).to(torch.int64)
            prod = (int_scale * identity.value).to(torch.int64)
            round_to_nearest = 1 << (self.fixed_point_bits - 1)
            prod_rounded = prod + torch.sign(prod.float()).to(torch.int64) * round_to_nearest
            approx_rescaled = (prod_rounded >> self.fixed_point_bits).to(identity.value.dtype)
            identity_rescaled = exact_rescaled + (approx_rescaled - exact_rescaled).detach()
        elif self.scale_approximation == ScaleApproximation.POWER_OF_TWO:
            # Approximate scale ratio to nearest power of two
            tiny = torch.finfo(scale_ratio.dtype).tiny
            scale_abs = scale_ratio.abs().clamp_min(tiny)
            nearest_exp = torch.round(torch.log2(scale_abs))
            power_of_two_scale = torch.pow(2, nearest_exp)
            identity_rescaled = identity.value * power_of_two_scale
        else:
            identity_rescaled = identity.value * scale_ratio
        result = QuantTensor(x.value + identity_rescaled, x.scale, None)
        # Apply final ReLU
        return self.activ(result)
    
    def extra_repr(self):
        return f"resize_identity={self.resize_identity}"
    
    def train(self, mode=True):
        """
        Training mode with proper BatchNorm handling.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

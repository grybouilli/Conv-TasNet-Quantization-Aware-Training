"""
Base classes for quantizers in the hatorch quantization framework.

This module provides abstract base classes that define the interface for
all quantizers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torch import Tensor

from hatorch.observers.observers import Observer
from hatorch.modules.quantized_tensor import QuantTensor


class BaseQuantizer(nn.Module, ABC):
    """
    Abstract base class for all quantizers.
    
    Quantizers implement the quantization operation with learnable or fixed parameters.
    They can use observers for initialization and support both training and inference modes.
    
    Args:
        observer: Optional observer for automatic initialization
        enabled: Whether quantization is enabled (default: True)
    """
    
    def __init__(
        self,
        is_activation: bool = False,
        symmetric: bool = True,
        per_channel: bool = False,
        num_channels: Optional[int] = None,
        channels_axis: Optional[int] = None,
        input_dim: Optional[int] = None,
        enabled: bool = True
    ):
        super().__init__()
        self.is_activation = is_activation
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.num_channels = num_channels
        self.channels_axis = channels_axis
        self.enabled = enabled
        self.register_buffer('is_initialized', torch.tensor(False))
        self._calibration_mode = False
    
    @abstractmethod
    def forward(self, x: Tensor, in_scale: Optional[Tensor] = None) -> QuantTensor:
        """
        Apply quantization to input tensor.
        
        Args:
            x: Input tensor to quantize
            in_scale: Optional input scale for quantization
            
        Returns:
            QuantTensor with quantized value and metadata
        """
        pass
    
    @abstractmethod
    def extra_repr(self) -> str:
        """Return extra representation string for print/logging."""
        pass
    
    def enable(self) -> None:
        """Enable quantization."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable quantization (pass-through mode)."""
        self.enabled = False
    
    def calibration_mode_(self, mode: bool = True) -> 'BaseQuantizer':
        """
        Set calibration mode.
        
        In calibration mode, the quantizer collects statistics using its observer
        instead of performing quantization.
        
        Args:
            mode: Whether to enable calibration mode
            
        Returns:
            Self for method chaining
        """
        self._calibration_mode = mode
        return self
    
    @property
    def calibration_mode(self) -> bool:
        """Check if quantizer is in calibration mode."""
        return self._calibration_mode
    
    def observe_tensor(self, x: Tensor, per_channel: bool) -> None:
        """
        Observe tensor for statistics collection.
        
        Args:
            x: Tensor to observe
            per_channel: Whether to observe per channel
        """
        if self.observer is not None:
            self.observer.observe(x, per_channel)
    
    def initialize_from_observer(self) -> None:
        """
        Initialize quantization parameters from observer statistics.
        
        Should be called after calibration phase.
        """
        if self.observer is None:
            raise RuntimeError("No observer attached to this quantizer")
        
        qparams = self.observer.calculate_qparams()
        self._initialize_from_qparams(qparams)
        self.is_initialized.fill_(True)
    
    @abstractmethod
    def _initialize_from_qparams(self, qparams: Dict[str, Any]) -> None:
        """
        Initialize internal parameters from quantization parameters.
        
        Args:
            qparams: Dictionary of quantization parameters from observer
        """
        pass

    @abstractmethod
    def onnx_export_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class IdentityQuantizer(BaseQuantizer):
    """
    Identity quantizer that performs no quantization (pass-through).
    
    Useful as a placeholder when no quantization is desired for certain tensors.
    """
    
    def __init__(self):
        super().__init__(enabled=True)
    
    def forward(self, x: Tensor, in_scale: Optional[Tensor] = None) -> QuantTensor:
        """Return input as QuantTensor with identity scale."""
        # TODO: maybe return the scale instead
        #assert in_scale is None, "IdentityQuantizer does not support input scale"
        # Wrap plain tensor as QuantTensor with identity quantization
        return QuantTensor(
            value=x,
            scale=in_scale,
            zero_point=None,
            bit_width=None,
            signed=None,
        )
    
    def extra_repr(self) -> str:
        return "Identity (no quantization)"
    
    def _initialize_from_qparams(self, qparams: Dict[str, Any]) -> None:
        """No parameters to initialize."""
        pass

    def onnx_export_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

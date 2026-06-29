"""
Observer classes for quantization statistics collection.

This module provides observer implementations that collect statistics
during calibration to initialize quantization parameters.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
from torch import Tensor


class Observer(ABC):
    """
    Abstract base class for observers that collect statistics for quantizer initialization.
    
    Observers track min/max values or other statistics during calibration to determine
    appropriate quantization parameters.
    """
    
    def __init__(self):
        self.min_val: Optional[Tensor] = None
        self.max_val: Optional[Tensor] = None
        self.mean_val: Optional[Tensor] = None
        self.std_val: Optional[Tensor] = None
        self.count = 0
    
    @abstractmethod
    def observe(self, x: Tensor, per_channel: bool = False) -> None:
        """
        Observe a tensor and update internal statistics.
        
        Args:
            x: Input tensor to observe
            per_channel: If True, compute statistics per channel (dim 0). If False, compute per tensor.
        """
        pass
    
    @abstractmethod
    def calculate_qparams(self) -> Dict[str, Any]:
        """
        Calculate quantization parameters based on observed statistics.
        
        Returns:
            Dictionary containing quantization parameters (e.g., scale, zero_point, qn, qp)
        """
        pass
    
    def reset(self) -> None:
        """Reset observer statistics."""
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
        self.count = 0

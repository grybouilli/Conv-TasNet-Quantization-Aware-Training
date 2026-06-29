from typing import Optional, Dict, Any
import torch
from torch import Tensor

from hatorch.observers.observers import Observer


class MinMaxObserver(Observer):
    """
    Observer that tracks min and max values across observed tensors.

    Args:
        percentile: Optional percentile for outlier rejection (e.g., 0.01 for 1st/99th percentile)
    """

    def __init__(self, percentile: Optional[float] = None):
        super().__init__()
        self.percentile = percentile

    def observe(self, x: Tensor, per_channel: bool = False) -> None:
        """Update min/max statistics with new tensor."""
        with torch.no_grad():
            if per_channel:
                # Compute statistics per channel (dim 0)
                # Reduce over all dims except 0
                reduce_dims = list(range(1, x.ndim)) if x.ndim > 1 else None
                
                if self.percentile is not None and reduce_dims:
                    # Flatten each channel separately
                    min_vals = []
                    max_vals = []
                    for i in range(x.shape[0]):
                        channel_data = x[i].flatten()
                        min_vals.append(torch.quantile(channel_data, self.percentile))
                        max_vals.append(torch.quantile(channel_data, 1.0 - self.percentile))
                    min_val = torch.stack(min_vals)
                    max_val = torch.stack(max_vals)
                else:
                    if reduce_dims:
                        min_val = x.amin(dim=reduce_dims)
                        max_val = x.amax(dim=reduce_dims)
                    else:
                        min_val = x
                        max_val = x
                
                mean_val = x.mean(dim=reduce_dims) if reduce_dims else x.mean()
                std_val = x.std(dim=reduce_dims) if reduce_dims else x.std()
            else:
                # Compute statistics per tensor
                if self.percentile is not None:
                    # Use percentile to ignore outliers
                    x_flat = x.flatten()
                    min_val = torch.quantile(x_flat, self.percentile)
                    max_val = torch.quantile(x_flat, 1.0 - self.percentile)
                else:
                    min_val = x.min()
                    max_val = x.max()
                
                mean_val = x.mean()
                std_val = x.std()

            if self.min_val is None:
                self.min_val = min_val
                self.max_val = max_val
                self.mean_val = mean_val
                self.std_val = std_val
            else:
                self.min_val = torch.min(self.min_val, min_val)
                self.max_val = torch.max(self.max_val, max_val)
                # Running average for mean and std
                self.mean_val = (self.mean_val * self.count + mean_val) / (self.count + 1)
                self.std_val = (self.std_val * self.count + std_val) / (self.count + 1)

            self.count += 1

    def calculate_qparams(self) -> Dict[str, Any]:
        """Calculate scale and range based on observed min/max."""
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Observer has not observed any data yet")

        return {
            'min_val': self.min_val,
            'max_val': self.max_val,
            'mean': self.mean_val,
            'std': self.std_val,
        }

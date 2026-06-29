from enum import Enum


class ScaleApproximation(Enum):
    """Scale approximation methods for hardware-efficient quantization."""
    NONE = "none"  # No approximation, use exact floating-point scale
    FIXED_POINT = "fixed_point"  # Fixed-point approximation (multiply + shift)
    POWER_OF_TWO = "power_of_two"  # Round scale to nearest power of 2

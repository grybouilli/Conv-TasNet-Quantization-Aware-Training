import pytest
import torch
from hatorch.quantizers import LNS1

test_device = "cpu"


def test_init_lns1():
    with pytest.raises(AssertionError):
        LNS1(2.0, 2, 3)
    with pytest.raises(AssertionError):
        LNS1(2.0, 0, 0)

    with pytest.raises(AssertionError):
        LNS1(-1.0, 2, -2)

    with pytest.raises(AssertionError):
        LNS1(0.0, 2, -2)


def test_lns_quantizer():
    base = 2.0
    msb = 1
    lsb = 0
    lns = LNS1(base, msb, lsb)  # [-1, -2^(-3)] U [2^(-3), 1]
    lns_constants = lns.initialize_params(torch.empty(0), torch.empty(0))
    ulp_ufix = lns_constants["ulp_ufix"]
    max_ufix = lns_constants["max_ufix"]
    min_ufix = lns_constants["min_ufix"]
    lns_zero = lns_constants["lns_zero"]

    x = torch.tensor(
        [
            -(10**3),
            -1.0,
            -0.5,
            -0.4,
            -0.25,
            -0.13,
            -0.125,
            -0.1,
            0.0,
            0.1,
            0.125,
            0.135,
            0.25,
            0.5,
            1.0,
            (10**4),
        ]
    )

    q_x = lns.quantizer.apply(x, base, ulp_ufix, max_ufix, min_ufix, lns_zero)
    q_x_expected = torch.tensor(
        [
            -1.0,  # -2^0
            -1.0,  # -2^0
            -0.5,  # -2^(-1)
            -0.5,  # -2^(-1)
            -0.25,  # -2^(-2)
            -0.125,  # -2^(-3)
            -0.125,  # -2^(-3)
            -0.125,  # -2^(-3)
            0.0,  # 0
            0.125,  # -2^(-3)
            0.125,  # -2^(-3)
            0.125,  # -2^(-3)
            0.25,  # 2^(-2)
            0.5,  # 2^(-1)
            1.0,  # 2^0
            1.0,  # 2^0
        ]
    )
    assert torch.equal(q_x, q_x_expected)

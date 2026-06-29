import pytest
import torch

from hatorch.quantizers import StepExactQuantizer, TIE_BREAK_LEFT, TIE_BREAK_RIGHT


def test_init_step_exact_validations():
    # levels 2D -> error
    levels_2d = torch.tensor([[0.0], [1.0], [2.0]])
    boundaries = torch.tensor([0.5, 1.5])
    directions = torch.tensor([TIE_BREAK_LEFT, TIE_BREAK_LEFT])
    with pytest.raises(ValueError):
        StepExactQuantizer(levels_2d, boundaries, directions)

    # boundaries not increasing -> error
    levels = torch.tensor([0.0, 1.0, 2.0])
    boundaries_bad = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError):
        StepExactQuantizer(levels, boundaries_bad, directions)

    # directions shape mismatch -> error
    boundaries_ok = torch.tensor([0.5, 1.5])
    directions_bad = torch.tensor([TIE_BREAK_LEFT])
    with pytest.raises(ValueError):
        StepExactQuantizer(levels, boundaries_ok, directions_bad)

    # directions not in {TIE_BREAK_LEFT, TIE_BREAK_RIGHT} -> error
    directions_bad_vals = torch.tensor([TIE_BREAK_LEFT, 2])
    with pytest.raises(ValueError):
        StepExactQuantizer(levels, boundaries_ok, directions_bad_vals)


def test_step_exact_forward():
    levels = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    boundaries = (levels[:-1] + levels[1:]) / 2
    directions = torch.tensor([TIE_BREAK_LEFT, TIE_BREAK_LEFT, TIE_BREAK_LEFT])

    x = torch.tensor([-10.0, -1.9, -1.5, -1.4, -0.25, 0.0, 0.5, 1.0, 2.0])

    q = StepExactQuantizer(levels, boundaries, directions, backward="STE")
    y = q.quantizer.apply(x, levels, boundaries, directions)

    y_expected = torch.tensor([-2.0, -2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert torch.equal(y, y_expected)


def test_backward_ste():
    levels = torch.tensor([-1.0, 0.0, 1.0])
    boundaries = torch.tensor([-0.5, 0.5])
    directions = torch.tensor([TIE_BREAK_LEFT, TIE_BREAK_LEFT])

    x = torch.linspace(-1.0, 1.0, 9, requires_grad=True)
    q = StepExactQuantizer(levels, boundaries, directions, backward="STE")
    y = q.quantizer.apply(x, levels, boundaries, directions)
    loss = y.sum()
    loss.backward()

    assert torch.allclose(x.grad, torch.ones_like(x))


def test_backward_pwl():
    levels = torch.tensor([-1.0, 0.0, 1.0])
    boundaries = torch.tensor([-0.5, 0.5])
    directions = torch.tensor([TIE_BREAK_LEFT, TIE_BREAK_LEFT])

    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    q = StepExactQuantizer(levels, boundaries, directions, backward="PWL")
    y = q.quantizer.apply(x, levels, boundaries, directions)
    loss = y.sum()
    loss.backward()

    expected = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
    assert torch.allclose(x.grad, expected)


def test_backward_mad():
    levels = torch.tensor([-1.0, 0.0, 1.0])
    boundaries = torch.tensor([-0.5, 0.5])
    directions = torch.tensor([TIE_BREAK_LEFT, TIE_BREAK_LEFT])

    x = torch.tensor([-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0], requires_grad=True)
    q = StepExactQuantizer(levels, boundaries, directions, backward="MAD")
    y = q.quantizer.apply(x, levels, boundaries, directions)
    loss = y.sum()
    loss.backward()

    expected = torch.tensor([0.25, 0.5, 1.0, 1.0, 1.0, 0.5, 0.25])
    assert torch.allclose(x.grad, expected)

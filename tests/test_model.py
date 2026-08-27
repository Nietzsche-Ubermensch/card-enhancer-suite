import pytest

torch = pytest.importorskip("torch")

from elan.model import ELAN, ELANLight  # noqa: E402


def test_elan_forward_shape():
    model = ELAN(upscale=2, channels=24, n_blocks=2, num_heads=3, n_groups=4)
    x = torch.rand(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 3, 64, 64)


def test_elan_light_forward_shape():
    model = ELANLight(upscale=2, channels=24, n_blocks=2, num_heads=3)
    x = torch.rand(2, 3, 24, 24)
    y = model(x)
    assert y.shape == (2, 3, 48, 48)


def test_elan_backward_flows():
    model = ELAN(upscale=2, channels=24, n_blocks=2, num_heads=3, n_groups=4)
    x = torch.rand(1, 3, 16, 16, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_elan_handles_odd_input_sizes():
    # Window padding: 17 is not a multiple of the 8/16 window sizes.
    model = ELAN(upscale=2, channels=24, n_blocks=4, num_heads=3, n_groups=4)
    x = torch.rand(1, 3, 17, 21)
    y = model(x)
    assert y.shape == (1, 3, 34, 42)


def test_invalid_group_split_raises():
    with pytest.raises(ValueError):
        ELAN(upscale=2, channels=30, n_blocks=1, n_groups=4)

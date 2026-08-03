"""Shape, causality and configurability tests for the TCN encoders (F5 §1)."""

import pytest
import torch

from flatjepa.models.encoders import (
    ActionEncoder,
    ActionEncoderConfig,
    CausalConv1d,
    StateEncoder,
    StateEncoderConfig,
    TCNEncoder,
)

H, T, B = 10, 20, 4


def test_state_encoder_shapes():
    enc = StateEncoder(StateEncoderConfig(state_dim=9, latent_dim=24))
    out = enc(torch.randn(B, H, 9))
    assert out.shape == (B, 24)


def test_state_encoder_batched_windows():
    """The (B, T, H, D) path used to encode all T target windows in one pass."""
    enc = StateEncoder(StateEncoderConfig(state_dim=6, latent_dim=16))
    out = enc(torch.randn(B, T, H, 6))
    assert out.shape == (B, T, 16)


@pytest.mark.parametrize("latent_dim", [4, 9, 16, 24, 64])
def test_latent_width_is_configurable(latent_dim):
    """F5 §1: the latent width is a swept parameter, never hard-coded at 24."""
    enc = StateEncoder(StateEncoderConfig(latent_dim=latent_dim))
    assert enc.latent_dim == latent_dim
    assert enc(torch.randn(B, H, 9)).shape == (B, latent_dim)


def test_default_channels_match_spec():
    assert StateEncoderConfig().channels == (8, 8, 16)
    assert ActionEncoderConfig().channels == (4, 4, 8)
    assert ActionEncoderConfig().embed_dim == 8


def test_action_encoder_shapes():
    enc = ActionEncoder(ActionEncoderConfig(action_dim=3, embed_dim=8))
    out = enc(torch.randn(B, H, 3), torch.randn(B, T, 3))
    assert out.shape == (B, T, 8)
    assert enc(torch.randn(B, H, 3)).shape == (B, H, 8)


def test_receptive_field_covers_history():
    enc = StateEncoder(StateEncoderConfig())
    assert enc.receptive_field >= H


def test_causal_conv_ignores_future():
    conv = CausalConv1d(2, 3, kernel_size=3, dilation=2)
    x = torch.randn(1, 2, 8)
    y = conv(x)
    assert y.shape == x.shape[:1] + (3,) + x.shape[2:]
    x2 = x.clone()
    x2[..., 5:] += 10.0  # perturb the future only
    y2 = conv(x2)
    assert torch.allclose(y[..., :5], y2[..., :5], atol=1e-6)


def test_action_embeddings_are_causal():
    """Perturbing a late future action must not change earlier action embeddings — otherwise the
    predictor would see actions it has not been given yet."""
    enc = ActionEncoder(ActionEncoderConfig()).eval()
    hist = torch.randn(2, H, 3)
    fut = torch.randn(2, T, 3)
    with torch.no_grad():
        z = enc(hist, fut)
        fut2 = fut.clone()
        fut2[:, 10:] += 5.0
        z2 = enc(hist, fut2)
    assert torch.allclose(z[:, :10], z2[:, :10], atol=1e-6)
    assert not torch.allclose(z[:, 10:], z2[:, 10:], atol=1e-6)


def test_normalization_does_not_pool_over_time():
    """ChannelLayerNorm must normalize per time step; a norm over (C, L) would leak the future."""
    enc = TCNEncoder(input_dim=2, channels=(4, 4), output_dim=3, norm=True).eval()
    x = torch.randn(1, 12, 2)
    with torch.no_grad():
        y = enc(x, pool="sequence")
        x2 = x.clone()
        x2[:, 6:] *= 20.0
        y2 = enc(x2, pool="sequence")
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)


def test_variable_sequence_length():
    enc = TCNEncoder(input_dim=3, channels=(4, 8), output_dim=5)
    for length in (1, 4, 30):
        assert enc(torch.randn(2, length, 3), pool="sequence").shape == (2, length, 5)
        assert enc(torch.randn(2, length, 3)).shape == (2, 5)


def test_rejects_wrong_input_dim():
    enc = TCNEncoder(input_dim=3, channels=(4,), output_dim=2)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 5, 4))
    with pytest.raises(ValueError):
        enc(torch.randn(2, 5))


def test_encoders_are_differentiable():
    enc = StateEncoder(StateEncoderConfig(latent_dim=8))
    x = torch.randn(2, H, 9, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

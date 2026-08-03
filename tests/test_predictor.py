"""Tests for the GRU latent predictor (F5 §1)."""

import pytest
import torch

from flatjepa.models.predictor import GRUPredictor, PredictorConfig

B, T = 4, 20


def test_predictor_shapes():
    pred = GRUPredictor(PredictorConfig(latent_dim=24, action_embed_dim=8))
    out = pred(torch.randn(B, 24), torch.randn(B, T, 8))
    assert out.shape == (B, T, 24)


@pytest.mark.parametrize("latent_dim", [4, 9, 24, 48])
def test_latent_width_is_configurable(latent_dim):
    pred = GRUPredictor(PredictorConfig(latent_dim=latent_dim))
    assert pred(torch.randn(B, latent_dim), torch.randn(B, T, 8)).shape == (B, T, latent_dim)


def test_hidden_dim_may_differ_from_latent_dim():
    pred = GRUPredictor(PredictorConfig(latent_dim=9, hidden_dim=32))
    assert pred(torch.randn(B, 9), torch.randn(B, T, 8)).shape == (B, T, 9)


def test_sequence_call_matches_step_recursion():
    """The batched nn.GRU call must equal the literal unroll s̃_k = Pred(s̃_{k-1}, z_{k-1})."""
    pred = GRUPredictor(PredictorConfig(latent_dim=12)).eval()
    context = torch.randn(B, 12)
    actions = torch.randn(B, T, 8)
    with torch.no_grad():
        batched = pred(context, actions)
        latent, hidden = context, None
        stepped = []
        for k in range(T):
            latent, hidden = pred.step(latent, actions[:, k], hidden)
            stepped.append(latent)
        stepped = torch.stack(stepped, dim=1)
    assert torch.allclose(batched, stepped, atol=1e-6)


def test_context_actually_conditions_the_rollout():
    pred = GRUPredictor(PredictorConfig(latent_dim=12)).eval()
    actions = torch.randn(B, T, 8)
    with torch.no_grad():
        a = pred(torch.zeros(B, 12), actions)
        b = pred(torch.randn(B, 12), actions)
    assert not torch.allclose(a, b, atol=1e-4)


def test_actions_actually_drive_the_rollout():
    pred = GRUPredictor(PredictorConfig(latent_dim=12)).eval()
    context = torch.randn(B, 12)
    with torch.no_grad():
        a = pred(context, torch.zeros(B, T, 8))
        b = pred(context, torch.randn(B, T, 8))
    assert not torch.allclose(a, b, atol=1e-4)


def test_rejects_wrong_action_dim():
    pred = GRUPredictor(PredictorConfig(latent_dim=12, action_embed_dim=8))
    with pytest.raises(ValueError):
        pred(torch.randn(B, 12), torch.randn(B, T, 5))


def test_predictor_is_differentiable():
    pred = GRUPredictor(PredictorConfig(latent_dim=8))
    context = torch.randn(B, 8, requires_grad=True)
    pred(context, torch.randn(B, T, 8)).sum().backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()

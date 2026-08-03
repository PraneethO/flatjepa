"""Tests for the assembled JEPA core and its combined objective (F5 §1–§2, §6)."""

import pytest
import torch

from flatjepa.models.diagnostics import latent_diagnostics
from flatjepa.models.jepa import FlatJEPA, JEPAConfig
from flatjepa.models.sigreg import SIGRegConfig

B, H, T, DS, DA = 6, 10, 20, 9, 3


def _batch(b=B, h=H, t=T, ds=DS, da=DA, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "state_hist": torch.randn(b, h, ds, generator=g),
        "action_hist": torch.randn(b, h, da, generator=g),
        "state_future": torch.randn(b, t, ds, generator=g),
        "action_future": torch.randn(b, t, da, generator=g),
    }


def _config(**kw):
    base = dict(history=H, horizon=T, sigreg=SIGRegConfig(num_slices=32))
    base.update(kw)
    return JEPAConfig(**base)


def test_forward_shapes():
    model = FlatJEPA(_config(latent_dim=24))
    out = model(**_batch())
    assert out["context"].shape == (B, 24)
    assert out["target_latents"].shape == (B, T, 24)
    assert out["predicted_latents"].shape == (B, T, 24)
    assert out["per_step_loss"].shape == (T,)
    assert out["loss"].shape == ()
    assert out["loss_pred"].shape == ()
    assert out["loss_sigreg"].shape == ()


@pytest.mark.parametrize("latent_dim", [4, 9, 16, 24, 48])
def test_latent_width_sweeps(latent_dim):
    """F5 §6: latent width configurable and swept — nothing may hard-code 24."""
    model = FlatJEPA(_config(latent_dim=latent_dim))
    out = model(**_batch())
    assert out["context"].shape[-1] == latent_dim
    assert out["predicted_latents"].shape[-1] == latent_dim


@pytest.mark.parametrize("history,horizon", [(10, 20), (5, 3), (16, 8)])
def test_window_sizes_configurable(history, horizon):
    model = FlatJEPA(_config(history=history, horizon=horizon, latent_dim=12))
    out = model(**_batch(h=history, t=horizon))
    assert out["predicted_latents"].shape == (B, horizon, 12)


def test_target_windows_slide_correctly():
    """Target k must be the H-step window ending at t+k, built from history then future."""
    model = FlatJEPA(_config(latent_dim=8, history=4, horizon=3))
    hist = torch.arange(4.0).reshape(1, 4, 1).expand(1, 4, DS).contiguous()
    fut = torch.arange(4.0, 7.0).reshape(1, 3, 1).expand(1, 3, DS).contiguous()
    windows = model.target_windows(hist, fut)
    assert windows.shape == (1, 3, 4, DS)
    assert torch.allclose(windows[0, 0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(windows[0, 2, :, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))


def test_loss_decomposition():
    model = FlatJEPA(_config(latent_dim=16, lambda_sig=0.5))
    out = model(**_batch())
    expected = out["loss_pred"] + 0.5 * out["loss_sigreg"]
    assert torch.allclose(out["loss"], expected)
    assert torch.allclose(out["loss_pred"], out["per_step_loss"].mean())


def test_lambda_zero_excludes_sigreg_from_the_graph_but_still_reports_it():
    """λ_sig = 0 is a real experimental arm (F5 §3), not a disabled feature: the penalty must
    still be logged, and it must not contribute gradient."""
    model = FlatJEPA(_config(latent_dim=16, lambda_sig=0.0))
    out = model(**_batch())
    assert torch.allclose(out["loss"], out["loss_pred"])
    assert float(out["loss_sigreg"]) > 0.0
    assert not out["loss_sigreg"].requires_grad


def test_target_mask_excludes_steps():
    model = FlatJEPA(_config(latent_dim=8, lambda_sig=0.0)).eval()
    batch = _batch()
    mask = torch.ones(B, T)
    mask[:, 5:] = 0.0
    with torch.no_grad():
        masked = model(**batch, target_mask=mask)
        full = model(**batch)
    assert torch.allclose(masked["loss_pred"], full["per_step_loss"][:5].mean(), atol=1e-5)


def test_gradients_reach_every_submodule():
    """No stop-gradient and no EMA copy (F5 §1): gradients must flow through the target branch too,
    which is precisely why the constant solution is reachable and SIGReg is needed."""
    model = FlatJEPA(_config(latent_dim=16, lambda_sig=1.0))
    model(**_batch())["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_single_encoder_serves_both_branches():
    """F5 §1: one Enc_θ produces context and targets — no EMA copy, no second encoder."""
    from flatjepa.models.encoders import StateEncoder

    model = FlatJEPA(_config(latent_dim=8)).eval()
    state_encoders = [m for m in model.modules() if isinstance(m, StateEncoder)]
    assert len(state_encoders) == 1

    # Both branches must be the same parameters, so perturbing the encoder moves context and
    # targets together — an EMA/stop-gradient target would decouple them.
    batch = _batch()
    with torch.no_grad():
        before = (model.encode_context(batch["state_hist"]),
                  model.encode_targets(batch["state_hist"], batch["state_future"]))
        model.state_encoder.tcn.head.bias.add_(1.0)
        after = (model.encode_context(batch["state_hist"]),
                 model.encode_targets(batch["state_hist"], batch["state_future"]))
    assert torch.allclose(after[0] - before[0], torch.ones_like(before[0]))
    assert torch.allclose(after[1] - before[1], torch.ones_like(before[1]))


def test_constant_encoder_gives_zero_prediction_loss():
    """The degenerate solution F5 §3 warns about: if Enc_θ ≡ c the prediction loss is exactly zero
    while the representation carries nothing.  Verifying it here is what makes the collapse
    diagnostics and SIGReg meaningful rather than assumed."""
    model = FlatJEPA(_config(latent_dim=8, lambda_sig=1.0)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()  # encoder outputs 0 for every input; the zeroed GRU keeps the latent at 0
        out = model(**_batch())
    assert float(out["loss_pred"]) == pytest.approx(0.0, abs=1e-12)
    assert float(out["loss_sigreg"]) > 0.1  # ...but SIGReg still objects, loudly
    diag = latent_diagnostics(out["target_latents"])
    assert diag.effective_rank < 1e-6


def test_sigreg_on_modes():
    for mode in ("encoder", "targets", "context", "all"):
        model = FlatJEPA(_config(latent_dim=8, sigreg_on=mode))
        assert float(model(**_batch())["loss_sigreg"].detach()) >= 0.0
    with pytest.raises(ValueError):
        _config(sigreg_on="nonsense")


def test_shape_validation():
    model = FlatJEPA(_config(latent_dim=8))
    batch = _batch()
    with pytest.raises(ValueError):
        model(**{**batch, "state_hist": torch.randn(B, H + 1, DS)})
    with pytest.raises(ValueError):
        model(**{**batch, "action_future": torch.randn(B, T, DA + 1)})


def test_model_scale_is_reported():
    """F5 §5 makes model size a deliberate choice; the count should be inspectable."""
    model = FlatJEPA(_config(latent_dim=24))
    assert 1_000 < model.num_parameters() < 1_000_000


def test_config_accepts_raw_yaml_mapping_for_sigreg():
    cfg = JEPAConfig(latent_dim=8, sigreg={"num_slices": 16, "num_points": 9})
    assert isinstance(cfg.sigreg, SIGRegConfig)
    assert cfg.sigreg.num_slices == 16

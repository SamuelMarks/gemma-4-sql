"""Unit and behavioral tests for PyTorch Direct Preference Optimization (DPO)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState
from tests.conftest import requires_torch

pytestmark = requires_torch


def test_real_pytorch_dpo_loss_logps_and_step() -> None:
    """Verify DPO loss, token log-probability gathering, and optimization step with real PyTorch tensors."""
    import torch

    import gemma_4_sql.backends.pytorch.dpo as dpo_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

    # (b) Verify loss == math.log(2.0) when policy and reference log-probs are equal
    eq_logp = torch.tensor([-1.5, -1.5])
    eq_loss, eq_ch_rew, eq_re_rew = dpo_module.dpo_loss(eq_logp, eq_logp, eq_logp, eq_logp, beta=0.1)
    assert float(eq_loss) == pytest.approx(math.log(2.0), rel=1e-5)
    assert float(eq_ch_rew.mean().item()) == pytest.approx(0.0)
    assert float(eq_re_rew.mean().item()) == pytest.approx(0.0)

    # (a) Verify chosen_rewards > rejected_rewards when policy favors chosen over reference
    p_chosen = torch.tensor([-1.0, -1.2], requires_grad=True)
    p_rejected = torch.tensor([-2.0, -2.5], requires_grad=True)
    r_chosen = torch.tensor([-1.1, -1.3])
    r_rejected = torch.tensor([-1.9, -2.4])

    loss, chosen_rew, rejected_rew = dpo_module.dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, beta=0.1)
    assert float(loss) > 0.0
    assert float(chosen_rew.mean().item()) > float(rejected_rew.mean().item())

    # (c) Verify real parameter updates reduce step loss on a miniature model
    tiny_cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        num_experts=1,
    )
    policy_model = Gemma4ForCausalLM(tiny_cfg)
    ref_model = Gemma4ForCausalLM(tiny_cfg)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-2)
    chosen_inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    chosen_labels = torch.tensor([[2, 3, 0]], dtype=torch.long)
    rejected_inputs = torch.tensor([[1, 4, 5]], dtype=torch.long)
    rejected_labels = torch.tensor([[4, 5, 0]], dtype=torch.long)
    batch: dict[str, dpo_module.TorchDPOBatchValue] = {
        "chosen_inputs": chosen_inputs,
        "chosen_labels": chosen_labels,
        "rejected_inputs": rejected_inputs,
        "rejected_labels": rejected_labels,
    }

    logps = dpo_module._compute_pytorch_logps(policy_model, chosen_inputs, chosen_labels)
    assert logps.shape == (1,)
    assert float(logps.item()) < 0.0

    logps_shifted = dpo_module._compute_pytorch_logps(policy_model, chosen_inputs, None)
    assert logps_shifted.shape == (1,)

    loss_1 = float(dpo_module._run_dpo_step(policy_model, ref_model, optimizer, batch, 0.5))
    loss_2 = float(dpo_module._run_dpo_step(policy_model, ref_model, optimizer, batch, 0.5))
    assert loss_1 > 0.0
    assert loss_2 < loss_1


def test_dpo_loss_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dpo_loss raises DependencyMissingError when PyTorch is unavailable."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    monkeypatch.setattr(dpo_module, "torch", None)
    with pytest.raises(DependencyMissingError, match=r"PyTorch DPO dependencies are missing\."):
        dpo_module.dpo_loss(1.0, 1.0, 1.0, 1.0, 0.1)


def test_run_dpo_step_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_dpo_step handles non-tensor stubs and objects without zero_grad/backward/step."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    mock_policy = MagicMock()
    mock_ref = MagicMock()
    mock_optim = MagicMock()

    mock_policy.return_value.mean.return_value = "mean"
    mock_ref.return_value = "no_mean"

    batch: dict[str, dpo_module.TorchDPOBatchValue] = {"chosen_inputs": "c", "rejected_inputs": "r"}

    with patch("gemma_4_sql.backends.pytorch.dpo.dpo_loss") as mock_loss:
        mock_l = MagicMock()
        mock_loss.return_value = (mock_l, 0, 0)

        res = dpo_module._run_dpo_step(mock_policy, mock_ref, mock_optim, batch, 0.1)
        assert res == mock_l
        mock_optim.zero_grad.assert_called_once()
        mock_l.backward.assert_called_once()
        mock_optim.step.assert_called_once()

        del mock_optim.zero_grad
        del mock_l.backward
        del mock_optim.step

        res2 = dpo_module._run_dpo_step(mock_policy, mock_ref, mock_optim, batch, 0.1)
        assert res2 == mock_l


def test_run_training_epochs() -> None:
    """Verify _run_training_epochs handles both tensor-like and float step outputs."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    state: TrainerState[None, None, None, dict[str, dpo_module.TorchDPOBatchValue]] = TrainerState(
        dataloader=[{"a": 1}, {"a": 2}],
        epochs=2,
    )

    with patch("gemma_4_sql.backends.pytorch.dpo._run_dpo_step") as mock_step:
        mock_l = MagicMock()
        mock_l.item.return_value = 0.5
        mock_step.return_value = mock_l

        res = dpo_module._run_training_epochs(state)
        assert res == 0.5
        assert mock_step.call_count == 4

        mock_step.return_value = 0.25
        assert dpo_module._run_training_epochs(state) == 0.25


def test_run_dpo_and_load_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_dpo and _load_dpo_models support native and HF models and handle errors."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config

    config = DPOConfig(model_name="model", dataset="dataset")
    tiny_cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        num_experts=1,
    )

    # Test pytorch_native model loading
    p_mod, r_mod = dpo_module._load_dpo_models("nonexistent_native", backend_alias="pytorch_native", model_config=tiny_cfg)
    assert p_mod is not None
    assert r_mod is not None

    # Test HF loader fallback
    mock_auto = MagicMock()
    mock_auto.from_pretrained.return_value = MagicMock()
    monkeypatch.setattr(dpo_module, "AutoModelForCausalLM", mock_auto)
    monkeypatch.setattr(dpo_module, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]

    with (
        patch.object(dpo_module.optim, "AdamW", return_value=MagicMock()),
        patch.object(dpo_module, "build_dataloader") as mock_build,
        patch.object(dpo_module, "_run_training_epochs") as mock_run,
    ):
        mock_build.return_value = {"loader": [1, 2]}
        mock_run.return_value = 0.5

        res = dpo_module.run_dpo(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

        mock_build.return_value = {}
        res_bad_dl = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res_bad_dl["status"]

        mock_build.return_value = {"loader": [1]}
        mock_run.side_effect = RuntimeError("epoch error")
        res_err = dpo_module.run_dpo(config)
        assert "failed: epoch error" in res_err["status"]
        with pytest.raises(TrainingError, match="epoch error"):
            dpo_module.run_dpo(config, raise_on_error=True)

    # Test model load failure when both gemma4 and AutoModelForCausalLM are None
    monkeypatch.setattr(dpo_module, "AutoModelForCausalLM", None)
    res_load_fail = dpo_module.run_dpo(config)
    assert "failed: Failed to load model" in res_load_fail["status"]

    monkeypatch.setattr(dpo_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        dpo_module.run_dpo(config)

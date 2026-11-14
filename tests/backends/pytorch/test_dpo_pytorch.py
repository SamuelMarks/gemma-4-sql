"""Unit and behavioral tests for PyTorch Direct Preference Optimization (DPO) without MagicMock."""

from __future__ import annotations

import math
from typing import Self
from unittest.mock import patch

import pytest
import torch

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState
from tests.conftest import requires_torch

pytestmark = requires_torch


class _TrackingOptimizer:
    """Typed PyTorch optimizer double recording `zero_grad` and `step` calls."""

    def __init__(self) -> None:
        """Initialize call counters."""
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        """Increment `zero_grad_calls`."""
        self.zero_grad_calls += 1

    def step(self) -> None:
        """Increment `step_calls`."""
        self.step_calls += 1


class _ScalarItemLoss:
    """Typed loss stub implementing `ScalarItemProtocol`."""

    def __init__(self, val: float) -> None:
        """Store scalar value."""
        self._val = val

    def item(self) -> float:
        """Return stored float value."""
        return self._val


def test_real_pytorch_dpo_loss_logps_and_step() -> None:
    """Verify DPO loss, token log-probability gathering, and optimization step with real PyTorch tensors."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

    eq_logp = torch.tensor([-1.5, -1.5])
    eq_loss, eq_ch_rew, eq_re_rew = dpo_module.dpo_loss(eq_logp, eq_logp, eq_logp, eq_logp, beta=0.1)
    assert float(eq_loss) == pytest.approx(math.log(2.0), rel=1e-5)
    assert float(eq_ch_rew.mean().item()) == pytest.approx(0.0)
    assert float(eq_re_rew.mean().item()) == pytest.approx(0.0)

    p_chosen = torch.tensor([-1.0, -1.2], requires_grad=True)
    p_rejected = torch.tensor([-2.0, -2.5], requires_grad=True)
    r_chosen = torch.tensor([-1.1, -1.3])
    r_rejected = torch.tensor([-1.9, -2.4])

    loss, chosen_rew, rejected_rew = dpo_module.dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, beta=0.1)
    assert float(loss) > 0.0
    assert float(chosen_rew.mean().item()) > float(rejected_rew.mean().item())

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


def test_run_dpo_step_fallback_branches() -> None:
    """Verify _compute_pytorch_logps and _run_dpo_step handle 1D, arithmetic, float, and string outputs."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    def _one_d_policy(_inputs: dpo_module.TorchDPOBatchValue) -> torch.Tensor:
        """Return 1D tensor so `_compute_pytorch_logps` takes the `.mean(dim=-1)` fallback."""
        return torch.tensor([-1.2, -0.8], requires_grad=True)

    class _ZeroDimTensorStub:
        """Tensor stub without `.mean()` that supports `SupportsFloat` and arithmetic with `torch.Tensor`."""

        def __float__(self) -> float:
            """Return `-1.0`."""
            return -1.0

        def __sub__(self, other: torch.Tensor) -> torch.Tensor:
            """Subtract `other` from `-1.0`.

            Returns:
                Difference tensor.
            """
            return torch.tensor(-1.0) - other

        def __rsub__(self, other: torch.Tensor) -> torch.Tensor:
            """Subtract `-1.0` from `other`.

            Returns:
                Reverse difference tensor.
            """
            return other - torch.tensor(-1.0)

    class _ArithmeticStub:
        """Minimal `ArithmeticTensorProtocol` stub testing the arithmetic fallback branch."""

        def __float__(self) -> float:
            """Return `1.0`."""
            return 1.0

        def __sub__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __rsub__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __mul__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __rmul__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __neg__(self) -> Self:
            """Return self."""
            return self

    def _scalar_ref(_inputs: dpo_module.TorchDPOBatchValue) -> _ZeroDimTensorStub:
        """Return `_ZeroDimTensorStub` so `_compute_pytorch_logps` takes the `SupportsFloat` fallback."""
        return _ZeroDimTensorStub()

    arith_val = _ArithmeticStub()
    assert float(arith_val) == 1.0
    assert (arith_val - 1.0) is arith_val
    assert (1.0 - arith_val) is arith_val
    assert (arith_val * 1.0) is arith_val
    assert (1.0 * arith_val) is arith_val
    assert (-arith_val) is arith_val

    def _arith_model(_x: dpo_module.TorchDPOBatchValue) -> _ArithmeticStub:
        """Return `arith_val`."""
        return arith_val

    def _str_model(_x: dpo_module.TorchDPOBatchValue) -> str:
        """Return `'non_numeric_str'`."""
        return "non_numeric_str"

    assert dpo_module._compute_pytorch_logps(_arith_model, None) is arith_val
    assert dpo_module._compute_pytorch_logps(_str_model, None) == 0.0

    tracking_opt = _TrackingOptimizer()
    batch: dict[str, dpo_module.TorchDPOBatchValue] = {
        "chosen_inputs": torch.tensor([[1, 2]]),
        "rejected_inputs": torch.tensor([[1, 3]]),
    }

    assert float(_ZeroDimTensorStub()) == -1.0
    assert float(_ZeroDimTensorStub() - torch.tensor(0.0)) == -1.0
    assert float(torch.tensor(0.0) - _ZeroDimTensorStub()) == 1.0
    neg_loss, _, _ = dpo_module.dpo_loss(-2.0, 0.0, 0.0, 0.0, beta=1.0)
    assert float(neg_loss) > 0.0
    res = dpo_module._run_dpo_step(_one_d_policy, _scalar_ref, tracking_opt, batch, 0.1)
    assert float(res) > 0.0
    assert tracking_opt.zero_grad_calls == 1
    assert tracking_opt.step_calls == 1

    # Also test optimizer=None and non-differentiable float loss branch
    res_no_opt = dpo_module._run_dpo_step(_scalar_ref, _scalar_ref, None, batch, 0.1)
    assert float(res_no_opt) > 0.0


def test_run_training_epochs() -> None:
    """Verify _run_training_epochs handles both ScalarItemProtocol and float step outputs."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    state: DPOLoopState[None, None, dict[str, dpo_module.TorchDPOBatchValue]] = DPOLoopState(
        dataloader=[{"a": 1}, {"a": 2}],
        policy_model=None,
        ref_model=None,
        optimizer=None,
        epochs=2,
    )

    with patch("gemma_4_sql.backends.pytorch.dpo._run_dpo_step", return_value=_ScalarItemLoss(0.5)) as mock_step:
        res = dpo_module._run_training_epochs(state)
        assert res == pytest.approx(0.5)
        assert mock_step.call_count == 4

    with patch("gemma_4_sql.backends.pytorch.dpo._run_dpo_step", return_value=0.25):
        assert dpo_module._run_training_epochs(state) == pytest.approx(0.25)


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

    p_mod, r_mod = dpo_module._load_dpo_models(
        "nonexistent_native", backend_alias="pytorch_native", model_config=tiny_cfg
    )
    assert p_mod is not None
    assert r_mod is not None

    def _none_attr(*_a: str | type, **_k: str) -> None:
        """Return None for import_optional_symbol."""
        return

    class _FakeHFAutoLoader:
        """Typed `AutoModelForCausalLM` double."""

        @staticmethod
        def from_pretrained(_name: str) -> torch.nn.Linear:
            """Return a tiny `torch.nn.Linear` module."""
            return torch.nn.Linear(4, 4)

    monkeypatch.setattr(dpo_module, "AutoModelForCausalLM", _FakeHFAutoLoader)
    monkeypatch.setattr(dpo_module, "import_optional_symbol", _none_attr)

    with (
        patch.object(dpo_module, "build_dataloader", return_value={"loader": [1, 2]}),
        patch.object(dpo_module, "_run_training_epochs", return_value=0.5),
    ):
        res = dpo_module.run_dpo(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

    with patch.object(dpo_module, "build_dataloader", return_value={}):
        res_bad_dl = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res_bad_dl["status"]

    with (
        patch.object(dpo_module, "build_dataloader", return_value={"loader": [1]}),
        patch.object(dpo_module, "_run_training_epochs", side_effect=RuntimeError("epoch error")),
    ):
        res_err = dpo_module.run_dpo(config)
        assert "failed: epoch error" in res_err["status"]
        with pytest.raises(TrainingError, match="epoch error"):
            dpo_module.run_dpo(config, raise_on_error=True)

    monkeypatch.setattr(dpo_module, "AutoModelForCausalLM", None)
    res_load_fail = dpo_module.run_dpo(config)
    assert "failed: Failed to load model" in res_load_fail["status"]

    monkeypatch.setattr(dpo_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        dpo_module.run_dpo(config)

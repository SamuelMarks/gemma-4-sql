"""Unit and behavioral tests for MLX Direct Preference Optimization (DPO)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_real_mlx_dpo_loss_logps_and_step() -> None:
    """Verify MLX dpo_loss, _compute_mlx_logps, and _run_dpo_step execute with real MLX arrays and nn.Module."""
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as optim

    import gemma_4_sql.backends.mlx.dpo as dpo_module

    class TinyMLXPolicy(mlx_nn.Module):
        """Minimal MLX causal language model for DPO testing."""

        def __init__(self) -> None:
            """Initialize TinyMLXPolicy with embedding and linear head."""
            super().__init__()
            self.embed = mlx_nn.Embedding(32, 8)
            self.head = mlx_nn.Linear(8, 32)

        def __call__(self, x: Any) -> Any:
            """Compute vocabulary logits from token IDs."""
            return self.head(self.embed(x))

    policy_model = TinyMLXPolicy()
    ref_model = TinyMLXPolicy()
    mx.eval(policy_model.parameters(), ref_model.parameters())

    optimizer = optim.AdamW(learning_rate=1e-2)
    batch: dict[str, Any] = {
        "chosen_inputs": mx.array([[1, 2, 3]], dtype=mx.int32),
        "chosen_labels": mx.array([[2, 3, 0]], dtype=mx.int32),
        "rejected_inputs": mx.array([[1, 4, 5]], dtype=mx.int32),
        "rejected_labels": mx.array([[4, 5, 0]], dtype=mx.int32),
    }

    logps = dpo_module._compute_mlx_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])  # type: ignore[arg-type]
    mx.eval(logps)
    assert logps.shape == (1,)
    assert float(logps.item()) < 0.0

    # Also test _compute_mlx_logps when labels is a Python list
    logps_list = dpo_module._compute_mlx_logps(policy_model, batch["chosen_inputs"], [[2, 3, 0]])  # type: ignore[arg-type]
    mx.eval(logps_list)
    assert logps_list.shape == (1,)

    step_loss = dpo_module._run_dpo_step(policy_model, ref_model, optimizer, batch, 0.1)  # type: ignore[arg-type]
    assert float(step_loss.item()) > 0.0


def test_dpo_loss_and_fallback_log_sigmoid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dpo_loss handles missing dependencies and fallback log-sigmoid implementations."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    mock_mx = MagicMock()
    mock_nn = MagicMock(spec=[])
    monkeypatch.setattr(dpo_module, "mx", mock_mx)
    monkeypatch.setattr(dpo_module, "mx_nn", mock_nn)

    with patch.object(dpo_module, "generic_dpo_loss") as mock_generic:
        mock_generic.return_value = (1.0, 2.0, 3.0)
        res = dpo_module.dpo_loss(1, 2, 3, 4, 0.1)  # type: ignore[arg-type]
        assert res == (1.0, 2.0, 3.0)

        log_sig_fn = mock_generic.call_args[0][5]
        mock_mx.logaddexp.side_effect = TypeError("no logaddexp on mock")
        mock_mx.negative.return_value = -1.0
        assert log_sig_fn(1.0) == -1.0

        del mock_mx.logaddexp
        del mock_mx.negative

        class NegatableScalar:
            """Scalar stub implementing unary negation."""

            def __neg__(self) -> str:
                """Return negated string marker."""
                return "negated_scalar"

        assert log_sig_fn(NegatableScalar()) == "negated_scalar"

    monkeypatch.setattr(dpo_module, "mx", None)
    assert dpo_module.dpo_loss(1, 2, 3, 4, 0.1) == (0.0, 0.0, 0.0)  # type: ignore[arg-type]


def test_run_dpo_step_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_dpo_step and _compute_mlx_logps fallback branches when value_and_grad or 2D logits are absent."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    mock_policy = MagicMock()
    mock_policy.return_value = MagicMock(ndim=1)
    mock_policy.return_value.mean.return_value = "mean_logps"

    mock_ref = MagicMock()
    mock_ref.return_value = "scalar_logits"

    monkeypatch.setattr(dpo_module, "nn", None)
    batch: dict[str, Any] = {"chosen_inputs": "ci", "rejected_inputs": "ri"}

    with patch.object(dpo_module, "dpo_loss", return_value=(0.75, 0.0, 0.0)):
        res = dpo_module._run_dpo_step(mock_policy, mock_ref, None, batch, 0.1)
        assert res == 0.75

    mock_nn = MagicMock()
    mock_nn.value_and_grad.return_value = lambda *_a: (0.5, "grads")
    monkeypatch.setattr(dpo_module, "nn", mock_nn)
    monkeypatch.setattr(dpo_module, "mx", None)
    res_no_opt = dpo_module._run_dpo_step(MagicMock(spec=[]), mock_ref, MagicMock(spec=[]), batch, 0.1)
    assert res_no_opt == 0.5


def test_run_training_epochs() -> None:
    """Verify _run_training_epochs averages step losses across multiple epochs."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    state = TrainerState(
        dataloader=[{"chosen_inputs": 1}, {"chosen_inputs": 2}],
        epochs=2,
        policy_model="p",
        ref_model="r",
        optimizer="o",
        beta=0.1,
    )

    with patch.object(dpo_module, "_run_dpo_step") as mock_step:
        mock_step.return_value = MagicMock(item=lambda: 0.5)
        loss = dpo_module._run_training_epochs(state)
        assert loss == 0.5
        assert mock_step.call_count == 4


def test_run_dpo_lifecycle_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_dpo handles model loading, dataloader validation, runtime errors, and missing dependencies."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    monkeypatch.setattr(dpo_module, "mlx", MagicMock())
    monkeypatch.setattr(dpo_module, "mx", MagicMock())
    monkeypatch.setattr(dpo_module, "nn", MagicMock())
    monkeypatch.setattr(dpo_module, "optim", MagicMock())
    monkeypatch.setattr(dpo_module, "load", MagicMock(return_value=("model1", "tok1")))

    config = DPOConfig(model_name="model", dataset="dataset")

    with patch.object(dpo_module, "build_dataloader") as mock_build_dataloader:
        mock_build_dataloader.return_value = {"loader": [1, 2]}

        with patch.object(dpo_module, "_run_training_epochs") as mock_run_epochs:
            mock_run_epochs.return_value = 0.5
            res = dpo_module.run_dpo(config)
            assert res["status"] == "completed"
            assert res["final_loss"] == 0.5

            mock_run_epochs.side_effect = RuntimeError("error")
            res2 = dpo_module.run_dpo(config)
            assert "failed: error" in res2["status"]
            with pytest.raises(TrainingError, match="error"):
                dpo_module.run_dpo(config, raise_on_error=True)

        mock_build_dataloader.return_value = {}
        res3 = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res3["status"]

        mock_build_dataloader.return_value = {"loader": MagicMock(spec=[])}
        res4 = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res4["status"]

    monkeypatch.setattr(dpo_module, "mlx", None)
    with pytest.raises(DependencyMissingError):
        dpo_module.run_dpo(config)

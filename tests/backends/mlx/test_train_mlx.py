"""Unit and behavioral tests for the MLX training pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import TrainerState, TrainingConfig
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_real_mlx_training_epochs_and_execute_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_training_epochs and _execute_train run real MLX forward/backward updates on a tiny nn.Module."""
    import mlx.core as mx
    import mlx.nn as mlx_nn

    import gemma_4_sql.backends.mlx.train as train_module

    class TinyMLXModel(mlx_nn.Module):
        """Minimal MLX language model for supervised fine-tuning tests."""

        def __init__(self) -> None:
            """Initialize TinyMLXModel with embedding and output projection."""
            super().__init__()
            self.embed = mlx_nn.Embedding(32, 8)
            self.head = mlx_nn.Linear(8, 32)

        def __call__(self, x: Any) -> Any:
            """Compute logits from input token IDs."""
            return self.head(self.embed(x))

    tiny_model = TinyMLXModel()
    mx.eval(tiny_model.parameters())
    old_weight = mx.array(tiny_model.head.weight)
    mx.eval(old_weight)

    monkeypatch.setattr(train_module, "load", lambda _m: (tiny_model, "tok"))
    batch = {"inputs": [[1, 2, 3]], "targets": [[2, 3, 4]]}
    monkeypatch.setattr(train_module, "build_dataloader", lambda *_a, **_k: {"loader": [batch]})

    status, final_loss = train_module._execute_train("tiny", "ds", epochs=1, learning_rate=1e-2, batch_size=1)
    assert status == "completed"
    assert final_loss > 0.0
    assert not mx.allclose(old_weight, tiny_model.head.weight).item()


def test_run_training_epochs_scalar_loss() -> None:
    """Verify _run_training_epochs handles plain float loss returns without .item()."""
    import gemma_4_sql.backends.mlx.train as train_module

    state: Any = TrainerState(
        dataloader=[{"inputs": [[1, 2]], "targets": [[2, 3]]}],
        epochs=1,
        policy_model=MagicMock(),
        optimizer=MagicMock(),
        train_step=lambda *_a: (0.25, "grads"),
    )
    loss = train_module._run_training_epochs(state)
    assert loss == 0.25


def test_execute_train_invalid_dataloader_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train validates the dataloader and raises DependencyMissingError when MLX is missing."""
    import gemma_4_sql.backends.mlx.train as train_module

    monkeypatch.setattr(train_module, "load", lambda _m: MagicMock())
    monkeypatch.setattr(train_module, "build_dataloader", lambda *_a, **_k: {})
    with pytest.raises(ValueError, match="Invalid dataloader"):
        train_module._execute_train("m", "d", 1, 0.1)

    monkeypatch.setattr(train_module, "mx", None)
    with pytest.raises(DependencyMissingError):
        train_module._execute_train("m", "d", 1, 0.1)


def test_train_model_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model returns structured results and captures runtime errors."""
    import gemma_4_sql.backends.mlx.train as train_module

    config = TrainingConfig(model_name="model", dataset="dataset")

    with patch.object(train_module, "_execute_train") as mock_exec:
        mock_exec.return_value = ("completed", 0.5)
        res = train_module.train_model(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

        mock_exec.side_effect = TypeError("err")
        res2 = train_module.train_model(config)
        assert "failed: err" in res2["status"]

        mock_exec.side_effect = RuntimeError("err")
        res4 = train_module.train_model(config)
        assert "failed: err" in res4["status"]
        with pytest.raises(TrainingError, match="err"):
            train_module.train_model(config, raise_on_error=True)

    monkeypatch.setattr(train_module, "mx", None)
    with pytest.raises(DependencyMissingError):
        train_module.train_model(config)

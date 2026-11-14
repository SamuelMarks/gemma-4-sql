"""Unit and behavioral tests for the MLX training pipeline without MagicMock."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, ETLError, TrainingError
from gemma_4_sql.type_hints import MlxArrayProtocol, MlxTrainerState, TrainingConfig
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_real_mlx_training_epochs_and_execute_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_training_epochs and _execute_train run real MLX forward/backward updates on a tiny nn.Module."""
    import mlx.core as mx
    import mlx.nn as mlx_nn

    import gemma_4_sql.backends.mlx.train as train_module
    from gemma_4_sql.type_hints import ETLConfig

    class TinyMLXModel(mlx_nn.Module):
        """Minimal MLX language model for supervised fine-tuning tests."""

        def __init__(self) -> None:
            """Initialize TinyMLXModel with embedding and output projection."""
            super().__init__()
            self.embed = mlx_nn.Embedding(32, 8)
            self.head = mlx_nn.Linear(8, 32)

        def __call__(self, x: mx.array) -> mx.array:
            """Compute logits from input token IDs.

            Returns:
                Vocabulary logits array.
            """
            return self.head(self.embed(x))

    from tests.utils.tensor_helpers import mx_eval, mx_eval_model

    tiny_model = TinyMLXModel()
    mx_eval_model(tiny_model)
    old_weight = mx.array(tiny_model.head.weight)
    mx_eval(old_weight)

    def _load_tiny(_m: str) -> tuple[TinyMLXModel, str]:
        """Return `(tiny_model, 'tok')`."""
        return (tiny_model, "tok")

    batch = {"inputs": [[1, 2, 3]], "targets": [[2, 3, 4]]}
    batch_mismatched = {"inputs": [[1, 2, 3, 4]], "targets": [[2, 3]]}

    def _mlx_loader(*_a: ETLConfig) -> dict[str, list[dict[str, list[list[int]]]]]:
        """Return 2-batch MLX loader."""
        return {"loader": [batch, batch_mismatched]}

    monkeypatch.setattr(train_module, "load", _load_tiny)
    monkeypatch.setattr(train_module, "build_dataloader", _mlx_loader)

    status, final_loss = train_module._execute_train("tiny", "ds", epochs=1, learning_rate=1e-2, batch_size=1)
    assert status == "completed"
    assert final_loss > 0.0
    assert not mx.allclose(old_weight, tiny_model.head.weight).item()

    # Also exercise offline fallback dataloader when build_dataloader raises ETLError on synthetic dataset
    with patch.object(train_module, "build_dataloader", side_effect=ETLError("offline")):
        status_fb, loss_fb = train_module._execute_train(
            "tiny", "synthetic", epochs=1, learning_rate=1e-2, batch_size=1
        )
        assert status_fb == "completed"
        assert loss_fb > 0.0


def test_run_training_epochs_scalar_loss() -> None:
    """Verify _run_training_epochs handles plain float and non-numeric loss returns without .item()."""
    import gemma_4_sql.backends.mlx.train as train_module

    class _StubMlxTrainModel:
        """Minimal `_MlxTrainModelProtocol` double for scalar loss testing."""

        def parameters(self) -> dict[str, MlxArrayProtocol]:
            """Return empty parameters dict."""
            return {}

    lin_model = _StubMlxTrainModel()
    assert lin_model.parameters() == {}

    class _NoopMlxOptimizer:
        """Typed `_MlxOptimizerProtocol` double."""

        state: dict[str, MlxArrayProtocol] = {}

        def update(self, _model: _StubMlxTrainModel, _grads: train_module.MlxGradTree) -> None:
            """No-op update."""

    def _float_lg(*_a: _StubMlxTrainModel | MlxArrayProtocol) -> tuple[float, dict[str, MlxArrayProtocol]]:
        """Return `(0.25, {})`."""
        return (0.25, {})

    def _str_lg(*_a: _StubMlxTrainModel | MlxArrayProtocol) -> tuple[str, dict[str, str]]:
        """Return `('non_num', {'k': 'v'})`."""
        return ("non_num", {"k": "v"})

    state: MlxTrainerState[_StubMlxTrainModel, _NoopMlxOptimizer, dict[str, list[list[int]]]] = MlxTrainerState(
        dataloader=[{"inputs": [[1, 2]], "targets": [[2, 3]]}],
        epochs=1,
        model=lin_model,
        optimizer=_NoopMlxOptimizer(),
        loss_and_grad_fn=_float_lg,
    )
    assert train_module._run_training_epochs(state) == 0.25

    state_str: MlxTrainerState[_StubMlxTrainModel, _NoopMlxOptimizer, dict[str, list[list[int]]]] = MlxTrainerState(
        dataloader=[{"inputs": [[1, 2]], "targets": [[2, 3]]}],
        epochs=1,
        model=lin_model,
        optimizer=_NoopMlxOptimizer(),
        loss_and_grad_fn=_str_lg,
    )
    assert train_module._run_training_epochs(state_str) == 0.0


def test_execute_train_invalid_dataloader_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train validates the dataloader and raises DependencyMissingError when MLX is missing."""
    import mlx.nn as mlx_nn

    import gemma_4_sql.backends.mlx.train as train_module
    from gemma_4_sql.type_hints import ETLConfig

    lin = mlx_nn.Linear(4, 4)

    def _load_lin(_m: str) -> mlx_nn.Linear:
        """Return `lin`."""
        return lin

    def _empty_loader(*_a: ETLConfig) -> dict[str, None]:
        """Return empty loader dict."""
        return {}

    monkeypatch.setattr(train_module, "load", _load_lin)
    monkeypatch.setattr(train_module, "build_dataloader", _empty_loader)
    with pytest.raises(ValueError, match="Invalid dataloader"):
        train_module._execute_train("m", "d", 1, 0.1)

    monkeypatch.setattr(train_module, "mx", None)
    with pytest.raises(DependencyMissingError):
        train_module._execute_train("m", "d", 1, 0.1)


def test_train_model_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model returns structured results and captures runtime errors."""
    import gemma_4_sql.backends.mlx.train as train_module

    config = TrainingConfig(model_name="model", dataset="dataset")

    with patch.object(train_module, "_execute_train", return_value=("completed", 0.5)):
        res = train_module.train_model(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

    with (
        patch.object(train_module, "_execute_train", side_effect=TypeError("err")),
        pytest.raises(TypeError, match="err"),
    ):
        train_module.train_model(config)

    with patch.object(train_module, "_execute_train", side_effect=RuntimeError("err")):
        res4 = train_module.train_model(config)
        assert "failed: err" in res4["status"]
        with pytest.raises(TrainingError, match="err"):
            train_module.train_model(config, raise_on_error=True)

    monkeypatch.setattr(train_module, "mx", None)
    with pytest.raises(DependencyMissingError):
        train_module.train_model(config)

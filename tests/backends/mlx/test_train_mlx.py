"""Tests for mlx train."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainingConfig


def test_mlx_train_imports():
    """Test mlx train imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx.nn": None, "mlx.optimizers": None, "mlx_lm": None}):
        import gemma_4_sql.backends.mlx.train as train_module

        importlib.reload(train_module)
        assert train_module.mx is None
        assert train_module.nn is None
        assert train_module.optim is None
        assert train_module.load is None

    with patch.dict(sys.modules, {"mlx": MagicMock(), "mlx.core": MagicMock(), "mlx.nn": MagicMock(), "mlx.optimizers": MagicMock(), "mlx_lm": MagicMock(load="load")}):
        importlib.reload(train_module)
        assert train_module.load == "load"
    importlib.reload(train_module)


def test_run_training_epochs():
    """Test _run_training_epochs."""
    import gemma_4_sql.backends.mlx.train as train_module

    state = MagicMock()
    state.dataloader = [{"inputs": [1, 2], "targets": [3, 4]}]
    state.epochs = 2
    state.policy_model = MagicMock()
    state.optimizer = MagicMock()

    loss_mock = MagicMock()
    loss_mock.item.return_value = 0.5
    state.train_step.return_value = (loss_mock, "grads")

    train_module.mx = MagicMock()

    loss = train_module._run_training_epochs(state)
    assert loss == 0.5


def test_execute_train():
    """Test _execute_train."""
    import gemma_4_sql.backends.mlx.train as train_module

    train_module.mx = MagicMock()
    train_module.nn = MagicMock()
    train_module.optim = MagicMock()
    train_module.load = MagicMock(return_value=MagicMock())

    with patch("gemma_4_sql.backends.mlx.train.build_dataloader") as mock_build:
        mock_build.return_value = {"loader": [1]}
        with patch("gemma_4_sql.backends.mlx.train._run_training_epochs") as mock_run:
            mock_run.return_value = 0.5

            stat, loss = train_module._execute_train("m", "d", 1, 0.1)
            assert stat == "completed"
            assert loss == 0.5

            # test value_and_grad callable
            loss_fn = train_module.nn.value_and_grad.call_args[0][1]
            mock_model = MagicMock()
            mock_model.return_value = "logits"
            train_module.nn.losses.cross_entropy.return_value = "loss"
            assert loss_fn(mock_model, "in", "tar") == "loss"

        mock_build.return_value = {}
        with pytest.raises(ValueError):
            train_module._execute_train("m", "d", 1, 0.1)

    train_module.mx = None
    with pytest.raises(DependencyMissingError):
        train_module._execute_train("m", "d", 1, 0.1)


def test_train_model():
    """Test train_model."""
    import gemma_4_sql.backends.mlx.train as train_module

    train_module.mx = MagicMock()
    train_module.nn = MagicMock()
    train_module.optim = MagicMock()
    train_module.load = MagicMock()

    config = TrainingConfig(model_name="model")

    with patch("gemma_4_sql.backends.mlx.train._execute_train") as mock_exec:
        mock_exec.return_value = ("completed", 0.5)

        res = train_module.train_model(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

        mock_exec.side_effect = TypeError("err")
        res2 = train_module.train_model(config)
        assert "failed: err" in res2["status"]  # Wait! TypeError inside the try falls back to _execute_train without batch_size!

        # We need to make the fallback return something
        mock_exec.side_effect = [TypeError("err"), ("completed", 0.5)]
        res3 = train_module.train_model(config)
        assert res3["status"] == "completed"

        mock_exec.side_effect = RuntimeError("err")
        res4 = train_module.train_model(config)
        assert "failed: err" in res4["status"]

    train_module.mx = None
    with pytest.raises(DependencyMissingError):
        train_module.train_model(config)

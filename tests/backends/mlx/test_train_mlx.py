"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.mlx.train import (
    _execute_train,
    _run_training_epochs,
    train_model,
)
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainerState


def test_run_training_epochs():
    """Docstring for test_run_training_epochs."""
    mock_dataloader = [{"inputs": [1, 2], "targets": [3, 4]}]
    mock_model = MagicMock()
    mock_model.parameters.return_value = "params"
    mock_optimizer = MagicMock()
    mock_optimizer.state = "state"

    mock_loss = MagicMock()
    mock_loss.item.return_value = 0.5
    mock_train_step = MagicMock(return_value=(mock_loss, "grads"))

    state = TrainerState(
        dataloader=mock_dataloader,
        epochs=1,
        policy_model=mock_model,
        optimizer=mock_optimizer,
        train_step=mock_train_step,
    )

    with patch("gemma_4_sql.backends.mlx.train.mx") as mock_mx:
        mock_mx.array.side_effect = lambda x: x
        final_loss = _run_training_epochs(state)
        assert final_loss == 0.5
        mock_train_step.assert_called_once_with(mock_model, [1, 2], [3, 4])
        mock_optimizer.update.assert_called_once_with(mock_model, "grads")
        mock_mx.eval.assert_called_once_with("params", "state")


def test_run_training_epochs_loss_no_item():
    """Docstring for test_run_training_epochs_loss_no_item."""
    mock_dataloader = [{"inputs": [1, 2], "targets": [3, 4]}]
    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    mock_train_step = MagicMock(return_value=(0.5, "grads"))

    state = TrainerState(
        dataloader=mock_dataloader,
        epochs=1,
        policy_model=mock_model,
        optimizer=mock_optimizer,
        train_step=mock_train_step,
    )
    with patch("gemma_4_sql.backends.mlx.train.mx") as mock_mx:
        mock_mx.array.side_effect = lambda x: x
        final_loss = _run_training_epochs(state)
        assert final_loss == 0.5


def test_execute_train_missing_deps():
    """Docstring for test_execute_train_missing_deps."""
    with patch("gemma_4_sql.backends.mlx.train.mx", None), pytest.raises(DependencyMissingError):
        _execute_train("model", "dataset", 1, 0.01)


def test_execute_train_success():
    """Docstring for test_execute_train_success."""
    mock_load = MagicMock(return_value="model")
    mock_nn = MagicMock()
    mock_optim = MagicMock()
    mock_optim.AdamW.return_value = "optimizer"

    mock_loader = MagicMock()
    # To bypass __iter__ check, mock_loader must have __iter__
    mock_loader.__iter__.return_value = []

    with (
        patch("gemma_4_sql.backends.mlx.train.load", mock_load),
        patch("gemma_4_sql.backends.mlx.train.nn", mock_nn),
        patch("gemma_4_sql.backends.mlx.train.optim", mock_optim),
        patch("gemma_4_sql.backends.mlx.train.mx", MagicMock()),
        patch("gemma_4_sql.backends.mlx.train.build_dataloader", return_value={"loader": mock_loader}),
        patch("gemma_4_sql.backends.mlx.train._run_training_epochs", return_value=0.5),
    ):
        status, loss = _execute_train("model", "dataset", 1, 0.01)
        assert status == "completed"
        assert loss == 0.5

        # also test the loss_fn inside
        loss_fn = mock_nn.value_and_grad.call_args.args[1]
        mock_model_t = MagicMock()
        mock_model_t.return_value = "logits"
        mock_nn.losses.cross_entropy.return_value = 0.5
        res = loss_fn(mock_model_t, "inputs", "targets")
        assert res == 0.5
        mock_model_t.assert_called_once_with("inputs")
        mock_nn.losses.cross_entropy.assert_called_once_with("logits", "targets", reduction="mean")


def test_execute_train_tuple_load():
    """Docstring for test_execute_train_tuple_load."""
    mock_load = MagicMock(return_value=("model", "tok"))
    mock_nn = MagicMock()
    mock_optim = MagicMock()
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = []

    with (
        patch("gemma_4_sql.backends.mlx.train.load", mock_load),
        patch("gemma_4_sql.backends.mlx.train.nn", mock_nn),
        patch("gemma_4_sql.backends.mlx.train.optim", mock_optim),
        patch("gemma_4_sql.backends.mlx.train.mx", MagicMock()),
        patch("gemma_4_sql.backends.mlx.train.build_dataloader", return_value={"loader": mock_loader}),
        patch("gemma_4_sql.backends.mlx.train._run_training_epochs", return_value=0.5),
    ):
        status, _loss = _execute_train("model", "dataset", 1, 0.01)
        assert status == "completed"


def test_execute_train_invalid_loader():
    """Docstring for test_execute_train_invalid_loader."""
    mock_load = MagicMock(return_value="model")
    mock_nn = MagicMock()
    mock_optim = MagicMock()
    with (
        patch("gemma_4_sql.backends.mlx.train.load", mock_load),
        patch("gemma_4_sql.backends.mlx.train.nn", mock_nn),
        patch("gemma_4_sql.backends.mlx.train.optim", mock_optim),
        patch("gemma_4_sql.backends.mlx.train.mx", MagicMock()),
        patch("gemma_4_sql.backends.mlx.train.build_dataloader", return_value={"loader": None}),
        pytest.raises(ValueError, match="Invalid dataloader"),
    ):
        _execute_train("model", "dataset", 1, 0.01)


class DummyConfig:
    """Docstring for DummyConfig."""


def test_train_model_missing_deps():
    """Docstring for test_train_model_missing_deps."""
    config = DummyConfig()
    with patch("gemma_4_sql.backends.mlx.train.mx", None), pytest.raises(DependencyMissingError):
        train_model(config)


def test_train_model_success():
    """Docstring for test_train_model_success."""
    config = DummyConfig()
    with (
        patch("gemma_4_sql.backends.mlx.train.mx", MagicMock()),
        patch("gemma_4_sql.backends.mlx.train.nn", MagicMock()),
        patch("gemma_4_sql.backends.mlx.train.optim", MagicMock()),
        patch("gemma_4_sql.backends.mlx.train.load", MagicMock()),
        patch("gemma_4_sql.backends.mlx.train._execute_train", return_value=("completed", 0.5)),
    ):
        res = train_model(config, distributed_strategy="fsdp")
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5
        assert res["distributed_strategy"] == "fsdp"


def test_train_model_typeerror_fallback():
    """Docstring for test_train_model_typeerror_fallback."""
    config = DummyConfig()
    config.batch_size = 4

    mock_execute = MagicMock()
    mock_execute.side_effect = [TypeError("wrong args"), ("completed", 0.5)]

    with patch("gemma_4_sql.backends.mlx.train.mx", MagicMock()), patch("gemma_4_sql.backends.mlx.train.nn", MagicMock()), patch("gemma_4_sql.backends.mlx.train.optim", MagicMock()), patch("gemma_4_sql.backends.mlx.train.load", MagicMock()), patch("gemma_4_sql.backends.mlx.train._execute_train", mock_execute):
        res = train_model(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5
        mock_execute.assert_called_with("gemma-4", "dummy", 1, 1e-05)


def test_module_import_success():
    """Docstring for test_module_import_success."""
    import importlib

    mock_mlx = MagicMock()
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.load = "mocked_load"

    with patch.dict("sys.modules", {"mlx": mock_mlx, "mlx.core": mock_mlx, "mlx.nn": mock_mlx, "mlx.optimizers": mock_mlx, "mlx_lm": mock_mlx_lm}):
        import gemma_4_sql.backends.mlx.train

        importlib.reload(gemma_4_sql.backends.mlx.train)
        assert gemma_4_sql.backends.mlx.train.load == "mocked_load"

    # reload without them to restore state
    with patch.dict("sys.modules", {"mlx": None, "mlx.core": None, "mlx.nn": None, "mlx.optimizers": None, "mlx_lm": None}):
        importlib.reload(gemma_4_sql.backends.mlx.train)

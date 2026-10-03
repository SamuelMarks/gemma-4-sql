"""Tests for Keras DPO."""

import builtins
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras.dpo import (
    _compute_logps,
    _execute_dpo,
    _get_train_step_fn,
    _run_training_epochs,
    dpo_loss,
    run_dpo,
)
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState

original_import = builtins.__import__


def _mock_import(name, *args, **kwargs):
    if name == "keras_nlp.models":
        raise ImportError
    return original_import(name, *args, **kwargs)


def test_dpo_loss_missing_tf():
    with patch("gemma_4_sql.backends.keras.dpo.tf", None):
        assert dpo_loss(None, None, None, None) == (0.0, 0.0, 0.0)


def test_dpo_loss_with_tf():
    mock_tf = MagicMock()
    mock_tf.math.log_sigmoid = MagicMock()
    with patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf), patch("gemma_4_sql.backends.keras.dpo.generic_dpo_loss", return_value=(1, 2, 3)) as mock_generic:
        res = dpo_loss(1, 2, 3, 4, 0.2)
        assert res == (1, 2, 3)
        mock_generic.assert_called_once_with(1, 2, 3, 4, 0.2, mock_tf.math.log_sigmoid)


def test_compute_logps_missing_tf():
    with patch("gemma_4_sql.backends.keras.dpo.tf", None):
        assert _compute_logps(None, None, None) == 0.0


def test_compute_logps_with_tf():
    mock_tf = MagicMock()
    mock_tf.nn.log_softmax.return_value = MagicMock()
    mock_tf.expand_dims.return_value = MagicMock()
    mock_tf.gather.return_value = MagicMock()
    mock_tf.squeeze.return_value = MagicMock()
    mock_tf.cast.return_value = MagicMock()
    mock_tf.reduce_sum.return_value = MagicMock()

    mock_model = MagicMock()
    mock_model.return_value = MagicMock()

    with patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf):
        res = _compute_logps(mock_model, MagicMock(), MagicMock())
        assert res == mock_tf.reduce_sum.return_value


def test_get_train_step_fn_missing_tf():
    with patch("gemma_4_sql.backends.keras.dpo.tf", None):
        fn = _get_train_step_fn(None, None, None, 0.1)
        assert fn(None) == 0.0


def test_get_train_step_fn_with_tf():
    mock_tf = MagicMock()

    class DummyTape:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def gradient(self, *args):
            return [1, 2]

    mock_tf.GradientTape = DummyTape
    mock_tf.function = lambda x: x

    mock_policy = MagicMock()
    mock_policy.trainable_variables = [1, 2]
    mock_ref = MagicMock()
    mock_opt = MagicMock()

    with patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf), patch("gemma_4_sql.backends.keras.dpo._compute_logps", return_value=1), patch("gemma_4_sql.backends.keras.dpo.dpo_loss", return_value=(10.0, 0, 0)):
        fn = _get_train_step_fn(mock_policy, mock_ref, mock_opt, 0.1)

        batch = {"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}
        loss = fn(batch)
        assert loss == 10.0
        mock_opt.apply_gradients.assert_called_once()


def test_run_training_epochs():
    mock_loader = [[1, 2], [3, 4]]

    class DummyLoss:
        def numpy(self):
            return 1.5

    def mock_train_step(batch):
        return DummyLoss()

    state = TrainerState(dataloader=mock_loader, epochs=2, train_step=mock_train_step)
    loss = _run_training_epochs(state)
    assert loss == 1.5


def test_run_training_epochs_no_numpy():
    mock_loader = [[1]]

    def mock_train_step(batch):
        return 2.5

    state = TrainerState(dataloader=mock_loader, epochs=1, train_step=mock_train_step)
    loss = _run_training_epochs(state)
    assert loss == 2.5

    def mock_train_step_err(batch):
        class ErrLoss:
            def __float__(self):
                raise ValueError

        return ErrLoss()

    state = TrainerState(dataloader=mock_loader, epochs=1, train_step=mock_train_step_err)
    loss = _run_training_epochs(state)
    assert loss == 0.0

    def mock_train_step_err_type(batch):
        class ErrLoss:
            def __float__(self):
                raise TypeError

        return ErrLoss()

    state = TrainerState(dataloader=mock_loader, epochs=1, train_step=mock_train_step_err_type)
    loss = _run_training_epochs(state)
    assert loss == 0.0


def test_execute_dpo_missing_keras():
    with patch("gemma_4_sql.backends.keras.dpo.keras", None), pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        _execute_dpo("model", "dataset", 0.1, 1, 0.01)


def test_execute_dpo_import_error():
    mock_keras = MagicMock()
    mock_tf = MagicMock()
    with patch("gemma_4_sql.backends.keras.dpo.keras", mock_keras), patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf), patch("builtins.__import__", side_effect=_mock_import), pytest.raises(ValueError, match="Failed to load Keras model"):
        _execute_dpo("model", "dataset", 0.1, 1, 0.01)


def test_execute_dpo_invalid_dataloader():
    mock_keras = MagicMock()
    mock_tf = MagicMock()
    mock_cls = MagicMock()

    def _mock_import_success(name, *args, **kwargs):
        if name == "keras_nlp.models":
            return mock_cls
        return original_import(name, *args, **kwargs)

    with patch("gemma_4_sql.backends.keras.dpo.keras", mock_keras), patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf), patch("builtins.__import__", side_effect=_mock_import_success), patch("gemma_4_sql.backends.keras.dpo.build_dataloader", return_value={"loader": None}):
        with pytest.raises(ValueError, match="Invalid dataloader"):
            _execute_dpo("model", "dataset", 0.1, 1, 0.01)


def test_execute_dpo_success():
    mock_keras = MagicMock()
    mock_tf = MagicMock()
    mock_cls = MagicMock()

    def _mock_import_success(name, *args, **kwargs):
        if name == "keras_nlp.models":
            return mock_cls
        return original_import(name, *args, **kwargs)

    with patch("gemma_4_sql.backends.keras.dpo.keras", mock_keras), patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf), patch("builtins.__import__", side_effect=_mock_import_success), patch("gemma_4_sql.backends.keras.dpo.build_dataloader", return_value={"loader": [1, 2]}):
        with patch("gemma_4_sql.backends.keras.dpo._get_train_step_fn"):
            with patch("gemma_4_sql.backends.keras.dpo._run_training_epochs", return_value=5.0):
                status, loss = _execute_dpo("model", "dataset", 0.1, 1, 0.01)
                assert status == "completed"
                assert loss == 5.0


def test_run_dpo_missing_keras():
    with patch("gemma_4_sql.backends.keras.dpo.keras", None), pytest.raises(DependencyMissingError, match="Keras DPO dependencies are missing"):
        run_dpo(DPOConfig(model_name="a", dataset="b", beta=0.1, epochs=1, learning_rate=0.01, batch_size=2))


def test_run_dpo_success():
    mock_keras = MagicMock()
    mock_tf = MagicMock()
    with patch("gemma_4_sql.backends.keras.dpo.keras", mock_keras), patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf), patch("gemma_4_sql.backends.keras.dpo._execute_dpo", return_value=("completed", 1.0)):
        res = run_dpo(DPOConfig(model_name="a", dataset="b", beta=0.1, epochs=1, learning_rate=0.01, batch_size=2))
        assert res["status"] == "completed"
        assert res["final_loss"] == 1.0


def test_run_dpo_failure():
    mock_keras = MagicMock()
    mock_tf = MagicMock()
    with patch("gemma_4_sql.backends.keras.dpo.keras", mock_keras), patch("gemma_4_sql.backends.keras.dpo.tf", mock_tf), patch("gemma_4_sql.backends.keras.dpo._execute_dpo", side_effect=ValueError("Test Error")):
        res = run_dpo(DPOConfig(model_name="a", dataset="b", beta=0.1, epochs=1, learning_rate=0.01, batch_size=2))
        assert res["status"] == "failed: Test Error"

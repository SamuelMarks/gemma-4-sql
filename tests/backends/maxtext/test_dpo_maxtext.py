"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext.dpo import _compute_logps, _dpo_step_loss, _execute_dpo, _get_train_step_fn, _run_training_epochs, dpo_loss, run_dpo
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState


@pytest.fixture
def mock_dpo_deps():
    """Docstring for mock_dpo_deps."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_optax = MagicMock()
    mock_Gemma4Model = MagicMock()

    with patch("gemma_4_sql.backends.maxtext.dpo.jax", mock_jax), patch("gemma_4_sql.backends.maxtext.dpo.jnp", mock_jnp), patch("gemma_4_sql.backends.maxtext.dpo.optax", mock_optax), patch("gemma_4_sql.backends.maxtext.dpo.Gemma4Model", mock_Gemma4Model):
        yield mock_jax, mock_jnp, mock_optax, mock_Gemma4Model


def test_dpo_loss():
    """Docstring for test_dpo_loss."""
    with patch("gemma_4_sql.backends.maxtext.dpo.jax_dpo_loss", return_value=(1, 2, 3)) as mock_jax_dpo:
        res = dpo_loss(1, 2, 3, 4, 0.5)
        assert res == (1, 2, 3)
        mock_jax_dpo.assert_called_once_with(1, 2, 3, 4, 0.5)


def test_compute_logps(mock_dpo_deps):
    """Docstring for test_compute_logps."""
    _, mock_jnp, _, _ = mock_dpo_deps
    model = MagicMock()
    model.apply.return_value = 5
    mock_jnp.sum.return_value = 10

    res = _compute_logps(model, {"p": 1}, 2, 3)
    assert res == 10
    model.apply.assert_called_once_with({"p": 1}, 2)
    # mock_jnp.sum was called with 5 * 3 = 15. The exact call is complex to assert due to magicmock mult, just ensure it was called.
    mock_jnp.sum.assert_called_once()


def test_dpo_step_loss():
    """Docstring for test_dpo_step_loss."""
    with patch("gemma_4_sql.backends.maxtext.dpo._compute_logps", side_effect=[1, 2, 3, 4]) as mock_comp, patch("gemma_4_sql.backends.maxtext.dpo.dpo_loss", return_value=(10, 0, 0)) as mock_loss:
        batch = {"chosen_inputs": "ci", "chosen_labels": "cl", "rejected_inputs": "ri", "rejected_labels": "rl"}
        res = _dpo_step_loss("pm", "pp", "rm", "rp", batch, 0.1)
        assert res == 10
        mock_comp.assert_any_call("pm", "pp", "ci", "cl")
        mock_comp.assert_any_call("pm", "pp", "ri", "rl")
        mock_comp.assert_any_call("rm", "rp", "ci", "cl")
        mock_comp.assert_any_call("rm", "rp", "ri", "rl")
        mock_loss.assert_called_once_with(1, 2, 3, 4, 0.1)


def test_get_train_step_fn(mock_dpo_deps):
    """Docstring for test_get_train_step_fn."""
    mock_jax, _, mock_optax, _ = mock_dpo_deps

    mock_jax.value_and_grad.return_value = lambda *args: (0.5, "grads")
    mock_jax.jit = lambda x: x

    optimizer = MagicMock()
    optimizer.update.return_value = ("updates", "new_opt_state")
    mock_optax.apply_updates.return_value = "new_policy_params"

    train_step = _get_train_step_fn("pm", "rm", optimizer, 0.1)

    with patch("gemma_4_sql.backends.maxtext.dpo._dpo_step_loss", return_value=0.5):
        policy_params, opt_state, loss = train_step("pp", "rp", "os", "batch")

    assert policy_params == "new_policy_params"
    assert opt_state == "new_opt_state"
    assert loss == 0.5
    optimizer.update.assert_called_once_with("grads", "os", "pp")


def test_get_train_step_fn_no_jit():
    """Docstring for test_get_train_step_fn_no_jit."""
    mock_jax = MagicMock()
    mock_jax.value_and_grad.return_value = lambda *args: (0.5, "grads")
    del mock_jax.jit
    optimizer = MagicMock()
    optimizer.update.return_value = ("updates", "new_opt_state")

    with patch("gemma_4_sql.backends.maxtext.dpo.jax", mock_jax), patch("gemma_4_sql.backends.maxtext.dpo.optax") as mock_optax:
        mock_optax.apply_updates.return_value = "new_policy_params"
        train_step = _get_train_step_fn("pm", "rm", optimizer, 0.1)

        with patch("gemma_4_sql.backends.maxtext.dpo._dpo_step_loss", return_value=0.5):
            _policy_params, _opt_state, loss = train_step("pp", "rp", "os", "batch")
        assert loss == 0.5


def test_run_training_epochs():
    """Docstring for test_run_training_epochs."""

    class MockLoss:
        """Docstring for MockLoss."""

        def item(self):
            """Docstring for item."""
            return 0.5

    def mock_train_step(pp, rp, os, b):
        """Docstring for mock_train_step."""
        return pp, os, MockLoss()

    state = TrainerState(dataloader=[1, 2], epochs=1, train_step=mock_train_step, policy_params="pp", ref_params="rp", opt_state="os")

    with patch("gemma_4_sql.backends.maxtext.dpo.generic_run_training_epochs", return_value=0.5) as mock_run:
        pp, os, loss = _run_training_epochs(state)
        assert pp == "pp"
        assert os == "os"
        assert loss == 0.5

        # Test the callback
        cb = mock_run.call_args[0][2]
        cb_loss = cb({"b": 1})
        assert cb_loss == 0.5


def test_run_training_epochs_no_item():
    """Docstring for test_run_training_epochs_no_item."""

    def mock_train_step(pp, rp, os, b):
        """Docstring for mock_train_step."""
        return pp, os, 0.5

    state = TrainerState(dataloader=[1, 2], epochs=1, train_step=mock_train_step, policy_params="pp", ref_params="rp", opt_state="os")

    with patch("gemma_4_sql.backends.maxtext.dpo.generic_run_training_epochs", return_value=0.5) as mock_run:
        _run_training_epochs(state)
        cb = mock_run.call_args[0][2]
        cb_loss = cb({"b": 1})
        assert cb_loss == 0.5


def test_execute_dpo(mock_dpo_deps):
    """Docstring for test_execute_dpo."""
    mock_jax, _, _, mock_Gemma4Model = mock_dpo_deps

    mock_jax.distributed.initialize.return_value = None

    mock_pm = MagicMock()
    mock_rm = MagicMock()
    mock_Gemma4Model.side_effect = [mock_pm, mock_rm]

    with patch("gemma_4_sql.backends.maxtext.dpo.build_dataloader", return_value={"loader": [1, 2]}), patch("gemma_4_sql.backends.maxtext.dpo._get_train_step_fn"), patch("gemma_4_sql.backends.maxtext.dpo._run_training_epochs", return_value=("pp", "os", 0.5)):
        status, loss = _execute_dpo("model", "dataset", 0.1, 1, 1e-5, 2)
        assert status == "completed"
        assert loss == 0.5


def test_execute_dpo_init_error(mock_dpo_deps):
    """Docstring for test_execute_dpo_init_error."""
    mock_jax, _, _, mock_Gemma4Model = mock_dpo_deps
    mock_jax.distributed.initialize.side_effect = RuntimeError("init fail")

    mock_pm = MagicMock()
    mock_rm = MagicMock()
    mock_Gemma4Model.side_effect = [mock_pm, mock_rm]

    with patch("gemma_4_sql.backends.maxtext.dpo.build_dataloader", return_value={"loader": [1, 2]}), patch("gemma_4_sql.backends.maxtext.dpo._get_train_step_fn"), patch("gemma_4_sql.backends.maxtext.dpo._run_training_epochs", return_value=("pp", "os", 0.5)):
        status, _loss = _execute_dpo("model", "dataset", 0.1, 1, 1e-5, 2)
        assert status == "completed"


def test_execute_dpo_invalid_loader(mock_dpo_deps):
    """Docstring for test_execute_dpo_invalid_loader."""
    _mock_jax, _, _, mock_Gemma4Model = mock_dpo_deps
    mock_Gemma4Model.side_effect = [MagicMock(), MagicMock()]

    with patch("gemma_4_sql.backends.maxtext.dpo.build_dataloader", return_value={"loader": None}), pytest.raises(ValueError, match="Invalid dataloader for dataset"):
        _execute_dpo("model", "dataset", 0.1, 1, 1e-5, 2)


def test_run_dpo_success():
    """Docstring for test_run_dpo_success."""
    config = DPOConfig(model_name="model", dataset="dataset")

    with (
        patch("gemma_4_sql.backends.maxtext.dpo.jax", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo.jnp", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo.optax", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo.Gemma4Model", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo._execute_dpo", return_value=("completed", 0.5)),
    ):
        res = run_dpo(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5


def test_run_dpo_missing_deps():
    """Docstring for test_run_dpo_missing_deps."""
    config = DPOConfig(model_name="model", dataset="dataset")
    with patch("gemma_4_sql.backends.maxtext.dpo.jax", None), pytest.raises(DependencyMissingError, match="MaxText dependencies are missing."):
        run_dpo(config)


def test_run_dpo_execution_error():
    """Docstring for test_run_dpo_execution_error."""
    config = DPOConfig(model_name="model", dataset="dataset")

    with (
        patch("gemma_4_sql.backends.maxtext.dpo.jax", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo.jnp", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo.optax", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo.Gemma4Model", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.dpo._execute_dpo", side_effect=RuntimeError("exec fail")),
    ):
        res = run_dpo(config)
        assert res["status"] == "failed: exec fail"

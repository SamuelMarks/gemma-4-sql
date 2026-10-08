"""Module docstring."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Docstring for mock_dependencies."""
    mock_jax = MagicMock()
    mock_jnn = MagicMock()
    mock_jnp = MagicMock()
    mock_optax = MagicMock()
    mock_nnx = MagicMock()
    mock_gemma4_config = MagicMock()
    mock_gemma4_model = MagicMock()

    mock_gemma4_mod = MagicMock()
    mock_gemma4_mod.Gemma4Config = mock_gemma4_config
    mock_gemma4_mod.Gemma4ForCausalLM = mock_gemma4_model

    mock_jnn.log_sigmoid.return_value = "log_sigmoid_out"
    mock_jnn.log_softmax.return_value = MagicMock()

    mock_jnp.expand_dims.return_value = MagicMock()
    mock_jnp.take_along_axis.return_value = MagicMock()
    mock_jnp.squeeze.return_value = MagicMock()
    # jnp.sum needs to return exactly what we expect

    def mock_value_and_grad(f):
        """Docstring for mock_value_and_grad."""

        def wrapper(*args, **kwargs):
            """Docstring for wrapper."""
            return (f(*args, **kwargs), "mock_grads")

        return wrapper

    mock_nnx.value_and_grad.side_effect = mock_value_and_grad
    mock_nnx.jit.side_effect = lambda f: f

    with patch.dict(
        sys.modules,
        {
            "jax": mock_jax,
            "jax.nn": mock_jnn,
            "jax.numpy": mock_jnp,
            "optax": mock_optax,
            "flax": MagicMock(nnx=mock_nnx),
            "flax.nnx": mock_nnx,
            "gemma_4_sql.backends.jax.gemma4": mock_gemma4_mod,
        },
    ):
        yield mock_jax, mock_jnn, mock_jnp, mock_optax, mock_nnx, mock_gemma4_config, mock_gemma4_model


def reload_module():
    """Docstring for reload_module."""
    import gemma_4_sql.backends.jax.dpo as jax_dpo

    importlib.reload(jax_dpo)
    return jax_dpo


def test_missing_dependencies():
    """Docstring for test_missing_dependencies."""
    with patch.dict(sys.modules, {"jax": None, "jax.nn": None, "jax.numpy": None, "optax": None, "flax": None, "flax.nnx": None}):
        jax_dpo = reload_module()

        loss, _, _ = jax_dpo.dpo_loss(None, None, None, None)
        assert loss == 0.0

        with pytest.raises(Exception, match="JAX DPO dependencies are missing."):
            jax_dpo.run_dpo(MagicMock())


def test_dpo_loss():
    """Docstring for test_dpo_loss."""
    jax_dpo = reload_module()

    with patch("gemma_4_sql.backends.jax.dpo.generic_dpo_loss", return_value=("loss", "chosen", "rejected")):
        l, c, r = jax_dpo.dpo_loss("a", "b", "c", "d")
        assert l == "loss"
        assert c == "chosen"
        assert r == "rejected"


def test_compute_logps():
    """Docstring for test_compute_logps."""
    jax_dpo = reload_module()

    mock_model = MagicMock()
    mock_model.return_value = "logits"

    jax_dpo.jnp.sum = MagicMock(return_value=1.0)

    logps = jax_dpo._compute_logps(mock_model, MagicMock(), MagicMock())
    assert logps == 1.0


def test_dpo_step_loss():
    """Docstring for test_dpo_step_loss."""
    jax_dpo = reload_module()

    mock_batch = {"chosen_inputs": "ci", "rejected_inputs": "ri", "chosen_labels": "cl", "rejected_labels": "rl"}

    with patch.object(jax_dpo, "_compute_logps", return_value="logp"), patch.object(jax_dpo, "dpo_loss", return_value=("step_loss", None, None)):
        loss = jax_dpo._dpo_step_loss(MagicMock(), MagicMock(), mock_batch, 0.1)
        assert loss == "step_loss"

        mock_batch2 = {"chosen_input_ids": "ci", "rejected_input_ids": "ri", "chosen_labels": "cl", "rejected_labels": "rl"}
        loss2 = jax_dpo._dpo_step_loss(MagicMock(), MagicMock(), mock_batch2, 0.1)
        assert loss2 == "step_loss"


def test_get_train_step_fn():
    """Docstring for test_get_train_step_fn."""
    jax_dpo = reload_module()

    train_step = jax_dpo._get_train_step_fn(0.1)

    mock_optimizer = MagicMock()
    mock_batch = {"chosen_inputs": "ci", "rejected_inputs": "ri", "chosen_labels": "cl", "rejected_labels": "rl"}

    with patch.object(jax_dpo, "_dpo_step_loss", return_value=1.0):
        loss = train_step(MagicMock(), MagicMock(), mock_optimizer, mock_batch)
        assert loss == 1.0
        mock_optimizer.update.assert_called_with("mock_grads")


def test_get_train_step_fn_no_nnx():
    """Docstring for test_get_train_step_fn_no_nnx."""
    jax_dpo = reload_module()

    del jax_dpo.nnx.value_and_grad
    del jax_dpo.nnx.jit

    train_step = jax_dpo._get_train_step_fn(0.1)

    mock_optimizer = MagicMock()
    del mock_optimizer.update

    loss = train_step(MagicMock(), MagicMock(), mock_optimizer, MagicMock())
    assert loss == 0.0


def test_run_training_epochs():
    """Docstring for test_run_training_epochs."""
    jax_dpo = reload_module()

    class MockLoss:
        """Docstring for MockLoss."""

        def item(self):
            """Docstring for item."""
            return 0.5

    mock_state = MagicMock()
    mock_state.dataloader = ["batch1", "batch2"]
    mock_state.epochs = 2
    mock_state.train_step = MagicMock(return_value=MockLoss())

    final_loss = jax_dpo._run_training_epochs(mock_state)
    assert final_loss == 0.5

    mock_state.train_step.return_value = 0.25
    final_loss = jax_dpo._run_training_epochs(mock_state)
    assert final_loss == 0.25


def test_execute_dpo():
    """Docstring for test_execute_dpo."""
    jax_dpo = reload_module()

    with patch("gemma_4_sql.backends.jax.dpo.build_dataloader") as mock_build_dl:
        mock_build_dl.return_value = {"loader": ["batch1"]}

        with patch.object(jax_dpo, "_run_training_epochs", return_value=0.5):
            status, loss = jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)
            assert status == "completed"
            assert loss == 0.5

        mock_build_dl.return_value = {"loader": None}
        with pytest.raises(ValueError, match="Invalid dataloader"):
            jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)

        jax_dpo.optax = None
        with pytest.raises(Exception, match="JAX dependencies are missing for DPO."):
            jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)


def test_run_dpo():
    """Docstring for test_run_dpo."""
    jax_dpo = reload_module()

    mock_config = MagicMock()

    with patch.object(jax_dpo, "_execute_dpo", return_value=("completed", 0.5)):
        res = jax_dpo.run_dpo(mock_config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

    with patch.object(jax_dpo, "_execute_dpo", side_effect=ValueError("Test Error")):
        res = jax_dpo.run_dpo(mock_config)
        assert res["status"] == "failed: Test Error"

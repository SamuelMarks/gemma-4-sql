"""Unit and behavioral tests for dpo jax."""

import importlib
import sys
from unittest.mock import MagicMock, patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import pytest


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Verify behavior of mock_dependencies."""
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
        """Verify behavior of mock_value_and_grad."""

        def wrapper(*args, **kwargs):
            """Verify behavior of wrapper."""
            return (f(*args, **kwargs), "mock_grads")

        return wrapper

    mock_nnx.value_and_grad.side_effect = mock_value_and_grad
    mock_nnx.jit.side_effect = lambda f: f
    mock_flax = MagicMock()
    mock_flax.nnx = mock_nnx

    import gemma_4_sql.backends.jax.gemma4 as real_gemma4_pkg

    with (
        patch.dict(
            sys.modules,
            {
                "jax": mock_jax,
                "jax.nn": mock_jnn,
                "jax.numpy": mock_jnp,
                "optax": mock_optax,
                "flax": mock_flax,
                "flax.nnx": mock_nnx,
                "gemma_4_sql.backends.jax.gemma4": mock_gemma4_mod,
            },
        ),
        patch.object(real_gemma4_pkg, "Gemma4Config", mock_gemma4_config),
        patch.object(real_gemma4_pkg, "Gemma4ForCausalLM", mock_gemma4_model),
    ):
        yield mock_jax, mock_jnn, mock_jnp, mock_optax, mock_nnx, mock_gemma4_config, mock_gemma4_model


def reload_module():
    """Verify behavior of reload_module."""
    import gemma_4_sql.backends.jax.dpo as jax_dpo

    importlib.reload(jax_dpo)
    jax_dpo.jax = sys.modules.get("jax")
    jax_dpo.jnn = sys.modules.get("jax.nn")
    jax_dpo.jnp = sys.modules.get("jax.numpy")
    jax_dpo.optax = sys.modules.get("optax")
    jax_dpo.nnx = sys.modules.get("flax.nnx")
    g4_mod = sys.modules.get("gemma_4_sql.backends.jax.gemma4")
    jax_dpo.Gemma4Config = getattr(g4_mod, "Gemma4Config", None)
    jax_dpo.Gemma4ForCausalLM = getattr(g4_mod, "Gemma4ForCausalLM", None)
    return jax_dpo


def test_missing_dependencies():
    """Verify behavior of test_missing_dependencies."""
    with patch.dict(sys.modules, {"jax": None, "jax.nn": None, "jax.numpy": None, "optax": None, "flax": None, "flax.nnx": None}):
        jax_dpo = reload_module()

        loss, _, _ = jax_dpo.dpo_loss(None, None, None, None)
        assert loss == 0.0

        with pytest.raises(Exception, match="JAX DPO dependencies are missing."):
            jax_dpo.run_dpo(MagicMock())


def test_dpo_loss():
    """Verify behavior of test_dpo_loss."""
    jax_dpo = reload_module()

    with patch("gemma_4_sql.backends.jax.dpo.generic_dpo_loss", return_value=("loss", "chosen", "rejected")):
        l, c, r = jax_dpo.dpo_loss("a", "b", "c", "d")
        assert l == "loss"
        assert c == "chosen"
        assert r == "rejected"


def test_compute_logps():
    """Verify behavior of test_compute_logps."""
    jax_dpo = reload_module()

    mock_model = MagicMock()
    mock_model.return_value = "logits"

    jax_dpo.jnp.sum = MagicMock(return_value=1.0)

    logps = jax_dpo._compute_logps(mock_model, MagicMock(), MagicMock())
    assert logps == 1.0


def test_dpo_step_loss():
    """Verify behavior of test_dpo_step_loss."""
    jax_dpo = reload_module()

    mock_batch = {"chosen_inputs": "ci", "rejected_inputs": "ri", "chosen_labels": "cl", "rejected_labels": "rl"}

    with patch.object(jax_dpo, "_compute_logps", return_value="logp"), patch.object(jax_dpo, "dpo_loss", return_value=("step_loss", None, None)):
        loss = jax_dpo._dpo_step_loss(MagicMock(), MagicMock(), mock_batch, 0.1)
        assert loss == "step_loss"

        mock_batch2 = {"chosen_input_ids": "ci", "rejected_input_ids": "ri", "chosen_labels": "cl", "rejected_labels": "rl"}
        loss2 = jax_dpo._dpo_step_loss(MagicMock(), MagicMock(), mock_batch2, 0.1)
        assert loss2 == "step_loss"


def test_get_train_step_fn():
    """Verify behavior of test_get_train_step_fn."""
    jax_dpo = reload_module()

    train_step = jax_dpo._get_train_step_fn(0.1)

    mock_optimizer = MagicMock()
    mock_batch = {"chosen_inputs": "ci", "rejected_inputs": "ri", "chosen_labels": "cl", "rejected_labels": "rl"}

    with patch.object(jax_dpo, "_dpo_step_loss", return_value=1.0):
        loss = train_step(MagicMock(), MagicMock(), mock_optimizer, mock_batch)
        assert loss == 1.0
        mock_optimizer.update.assert_called_with("mock_grads")


def test_get_train_step_fn_no_nnx():
    """Verify behavior of test_get_train_step_fn_no_nnx."""
    jax_dpo = reload_module()

    del jax_dpo.nnx.value_and_grad
    del jax_dpo.nnx.jit

    train_step = jax_dpo._get_train_step_fn(0.1)

    mock_optimizer = MagicMock()
    del mock_optimizer.update

    loss = train_step(MagicMock(), MagicMock(), mock_optimizer, MagicMock())
    assert loss == 0.0


def test_run_training_epochs():
    """Verify behavior of test_run_training_epochs."""
    jax_dpo = reload_module()

    class MockLoss:
        """Verify behavior of MockLoss."""

        def item(self):
            """Verify behavior of item."""
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
    """Verify behavior of test_execute_dpo."""
    jax_dpo = reload_module()

    with patch("gemma_4_sql.backends.jax.dpo.build_dataloader") as mock_build_dl:
        mock_build_dl.return_value = {"loader": ["batch1"]}

        with patch.object(jax_dpo, "_run_training_epochs", return_value=0.5):
            status, loss = jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)
            assert status == "completed"
            assert loss == 0.5
            del jax_dpo.Gemma4Config.from_preset
            assert jax_dpo._resolve_gemma4_config("model") is not None

        mock_build_dl.return_value = {"loader": None}
        with pytest.raises(ValueError, match="Invalid dataloader"):
            jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)

        jax_dpo.optax = None
        with pytest.raises(Exception, match="JAX dependencies are missing for DPO."):
            jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)


def test_run_dpo():
    """Verify behavior of test_run_dpo."""
    jax_dpo = reload_module()

    mock_config = MagicMock()

    with patch.object(jax_dpo, "_execute_dpo", return_value=("completed", 0.5)):
        res = jax_dpo.run_dpo(mock_config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

    with patch.object(jax_dpo, "_execute_dpo", side_effect=ValueError("Test Error")):
        res = jax_dpo.run_dpo(mock_config)
        assert res["status"] == "failed: Test Error"


def test_real_tiny_jax_dpo_loss_and_logps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _compute_logps and dpo_loss compute finite values on a real tiny Gemma4ForCausalLM."""
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp
    from flax import nnx

    import gemma_4_sql.backends.jax.dpo as jax_dpo
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM

    monkeypatch.setattr(jax_dpo, "jax", jax)
    monkeypatch.setattr(jax_dpo, "jnn", jnn)
    monkeypatch.setattr(jax_dpo, "jnp", jnp)
    monkeypatch.setattr(jax_dpo, "nnx", nnx)

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    policy_model = Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))
    ref_model = Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(1))
    batch = {
        "chosen_inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "chosen_labels": jnp.array([[2, 3, 4]], dtype=jnp.int32),
        "rejected_inputs": jnp.array([[1, 5, 6]], dtype=jnp.int32),
        "rejected_labels": jnp.array([[5, 6, 7]], dtype=jnp.int32),
    }
    step_loss = jax_dpo._dpo_step_loss(policy_model, ref_model, batch, 0.1)
    assert float(step_loss) > 0.0

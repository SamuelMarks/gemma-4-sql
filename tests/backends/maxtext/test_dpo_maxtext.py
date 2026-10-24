"""Unit and behavioral tests for MaxText Direct Preference Optimization (DPO)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext.dpo import (
    _compute_logps,
    _dpo_step_loss,
    _execute_dpo,
    _get_train_step_fn,
    _run_training_epochs,
    dpo_loss,
    run_dpo,
)
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState
from tests.conftest import requires_jax

pytestmark = requires_jax


def test_real_maxtext_dpo_logps_loss_and_train_step() -> None:
    """Verify MaxText _compute_logps, _dpo_step_loss, and _get_train_step_fn with real JAX arrays and Optax."""
    import jax.numpy as jnp
    import optax

    class TinyStatelessModel:
        """Minimal stateless Flax/MaxText-style model."""

        def apply(self, params: dict[str, Any], inputs: Any) -> Any:
            """Project token embeddings to vocabulary logits."""
            emb = jnp.take(params["embed"], inputs, axis=0)
            return jnp.dot(emb, params["head"])

    model = TinyStatelessModel()
    policy_params: dict[str, Any] = {
        "embed": jnp.ones((16, 4), dtype=jnp.float32) * 0.1,
        "head": jnp.ones((4, 16), dtype=jnp.float32) * 0.1,
    }
    ref_params: dict[str, Any] = {
        "embed": jnp.ones((16, 4), dtype=jnp.float32) * 0.1,
        "head": jnp.ones((4, 16), dtype=jnp.float32) * 0.1,
    }
    batch: dict[str, Any] = {
        "chosen_inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "chosen_labels": jnp.array([[2, 3, 0]], dtype=jnp.int32),
        "rejected_inputs": jnp.array([[1, 4, 5]], dtype=jnp.int32),
        "rejected_labels": jnp.array([[4, 5, 0]], dtype=jnp.int32),
    }

    model_any: Any = model
    pp_any: Any = policy_params
    rp_any: Any = ref_params
    logps: Any = _compute_logps(model_any, pp_any, batch["chosen_inputs"], batch["chosen_labels"])
    assert logps.shape == (1,)
    assert float(logps[0]) < 0.0

    step_loss: Any = _dpo_step_loss(model_any, pp_any, model_any, rp_any, batch, 0.1)
    assert float(step_loss) > 0.0

    optimizer = optax.adamw(1e-2)
    opt_state = optimizer.init(policy_params)
    train_step = _get_train_step_fn(model_any, model_any, optimizer, 0.1)

    state: Any = TrainerState(
        dataloader=[batch],
        epochs=1,
        train_step=train_step,
        policy_params=policy_params,
        ref_params=ref_params,
        opt_state=opt_state,
    )
    new_params, _new_opt_state, final_loss = _run_training_epochs(state)
    assert final_loss > 0.0
    assert not jnp.allclose(new_params["head"], policy_params["head"])


def test_compute_logps_scalar_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _compute_logps falls back to sum(logits * labels) when logits are 1D/scalar mocks."""
    import gemma_4_sql.backends.maxtext.dpo as mt_dpo

    mock_jnp = MagicMock()
    mock_jnp.sum.return_value = 10.0
    monkeypatch.setattr(mt_dpo, "jnp", mock_jnp)
    monkeypatch.setattr(mt_dpo, "jnn", None)

    model = MagicMock()
    model.apply.return_value = 5
    p_stub: Any = {"p": 1}
    i_stub: Any = 2
    l_stub: Any = 3
    res = _compute_logps(model, p_stub, i_stub, l_stub)
    assert res == 10.0


def test_dpo_loss_delegation() -> None:
    """Verify MaxText dpo_loss delegates to JAX dpo_loss."""
    t1: Any = 1
    t2: Any = 2
    t3: Any = 3
    t4: Any = 4
    with patch("gemma_4_sql.backends.maxtext.dpo.jax_dpo_loss", return_value=(1.0, 2.0, 3.0)) as mock_jax_dpo:
        res = dpo_loss(t1, t2, t3, t4, 0.5)
        assert res == (1.0, 2.0, 3.0)
        mock_jax_dpo.assert_called_once_with(1, 2, 3, 4, 0.5)


def test_get_train_step_fn_without_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_train_step_fn returns an un-jitted step callable when jax.jit is absent."""
    import gemma_4_sql.backends.maxtext.dpo as mt_dpo

    mock_jax = MagicMock(spec=["value_and_grad"])
    mock_jax.value_and_grad.return_value = lambda *_args: (0.5, "grads")
    monkeypatch.setattr(mt_dpo, "jax", mock_jax)

    optimizer = MagicMock()
    optimizer.update.return_value = ("updates", "new_opt_state")
    mock_optax = MagicMock()
    mock_optax.apply_updates.return_value = "new_policy_params"
    monkeypatch.setattr(mt_dpo, "optax", mock_optax)

    pm_stub: Any = "pm"
    rm_stub: Any = "rm"
    train_step = _get_train_step_fn(pm_stub, rm_stub, optimizer, 0.1)
    with patch("gemma_4_sql.backends.maxtext.dpo._dpo_step_loss", return_value=0.5):
        policy_params, opt_state, loss = train_step("pp", "rp", "os", "batch")
    assert policy_params == "new_policy_params"
    assert opt_state == "new_opt_state"
    assert loss == 0.5


def test_run_training_epochs_float_loss() -> None:
    """Verify _run_training_epochs handles plain float losses without .item()."""

    def mock_train_step(pp: Any, _rp: Any, os: Any, _b: Any) -> tuple[Any, Any, float]:
        """Return updated state tuple with float scalar loss."""
        return pp, os, 0.5

    state: Any = TrainerState(dataloader=[{"b": 1}], epochs=1, train_step=mock_train_step, policy_params="pp", ref_params="rp", opt_state="os")
    _pp, _os, loss = _run_training_epochs(state)
    assert loss == 0.5


def test_execute_dpo_and_run_dpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_dpo and run_dpo handle distributed initialization, dataloader validation, and errors."""
    import gemma_4_sql.backends.maxtext.dpo as mt_dpo

    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_optax = MagicMock()
    mock_gemma = MagicMock()

    monkeypatch.setattr(mt_dpo, "jax", mock_jax)
    monkeypatch.setattr(mt_dpo, "jnp", mock_jnp)
    monkeypatch.setattr(mt_dpo, "optax", mock_optax)
    monkeypatch.setattr(mt_dpo, "Gemma4Model", mock_gemma)

    with (
        patch("gemma_4_sql.backends.maxtext.dpo.build_dataloader", return_value={"loader": [1, 2]}),
        patch("gemma_4_sql.backends.maxtext.dpo._get_train_step_fn"),
        patch("gemma_4_sql.backends.maxtext.dpo._run_training_epochs", return_value=("pp", "os", 0.5)),
    ):
        status, loss = _execute_dpo("model", "dataset", 0.1, 1, 1e-5, 2)
        assert status == "completed"
        assert loss == 0.5

        mock_jax.distributed.initialize.side_effect = RuntimeError("init fail")
        status2, _ = _execute_dpo("model", "dataset", 0.1, 1, 1e-5, 2)
        assert status2 == "completed"

    with (
        patch("gemma_4_sql.backends.maxtext.dpo.build_dataloader", return_value={"loader": None}),
        pytest.raises(ValueError, match="Invalid dataloader for dataset"),
    ):
        _execute_dpo("model", "dataset", 0.1, 1, 1e-5, 2)

    config = DPOConfig(model_name="model", dataset="dataset")
    with patch("gemma_4_sql.backends.maxtext.dpo._execute_dpo", return_value=("completed", 0.5)):
        res = run_dpo(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

    with patch("gemma_4_sql.backends.maxtext.dpo._execute_dpo", side_effect=RuntimeError("exec fail")):
        res_err = run_dpo(config)
        assert res_err["status"] == "failed: exec fail"
        with pytest.raises(TrainingError, match="exec fail"):
            run_dpo(config, raise_on_error=True)

    monkeypatch.setattr(mt_dpo, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"MaxText dependencies are missing\."):
        run_dpo(config)

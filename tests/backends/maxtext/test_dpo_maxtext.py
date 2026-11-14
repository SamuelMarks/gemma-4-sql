"""Unit and behavioral tests for MaxText Direct Preference Optimization (DPO) without MagicMock."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from unittest.mock import patch

import jax
import optax
import pytest

import gemma_4_sql.backends.maxtext.dpo as mt_dpo
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
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
from gemma_4_sql.type_hints import DPOConfig, MaxTextDPOState
from tests.conftest import requires_jax

pytestmark = requires_jax


class _TinyStatelessModel:
    """Minimal stateless Flax/MaxText-style model."""

    def __init__(self, _name: str = "tiny") -> None:
        """Initialize stateless model."""

    def init(self, _rng: jax.Array, _inp: jax.Array) -> dict[str, jax.Array]:
        """Initialize embedding and output head arrays.

        Returns:
            A dictionary with `'embed'` and `'head'` JAX arrays.

        """
        return {
            "embed": jnp.ones((16, 4), dtype=jnp.float32) * 0.1,
            "head": jnp.ones((4, 16), dtype=jnp.float32) * 0.1,
        }

    def apply(self, params: Mapping[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Project token embeddings to vocabulary logits.

        Returns:
            Vocabulary logits array produced from `inputs`.

        """
        emb = jnp.take(params["embed"], inputs, axis=0)
        return jnp.dot(emb, params["head"])


def test_real_maxtext_dpo_logps_loss_and_train_step() -> None:
    """Verify MaxText `_compute_logps`, `_dpo_step_loss`, and `_get_train_step_fn` with real JAX arrays and Optax."""
    eq_logp = jnp.array([-1.5], dtype=jnp.float32)
    eq_loss, eq_ch_rew, eq_re_rew = dpo_loss(eq_logp, eq_logp, eq_logp, eq_logp, beta=0.1)
    assert float(eq_loss) == pytest.approx(math.log(2.0), rel=1e-5)
    assert float(eq_ch_rew[0]) == pytest.approx(0.0)
    assert float(eq_re_rew[0]) == pytest.approx(0.0)

    fav_loss, fav_ch_rew, fav_re_rew = dpo_loss(
        jnp.array([-1.0], dtype=jnp.float32),
        jnp.array([-3.0], dtype=jnp.float32),
        jnp.array([-2.0], dtype=jnp.float32),
        jnp.array([-2.0], dtype=jnp.float32),
        beta=0.5,
    )
    assert float(fav_loss) > 0.0
    assert float(fav_ch_rew[0]) > float(fav_re_rew[0])

    model = _TinyStatelessModel()
    policy_params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3), dtype=jnp.int32))
    ref_params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3), dtype=jnp.int32))
    batch: dict[str, jax.Array] = {
        "chosen_inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "chosen_labels": jnp.array([[2, 3, 0]], dtype=jnp.int32),
        "rejected_inputs": jnp.array([[1, 4, 5]], dtype=jnp.int32),
        "rejected_labels": jnp.array([[4, 5, 0]], dtype=jnp.int32),
    }

    logps = _compute_logps(model, policy_params, batch["chosen_inputs"], batch["chosen_labels"])
    assert isinstance(logps, jax.Array)
    assert logps.shape == (1,)
    assert float(logps[0]) < 0.0

    step_loss = _dpo_step_loss(model, policy_params, model, ref_params, batch, 0.1)
    assert float(step_loss) > 0.0

    optimizer = optax.adamw(1e-2)
    opt_state = optimizer.init(policy_params)
    train_step: mt_dpo.MaxTextDPOStepCallable[dict[str, jax.Array], optax.OptState, dict[str, jax.Array]] = (
        _get_train_step_fn(
            model,
            model,
            optimizer,
            0.1,
        )
    )

    state: MaxTextDPOState[dict[str, jax.Array], optax.OptState, dict[str, jax.Array]] = MaxTextDPOState(
        dataloader=[batch],
        epochs=1,
        train_step=train_step,
        policy_params=policy_params,
        ref_params=ref_params,
        opt_state=opt_state,
    )
    new_params, _new_opt_state, final_loss = _run_training_epochs(state)
    assert final_loss > 0.0
    new_head_val = new_params["head"]
    assert not jnp.allclose(new_head_val, policy_params["head"])
    updated_dict: dict[str, jax.Array] = {"embed": policy_params["embed"], "head": new_head_val}
    post_loss = float(_dpo_step_loss(model, updated_dict, model, ref_params, batch, 0.1))
    assert post_loss < float(step_loss)


def test_compute_logps_scalar_and_passthrough_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_compute_logps` falls back to `jnp.sum(logits * labels)` and passthrough when `jnn`/`jnp` are None."""

    class _ScalarApplyModel:
        """Stateless model returning scalar `float`."""

        def apply(self, _params: Mapping[str, float], _inputs: float) -> float:
            """Return `5.0`."""
            return 5.0

    monkeypatch.setattr(mt_dpo, "jnn", None)
    res_sum = _compute_logps(_ScalarApplyModel(), {"p": 1.0}, 2.0, 3.0)
    assert res_sum == 15.0

    monkeypatch.setattr(mt_dpo, "jnp", None)
    res_passthrough = _compute_logps(_ScalarApplyModel(), {"p": 1.0}, 2.0, 3.0)
    assert res_passthrough == 5.0


def test_dpo_loss_delegation() -> None:
    """Verify MaxText `dpo_loss` delegates to JAX `dpo_loss`."""
    with patch("gemma_4_sql.backends.maxtext.dpo.jax_dpo_loss", return_value=(1.0, 2.0, 3.0)) as mock_jax_dpo:
        res = dpo_loss(1.0, 2.0, 3.0, 4.0, 0.5)
        assert res == (1.0, 2.0, 3.0)
        mock_jax_dpo.assert_called_once_with(1.0, 2.0, 3.0, 4.0, 0.5)


def test_get_train_step_fn_without_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_get_train_step_fn` returns an un-jitted step callable when `jax.jit` is absent."""

    class _NoJitJax:
        """Typed `jax` double providing `value_and_grad` without `jit`."""

        def value_and_grad[P1, R1, B1, L1](
            self,
            fun: Callable[[P1, R1, B1], L1],
        ) -> Callable[[P1, R1, B1], tuple[L1, P1]]:
            """Evaluate `fun` and return `(loss, policy_params)` as synthetic grad.

            Returns:
                A callable returning `(loss, policy_params)` for inputs `(p, r, b)`.

            """

            def _grad_fn(p: P1, r: R1, b: B1) -> tuple[L1, P1]:
                """Return `(fun(p, r, b), p)`."""
                return (fun(p, r, b), p)

            return _grad_fn

    class _StubOptimizer:
        """Typed Optax optimizer double."""

        def init(self, params: dict[str, float]) -> str:
            """Return `'init_state'`."""
            _ = params
            return "init_state"

        def update(
            self,
            updates: dict[str, float],
            state: str,
            params: dict[str, float] | None = None,
        ) -> tuple[dict[str, float], str]:
            """Return `(updates, 'new_opt_state')`."""
            _ = (state, params)
            return (updates, "new_opt_state")

    class _StubOptax:
        """Typed Optax module double providing `apply_updates`."""

        def apply_updates(self, params: dict[str, float], updates: dict[str, float]) -> dict[str, float]:
            """Sum matching float keys.

            Returns:
                A dictionary with updated float parameter values.

            """
            return {k: params[k] + updates[k] for k in params}

    monkeypatch.setattr(mt_dpo, "jax", _NoJitJax())
    monkeypatch.setattr(mt_dpo, "optax", _StubOptax())
    assert _StubOptimizer().init({"w": 1.0}) == "init_state"

    train_step: mt_dpo.MaxTextDPOStepCallable[dict[str, float], str, dict[str, float]] = _get_train_step_fn(
        "pm",
        "rm",
        _StubOptimizer(),
        0.1,
    )
    with patch("gemma_4_sql.backends.maxtext.dpo._dpo_step_loss", return_value=0.5):
        policy_params, opt_state, loss = train_step({"w": 1.0}, {"w": 1.0}, "os", {"b": 1.0})
    assert policy_params == {"w": 2.0}
    assert opt_state == "new_opt_state"
    assert loss == 0.5


def test_run_training_epochs_float_loss() -> None:
    """Verify `_run_training_epochs` handles plain `float` losses without `.item()`."""

    def mock_train_step(pp: str, _rp: str, os: str, _b: dict[str, int]) -> tuple[str, str, float]:
        """Return updated state tuple with float scalar loss."""
        return pp, os, 0.5

    state: MaxTextDPOState[str, str, dict[str, int]] = MaxTextDPOState(
        dataloader=[{"b": 1}],
        epochs=1,
        train_step=mock_train_step,
        policy_params="pp",
        ref_params="rp",
        opt_state="os",
    )
    _pp, _os, loss = _run_training_epochs(state)
    assert loss == 0.5


def test_execute_dpo_and_run_dpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_execute_dpo` and `run_dpo` handle distributed initialization, real tiny model updates, and errors."""
    monkeypatch.setattr(mt_dpo, "Gemma4Model", _TinyStatelessModel)

    batch: dict[str, jax.Array] = {
        "chosen_inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "chosen_labels": jnp.array([[2, 3, 0]], dtype=jnp.int32),
        "rejected_inputs": jnp.array([[1, 4, 5]], dtype=jnp.int32),
        "rejected_labels": jnp.array([[4, 5, 0]], dtype=jnp.int32),
    }

    with (
        patch.object(jax.distributed, "initialize", side_effect=[None, RuntimeError("already init")]),
        patch("gemma_4_sql.backends.maxtext.dpo.build_dataloader", return_value={"loader": [batch]}),
    ):
        status, loss = _execute_dpo("model", "dataset", 0.1, 1, 1e-3, 1)
        assert status == "completed"
        assert loss > 0.0

        status2, loss2 = _execute_dpo("model", "dataset", 0.1, 1, 1e-3, 1)
        assert status2 == "completed"
        assert loss2 > 0.0

    with (
        patch.object(jax.distributed, "initialize"),
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

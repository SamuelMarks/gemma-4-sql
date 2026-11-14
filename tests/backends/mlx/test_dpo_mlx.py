"""Unit and behavioral tests for MLX Direct Preference Optimization (DPO) without MagicMock."""

from __future__ import annotations

import math
from typing import Self
from unittest.mock import patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, ETLError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState, MlxArrayProtocol
from tests.conftest import requires_mlx
from tests.utils.tensor_helpers import (
    TypedMlxLinearProtocol,
    TypedMlxSequentialProtocol,
    get_typed_mlx_nn,
    get_typed_mlx_optim,
    get_typed_mx,
    mx_eval,
    mx_eval_model,
)

pytestmark = requires_mlx


def test_real_mlx_dpo_loss_logps_and_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLX dpo_loss, _compute_mlx_logps, _run_dpo_step, and run_dpo with real MLX arrays."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    mx = get_typed_mx()
    mlx_nn = get_typed_mlx_nn()
    optim = get_typed_mlx_optim()

    def _build_policy() -> TypedMlxSequentialProtocol:
        """Build a real 2-layer MLX causal policy (`Embedding -> Linear`).

        Returns:
            Initialized `TypedMlxSequentialProtocol` model.
        """
        return mlx_nn.Sequential(mlx_nn.Embedding(32, 8), mlx_nn.Linear(8, 32))

    policy_model = _build_policy()
    ref_model = _build_policy()
    mx_eval_model(policy_model, ref_model)

    optimizer = optim.AdamW(learning_rate=1e-2)
    chosen_inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    chosen_labels = mx.array([[2, 3, 0]], dtype=mx.int32)
    rejected_inputs = mx.array([[1, 4, 5]], dtype=mx.int32)
    rejected_labels = mx.array([[4, 5, 0]], dtype=mx.int32)
    batch: dpo_module.MlxDPOBatch = {
        "chosen_inputs": chosen_inputs,
        "chosen_labels": chosen_labels,
        "rejected_inputs": rejected_inputs,
        "rejected_labels": rejected_labels,
    }

    logps = dpo_module._compute_mlx_logps(policy_model, chosen_inputs, chosen_labels)
    assert isinstance(logps, dpo_module.MlxArrayProtocol)
    mx_eval(logps)
    assert logps.shape == (1,)
    assert float(logps) < 0.0

    logps_list = dpo_module._compute_mlx_logps(policy_model, chosen_inputs, [[2, 3, 0]])
    assert isinstance(logps_list, dpo_module.MlxArrayProtocol)
    mx_eval(logps_list)
    assert logps_list.shape == (1,)

    logps_shifted = dpo_module._compute_mlx_logps(policy_model, chosen_inputs, None)
    assert isinstance(logps_shifted, dpo_module.MlxArrayProtocol)
    mx_eval(logps_shifted)
    assert logps_shifted.shape == (1,)

    # Also test MlxCausalLMProtocol with None input -> 0.0 fallback
    assert dpo_module._compute_mlx_logps(policy_model, None, None) == 0.0

    loss_step_1 = float(dpo_module._run_dpo_step(policy_model, ref_model, optimizer, batch, 0.5))
    loss_step_2 = float(dpo_module._run_dpo_step(policy_model, ref_model, optimizer, batch, 0.5))
    assert loss_step_1 > 0.0
    assert loss_step_2 < loss_step_1

    # Full run_dpo end-to-end execution with real MLX Sequential policy
    def _load_tiny(_m: str) -> tuple[TypedMlxSequentialProtocol, str]:
        """Return `(_build_policy(), 'tok')`."""
        return (_build_policy(), "tok")

    monkeypatch.setattr(dpo_module, "load", _load_tiny)
    with patch.object(dpo_module, "build_dataloader", side_effect=ETLError("offline fallback")):
        res_e2e = dpo_module.run_dpo(DPOConfig(model_name="tiny", dataset="synthetic-dpo", epochs=1))
        assert res_e2e["status"] == "completed"
        assert res_e2e["final_loss"] > 0.0


def test_dpo_loss_and_fallback_log_sigmoid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dpo_loss computes exact preference loss on real MLX arrays and checks missing dependencies."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    mx = get_typed_mx()

    eq_logp = mx.array([-1.5])
    eq_loss, eq_ch_r, eq_re_r = dpo_module.dpo_loss(eq_logp, eq_logp, eq_logp, eq_logp, 0.1)
    assert float(eq_loss) == pytest.approx(math.log(2.0), rel=1e-5)
    assert float(eq_ch_r) == pytest.approx(0.0)
    assert float(eq_re_r) == pytest.approx(0.0)

    real_loss, chosen_r, rejected_r = dpo_module.dpo_loss(
        mx.array([-1.0]),
        mx.array([-3.0]),
        mx.array([-2.0]),
        mx.array([-2.0]),
        0.5,
    )
    assert float(real_loss) > 0.0
    assert float(chosen_r) > float(rejected_r)

    monkeypatch.setattr(dpo_module, "mx", None)
    with pytest.raises(DependencyMissingError, match=r"MLX DPO dependencies are missing\."):
        dpo_module.dpo_loss(eq_logp, eq_logp, eq_logp, eq_logp, 0.1)


def test_run_dpo_step_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_dpo_step and _compute_mlx_logps fallback branches without MagicMock."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    mx = get_typed_mx()

    class _ArithMeanStub:
        """Minimal `ArithmeticTensorProtocol` stub."""

        def __float__(self) -> float:
            """Return `1.0`."""
            return 1.0

        def __sub__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __rsub__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __mul__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __rmul__(self, _other: Self | float, /) -> Self:
            """Return self."""
            return self

        def __neg__(self) -> Self:
            """Return self."""
            return self

    arith_inst = _ArithMeanStub()
    assert float(arith_inst) == 1.0 and (arith_inst - 1.0) is arith_inst and (1.0 - arith_inst) is arith_inst
    assert (arith_inst * 1.0) is arith_inst and (1.0 * arith_inst) is arith_inst and (-arith_inst) is arith_inst

    def _one_d_mlx_model(_x: dpo_module.MlxDPOBatchValue) -> MlxArrayProtocol:
        """Return 1D MLX array to exercise `raw_logits.mean(axis=-1)`."""
        return mx.array([1.0, 3.0])

    def _arith_model(_x: dpo_module.MlxDPOBatchValue) -> _ArithMeanStub:
        """Return `arith_inst`."""
        return arith_inst

    class _PureSupportsFloat:
        """Pure `SupportsFloat` stub without arithmetic operators."""

        def __float__(self) -> float:
            """Return `0.5`."""
            return 0.5

    def _float_model(_x: dpo_module.MlxDPOBatchValue) -> _PureSupportsFloat:
        """Return `_PureSupportsFloat()`."""
        return _PureSupportsFloat()

    def _str_model(_x: dpo_module.MlxDPOBatchValue) -> str:
        """Return `'str_logits'`."""
        return "str_logits"

    assert float(dpo_module._compute_mlx_logps(_one_d_mlx_model, None)) == pytest.approx(2.0)
    assert dpo_module._compute_mlx_logps(_arith_model, None) is arith_inst
    assert dpo_module._compute_mlx_logps(_float_model, None) == pytest.approx(0.5)
    assert dpo_module._compute_mlx_logps(_str_model, None) == 0.0

    monkeypatch.setattr(dpo_module, "nn", None)
    batch: dpo_module.MlxDPOBatch = {"chosen_inputs": "ci", "rejected_inputs": "ri"}

    with patch.object(dpo_module, "dpo_loss", return_value=(0.75, 0.0, 0.0)):
        res = dpo_module._run_dpo_step(_float_model, _str_model, None, batch, 0.1)
        assert res == 0.75

    class _TypedNNStub:
        """Typed `mlx.nn` stub without `object`."""

        @staticmethod
        def value_and_grad(
            _m: dpo_module.MlxDPOModel,
            _fn: dpo_module.Callable[..., float],
        ) -> dpo_module.Callable[..., tuple[float, dict[str, MlxArrayProtocol]]]:
            """Return callable yielding `(0.5, {})`."""

            def _inner(
                *_args: dpo_module.MlxDPOModel | dpo_module.MlxDPOBatch | float,
            ) -> tuple[float, dict[str, MlxArrayProtocol]]:
                """Return `(0.5, {})`."""
                return (0.5, {})

            return _inner

    monkeypatch.setattr(dpo_module, "nn", _TypedNNStub())
    monkeypatch.setattr(dpo_module, "mx", None)
    res_no_opt = dpo_module._run_dpo_step(_float_model, _str_model, "no_update_opt", batch, 0.1)
    assert res_no_opt == 0.5


def test_run_training_epochs_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_training_epochs and run_dpo error handling without MagicMock."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    mx = get_typed_mx()
    mlx_nn = get_typed_mlx_nn()

    def _dummy_policy(_x: dpo_module.MlxDPOBatchValue) -> float:
        """Return `0.0`."""
        return 0.0

    assert _dummy_policy(None) == 0.0

    state: DPOLoopState[dpo_module.MlxDPOModel, str, dict[str, int]] = DPOLoopState(
        dataloader=[{"chosen_inputs": 1}, {"chosen_inputs": 2}],
        epochs=2,
        policy_model=_dummy_policy,
        ref_model=_dummy_policy,
        optimizer="o",
        beta=0.1,
    )

    calls = 0

    def _step_arr(*_a: dpo_module.MlxDPOModel | str | dict[str, int] | float | None) -> MlxArrayProtocol:
        """Increment `calls` and return `mx.array(0.5)`.

        Returns:
            Scalar MLX loss array of `0.5`.
        """
        nonlocal calls
        calls += 1
        return mx.array(0.5)

    with patch.object(dpo_module, "_run_dpo_step", side_effect=_step_arr):
        loss = dpo_module._run_training_epochs(state)
        assert loss == 0.5
        assert calls == 4

    with patch.object(dpo_module, "_run_dpo_step", return_value=0.375):
        assert dpo_module._run_training_epochs(state) == 0.375

    with patch.object(dpo_module, "_run_dpo_step", return_value="non_float"):
        assert dpo_module._run_training_epochs(state) == 0.0

    lin = mlx_nn.Linear(4, 4)

    def _load_lin(_m: str) -> tuple[TypedMlxLinearProtocol, str]:
        """Return `(lin, 'tok1')`."""
        return (lin, "tok1")

    monkeypatch.setattr(dpo_module, "load", _load_lin)
    config = DPOConfig(model_name="model", dataset="dataset")

    with patch.object(dpo_module, "build_dataloader", return_value={"loader": [{"chosen_inputs": [1]}]}):
        with patch.object(dpo_module, "_run_training_epochs", side_effect=RuntimeError("error")):
            res2 = dpo_module.run_dpo(config)
            assert "failed: error" in res2["status"]
            with pytest.raises(TrainingError, match="error"):
                dpo_module.run_dpo(config, raise_on_error=True)

    with patch.object(dpo_module, "build_dataloader", return_value={}):
        res3 = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res3["status"]

    with patch.object(dpo_module, "build_dataloader", return_value={"loader": 123}):
        res4 = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res4["status"]

    monkeypatch.setattr(dpo_module, "mlx", None)
    with pytest.raises(DependencyMissingError):
        dpo_module.run_dpo(config)

"""Unit and behavioral tests for Keras Direct Preference Optimization (DPO)."""

from __future__ import annotations

import math
import types
from collections.abc import Iterable, Sequence
from unittest.mock import MagicMock

import pytest
import torch

from gemma_4_sql.backends.keras import dpo
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, KerasTrainerState
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_real_keras_dpo_loss_logps_and_train_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras dpo_loss, _compute_logps, and _get_train_step_fn with concrete tensor math and gradient updates."""

    class _ArrayTensor:
        """Concrete tensor wrapper implementing `TfTensorProtocol` and `ArithmeticTensorProtocol`."""

        def __init__(
            self,
            arr: torch.Tensor | Sequence[float] | Sequence[Sequence[int]] | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]] | float,
        ) -> None:
            """Initialize with a float32 tensor."""
            self._t: torch.Tensor = arr.to(torch.float32) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float32)

        @property
        def shape(self) -> tuple[int, ...]:
            """Provide tensor shape."""
            return tuple(self._t.shape)

        @property
        def ndim(self) -> int:
            """Provide number of dimensions."""
            return int(self._t.ndim)

        @property
        def dtype(self) -> str:
            """Provide dtype name."""
            return str(self._t.dtype)

        def numpy(self) -> _ArrayTensor:
            """Return self as array protocol."""
            return self

        def mean(self) -> _ArrayTensor:
            """Compute arithmetic mean."""
            return _ArrayTensor(self._t.mean())

        def __float__(self) -> float:
            """Convert scalar array to float."""
            return float(self._t.mean().item())

        def __index__(self) -> int:
            """Convert scalar array to integer index."""
            return int(self._t.mean().item())

        def __sub__(self, other: _ArrayTensor | float, /) -> _ArrayTensor:
            """Subtract another tensor or scalar."""
            rhs = other._t if isinstance(other, _ArrayTensor) else float(other)
            return _ArrayTensor(self._t - rhs)

        def __rsub__(self, other: _ArrayTensor | float, /) -> _ArrayTensor:
            """Right-subtract from scalar."""
            rhs = other._t if isinstance(other, _ArrayTensor) else float(other)
            return _ArrayTensor(rhs - self._t)

        def __add__(self, other: _ArrayTensor | float, /) -> _ArrayTensor:
            """Add another tensor or scalar."""
            rhs = other._t if isinstance(other, _ArrayTensor) else float(other)
            return _ArrayTensor(self._t + rhs)

        def __radd__(self, other: _ArrayTensor | float, /) -> _ArrayTensor:
            """Right-add to scalar."""
            rhs = other._t if isinstance(other, _ArrayTensor) else float(other)
            return _ArrayTensor(rhs + self._t)

        def __mul__(self, other: _ArrayTensor | float, /) -> _ArrayTensor:
            """Multiply by another tensor or scalar."""
            rhs = other._t if isinstance(other, _ArrayTensor) else float(other)
            return _ArrayTensor(self._t * rhs)

        def __rmul__(self, other: _ArrayTensor | float, /) -> _ArrayTensor:
            """Right-multiply by scalar."""
            rhs = other._t if isinstance(other, _ArrayTensor) else float(other)
            return _ArrayTensor(rhs * self._t)

        def __neg__(self) -> _ArrayTensor:
            """Negate elementwise."""
            return _ArrayTensor(-self._t)

    t_check = _ArrayTensor([[1.0, 2.0]])
    assert t_check.ndim == 2
    assert "float32" in t_check.dtype
    assert t_check.numpy() is t_check
    assert int(t_check) == 1
    assert float((t_check + 1.0) + (1.0 + t_check) + (t_check + t_check)) > 0.0
    assert float((t_check * 2.0) + (t_check * t_check) + (2.0 - t_check)) > 0.0

    class _TFStub:
        """Tensor-backed TensorFlow runtime stub for DPO mathematical verification."""

        int32 = "int32"
        float32 = "float32"

        class dtypes:
            """Dtypes namespace."""

            @staticmethod
            def cast(x: _ArrayTensor | bool, _dtype: str) -> _ArrayTensor:
                """Cast array or boolean mask to float32 `_ArrayTensor`."""
                if isinstance(x, _ArrayTensor):
                    return _ArrayTensor(x._t)
                return _ArrayTensor(1.0 if x else 0.0)

        class math:
            """Math namespace."""

            @staticmethod
            def log_sigmoid(x: _ArrayTensor | float) -> _ArrayTensor:
                """Compute elementwise `-log(1 + exp(-x))`."""
                t = x._t if isinstance(x, _ArrayTensor) else torch.tensor(x, dtype=torch.float32)
                return _ArrayTensor(torch.nn.functional.logsigmoid(t))

        class nn:
            """NN namespace."""

            @staticmethod
            def log_softmax(logits: _ArrayTensor, axis: int = -1) -> _ArrayTensor:
                """Compute log-softmax along `axis`."""
                return _ArrayTensor(torch.nn.functional.log_softmax(logits._t, dim=axis))

        @staticmethod
        def expand_dims(x: _ArrayTensor, axis: int = -1) -> _ArrayTensor:
            """Insert singleton axis."""
            return _ArrayTensor(torch.unsqueeze(x._t, dim=axis))

        @staticmethod
        def gather(params: _ArrayTensor, indices: _ArrayTensor, batch_dims: int = 0) -> _ArrayTensor:
            """Gather token log-probabilities along vocabulary axis."""
            _ = batch_dims
            return _ArrayTensor(torch.gather(params._t, dim=-1, index=indices._t.to(torch.int64)))

        @staticmethod
        def squeeze(x: _ArrayTensor, axis: int = -1) -> _ArrayTensor:
            """Remove singleton axis."""
            return _ArrayTensor(torch.squeeze(x._t, dim=axis))

        @staticmethod
        def reduce_sum(x: _ArrayTensor, axis: int = -1) -> _ArrayTensor:
            """Sum along `axis`."""
            return _ArrayTensor(torch.sum(x._t, dim=axis))

        @staticmethod
        def reduce_mean(x: _ArrayTensor | float) -> _ArrayTensor:
            """Mean across all elements."""
            t = x._t if isinstance(x, _ArrayTensor) else torch.tensor(x, dtype=torch.float32)
            return _ArrayTensor(torch.mean(t))

        @staticmethod
        def function[FnT](fn: FnT) -> FnT:
            """Identity decorator for `tf.function`."""
            return fn

        class GradientTape:
            """Gradient tape for TinyTFModel variables."""

            def __enter__(self) -> _TFStub.GradientTape:
                """Enter tape context."""
                return self

            def __exit__(
                self,
                _exc_type: type[BaseException] | None = None,
                _exc_val: BaseException | None = None,
                _exc_tb: types.TracebackType | None = None,
            ) -> None:
                """Exit tape context."""

            @staticmethod
            def gradient(_loss: _ArrayTensor, variables: list[_ArrayTensor]) -> list[_ArrayTensor]:
                """Return non-zero gradient tensors matching variable shapes."""
                return [_ArrayTensor(torch.ones_like(v._t) * 0.01) for v in variables]

    monkeypatch.setattr(dpo, "tf", _TFStub)

    # (b) Verify mathematical DPO properties: equal logps -> loss == log(2.0)
    z = _ArrayTensor([-1.0])
    loss_eq, ch_rew_eq, re_rew_eq = dpo.dpo_loss(z, z, z, z, beta=0.1)
    assert float(loss_eq) == pytest.approx(math.log(2.0), rel=1e-5)
    assert float(ch_rew_eq) == pytest.approx(0.0)
    assert float(re_rew_eq) == pytest.approx(0.0)

    # (a) Verify chosen_rewards > rejected_rewards when policy favors chosen over reference
    fav_loss, fav_ch_rew, fav_re_rew = dpo.dpo_loss(
        _ArrayTensor([-1.0]),
        _ArrayTensor([-3.0]),
        _ArrayTensor([-2.0]),
        _ArrayTensor([-2.0]),
        beta=0.5,
    )
    assert float(fav_loss) > 0.0
    assert float(fav_ch_rew) > float(fav_re_rew)

    # Scalar float inputs (not isinstance of _ArrayTensor)
    loss_scalar, _, _ = dpo.dpo_loss(-1.0, -1.0, -1.0, -1.0, beta=0.1)
    assert float(loss_scalar) == pytest.approx(math.log(2.0), rel=1e-5)

    # (c) Verify real parameter updates reduce step loss on a miniature model
    class TinyTFModel:
        """Minimal causal LM for DPO logps and step testing."""

        def __init__(self, bias_chosen: float = 0.0) -> None:
            """Initialize TinyTFModel with trainable projection weight."""
            self.w = _ArrayTensor([[0.0] * 8 for _ in range(8)])
            self.trainable_variables = [self.w]
            self.bias_chosen = bias_chosen

        def __call__(self, x: _ArrayTensor) -> _ArrayTensor:
            """Return vocabulary logits of shape `(batch, seq, 8)`."""
            b, s = x.shape
            rows = [[[self.bias_chosen if c in (2, 3) else 0.0 for c in range(8)] for _ in range(s)] for _ in range(b)]
            return _ArrayTensor(rows)

    policy_model = TinyTFModel(bias_chosen=0.5)
    ref_model = TinyTFModel(bias_chosen=0.0)

    batch: dict[str, _ArrayTensor] = {
        "chosen_inputs": _ArrayTensor([[1, 2, 3]]),
        "chosen_labels": _ArrayTensor([[2, 3, 0]]),
        "rejected_inputs": _ArrayTensor([[1, 4, 5]]),
        "rejected_labels": _ArrayTensor([[4, 5, 0]]),
    }

    logps = dpo._compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])
    assert tuple(logps.shape) == (1,)
    assert float(logps) < 0.0

    applied_grads: list[_ArrayTensor] = []

    class SimpleTFOptimizer:
        """Minimal optimizer applying gradients and increasing policy preference."""

        def apply_gradients(self, grads_and_vars: Iterable[tuple[_ArrayTensor, _ArrayTensor]]) -> None:
            """Apply gradient updates to variables."""
            for g, _v in grads_and_vars:
                applied_grads.append(g)
            policy_model.bias_chosen += 0.5

    step_fn = dpo._get_train_step_fn(policy_model, ref_model, SimpleTFOptimizer(), 0.5)
    loss_1 = float(step_fn(batch))
    loss_2 = float(step_fn(batch))
    assert loss_1 > 0.0
    assert loss_2 < loss_1
    assert len(applied_grads) == 2

    state: KerasTrainerState[TinyTFModel, SimpleTFOptimizer, dict[str, _ArrayTensor]] = KerasTrainerState(
        dataloader=[batch],
        epochs=1,
        train_step=step_fn,
    )
    final_loss = dpo._run_training_epochs(state)
    assert final_loss > 0.0


def test_dpo_loss_and_logps_missing_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dpo_loss, _compute_logps, and _get_train_step_fn raise DependencyMissingError when TensorFlow is None."""
    monkeypatch.setattr(dpo, "tf", None)
    with pytest.raises(DependencyMissingError, match=r"Keras DPO dependencies are missing\."):
        dpo.dpo_loss(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(DependencyMissingError, match=r"Keras DPO dependencies are missing\."):
        dpo._compute_logps(MagicMock(), 1.0, 1.0)
    with pytest.raises(DependencyMissingError, match=r"Keras DPO dependencies are missing\."):
        dpo._get_train_step_fn(MagicMock(), MagicMock(), MagicMock(), 0.1)


def test_run_training_epochs_loss_variants() -> None:
    """Verify _run_training_epochs handles float and non-numeric step return values."""
    state_float: KerasTrainerState[None, None, dict[str, int]] = KerasTrainerState(
        dataloader=[{"b": 1}],
        epochs=1,
        train_step=lambda _b: 2.0,  # pyright: ignore[reportUnknownLambdaType]
    )
    assert dpo._run_training_epochs(state_float) == 2.0

    state_str: KerasTrainerState[None, None, dict[str, int]] = KerasTrainerState(
        dataloader=[{"b": 1}],
        epochs=1,
        train_step=lambda _b: "non_numeric",  # pyright: ignore[reportUnknownLambdaType]
    )
    assert dpo._run_training_epochs(state_str) == 0.0


def test_execute_dpo_and_run_dpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_dpo and run_dpo handle missing dependencies, preset loading, and dataloader validation."""
    monkeypatch.setattr(dpo, "keras", None)
    with pytest.raises(DependencyMissingError):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    mock_keras = MagicMock()
    monkeypatch.setattr(dpo, "keras", mock_keras)
    monkeypatch.setattr(dpo, "tf", MagicMock())

    monkeypatch.setattr(dpo, "GemmaCausalLM", None)
    monkeypatch.setattr(dpo, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]
    with pytest.raises(ValueError, match="Failed to load Keras model"):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = MagicMock()
    monkeypatch.setattr(dpo, "GemmaCausalLM", mock_cls)

    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={"loader": None}))
    with pytest.raises(ValueError, match="Invalid dataloader"):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={"loader": [{"b": 1}]}))
    monkeypatch.setattr(dpo, "_run_training_epochs", MagicMock(return_value=1.5))
    status, loss = dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)
    assert status == "completed"
    assert loss == 1.5

    cfg = DPOConfig(model_name="m", dataset="d")
    monkeypatch.setattr(dpo, "keras", None)
    with pytest.raises(DependencyMissingError):
        dpo.run_dpo(cfg)

    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", MagicMock())
    monkeypatch.setattr(dpo, "_execute_dpo", MagicMock(return_value=("completed", 0.75)))
    res_ok = dpo.run_dpo(cfg)
    assert res_ok["status"] == "completed"
    assert res_ok["final_loss"] == 0.75

    monkeypatch.setattr(dpo, "_execute_dpo", MagicMock(side_effect=ValueError("simulated dpo failure")))
    res_err = dpo.run_dpo(cfg)
    assert "failed: simulated dpo failure" in str(res_err["status"])
    with pytest.raises(TrainingError, match="simulated dpo failure"):
        dpo.run_dpo(cfg, raise_on_error=True)

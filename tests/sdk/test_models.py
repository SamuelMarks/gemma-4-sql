"""Behavioral unit tests for the Models SDK module."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk import models
from gemma_4_sql.sdk.models import posttrain_model, pretrain_model, sft_model, train_from_scratch
from gemma_4_sql.type_hints import TrainingConfig
from tests.utils.test_utils import RecordingBackendDouble


def test_pretrain_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `pretrain_model` routes configurations across JAX, PyTorch, Keras, MaxText, and MLX."""
    from gemma_4_sql.sdk.protocols import BackendProtocol
    from gemma_4_sql.sdk.registry import get_backend

    real_get_backend = get_backend
    jax_double = RecordingBackendDouble("jax")
    mlx_double = RecordingBackendDouble("mlx")
    pytorch_err_double = RecordingBackendDouble(
        "pytorch",
        error_to_raise=DependencyMissingError("PyTorch dependencies are missing."),
    )

    def _dispatch_backend(name: str) -> BackendProtocol:
        """Return recording doubles for jax/mlx/pytorch or real backend for keras/maxtext."""
        if name == "jax":
            return jax_double
        if name == "mlx":
            return mlx_double
        if name == "pytorch":
            return pytorch_err_double
        return real_get_backend(name)

    monkeypatch.setattr(models, "get_backend", _dispatch_backend)

    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    assert res["backend"] == "jax"
    assert res["action"] == "pretrain"
    assert res["model"] == "my-model"
    assert len(jax_double.train_configs) == 1
    assert jax_double.train_configs[0].epochs == 2

    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="pytorch"))

    import gemma_4_sql.backends.keras.train as ktrain

    monkeypatch.setattr(ktrain, "tf", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))

    class DummyPreprocessor:
        """Preprocessor stub with sequence_length."""

        sequence_length: int = 128

    class DummyFitHistory:
        """Fit history stub with decreasing loss."""

        def __init__(self, epochs: int) -> None:
            """Initialize history with loss scaled by epochs."""
            self.history: dict[str, list[float]] = {"loss": [0.4 / max(1, epochs)]}

    class DummyKerasModel:
        """Minimal Keras model stub for behavioral routing through ktrain.train_model."""

        def __init__(self) -> None:
            """Initialize model stub with preprocessor."""
            self.preprocessor = DummyPreprocessor()

        def compile(self, **_kw: str | float | bool | None) -> None:
            """No-op compile."""

        def fit(
            self,
            _ds: Iterable[Mapping[str, list[list[int]]]],
            epochs: int = 1,
        ) -> DummyFitHistory:
            """Return history stub with decreasing loss."""
            return DummyFitHistory(epochs)

    class DummyGemmaCausalLM:
        """Stub for GemmaCausalLM.from_preset."""

        @staticmethod
        def from_preset(_name: str) -> DummyKerasModel:
            """Return a DummyKerasModel instance."""
            return DummyKerasModel()

    class DummyKerasMod:
        """Stub for keras module with AdamW and SparseCategoricalCrossentropy."""

        class optimizers:
            """Stub optimizers namespace."""

            @staticmethod
            def AdamW(**_kw: float | str) -> str:
                """Return optimizer stub."""
                return "adamw"

        class losses:
            """Stub losses namespace."""

            @staticmethod
            def SparseCategoricalCrossentropy(**_kw: bool | str) -> str:
                """Return loss stub."""
                return "loss"

    class DummyTF:
        """Stub for tensorflow module with distribute.MirroredStrategy."""

        class distribute:
            """Stub distribute namespace."""

            class MirroredStrategy:
                """Context-manager stub for MirroredStrategy."""

                def scope(self) -> AbstractContextManager[None]:
                    """Return a nullcontext scope."""
                    return nullcontext()

    monkeypatch.setattr(ktrain, "keras", DummyKerasMod)
    monkeypatch.setattr(ktrain, "tf", DummyTF)
    monkeypatch.setattr(ktrain, "GemmaCausalLM", DummyGemmaCausalLM)
    monkeypatch.setattr(ktrain, "build_dataloader", lambda _cfg: {"loader": [{"inputs": [[1, 2]], "targets": [[2, 3]]}]})  # pyright: ignore[reportUnknownLambdaType]

    res_k = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))
    assert res_k["backend"] == "keras"
    assert res_k["action"] == "pretrain"
    assert res_k["status"] == "completed"
    assert res_k["final_loss"] == pytest.approx(0.2)

    import gemma_4_sql.backends.maxtext.train as mx_train

    monkeypatch.setattr(mx_train, "jax", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="maxtext"))

    res_mlx = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="mlx"))
    assert res_mlx["backend"] == "mlx"
    assert len(mlx_double.train_configs) == 1
    assert mlx_double.train_configs[0].action == "pretrain"


def test_pretrain_model_error() -> None:
    """Verify `pretrain_model` raises `ConfigurationError` with an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="unknown"))


def test_models_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `pretrain_model`, `train_from_scratch`, `sft_model`, and `posttrain_model` populate default configs without mutating caller configs."""
    recording_double = RecordingBackendDouble("jax")
    monkeypatch.setattr(models, "get_backend", lambda _b: recording_double)  # pyright: ignore[reportUnknownLambdaType]

    res0 = pretrain_model()
    assert res0["action"] == "pretrain"
    assert recording_double.train_configs[-1].action == "pretrain"
    assert recording_double.train_configs[-1].backend == "maxtext"

    res1 = train_from_scratch()
    assert res1["action"] == "train_from_scratch"
    assert recording_double.train_configs[-1].action == "train_from_scratch"

    res2 = sft_model()
    assert res2["action"] == "sft"
    assert recording_double.train_configs[-1].action == "sft"

    res3 = posttrain_model()
    assert res3["action"] == "posttrain"
    assert recording_double.train_configs[-1].action == "posttrain"
    assert recording_double.train_configs[-1].backend == "keras"


def test_sft_and_train_from_scratch_real_tiny_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `sft_model` and `train_from_scratch` execute a real 1-step JAX training loop on a tiny Gemma 4 model."""
    import jax.numpy as jnp

    import gemma_4_sql.backends.jax.train as jax_train

    monkeypatch.setattr(
        jax_train,
        "build_dataloader",
        lambda _cfg: {  # pyright: ignore[reportUnknownLambdaType]
            "loader": [
                {
                    "inputs": jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
                    "targets": jnp.array([[2, 3, 4, 5]], dtype=jnp.int32),
                }
            ]
        },
    )
    res_sft = sft_model(
        TrainingConfig(
            action="sft",
            model_name="gemma-4-tiny",
            dataset="synthetic",
            epochs=1,
            learning_rate=1e-3,
            batch_size=1,
            backend="jax",
        )
    )
    assert res_sft["backend"] == "jax"
    assert res_sft["action"] == "sft"
    assert res_sft["status"] == "completed"
    assert res_sft["final_loss"] > 0.0

    res_scratch = train_from_scratch(
        TrainingConfig(
            action="train_from_scratch",
            model_name="gemma-4-tiny",
            dataset="synthetic",
            epochs=1,
            learning_rate=1e-3,
            batch_size=1,
            backend="jax",
        )
    )
    assert res_scratch["action"] == "train_from_scratch"
    assert res_scratch["status"] == "completed"
    assert res_scratch["final_loss"] > 0.0

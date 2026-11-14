"""Unit tests for protocol runtime checking, invariant validation, and registries in `gemma_4_sql.type_hints`."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Iterator, Mapping, Sequence

import pytest

from gemma_4_sql.backends.keras.adapter import KerasBackend
from gemma_4_sql.backends.maxtext.adapter import MaxTextBackend
from gemma_4_sql.backends.mlx.adapter import MLXBackend
from gemma_4_sql.backends.pytorch.adapter import PyTorchBackend
from gemma_4_sql.exceptions import ConfigurationError
from gemma_4_sql.sdk.protocols import BackendProtocol, InferenceProtocol, TrainingProtocol
from gemma_4_sql.tokenization import ByteFallbackTokenizer, SQLTokenizer
from gemma_4_sql.type_hints import (
    CausalLMProtocol,
    DataLoaderProtocol,
    DPOConfig,
    DPOLoopState,
    EmbedderModelProtocol,
    ETLConfig,
    FastAPIAppProtocol,
    FlaxNNXModelProtocol,
    ForwardArg,
    JaxCausalLMProtocol,
    JaxDPOState,
    JaxTrainingState,
    JSONValue,
    KerasCausalLMProtocol,
    KerasSamplerProtocol,
    KerasTrainerState,
    MaxTextCausalLMProtocol,
    MaxTextDPOState,
    MaxTextTrainerState,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    MlxTrainerState,
    NDArrayProtocol,
    QuantizedModelArtifact,
    ServeHandle,
    ServeResultDict,
    TensorProtocol,
    TokenizerProtocol,
    TorchCausalLMProtocol,
    TorchTrainerState,
    TrainingConfig,
    clear_quantized_artifact_registry,
    extract_bool_kwarg,
    extract_float_kwarg,
    extract_int_kwarg,
    extract_str_kwarg,
    get_quantized_artifact,
    register_quantized_artifact,
)
from tests.conftest import requires_jax


def test_config_post_init_invariants() -> None:
    """Verify `DPOConfig`, `ETLConfig`, and `TrainingConfig` reject invalid hyperparameters in `__post_init__`."""
    with pytest.raises(ConfigurationError, match="beta must be positive"):
        DPOConfig(model_name="m", dataset="d", beta=0.0)
    with pytest.raises(ConfigurationError, match="epochs must be non-negative"):
        DPOConfig(model_name="m", dataset="d", epochs=-1)
    with pytest.raises(ConfigurationError, match="learning_rate must be positive"):
        DPOConfig(model_name="m", dataset="d", learning_rate=0.0)
    with pytest.raises(ConfigurationError, match="batch_size must be positive"):
        DPOConfig(model_name="m", dataset="d", batch_size=0)

    with pytest.raises(ConfigurationError, match="batch_size must be positive"):
        ETLConfig(dataset_name="d", split="train", batch_size=0)

    with pytest.raises(ConfigurationError, match="epochs must be non-negative"):
        TrainingConfig(epochs=-1)
    with pytest.raises(ConfigurationError, match="learning_rate must be positive"):
        TrainingConfig(learning_rate=-1e-4)
    with pytest.raises(ConfigurationError, match="batch_size must be positive"):
        TrainingConfig(batch_size=0)


def test_type_hints_dataclasses_and_defaults() -> None:
    """Verify configuration, ServeHandle, and framework-specific state dataclasses instantiate cleanly."""
    dpo_cfg = DPOConfig(model_name="gemma-4", dataset="sql_dpo")
    assert dpo_cfg.beta == 0.1
    assert dpo_cfg.extra_kwargs == {}

    etl_cfg = ETLConfig(dataset_name="spider", split="train")
    assert etl_cfg.batch_size == 32
    assert etl_cfg.stage == "sft"

    train_cfg = TrainingConfig()
    assert train_cfg.model_name == "gemma-4"
    assert train_cfg.extra_kwargs == {}

    def _scalar_step(*_a: str | int) -> float:
        """Return 0.0 scalar loss."""
        return 0.0

    def _mt_train_step(*_a: dict[str, int] | int) -> tuple[dict[str, int], dict[str, int], float]:
        """Return updated `(params, opt_state, loss)`."""
        return ({"w": 1}, {"s": 0}, 0.0)

    def _mlx_grad_step(*_a: str | int) -> tuple[float, dict[str, float]]:
        """Return `(loss, grads)`."""
        return (0.5, {"w": 1.0})

    jax_train_state: JaxTrainingState[str, str, int] = JaxTrainingState(
        dataloader=[1, 2],
        model="m",
        optimizer="opt",
        train_step=_scalar_step,
        epochs=2,
        sharding="shard",
    )
    assert jax_train_state.epochs == 2
    assert jax_train_state.model == "m"
    assert jax_train_state.sharding == "shard"

    jax_dpo_state: JaxDPOState[str, str, int] = JaxDPOState(
        dataloader=[1],
        policy_model="pm",
        ref_model="rm",
        optimizer="opt",
        train_step=_scalar_step,
    )
    assert jax_dpo_state.beta == 0.1

    dpo_state: DPOLoopState[str, str, int] = DPOLoopState(
        dataloader=[1],
        epochs=1,
        policy_model="pm",
        ref_model="rm",
        optimizer="opt",
    )
    assert dpo_state.beta == 0.1
    assert dpo_state.policy_model == "pm"

    mt_train_state: MaxTextTrainerState[dict[str, int], dict[str, int], int] = MaxTextTrainerState(
        dataloader=[1],
        params={"w": 1},
        opt_state={"s": 0},
        train_step=_mt_train_step,
        epochs=1,
    )
    assert mt_train_state.epochs == 1

    mt_dpo_state: MaxTextDPOState[dict[str, int], dict[str, int], int] = MaxTextDPOState(
        dataloader=[1],
        policy_params={"w": 1},
        ref_params={"w": 1},
        opt_state={"s": 0},
        train_step=_mt_train_step,
        epochs=1,
    )
    assert mt_dpo_state.beta == 0.1

    torch_state: TorchTrainerState[str, str, int] = TorchTrainerState(
        dataloader=[1],
        epochs=1,
        model="m",
        optimizer="opt",
        criterion=_scalar_step,
    )
    assert torch_state.device == "cpu"

    keras_state: KerasTrainerState[str, str, int] = KerasTrainerState(
        dataloader=[1],
        epochs=1,
        train_step=_scalar_step,
        model="m",
        optimizer="opt",
    )
    assert keras_state.epochs == 1

    mlx_state: MlxTrainerState[str, str, int] = MlxTrainerState(
        dataloader=[1],
        model="m",
        optimizer="opt",
        loss_and_grad_fn=_mlx_grad_step,
        epochs=3,
    )
    assert mlx_state.epochs == 3
    assert jax_train_state.train_step(1) == 0.0
    assert mt_train_state.train_step(1) == ({"w": 1}, {"s": 0}, 0.0)
    assert mlx_state.loss_and_grad_fn(1) == (0.5, {"w": 1.0})

    serve_meta: ServeResultDict = {
        "backend": "jax",
        "model": "gemma-4-tiny",
        "port": 8000,
        "max_batch_size": 16,
        "status": "running_jax_serve",
        "status_code": "success",
        "mode": "continuous_batching",
    }
    serve_handle = ServeHandle(metadata=serve_meta, app=None)
    assert serve_handle.metadata["port"] == 8000
    assert serve_handle.app is None

    class _DummyModel:
        """Minimal weakrefable hashable model object for QuantizedModelArtifact tests."""

    dummy_model = _DummyModel()
    assert get_quantized_artifact(dummy_model) is None
    artifact_default = QuantizedModelArtifact(model=dummy_model, method="int8")
    assert artifact_default.scales == {}
    artifact = register_quantized_artifact(dummy_model, method="awq", scales={"w": 0.5})
    assert isinstance(artifact, QuantizedModelArtifact)
    assert artifact.is_quantized is True
    assert artifact.method == "awq"
    assert artifact.scales == {"w": 0.5}
    fetched = get_quantized_artifact(dummy_model)
    assert fetched is not None
    assert fetched.model is dummy_model
    assert fetched.method == "awq"
    assert fetched.scales == {"w": 0.5}

    # Verify WeakKeyDictionary cleans up automatically when an object is garbage collected
    ephemeral_model = _DummyModel()
    register_quantized_artifact(ephemeral_model, method="gptq")
    del ephemeral_model
    gc.collect()

    # Verify unhashable weakrefable MlxCausalLMProtocol models (like mlx.nn.Module subclasses)
    class _UnhashableMlxModel(list[int]):
        """Unhashable weakrefable list subclass implementing `MlxCausalLMProtocol`."""

        def __call__(self, *_a: ForwardArg, **_kw: ForwardArg) -> float:
            """Execute forward pass.

            Returns:
                Scalar output value.

            """
            return 1.0

        def parameters(self) -> dict[str, MlxArrayProtocol | TensorProtocol]:
            """Return parameters."""
            return {}

        def update(
            self,
            parameters: Mapping[
                str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]
            ],
        ) -> None:
            """Update parameters."""
            _ = parameters

        def freeze(self) -> None:
            """Freeze parameters."""

    unhashable_mlx = _UnhashableMlxModel()
    assert unhashable_mlx() == 1.0
    assert unhashable_mlx.parameters() == {}
    unhashable_mlx.update({})
    unhashable_mlx.freeze()
    register_quantized_artifact(unhashable_mlx, method="int4", scales={"l0": 0.25})
    register_quantized_artifact(unhashable_mlx, method="awq", scales={"l0": 0.125})
    other_unhashable = _UnhashableMlxModel()
    assert get_quantized_artifact(other_unhashable) is None
    fetched_mlx = get_quantized_artifact(unhashable_mlx)
    assert fetched_mlx is not None
    assert fetched_mlx.method == "awq"
    del unhashable_mlx
    gc.collect()

    class _SlottedUnhashableMlxModel(list[int]):
        """Unhashable non-weakrefable `MlxCausalLMProtocol` double (`list` with `__slots__ = ()`)."""

        __slots__ = ()

        def __call__(self, *_a: ForwardArg, **_kw: ForwardArg) -> float:
            """Execute forward pass.

            Returns:
                Scalar output value.

            """
            return 1.0

        def parameters(self) -> dict[str, MlxArrayProtocol | TensorProtocol]:
            """Return parameters."""
            return {}

        def update(
            self,
            parameters: Mapping[
                str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]
            ],
        ) -> None:
            """Update parameters."""
            _ = parameters

        def freeze(self) -> None:
            """Freeze parameters."""

    slotted_mlx = _SlottedUnhashableMlxModel()
    assert slotted_mlx() == 1.0
    assert slotted_mlx.parameters() == {}
    slotted_mlx.update({})
    slotted_mlx.freeze()
    wrapped_slotted_mlx = register_quantized_artifact(slotted_mlx, method="int8")
    assert get_quantized_artifact(slotted_mlx) is None
    assert get_quantized_artifact(wrapped_slotted_mlx) is not None

    # Non-weakrefable hashable objects (e.g., __slots__ without __weakref__) do not leak into global registries;
    # callers compose them via the returned QuantizedModelArtifact wrapper.
    class _SlottedModel:
        """Model without __weakref__ slot to exercise non-weakrefable wrapper composition."""

        __slots__ = ("weight",)

        def __init__(self) -> None:
            """Initialize slotted model."""
            self.weight = 1.0

    slotted = _SlottedModel()
    wrapped_slotted = register_quantized_artifact(slotted, method="int4")
    assert get_quantized_artifact(slotted) is None
    unwrapped = get_quantized_artifact(wrapped_slotted)
    assert unwrapped is not None
    assert unwrapped.model is slotted
    assert unwrapped.method == "int4"

    # Non-hashable non-MLX structure (e.g., MaxText PyTree dict) also composes cleanly via wrapper
    pytree: dict[str, list[float]] = {"params": [1.0, 2.0]}
    wrapped_pytree = register_quantized_artifact(pytree, method="int8")
    assert get_quantized_artifact(pytree) is None
    fetched_pytree = get_quantized_artifact(wrapped_pytree)
    assert fetched_pytree is not None
    assert fetched_pytree.model is pytree

    clear_quantized_artifact_registry()
    assert get_quantized_artifact(dummy_model) is None

    from gemma_4_sql.type_hints import KwargValue

    kw_map: dict[str, KwargValue] = {
        "i_int": 4,
        "i_str": "8",
        "i_bool": True,
        "i_bad": [1, 2],
        "f_float": 0.25,
        "f_bool": False,
        "f_bad": {"a": 1},
        "s_val": "hello",
        "b_str_true": "yes",
        "b_str_false": "no",
        "b_bool": True,
    }
    assert extract_int_kwarg(kw_map, "missing", 3) == 3
    assert extract_int_kwarg(kw_map, "i_int", 3) == 4
    assert extract_int_kwarg(kw_map, "i_str", 3) == 8
    assert extract_int_kwarg(kw_map, "i_bool", 3) == 1
    assert extract_int_kwarg(kw_map, "i_bad", 3) == 3

    assert extract_float_kwarg(kw_map, "missing", 0.5) == 0.5
    assert extract_float_kwarg(kw_map, "f_float", 0.5) == 0.25
    assert extract_float_kwarg(kw_map, "f_bool", 0.5) == 0.0
    assert extract_float_kwarg(kw_map, "f_bad", 0.5) == 0.5

    assert extract_str_kwarg(kw_map, "missing", "def") == "def"
    assert extract_str_kwarg(kw_map, "s_val", "def") == "hello"

    from gemma_4_sql.type_hints import extract_status_kwargs

    assert extract_status_kwargs(kw_map) == {
        "i_int": 4,
        "i_str": "8",
        "i_bool": True,
        "f_float": 0.25,
        "f_bool": False,
        "s_val": "hello",
        "b_str_true": "yes",
        "b_str_false": "no",
        "b_bool": True,
    }

    assert extract_bool_kwarg(kw_map, "missing", False) is False
    assert extract_bool_kwarg(kw_map, "b_str_true", False) is True
    assert extract_bool_kwarg(kw_map, "b_str_false", True) is False
    assert extract_bool_kwarg(kw_map, "b_bool", False) is True

    from gemma_4_sql.type_hints import run_coroutine_sync, shutdown_sync_coro_executor

    async def _sample_coro() -> int:
        """Return sample integer from async coroutine."""
        return 7

    shutdown_sync_coro_executor()
    assert run_coroutine_sync(_sample_coro) == 7

    async def _nested_caller() -> int:
        """Invoke run_coroutine_sync while an event loop is already running.

        Returns:
            Result from `run_coroutine_sync`.

        """
        return run_coroutine_sync(_sample_coro)

    assert asyncio.run(_nested_caller()) == 7
    assert asyncio.run(_nested_caller()) == 7
    shutdown_sync_coro_executor()
    shutdown_sync_coro_executor()


def test_runtime_checkable_protocols_with_real_components() -> None:
    """Verify runtime_checkable protocols distinguish framework models and reject non-conforming objects."""
    sql_tok = SQLTokenizer()
    assert isinstance(sql_tok, TokenizerProtocol)
    assert not isinstance(ByteFallbackTokenizer(), TokenizerProtocol)

    for adapter_cls in (PyTorchBackend, KerasBackend, MaxTextBackend, MLXBackend):
        adapter = adapter_cls()
        assert isinstance(adapter, TrainingProtocol)
        assert isinstance(adapter, InferenceProtocol)
        assert isinstance(adapter, BackendProtocol)

    class DummyLoader:
        """Conforming DataLoaderProtocol stub."""

        def __iter__(self) -> Iterator[dict[str, list[int]]]:
            """Return iterator."""
            return iter([{"inputs": [1]}])

        def __len__(self) -> int:
            """Return length."""
            return 1

    loader = DummyLoader()
    assert isinstance(loader, DataLoaderProtocol)
    assert len(loader) == 1
    assert list(loader) == [{"inputs": [1]}]

    class DummyApp:
        """Conforming FastAPIAppProtocol stub."""

        async def __call__(
            self,
            scope: Mapping[str, JSONValue],
            receive: None,
            send: None,
        ) -> None:
            """Handle ASGI scope."""
            _ = (scope, receive, send)

        def post(self, path: str, **_kwargs: str | bool | None) -> str:
            """Register POST route.

            Returns:
                Registered route path.

            """
            return path

        def get(self, path: str, **_kwargs: str | bool | None) -> str:
            """Register GET route.

            Returns:
                Registered route path.

            """
            return path

    app = DummyApp()
    assert isinstance(app, FastAPIAppProtocol)
    assert app.post("/generate") == "/generate"
    assert app.get("/health") == "/health"
    asyncio.run(app({}, None, None))

    class DummyEmbedder:
        """Conforming EmbedderModelProtocol stub."""

        def encode(self, *args: str | Sequence[str], **kwargs: int | bool | str | None) -> list[float]:
            """Return embedding vector."""
            _ = (args, kwargs)
            return [1.0, 0.0]

    embedder = DummyEmbedder()
    assert isinstance(embedder, EmbedderModelProtocol)
    assert embedder.encode("test") == [1.0, 0.0]

    class DummyMaxTextModel:
        """Conforming MaxTextCausalLMProtocol stub."""

        def init(self, *args: int | None, **kwargs: int | bool | None) -> dict[str, TensorProtocol]:
            """Return initial parameter dict."""
            _ = (args, kwargs)
            return {}

        def apply(self, *args: int | None, **kwargs: int | bool | None) -> float:
            """Apply forward pass.

            Returns:
                Scalar forward pass output.

            """
            _ = (args, kwargs)
            return 0.0

    mt_model = DummyMaxTextModel()
    assert isinstance(mt_model, MaxTextCausalLMProtocol)
    assert mt_model.init() == {}
    assert mt_model.apply() == 0.0

    class DummyFlaxTrunk:
        """Conforming FlaxNNXModelProtocol stub without `.model` attribute."""

        config = None

        def __call__(self, *args: int | None, **kwargs: int | None) -> float:
            """Run forward pass.

            Returns:
                Scalar forward pass output.

            """
            _ = (args, kwargs)
            return 0.5

    trunk = DummyFlaxTrunk()
    assert isinstance(trunk, FlaxNNXModelProtocol)
    assert not isinstance(trunk, JaxCausalLMProtocol)
    assert trunk() == 0.5

    class DummyJaxModel:
        """Conforming JaxCausalLMProtocol stub."""

        config = None
        model = None

        def __call__(self, *args: int | None, **kwargs: int | None) -> float:
            """Run forward pass.

            Returns:
                Scalar forward pass output.

            """
            _ = (args, kwargs)
            return 1.0

    jax_model = DummyJaxModel()
    assert isinstance(jax_model, FlaxNNXModelProtocol)
    assert isinstance(jax_model, JaxCausalLMProtocol)
    assert jax_model() == 1.0

    class DummyMlxModel:
        """Conforming MlxCausalLMProtocol stub."""

        def __call__(self, *args: int | None, **kwargs: int | None) -> float:
            """Run forward pass.

            Returns:
                Scalar forward pass output.

            """
            _ = (args, kwargs)
            return 1.0

        def parameters(self) -> dict[str, TensorProtocol]:
            """Return parameter dictionary."""
            return {}

        def update(self, parameters: dict[str, TensorProtocol]) -> None:
            """Update parameters."""
            _ = parameters

        def freeze(self) -> None:
            """Freeze parameters."""

    mlx_model = DummyMlxModel()
    assert isinstance(mlx_model, MlxCausalLMProtocol)
    assert mlx_model() == 1.0
    assert mlx_model.parameters() == {}
    mlx_model.update({})
    mlx_model.freeze()

    non_conforming = 42
    for proto in (
        TokenizerProtocol,
        DataLoaderProtocol,
        FastAPIAppProtocol,
        EmbedderModelProtocol,
        MaxTextCausalLMProtocol,
        FlaxNNXModelProtocol,
        JaxCausalLMProtocol,
        MlxCausalLMProtocol,
        KerasSamplerProtocol,
        NDArrayProtocol,
        TensorProtocol,
        CausalLMProtocol,
        TorchCausalLMProtocol,
        KerasCausalLMProtocol,
    ):
        assert not isinstance(non_conforming, proto)


@requires_jax
def test_jax_runtime_checkable_protocols() -> None:
    """Verify JAX arrays and JaxBackend satisfy runtime-checkable protocols when JAX is installed."""
    from gemma_4_sql.backends.jax.adapter import JaxBackend
    from gemma_4_sql.backends.jax.gemma4._jnp import jnp

    arr = jnp.zeros((2, 3), dtype=jnp.float32)
    assert isinstance(arr, NDArrayProtocol)
    assert isinstance(arr, TensorProtocol)
    jax_adapter = JaxBackend()
    assert isinstance(jax_adapter, TrainingProtocol)
    assert isinstance(jax_adapter, InferenceProtocol)
    assert isinstance(jax_adapter, BackendProtocol)

    from dataclasses import dataclass

    from gemma_4_sql.type_hints import get_quantized_artifact, register_quantized_artifact

    @dataclass
    class _UnhashableWeightHolder:
        """Mutable dataclass (`__hash__ = None`) satisfying `_WeightAttrTargetProtocol`."""

        weight: tuple[float, ...] = (1.0, 2.0)

    unhashable_holder = _UnhashableWeightHolder()
    art_unhashable = register_quantized_artifact(unhashable_holder, method="int8")
    assert art_unhashable.model is unhashable_holder
    assert get_quantized_artifact(unhashable_holder) is None

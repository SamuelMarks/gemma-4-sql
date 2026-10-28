"""Unit tests for protocol runtime checking and type definitions in `gemma_4_sql.type_hints`."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Iterator, Mapping, Sequence

import jax.numpy as jnp

from gemma_4_sql.backends.jax.adapter import JaxBackend
from gemma_4_sql.backends.keras.adapter import KerasBackend
from gemma_4_sql.backends.maxtext.adapter import MaxTextBackend
from gemma_4_sql.backends.mlx.adapter import MLXBackend
from gemma_4_sql.backends.pytorch.adapter import PyTorchBackend
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
    JaxCausalLMProtocol,
    JSONValue,
    KerasCausalLMProtocol,
    KerasModelProtocol,
    KerasTrainerState,
    MaxTextCausalLMProtocol,
    MaxTextModelProtocol,
    MaxTextTrainerState,
    MlxCausalLMProtocol,
    MLXModelProtocol,
    NDArrayProtocol,
    QuantizableModelProtocol,
    QuantizedModelArtifact,
    TensorProtocol,
    TokenizerProtocol,
    TorchCausalLMProtocol,
    TorchModelProtocol,
    TorchTrainerState,
    TrainableModelProtocol,
    TrainerState,
    TrainingConfig,
    TrainingLoopState,
    clear_quantized_artifact_registry,
    extract_bool_kwarg,
    extract_float_kwarg,
    extract_int_kwarg,
    extract_str_kwarg,
    get_quantized_artifact,
    register_quantized_artifact,
)


def test_type_hints_dataclasses_and_defaults() -> None:
    """Verify configuration and state dataclasses instantiate with expected defaults."""
    dpo_cfg = DPOConfig(model_name="gemma-4", dataset="sql_dpo")
    assert dpo_cfg.beta == 0.1
    assert dpo_cfg.extra_kwargs == {}

    etl_cfg = ETLConfig(dataset_name="spider", split="train")
    assert etl_cfg.batch_size == 32
    assert etl_cfg.stage == "sft"

    train_cfg = TrainingConfig()
    assert train_cfg.model_name == "gemma-4"
    assert train_cfg.extra_kwargs == {}

    loop_state: TrainingLoopState[str, str, int] = TrainingLoopState(
        dataloader=[1, 2],
        epochs=2,
        model="m",
        optimizer="opt",
        sharding="shard",
    )
    assert loop_state.epochs == 2
    assert loop_state.model == "m"
    assert loop_state.sharding == "shard"

    dpo_state: DPOLoopState[str, str, int] = DPOLoopState(
        dataloader=[1],
        epochs=1,
        policy_model="pm",
        ref_model="rm",
        optimizer="opt",
    )
    assert dpo_state.beta == 0.1
    assert dpo_state.policy_model == "pm"

    mt_state: MaxTextTrainerState[dict[str, int], dict[str, int], int] = MaxTextTrainerState(
        dataloader=[1],
        epochs=1,
        train_step=lambda *_a: ({"w": 1}, {"s": 0}, 0.0),
        policy_params={"w": 1},
        ref_params={"w": 1},
        opt_state={"s": 0},
    )
    assert mt_state.epochs == 1

    torch_state: TorchTrainerState[str, str, int] = TorchTrainerState(
        dataloader=[1],
        epochs=1,
        model="m",
        optimizer="opt",
        criterion=lambda *_a: 0.0,
    )
    assert torch_state.device == "cpu"

    keras_state: KerasTrainerState[str, str, int] = KerasTrainerState(
        dataloader=[1],
        epochs=1,
        train_step=lambda *a: a,
        model="m",
        optimizer="opt",
    )
    assert keras_state.epochs == 1

    unified: TrainerState[str, dict[str, int], str, int] = TrainerState(
        dataloader=[1],
        epochs=3,
        beta=0.2,
        loss_and_grad_fn=lambda *_a: (0.5, {"w": 1}),
    )
    assert unified.epochs == 3
    assert unified.beta == 0.2
    assert unified.loss_and_grad_fn is not None

    class _DummyModel:
        """Minimal model object for QuantizedModelArtifact registration tests."""

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

    # Verify weakref finalizer cleans up when an object is garbage collected
    ephemeral_model = _DummyModel()
    register_quantized_artifact(ephemeral_model, method="gptq")
    del ephemeral_model
    gc.collect()

    class _SlottedModel:
        """Model without __weakref__ slot to exercise fallback registry."""

        __slots__ = ("weight",)

        def __init__(self) -> None:
            """Initialize slotted model."""
            self.weight = 1.0

    slotted = _SlottedModel()
    register_quantized_artifact(slotted, method="int4")
    fetched_slotted = get_quantized_artifact(slotted)
    assert fetched_slotted is not None
    assert fetched_slotted.method == "int4"
    clear_quantized_artifact_registry()
    assert get_quantized_artifact(dummy_model) is None
    assert get_quantized_artifact(slotted) is None

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

    assert extract_bool_kwarg(kw_map, "missing", False) is False
    assert extract_bool_kwarg(kw_map, "b_str_true", False) is True
    assert extract_bool_kwarg(kw_map, "b_str_false", True) is False
    assert extract_bool_kwarg(kw_map, "b_bool", False) is True


def test_runtime_checkable_protocols_with_real_components() -> None:
    """Verify runtime_checkable protocols match real domain classes and reject non-conforming objects."""
    # Real tokenizers
    sql_tok = SQLTokenizer()
    assert isinstance(sql_tok, TokenizerProtocol)
    assert not isinstance(ByteFallbackTokenizer(), TokenizerProtocol)

    # Real JAX array satisfies NDArrayProtocol and TensorProtocol
    arr = jnp.zeros((2, 3), dtype=jnp.float32)
    assert isinstance(arr, NDArrayProtocol)
    assert isinstance(arr, TensorProtocol)

    # Real backend adapters satisfy TrainingProtocol, InferenceProtocol, and BackendProtocol
    for adapter_cls in (JaxBackend, PyTorchBackend, KerasBackend, MaxTextBackend, MLXBackend):
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
            """Register POST route."""
            return path

        def get(self, path: str, **_kwargs: str | bool | None) -> str:
            """Register GET route."""
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
            """Apply forward pass."""
            _ = (args, kwargs)
            return 0.0

    mt_model = DummyMaxTextModel()
    assert isinstance(mt_model, (MaxTextModelProtocol, MaxTextCausalLMProtocol))
    assert mt_model.init() == {}
    assert mt_model.apply() == 0.0

    class DummyJaxModel:
        """Conforming JaxCausalLMProtocol stub."""

        config = None
        model = None
        lm_head = None

        def __call__(self, *args: int | None, **kwargs: int | None) -> float:
            """Run forward pass."""
            _ = (args, kwargs)
            return 1.0

    jax_model = DummyJaxModel()
    assert isinstance(jax_model, (FlaxNNXModelProtocol, JaxCausalLMProtocol))
    assert jax_model() == 1.0

    class DummyMlxModel:
        """Conforming MlxCausalLMProtocol stub."""

        def __call__(self, *args: int | None, **kwargs: int | None) -> float:
            """Run forward pass."""
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
    assert isinstance(mlx_model, (MLXModelProtocol, MlxCausalLMProtocol))
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
        JaxCausalLMProtocol,
        MlxCausalLMProtocol,
        NDArrayProtocol,
        TensorProtocol,
        CausalLMProtocol,
        TorchModelProtocol,
        TorchCausalLMProtocol,
        KerasModelProtocol,
        KerasCausalLMProtocol,
        TrainableModelProtocol,
        QuantizableModelProtocol,
    ):
        assert not isinstance(non_conforming, proto)

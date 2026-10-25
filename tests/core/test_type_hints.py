"""Unit tests for protocol runtime checking and type definitions in `gemma_4_sql.type_hints`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
        train_step=lambda *a: ({"w": 1}, {"s": 0}, 0.0),
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
        criterion=lambda *a: 0.0,
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

    unified: TrainerState[str, str, str, int] = TrainerState(dataloader=[1], epochs=3, beta=0.2)
    assert unified.epochs == 3
    assert unified.beta == 0.2

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

    class _SlottedModel:
        """Model without __weakref__ slot to exercise fallback registry."""

        __slots__ = ("weight",)

        def __init__(self) -> None:
            """Initialize slotted model."""
            self.weight = 1.0

    slotted = _SlottedModel()
    slotted_art = register_quantized_artifact(slotted, method="int4")
    assert get_quantized_artifact(slotted) is slotted_art
    clear_quantized_artifact_registry()
    assert get_quantized_artifact(dummy_model) is None
    assert get_quantized_artifact(slotted) is None


def test_runtime_checkable_protocols() -> None:
    """Verify runtime_checkable protocols match conforming structural implementations and reject non-conforming objects."""

    class DummyTokenizer:
        """Conforming TokenizerProtocol stub."""

        def encode(self, text: str) -> list[int]:
            """Encode text."""
            return [len(text)]

        def decode(self, token_ids: Sequence[int]) -> str:
            """Decode tokens."""
            return str(list(token_ids))

    tok = DummyTokenizer()
    assert isinstance(tok, TokenizerProtocol)
    assert not isinstance(object(), TokenizerProtocol)
    assert tok.encode("abc") == [3]
    assert tok.decode([1, 2]) == "[1, 2]"

    class DummyLoader:
        """Conforming DataLoaderProtocol stub."""

        def __iter__(self) -> Any:
            """Return iterator."""
            return iter([{"inputs": [1]}])

        def __len__(self) -> int:
            """Return length."""
            return 1

    loader = DummyLoader()
    assert isinstance(loader, DataLoaderProtocol)
    assert not isinstance(object(), DataLoaderProtocol)
    assert len(loader) == 1
    assert list(loader) == [{"inputs": [1]}]

    class DummyApp:
        """Conforming FastAPIAppProtocol stub."""

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            """Handle ASGI scope."""
            _ = (scope, receive, send)

        def post(self, path: str, **kwargs: object) -> Any:
            """Register POST route."""
            return lambda fn: fn

        def get(self, path: str, **kwargs: object) -> Any:
            """Register GET route."""
            return lambda fn: fn

    app = DummyApp()
    assert isinstance(app, FastAPIAppProtocol)
    assert not isinstance(object(), FastAPIAppProtocol)
    assert callable(app.post("/generate"))
    assert callable(app.get("/health"))
    import asyncio

    asyncio.run(app({}, None, None))

    class DummyEmbedder:
        """Conforming EmbedderModelProtocol stub."""

        def encode(self, *args: object, **kwargs: object) -> list[float]:
            """Return embedding vector."""
            _ = (args, kwargs)
            return [1.0, 0.0]

    embedder = DummyEmbedder()
    assert isinstance(embedder, EmbedderModelProtocol)
    assert not isinstance(object(), EmbedderModelProtocol)
    assert embedder.encode("test") == [1.0, 0.0]

    class DummyMaxTextModel:
        """Conforming MaxTextCausalLMProtocol stub."""

        def init(self, *args: object, **kwargs: object) -> dict[str, Any]:
            """Return initial parameter dict."""
            _ = (args, kwargs)
            return {}

        def apply(self, *args: object, **kwargs: object) -> float:
            """Apply forward pass."""
            _ = (args, kwargs)
            return 0.0

    mt_model = DummyMaxTextModel()
    assert isinstance(mt_model, (MaxTextModelProtocol, MaxTextCausalLMProtocol))
    assert not isinstance(object(), MaxTextCausalLMProtocol)
    assert mt_model.init() == {}
    assert mt_model.apply() == 0.0

    class DummyJaxModel:
        """Conforming JaxCausalLMProtocol stub."""

        config = None
        model = None
        lm_head = None

        def __call__(self, *args: object, **kwargs: object) -> float:
            """Run forward pass."""
            _ = (args, kwargs)
            return 1.0

    jax_model = DummyJaxModel()
    assert isinstance(jax_model, (FlaxNNXModelProtocol, JaxCausalLMProtocol))
    assert not isinstance(object(), JaxCausalLMProtocol)
    assert jax_model() == 1.0

    class DummyMlxModel:
        """Conforming MlxCausalLMProtocol stub."""

        def __call__(self, *args: object, **kwargs: object) -> float:
            """Run forward pass."""
            _ = (args, kwargs)
            return 1.0

        def parameters(self) -> dict[str, Any]:
            """Return parameter dictionary."""
            return {}

        def update(self, parameters: dict[str, Any]) -> None:
            """Update parameters."""
            _ = parameters

        def freeze(self) -> None:
            """Freeze parameters."""

    mlx_model = DummyMlxModel()
    assert isinstance(mlx_model, (MLXModelProtocol, MlxCausalLMProtocol))
    assert not isinstance(object(), MlxCausalLMProtocol)
    assert mlx_model() == 1.0
    assert mlx_model.parameters() == {}
    mlx_model.update({})
    mlx_model.freeze()

    for proto in (
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
        assert not isinstance(object(), proto)

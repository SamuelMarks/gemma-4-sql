"""Unit tests for protocol runtime checking and type definitions in `gemma_4_sql.type_hints`."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

from gemma_4_sql.type_hints import (
    CausalLMProtocol,
    DPOLoopState,
    DPOTrajectoryState,
    FlaxNNXModelProtocol,
    KerasModelProtocol,
    KerasTrainerState,
    MaxTextModelProtocol,
    MaxTextTrainerState,
    MLXModelProtocol,
    NDArrayProtocol,
    TensorType,
    TokenizerProtocol,
    TorchModelProtocol,
    TorchTrainerState,
    TrainingLoopState,
)


def test_type_hints_protocols() -> None:
    """Verify runtime protocol checks against conforming and non-conforming implementations."""

    class ConformingArray:
        """Minimal array implementation satisfying NDArrayProtocol."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return array shape."""
            return (2, 2)

        @property
        def dtype(self) -> object:
            """Return array dtype."""
            return "float32"

        def tolist(self) -> list[object]:
            """Return nested list."""
            return [[1.0, 2.0], [3.0, 4.0]]

        def reshape(self, *_shape: int) -> ConformingArray:
            """Return reshaped array."""
            return self

        def transpose(self, *_axes: int) -> ConformingArray:
            """Return transposed array."""
            return self

        def astype(self, _dtype: object) -> ConformingArray:
            """Return cast array."""
            return self

        def flatten(self) -> ConformingArray:
            """Return flattened array."""
            return self

        def __array__(self) -> object:
            """Return self as array."""
            return self

    class ConformingTensor:
        """Minimal tensor implementation satisfying TensorType."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return tensor shape."""
            return (1, 2)

        @property
        def ndim(self) -> int:
            """Return tensor rank."""
            return 2

        @property
        def dtype(self) -> object:
            """Return tensor dtype."""
            return "float32"

        def reshape(self, *_shape: int) -> ConformingTensor:
            """Return reshaped tensor."""
            return self

        def transpose(self, *_axes: int) -> ConformingTensor:
            """Return transposed tensor."""
            return self

        def astype(self, _dtype: object) -> ConformingTensor:
            """Return cast tensor."""
            return self

        def mean(self, *_args: object, **_kwargs: object) -> ConformingTensor:
            """Return mean tensor."""
            return self

        def item(self) -> float:
            """Return scalar float."""
            return 1.0

        def __getitem__(self, _item: object, /) -> ConformingTensor:
            """Slice tensor."""
            return self

        def __sub__(self, _other: object) -> ConformingTensor:
            """Subtract tensor."""
            return self

        def __add__(self, _other: object) -> ConformingTensor:
            """Add tensor."""
            return self

        def __mul__(self, _other: object) -> ConformingTensor:
            """Multiply tensor."""
            return self

        def __rmul__(self, _other: object) -> ConformingTensor:
            """Right-multiply tensor."""
            return self

        def __neg__(self) -> ConformingTensor:
            """Negate tensor."""
            return self

    class ConformingTokenizer:
        """Minimal tokenizer satisfying TokenizerProtocol."""

        def encode(self, text: str) -> list[int]:
            """Encode string into byte ordinals."""
            return [ord(c) for c in text]

        def decode(self, token_ids: Sequence[int]) -> str:
            """Decode ordinals into string."""
            return "".join(chr(i) for i in token_ids)

    class ConformingTorchModel:
        """Minimal PyTorch model satisfying TorchModelProtocol and CausalLMProtocol."""

        def __call__(self, *_a: object, **_kw: object) -> Any:
            """Forward pass."""
            return None

        def generate(self, *_a: object, **_kw: object) -> Any:
            """Generate tokens."""
            return [1, 2]

        def parameters(self) -> list[Any]:
            """Return parameters."""
            return []

        def named_modules(self) -> list[tuple[str, Any]]:
            """Return named modules."""
            return []

        def train(self, mode: bool = True) -> ConformingTorchModel:
            """Set training mode."""
            _ = mode
            return self

        def to(self, *_a: object, **_kw: object) -> ConformingTorchModel:
            """Move to device."""
            return self

        def state_dict(self) -> dict[str, Any]:
            """Return state dict."""
            return {}

    class ConformingKerasModel:
        """Minimal Keras model satisfying KerasModelProtocol."""

        def __call__(self, *_a: object, **_kw: object) -> Any:
            """Forward pass."""
            return None

        def generate(self, *_a: object, **_kw: object) -> Any:
            """Generate text."""
            return "SELECT 1"

        def compile(self, *_a: object, **_kw: object) -> None:
            """Compile model."""

        def fit(self, *_a: object, **_kw: object) -> Any:
            """Fit model."""
            return None

        @property
        def layers(self) -> list[Any]:
            """Return layers."""
            return []

        def save(self, *_a: object, **_kw: object) -> None:
            """Save model."""

        def load_weights(self, *_a: object, **_kw: object) -> None:
            """Load weights."""

    class ConformingFlaxNNXModel:
        """Minimal Flax NNX model satisfying FlaxNNXModelProtocol."""

        def __call__(self, *_a: object, **_kw: object) -> Any:
            """Forward pass."""
            return None

        @property
        def config(self) -> dict[str, Any]:
            """Return config."""
            return {}

        @property
        def model(self) -> object:
            """Return inner model."""
            return object()

        @property
        def lm_head(self) -> object:
            """Return language modeling head."""
            return object()

    class ConformingMaxTextModel:
        """Minimal MaxText model satisfying MaxTextModelProtocol."""

        def init(self, *_a: object, **_kw: object) -> dict[str, Any]:
            """Initialize parameters."""
            return {}

        def apply(self, *_a: object, **_kw: object) -> Any:
            """Apply model."""
            return None

    class ConformingMLXModel:
        """Minimal MLX model satisfying MLXModelProtocol."""

        def __call__(self, *_a: object, **_kw: object) -> Any:
            """Forward pass."""
            return None

        def parameters(self) -> dict[str, Any]:
            """Return parameter tree."""
            return {}

        def update(self, _parameters: Any) -> None:
            """Update parameter tree."""

    arr = ConformingArray()
    assert isinstance(arr, NDArrayProtocol)
    assert arr.shape == (2, 2)
    assert arr.dtype == "float32"
    assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert arr.reshape(4) is arr
    assert arr.transpose(1, 0) is arr
    assert arr.astype("int32") is arr
    assert arr.flatten() is arr
    assert arr.__array__() is arr
    assert not isinstance(object(), NDArrayProtocol)

    t = ConformingTensor()
    assert isinstance(t, TensorType)
    assert t.shape == (1, 2)
    assert t.ndim == 2
    assert t.dtype == "float32"
    assert t.reshape(2, 1) is t
    assert t.transpose(1, 0) is t
    assert t.astype("float16") is t
    assert t.mean() is t
    assert t.item() == 1.0
    assert t[0] is t
    assert (t - 1) is t
    assert (t + 1) is t
    assert (t * 2) is t
    assert (2 * t) is t
    assert (-t) is t
    assert not isinstance(object(), TensorType)

    tok = ConformingTokenizer()
    assert isinstance(tok, TokenizerProtocol)
    assert tok.decode(tok.encode("SQL")) == "SQL"
    assert not isinstance(object(), TokenizerProtocol)

    pt_model = ConformingTorchModel()
    assert isinstance(pt_model, CausalLMProtocol)
    assert isinstance(pt_model, TorchModelProtocol)
    assert not isinstance(object(), TorchModelProtocol)

    k_model = ConformingKerasModel()
    assert isinstance(k_model, KerasModelProtocol)
    assert not isinstance(object(), KerasModelProtocol)

    nnx_model = ConformingFlaxNNXModel()
    assert isinstance(nnx_model, FlaxNNXModelProtocol)
    assert not isinstance(object(), FlaxNNXModelProtocol)

    mt_model = ConformingMaxTextModel()
    assert isinstance(mt_model, MaxTextModelProtocol)
    assert not isinstance(object(), MaxTextModelProtocol)

    mlx_model = ConformingMLXModel()
    assert isinstance(mlx_model, MLXModelProtocol)
    assert not isinstance(object(), MLXModelProtocol)

    # Verify typed trainer state dataclasses
    loop_state = TrainingLoopState(dataloader=[{"inputs": [1]}], epochs=2, model=pt_model)
    assert loop_state.epochs == 2
    dpo_state = DPOLoopState(policy_model=pt_model, ref_model=pt_model, dataloader=[], beta=0.2)
    assert dpo_state.beta == 0.2
    traj_state = DPOTrajectoryState(dataloader=[], epochs=3, beta=0.1)
    assert traj_state.epochs == 3
    mt_state = MaxTextTrainerState(dataloader=[], epochs=1, params={"w": 1})
    assert mt_state.params == {"w": 1}
    torch_state = TorchTrainerState(dataloader=[], model=pt_model, device="cpu")
    assert torch_state.device == "cpu"
    keras_state = KerasTrainerState(dataloader=[], model=k_model, epochs=4)
    assert keras_state.epochs == 4


def test_numpy_fallback() -> None:
    """Verify `type_hints.ndarray` falls back to `object` when `numpy` is unavailable."""
    import gemma_4_sql.type_hints as th

    with patch.dict(sys.modules, {"numpy": None}):
        importlib.reload(th)
        assert th.ndarray is object
    importlib.reload(th)

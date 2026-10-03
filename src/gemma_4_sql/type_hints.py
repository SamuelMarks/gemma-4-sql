"""Custom type hints for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypedDict, Union

__all__ = [
    "AudioInput",
    "DPOConfig",
    "ETLConfig",
    "ImageInput",
    "JSONDict",
    "JSONPrimitive",
    "JSONValue",
    "ModelType",
    "MultimodalInput",
    "TensorType",
    "TrainerState",
    "TrainingConfig",
]

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
JSONDict = dict[str, JSONValue]


class TensorType(Protocol):
    """Protocol defining the required interface for a tensor object."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor."""
        ...

    @property
    def dtype(self) -> Any:
        """Provide the data type of the tensor."""
        ...

    def reshape(self, *args: Any, **kwargs: Any) -> TensorType:
        """Reshape the tensor."""
        ...

    def transpose(self, *args: Any, **kwargs: Any) -> TensorType:
        """Transpose the tensor."""
        ...

    def astype(self, *args: Any, **kwargs: Any) -> TensorType:
        """Cast the tensor to a different data type."""
        ...

    def __getitem__(self, item: Any) -> TensorType:
        """Provide an item from the tensor."""
        ...

    def __sub__(self, other: Any) -> TensorType:
        """Subtract another tensor from this tensor."""
        ...

    def __neg__(self) -> TensorType:
        """Negate the tensor."""
        ...


class ModelType(Protocol):
    """Protocol defining the required interface for a model object."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the model."""
        ...

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate output from the model."""
        ...

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Compile the model."""
        ...

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Train the model."""
        ...

    def parameters(self) -> Any:
        """Provide the model's parameters."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Set the model's training mode."""
        ...

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Apply a function to the model."""
        ...

    def load_weights(self, *args: Any, **kwargs: Any) -> Any:
        """Load weights into the model."""
        ...

    def freeze(self) -> Any:
        """Freeze the model's parameters."""
        ...

    def named_modules(self) -> Any:
        """Provide the model's named modules."""
        ...

    @property
    def sampler(self) -> Any:
        """Provide the model's sampler."""
        ...

    @property
    def preprocessor(self) -> Any:
        """Provide the model's preprocessor."""
        ...

    @property
    def model(self) -> Any:
        """Provide the underlying model object."""
        ...

    @property
    def _quant_scales(self) -> Any:
        """Provide the quantization scales."""
        ...

    @property
    def _is_quantized(self) -> bool:
        """Provide whether the model is quantized."""
        ...

    @property
    def _quant_method(self) -> str:
        """Provide the quantization method used."""
        ...

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Encode the input."""
        ...


if TYPE_CHECKING:
    from numpy import ndarray

else:
    try:
        from numpy import ndarray
    except (ImportError, AttributeError):
        ndarray = object

# Multimodal input types
ImageInput = Union[str, Path, bytes, object, None]
AudioInput = Union[str, Path, bytes, object, None]


class MultimodalInput(TypedDict, total=False):
    """Structured multimodal input containing prompt and optional media modalities.

    Attributes:
        prompt: Natural language query or textual instruction.
        image: Image input path, binary bytes, or tensor representation.
        audio: Audio input path, binary bytes, or tensor representation.
        image_token_mask: Optional boolean alignment mask indicating image tokens.
        audio_token_mask: Optional boolean alignment mask indicating audio tokens.
        modality: Explicit modality selector ('text', 'vision', 'audio', 'multimodal').

    """

    prompt: str
    image: ImageInput
    audio: AudioInput
    image_token_mask: list[bool] | None
    audio_token_mask: list[bool] | None
    modality: str


@dataclass
class DPOConfig:
    """Config for DPO execution."""

    model_name: str
    dataset: str
    beta: float = 0.1
    epochs: int = 1
    learning_rate: float = 1e-05
    batch_size: int = 2


@dataclass
class ETLConfig:
    """Config for ETL execution."""

    dataset_name: str
    split: str
    batch_size: int = 32
    distributed: bool = False
    tokenizer_name: str | None = None
    duckdb_path: str | None = None
    duckdb_table: str | None = None
    modality: str = "text"
    image_column: str | None = None
    audio_column: str | None = None


@dataclass
class TrainingConfig:
    """Config for training execution."""

    action: str = ""
    model_name: str = "gemma-4"
    dataset: str = "dummy"
    epochs: int = 1
    learning_rate: float = 0.0001
    batch_size: int = 2
    backend: str = "jax"
    distributed_strategy: str = "none"
    modality: str = "text"
    extra_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class TrainerState:
    """State config for training loops."""

    dataloader: Iterable[object] | None = None
    epochs: int = 1
    train_step: Callable[..., Any] | None = None
    params: object = None
    opt_state: object = None
    policy_params: object = None
    ref_params: object = None
    policy_model: object = None
    ref_model: object = None
    optimizer: object = None
    criterion: Callable[..., Any] | None = None
    device: str | object | None = None
    dummy_batch: dict[str, Any] | None = None
    beta: float = 0.1
    dataset: str = ""
    learning_rate: float = 0.0
    extra_kwargs: dict[str, object] | None = None

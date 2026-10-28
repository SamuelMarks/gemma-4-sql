"""Strongly-typed protocols, TypedDicts, and configuration dataclasses for gemma-4-sql."""

from __future__ import annotations

import types
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Generic,
    NotRequired,
    Protocol,
    Self,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    TypedDict,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

from gemma_4_sql._types_extractors import (
    KwargScalar,
    KwargValue,
    TResult,
    extract_bool_kwarg,
    extract_float_kwarg,
    extract_int_kwarg,
    extract_str_kwarg,
    run_coroutine_sync,
)

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
JSONMapping = Mapping[str, JSONValue]
JSONDict = dict[str, JSONValue]

TensorIndex = Union[
    int,
    slice,
    types.EllipsisType,
    None,
    Sequence[int],
    Sequence[bool],
    "TensorProtocol",
    "NDArrayProtocol",
    tuple[
        Union[int, slice, types.EllipsisType, Sequence[int], Sequence[bool], "TensorProtocol", "NDArrayProtocol", None],
        ...,
    ],
]

type NestedScalarList = list[float] | list[int] | list[bool] | list[list[float]] | list[list[int]] | list[list[bool]] | list[list[list[float]]] | list[list[list[int]]] | list[list[list[list[float]]]]
type DTypeSpec = type[int | float | bool] | str | None


@runtime_checkable
class DTypeProtocol(Protocol):
    """Protocol representing a backend tensor data type descriptor."""

    def __str__(self) -> str:
        """Return the string representation of the data type."""
        ...


@runtime_checkable
class NDArrayProtocol(Protocol):
    """Protocol representing an n-dimensional backend array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the array."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the element data type of the array."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the array into nested Python lists."""
        ...

    def reshape(self, *shape: int | tuple[int, ...]) -> NDArrayProtocol:
        """Reshape the array without changing its data."""
        ...

    def transpose(self, *axes: int | tuple[int, ...]) -> NDArrayProtocol:
        """Permute the dimensions of the array."""
        ...

    def astype(self, dtype: DTypeProtocol | DTypeSpec) -> NDArrayProtocol:
        """Return a copy of the array cast to a specified type."""
        ...

    def flatten(self) -> NDArrayProtocol:
        """Return a copy of the array collapsed into one dimension."""
        ...

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        ...

    def __getitem__(self, item: int | slice | tuple[int | slice, ...], /) -> NDArrayProtocol:
        """Slice or index into the array."""
        ...

    def __setitem__(
        self,
        item: int | slice | tuple[int | slice, ...],
        value: NDArrayProtocol | float | bool | Sequence[float],
        /,
    ) -> None:
        """Assign values to a slice or index of the array."""
        ...

    def __sub__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Subtract an array or scalar from this array."""
        ...

    def __rsub__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-subtract this array from an array or scalar."""
        ...

    def __add__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Add an array or scalar to this array."""
        ...

    def __radd__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-add this array to an array or scalar."""
        ...

    def __mul__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Multiply this array by an array or scalar."""
        ...

    def __rmul__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-multiply this array by an array or scalar."""
        ...

    def __truediv__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Divide this array by an array or scalar."""
        ...

    def __rtruediv__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-divide an array or scalar by this array."""
        ...

    def __neg__(self) -> NDArrayProtocol:
        """Negate the array elementwise."""
        ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol representing a text/SQL tokenizer."""

    @overload
    def encode(self, text: str) -> list[int]:  # noqa: D418
        """Encode a single string into a list of integer token IDs."""

    @overload
    def encode(self, text: list[str] | tuple[str, ...]) -> list[list[int]]:  # noqa: D418
        """Encode a sequence of strings into a list of integer token ID lists."""

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        """Encode a text string or sequence of strings into integer token IDs.

        Args:
            text: Input text string or sequence of strings to tokenize.

        Returns:
            List of integer token IDs or list of token ID lists.

        """
        ...

    def decode(self, token_ids: Sequence[int], /) -> str:
        """Decode a sequence of integer token IDs back into a string.

        Args:
            token_ids: Sequence of integer token IDs.

        Returns:
            Decoded text string.

        """
        ...


@runtime_checkable
class HFTokenizerProtocol(Protocol):
    """Protocol representing a Hugging Face PreTrainedTokenizer instance."""

    @property
    def pad_token_id(self) -> int | None:
        """Provide the padding token ID."""
        ...

    @property
    def eos_token_id(self) -> int | None:
        """Provide the end-of-sequence token ID."""
        ...

    @property
    def bos_token_id(self) -> int | None:
        """Provide the beginning-of-sequence token ID."""
        ...

    def encode(self, text: str, **kwargs: bool | int | str | None) -> Sequence[int]:
        """Encode a text string into a sequence of integer token IDs.

        Args:
            text: Input text string to tokenize.
            **kwargs: Optional keyword arguments (such as `add_special_tokens`).

        Returns:
            Sequence of integer token IDs.

        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of integer token IDs back into a string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.

        """
        ...

    def get_vocab(self) -> Mapping[str, int]:
        """Return the vocabulary mapping token strings to integer IDs."""
        ...


@runtime_checkable
class TensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol defining the interface for framework tensor objects (JAX, PyTorch, TF, MLX)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor."""
        ...

    def reshape(self, *shape: int) -> Self:
        """Reshape the tensor to the given dimensions."""
        ...

    def transpose(self, ax0: int, ax1: int, /) -> Self:
        """Transpose two tensor axes."""
        ...

    def mean(self) -> Self:
        """Compute the arithmetic mean of the tensor elements."""
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element tensor."""
        ...

    def __float__(self) -> float:
        """Convert single-element tensor to Python float."""
        ...

    def __index__(self) -> int:
        """Convert scalar integer tensor to Python index."""
        ...

    def __getitem__(
        self,
        item: int | slice | types.EllipsisType | tuple[int | slice | types.EllipsisType, ...],
        /,
    ) -> Self:
        """Slice or index into the tensor."""
        ...

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor."""
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar."""
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add another tensor or scalar to this tensor."""
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar."""
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar."""
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor."""
        ...

    def __truediv__(self, other: Self | float, /) -> Self:
        """Divide this tensor by another tensor or scalar."""
        ...

    def __rtruediv__(self, other: Self | float, /) -> Self:
        """Right-divide a scalar or tensor by this tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise."""
        ...


TensorType = Union[TensorProtocol, "TorchTensorProtocol", "TfTensorProtocol", "MlxArrayProtocol", NDArrayProtocol]

type ForwardArg = TensorType | Sequence[int] | Sequence[Sequence[int]] | str | int | float | bool | None
type ForwardOutput = TensorProtocol | TorchTensorProtocol | MlxArrayProtocol | tuple[TensorProtocol | TorchTensorProtocol | MlxArrayProtocol, ...] | Mapping[str, TensorProtocol | TorchTensorProtocol | MlxArrayProtocol] | float


@runtime_checkable
class CausalLMProtocol(Protocol):
    """Protocol for causal language models supporting forward pass and autoregressive generation."""

    def __call__(
        self,
        *args: ForwardArg,
        **kwargs: ForwardArg,
    ) -> ForwardOutput:
        """Execute the forward pass of the causal language model."""
        ...


@runtime_checkable
class TorchParameterProtocol(Protocol):
    """Protocol representing a trainable PyTorch `nn.Parameter` or `torch.Tensor`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the parameter tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the parameter tensor."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the parameter tensor."""
        ...


@runtime_checkable
class TorchModelProtocol(Protocol):
    """Focused protocol for PyTorch causal language models."""

    def __call__(
        self,
        *args: ForwardArg,
        **kwargs: ForwardArg,
    ) -> ForwardOutput:
        """Execute the forward pass of the PyTorch module."""
        ...

    def parameters(self) -> Iterable[TorchParameterProtocol]:
        """Yield trainable parameter tensors."""
        ...

    def named_modules(self) -> Iterable[tuple[str, TorchModelProtocol | CausalLMProtocol]]:
        """Yield `(name, module)` pairs across the module hierarchy."""
        ...

    def train(self, mode: bool = True) -> Self:
        """Toggle training mode on the PyTorch model."""
        ...

    def state_dict(self) -> Mapping[str, TorchTensorProtocol | TensorProtocol]:
        """Return a dictionary containing references to the whole state of the module."""
        ...


TorchCausalLMProtocol = TorchModelProtocol


@runtime_checkable
class TorchTensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol for `torch.Tensor` objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor."""
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element tensor."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the tensor to a nested Python list or scalar."""
        ...

    def mean(self) -> Self:
        """Compute the arithmetic mean of the tensor elements."""
        ...

    def sum(self, dim: int = -1) -> Self:
        """Compute the sum of the tensor elements."""
        ...

    def unsqueeze(self, dim: int, /) -> Self:
        """Insert a singleton dimension at `dim`."""
        ...

    def squeeze(self, dim: int = -1, /) -> Self:
        """Remove a singleton dimension at `dim`."""
        ...

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        ...

    def __getitem__(self, item: int | slice | tuple[int | slice, ...], /) -> Self:
        """Slice or index into the tensor."""
        ...

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor."""
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar."""
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add another tensor or scalar to this tensor."""
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar."""
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar."""
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise."""
        ...


@runtime_checkable
class TorchOptimizerProtocol(Protocol):
    """Protocol for PyTorch optimizers supporting `.zero_grad()` and `.step()`."""

    def zero_grad(self) -> None:
        """Reset gradients of all optimized `torch.Tensor` parameters."""
        ...

    def step(self) -> TensorProtocol | float | None:
        """Perform a single optimization step."""
        ...


@runtime_checkable
class PeftModelProtocol(Protocol):
    """Protocol for a Hugging Face PEFT model supporting `.print_trainable_parameters()`, `.save_pretrained()`, and `.named_modules()`."""

    def print_trainable_parameters(self) -> None:
        """Log trainable vs. total parameter counts."""
        ...

    def save_pretrained(self, save_directory: str | Path, /) -> None:
        """Persist PEFT adapter weights and config to `save_directory`."""
        ...

    def named_modules(self) -> Iterable[tuple[str, TorchModelProtocol | CausalLMProtocol]]:
        """Yield `(name, module)` pairs across the wrapped PEFT hierarchy."""
        ...


@runtime_checkable
class VLLMCompletionOutputProtocol(Protocol):
    """Protocol for a single vLLM completion item exposing `.text`."""

    @property
    def text(self) -> str:
        """Provide the generated completion text."""
        ...


@runtime_checkable
class VLLMRequestOutputProtocol(Protocol):
    """Protocol for a vLLM streaming `RequestOutput` exposing `.outputs`."""

    @property
    def outputs(self) -> Sequence[VLLMCompletionOutputProtocol]:
        """Provide the sequence of candidate completion outputs."""
        ...


@runtime_checkable
class VLLMEngineProtocol(Protocol):
    """Protocol for a vLLM `AsyncLLMEngine` supporting `.generate(...)` and `.abort(...)`."""

    def generate(
        self,
        prompt: str,
        sampling_params: None,
        request_id: str,
        /,
    ) -> AsyncIterator[VLLMRequestOutputProtocol]:
        """Stream `VLLMRequestOutputProtocol` chunks for `prompt`."""
        ...

    async def abort(self, request_id: str, /) -> None:
        """Abort an in-flight vLLM request by `request_id`."""
        ...


@runtime_checkable
class KerasModelProtocol(CausalLMProtocol, Protocol):
    """Focused protocol for Keras 3 / KerasNLP causal language models."""

    def generate(
        self,
        *args: ForwardArg,
        **kwargs: ForwardArg,
    ) -> TensorProtocol | TfTensorProtocol | Sequence[int] | str:
        """Generate token sequences autoregressively."""
        ...

    def compile(
        self,
        *args: str | Callable[..., TensorProtocol] | None,
        **kwargs: str | float | Sequence[str] | Callable[..., TensorProtocol] | None,
    ) -> None:
        """Compile the model with loss and optimizer."""
        ...

    def fit(
        self,
        *args: Iterable[Mapping[str, TensorType]] | TensorType,
        **kwargs: float | bool | None,
    ) -> Mapping[str, Sequence[float]] | None:
        """Fit the model on a dataset iterator."""
        ...

    @property
    def layers(self) -> Sequence[KerasLayerProtocol]:
        """Provide the sequence of Keras layers."""
        ...

    @property
    def trainable_variables(self) -> Sequence[KerasVariableProtocol | TfTensorProtocol]:
        """Provide the sequence of trainable variables."""
        ...

    @property
    def sampler(self) -> str | Callable[..., TensorProtocol] | None:
        """Provide the model's token sampler."""
        ...

    @property
    def preprocessor(self) -> TokenizerProtocol | Callable[..., Mapping[str, TensorType]] | None:
        """Provide the model's input preprocessor."""
        ...

    def save(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Save model weights or checkpoint to disk."""
        ...

    def load_weights(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Load serialized weights into the model."""
        ...


KerasCausalLMProtocol = KerasModelProtocol


@runtime_checkable
class TfTensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol for TensorFlow `tf.Tensor` and Keras tensor objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the tensor shape tuple."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the rank (number of dimensions) of the tensor."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor."""
        ...

    def numpy(self) -> NDArrayProtocol:
        """Convert the TensorFlow tensor to an n-dimensional array."""
        ...

    def __sub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor."""
        ...

    def __rsub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar."""
        ...

    def __add__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Add another tensor or scalar to this tensor."""
        ...

    def __radd__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar."""
        ...

    def __mul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar."""
        ...

    def __rmul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise."""
        ...


@runtime_checkable
class KerasVariableProtocol(Protocol):
    """Protocol for a Keras 3 / TensorFlow variable supporting `.assign(...)`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the variable shape tuple."""
        ...

    def assign(self, value: NDArrayProtocol | TfTensorProtocol | TensorProtocol | SupportsFloat) -> None:
        """Assign a new array or tensor value to the variable in-place."""
        ...


@runtime_checkable
class KerasLayerProtocol(Protocol):
    """Protocol for a Keras 3 `Layer` exposing `.weights`, `.trainable_weights`, and `.trainable`."""

    trainable: bool

    @property
    def name(self) -> str:
        """Provide the layer name."""
        ...

    @property
    def weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide all variables belonging to the layer."""
        ...

    @property
    def trainable_weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide trainable variables belonging to the layer."""
        ...


@runtime_checkable
class KerasBackboneProtocol(Protocol):
    """Protocol for a KerasNLP `Backbone` supporting `.enable_lora(rank=...)` and `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol]:
        """Provide the sequence of backbone layers."""
        ...

    def enable_lora(self, *, rank: int) -> None:
        """Enable native KerasNLP LoRA with the given rank."""
        ...


@runtime_checkable
class TfModuleProtocol(Protocol):
    """Protocol for the `tensorflow` runtime module."""

    @property
    def int32(self) -> DTypeProtocol | str:
        """Provide `tf.int32` dtype."""
        ...

    @property
    def float32(self) -> DTypeProtocol | str:
        """Provide `tf.float32` dtype."""
        ...

    def cast(
        self,
        x: TfTensorProtocol | TensorProtocol | bool,
        dtype: DTypeProtocol | str,
        /,
    ) -> TfTensorProtocol:
        """Cast tensor `x` to `dtype`."""
        ...

    def expand_dims(self, input: TfTensorProtocol, axis: int = -1, /) -> TfTensorProtocol:
        """Insert a dimension of length 1 at `axis`."""
        ...

    def gather(
        self,
        params: TfTensorProtocol,
        indices: TfTensorProtocol,
        *,
        batch_dims: int = 0,
    ) -> TfTensorProtocol:
        """Gather slices from `params` according to `indices`."""
        ...

    def squeeze(self, input: TfTensorProtocol, axis: int = -1, /) -> TfTensorProtocol:
        """Remove dimensions of size 1 along `axis`."""
        ...

    def reduce_sum(self, input_tensor: TfTensorProtocol, axis: int = -1, /) -> TfTensorProtocol:
        """Compute the sum of elements across dimensions of a tensor."""
        ...

    def reduce_mean(self, input_tensor: TfTensorProtocol, /) -> TfTensorProtocol:
        """Compute the mean of elements across dimensions of a tensor."""
        ...


@runtime_checkable
class ModelConfigProtocol(Protocol):
    """Protocol representing a Gemma 4 model configuration object."""

    quantization_config: JSONValue | None

    @property
    def vocab_size(self) -> int:
        """Provide the vocabulary size."""
        ...

    @property
    def hidden_size(self) -> int:
        """Provide the hidden embedding size."""
        ...

    @property
    def num_hidden_layers(self) -> int:
        """Provide the number of transformer layers."""
        ...


@runtime_checkable
class FlaxNNXModelProtocol(Protocol):
    """Focused protocol for JAX/Flax NNX causal language models."""

    def __call__(
        self,
        *args: ForwardArg,
        **kwargs: ForwardArg,
    ) -> ForwardOutput:
        """Execute the forward pass of the NNX module."""
        ...


JaxModelProtocol = FlaxNNXModelProtocol
JaxCausalLMProtocol = FlaxNNXModelProtocol


@runtime_checkable
class JaxArrayProtocol(TensorProtocol, Protocol):
    """Protocol for `jax.Array` objects supporting `.astype(...)` and `.tolist()`."""

    def astype(self, dtype: DTypeProtocol | DTypeSpec, /) -> Self:
        """Cast the JAX array to `dtype`."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the JAX array into nested Python lists or a scalar."""
        ...


@runtime_checkable
class OrbaxCheckpointerProtocol[StateT](Protocol):
    """Protocol for an Orbax `PyTreeCheckpointer` supporting `.save(path, state)` and `.restore(path)`."""

    def save(self, directory: Path | str, item: StateT) -> None:
        """Persist a PyTree checkpoint to `directory`."""
        ...

    def restore(self, directory: Path | str) -> StateT:
        """Restore a PyTree checkpoint from `directory`."""
        ...


@runtime_checkable
class MaxTextModelProtocol(Protocol):
    """Focused protocol for stateless Flax Linen / MaxText models."""

    def init(
        self,
        *args: ForwardArg | Mapping[str, TensorType],
        **kwargs: ForwardArg,
    ) -> Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]]:
        """Initialize stateless model parameters with a PRNGKey and sample input."""
        ...

    def apply(
        self,
        *args: ForwardArg | Mapping[str, TensorType],
        **kwargs: ForwardArg,
    ) -> ForwardOutput:
        """Apply a stateless transformation or forward pass."""
        ...


MaxTextCausalLMProtocol = MaxTextModelProtocol
type PyTreeNode = Mapping[str, "PyTreeNode"] | JaxArrayProtocol | TensorProtocol


@runtime_checkable
class MlxArrayProtocol(SupportsFloat, SupportsInt, Protocol):
    """Protocol for Apple Silicon `mlx.core.array` objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the array."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the array."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the array."""
        ...

    @property
    def T(self) -> Self:
        """Provide the transpose of the array."""
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element array."""
        ...

    def tolist(self) -> list[int] | list[float] | list[list[int]] | list[list[float]] | float | int:
        """Convert the array to a nested Python list or scalar."""
        ...

    def mean(self, axis: int = -1) -> Self:
        """Compute the arithmetic mean of the array elements."""
        ...

    def sum(self, axis: int = -1) -> Self:
        """Compute the sum of the array elements."""
        ...

    def __getitem__(self, item: TensorIndex, /) -> Self:
        """Slice or index into the array."""
        ...

    def __sub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Subtract another array or scalar from this array."""
        ...

    def __rsub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-subtract this array from another array or scalar."""
        ...

    def __add__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Add another array or scalar to this array."""
        ...

    def __radd__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-add this array to another array or scalar."""
        ...

    def __mul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Multiply this array by another array or scalar."""
        ...

    def __rmul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-multiply this array by a scalar or array."""
        ...

    def __matmul__(self, other: Self, /) -> Self:
        """Matrix-multiply two arrays."""
        ...

    def __neg__(self) -> Self:
        """Negate the array elementwise."""
        ...


@runtime_checkable
class MlxCoreProtocol(Protocol):
    """Protocol for `mlx.core` (`mx`) runtime functions."""

    def array(
        self,
        val: Sequence[int] | Sequence[float] | Sequence[Sequence[int]] | Sequence[Sequence[float]] | MlxArrayProtocol | NDArrayProtocol,
        *,
        dtype: DTypeProtocol | DTypeSpec = None,
    ) -> MlxArrayProtocol:
        """Construct an MLX array."""
        ...

    def logsumexp(
        self,
        a: MlxArrayProtocol,
        axis: int = -1,
        keepdims: bool = False,
    ) -> MlxArrayProtocol:
        """Compute log-sum-exp along `axis`."""
        ...

    def take_along_axis(
        self,
        a: MlxArrayProtocol,
        indices: MlxArrayProtocol,
        axis: int = -1,
    ) -> MlxArrayProtocol:
        """Gather elements from `a` along `axis` at `indices`."""
        ...

    def expand_dims(
        self,
        a: MlxArrayProtocol,
        axis: int = -1,
    ) -> MlxArrayProtocol:
        """Insert a singleton dimension at `axis`."""
        ...

    def squeeze(
        self,
        a: MlxArrayProtocol,
        axis: int = -1,
    ) -> MlxArrayProtocol:
        """Remove a singleton dimension at `axis`."""
        ...

    def eval(self, *args: MlxArrayProtocol | Mapping[str, MlxArrayProtocol] | None) -> None:
        """Evaluate lazy MLX arrays and parameter trees."""
        ...


@runtime_checkable
class MlxNNLossesProtocol(Protocol):
    """Protocol for `mlx.nn.losses` exposing `log_sigmoid` and `cross_entropy`."""

    def log_sigmoid(self, x: MlxArrayProtocol | float) -> MlxArrayProtocol:
        """Compute elementwise log-sigmoid."""
        ...

    def cross_entropy(
        self,
        logits: MlxArrayProtocol,
        targets: MlxArrayProtocol,
        *,
        reduction: str = "mean",
    ) -> MlxArrayProtocol:
        """Compute cross-entropy loss."""
        ...


@runtime_checkable
class MlxNNProtocol(Protocol):
    """Protocol for `mlx.nn` exposing `losses`, `log_sigmoid`, `value_and_grad`, and `quantize`."""

    @property
    def losses(self) -> MlxNNLossesProtocol:
        """Provide the `mlx.nn.losses` submodule."""
        ...

    def log_sigmoid(self, x: MlxArrayProtocol | float) -> MlxArrayProtocol:
        """Compute elementwise log-sigmoid."""
        ...

    def value_and_grad(
        self,
        model: MLXModelProtocol,
        fn: Callable[..., MlxArrayProtocol],
    ) -> Callable[..., tuple[MlxArrayProtocol, dict[str, MlxArrayProtocol]]]:
        """Transform `fn` to compute value and gradients with respect to `model`."""
        ...

    def quantize(
        self,
        model: MLXModelProtocol,
        *,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        """Quantize linear layers of `model` in-place."""
        ...


@runtime_checkable
class MLXModelProtocol(Protocol):
    """Focused protocol for Apple Silicon MLX models."""

    def __call__(
        self,
        *args: ForwardArg | MlxArrayProtocol,
        **kwargs: ForwardArg | MlxArrayProtocol,
    ) -> ForwardOutput | MlxArrayProtocol:
        """Execute the forward pass of the MLX model."""
        ...

    def parameters(self) -> Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]]:
        """Return the parameter tree of the MLX model."""
        ...

    def update(
        self,
        parameters: Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]],
    ) -> Self | None:
        """Update the parameters of the MLX model with a parameter tree."""
        ...

    def freeze(self) -> Self | None:
        """Freeze base model parameters prior to adapter training."""
        ...


MlxModelProtocol = MLXModelProtocol
MlxCausalLMProtocol = MLXModelProtocol


@runtime_checkable
class EmbedderModelProtocol(Protocol):
    """Focused protocol for text and schema embedding models."""

    def encode(
        self,
        *args: str | Sequence[str],
        **kwargs: int | bool | str | None,
    ) -> NDArrayProtocol | Sequence[Sequence[float]] | Sequence[float]:
        """Encode text or features into dense vector embeddings."""
        ...


@runtime_checkable
class TrainableModelProtocol(CausalLMProtocol, Protocol):
    """Protocol for trainable models across JAX, PyTorch, Keras, MaxText, and MLX."""

    def compile(
        self,
        *args: str | Callable[..., TensorProtocol] | None,
        **kwargs: str | float | Callable[..., TensorProtocol] | None,
    ) -> None:
        """Compile the model with loss and optimizer (Keras)."""
        ...

    def fit(
        self,
        *args: Iterable[Mapping[str, TensorType]] | TensorType,
        **kwargs: float | bool | None,
    ) -> Mapping[str, Sequence[float]] | None:
        """Fit the model on a dataset iterator (Keras)."""
        ...

    def parameters(self) -> Iterable[TensorProtocol] | dict[str, TensorProtocol]:
        """Yield trainable parameter tensors (PyTorch/MLX)."""
        ...

    def train(self, mode: bool = True) -> Self:
        """Toggle training mode on the model (PyTorch)."""
        ...

    def apply(
        self,
        *args: Mapping[str, TensorType] | TensorType | Sequence[int] | int | None,
        **kwargs: TensorType | Sequence[int] | int | bool | None,
    ) -> TensorProtocol | tuple[TensorProtocol, ...]:
        """Apply a stateless transformation or forward pass (Flax/MaxText)."""
        ...

    def load_weights(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Load serialized weights into the model."""
        ...

    def save(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Save model weights or checkpoint to disk (Keras)."""
        ...

    def freeze(self) -> None:
        """Freeze base model parameters prior to adapter training (MLX)."""
        ...


@runtime_checkable
class QuantizableModelProtocol(TrainableModelProtocol, Protocol):
    """Protocol for models supporting weight quantization and layer inspection."""

    def named_modules(self) -> Iterable[tuple[str, CausalLMProtocol]]:
        """Yield `(name, module)` pairs across the model hierarchy."""
        ...

    @property
    def layers(self) -> Sequence[CausalLMProtocol]:
        """Provide the list of model layers."""
        ...

    @property
    def sampler(self) -> str | Callable[..., TensorProtocol] | None:
        """Provide the model's token sampler."""
        ...

    @property
    def preprocessor(self) -> TokenizerProtocol | Callable[..., Mapping[str, TensorType]] | None:
        """Provide the model's input preprocessor."""
        ...

    @property
    def model(self) -> CausalLMProtocol | None:
        """Provide the inner trunk model."""
        ...

    @property
    def config(self) -> Mapping[str, JSONValue] | None:
        """Provide the model configuration object."""
        ...

    def encode(
        self,
        *args: str | Sequence[str],
        **kwargs: int | bool | str | None,
    ) -> NDArrayProtocol | Sequence[Sequence[float]] | Sequence[float]:
        """Encode text or features into embeddings."""
        ...


@runtime_checkable
class CursorProtocol(Protocol):
    """Protocol representing a DB-API 2.0 compatible database cursor."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | bool | None:
        """Provide the cursor column description metadata."""
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] = (),
        /,
    ) -> Self:
        """Execute a SQL query on the cursor."""
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows of a query result set."""
        ...

    def close(self) -> None:
        """Close the cursor."""
        ...


@runtime_checkable
class DatabaseConnectionProtocol(Protocol):
    """Protocol representing a DB-API 2.0 or analytical database connection."""

    def cursor(self) -> CursorProtocol:
        """Return a new `CursorProtocol` object using the connection."""
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] = (),
        /,
    ) -> CursorProtocol:
        """Execute a SQL statement directly on the connection."""
        ...

    def commit(self) -> Self | None:
        """Commit any pending transaction to the database."""
        ...

    def rollback(self) -> Self | None:
        """Roll back any pending transaction."""
        ...

    def close(self) -> None:
        """Close the database connection immediately."""
        ...


@runtime_checkable
class AsyncDatabaseConnectionProtocol(Protocol):
    """Protocol representing an asynchronous database connection (`asyncpg`, `aiosqlite`, etc.)."""

    async def close(self) -> None:
        """Close the asynchronous database connection."""
        ...


@runtime_checkable
class DuckDBConnectionProtocol(DatabaseConnectionProtocol, Protocol):
    """Protocol for a DuckDB connection supporting `.create_function(...)` and `.cursor()`."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | bool | None:
        """Provide the cursor column description metadata."""
        ...

    def create_function(
        self,
        name: str,
        function: Callable[..., str],
        parameters: list[type[str]],
        return_type: type[str],
    ) -> None:
        """Register a Python scalar UDF with DuckDB."""
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = ...,
        /,
    ) -> Self:
        """Execute a SQL statement directly on the DuckDB connection."""
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows of a query result set."""
        ...

    def cursor(self) -> DuckDBConnectionProtocol:
        """Return a thread-local duplicate DuckDB cursor/connection."""
        ...


TQuantModel = TypeVar("TQuantModel")


def _empty_scales_dict() -> dict[str, float]:
    """Return an empty quantization scale dictionary."""
    return {}


@dataclass
class QuantizedModelArtifact(Generic[TQuantModel]):
    """Strongly-typed wrapper recording quantization state and scales for a model."""

    model: TQuantModel
    method: str
    is_quantized: bool = True
    scales: dict[str, float] = field(default_factory=_empty_scales_dict)


@dataclass
class _QuantizedMetadata:
    """Internal metadata record stored in `_QUANTIZED_ARTIFACT_WEAK_REGISTRY` without a strong model cycle."""

    method: str
    is_quantized: bool
    scales: dict[str, float]


_QUANTIZED_ARTIFACT_WEAK_REFS: dict[int, _QuantizedMetadata] = {}
_QUANTIZED_ARTIFACT_FALLBACK_REGISTRY: dict[int, _QuantizedMetadata] = {}


def clear_quantized_artifact_registry() -> None:
    """Clear all cached entries from the quantized model artifact registry."""
    _QUANTIZED_ARTIFACT_WEAK_REFS.clear()
    _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY.clear()


def register_quantized_artifact(
    model: TQuantModel,
    method: str,
    scales: dict[str, float] | None = None,
) -> QuantizedModelArtifact[TQuantModel]:
    """Register quantization metadata for a model in the memory-safe artifact registry.

    Args:
        model: Target model instance.
        method: Applied quantization method identifier ('int8', 'int4', 'awq', etc.).
        scales: Optional per-layer or per-parameter quantization scale factors.

    Returns:
        Registered `QuantizedModelArtifact` wrapping `model`.

    """
    resolved_scales = dict(scales) if scales is not None else {}
    meta = _QuantizedMetadata(
        method=method,
        is_quantized=True,
        scales=resolved_scales,
    )
    model_id = id(model)

    def _on_collect() -> None:
        """Remove `model_id` from `_QUANTIZED_ARTIFACT_WEAK_REFS` upon garbage collection."""
        _QUANTIZED_ARTIFACT_WEAK_REFS.pop(model_id, None)

    try:
        weakref.finalize(model, _on_collect)
        _QUANTIZED_ARTIFACT_WEAK_REFS[model_id] = meta
    except TypeError:
        _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY[model_id] = meta
    return QuantizedModelArtifact(
        model=model,
        method=method,
        is_quantized=True,
        scales=resolved_scales,
    )


def get_quantized_artifact(model: TQuantModel) -> QuantizedModelArtifact[TQuantModel] | None:
    """Retrieve the registered `QuantizedModelArtifact` for a quantized model instance.

    Args:
        model: Target model instance to inspect.

    Returns:
        Registered `QuantizedModelArtifact` if the model was quantized, or `None`.

    """
    model_id = id(model)
    meta = _QUANTIZED_ARTIFACT_WEAK_REFS.get(model_id) or _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY.get(model_id)
    if meta is not None:
        return QuantizedModelArtifact(
            model=model,
            method=meta.method,
            is_quantized=meta.is_quantized,
            scales=dict(meta.scales),
        )
    return None


TrainableModelType = Union[
    JaxCausalLMProtocol,
    TorchCausalLMProtocol,
    KerasCausalLMProtocol,
    MaxTextCausalLMProtocol,
    MlxCausalLMProtocol,
    TrainableModelProtocol,
]

QuantizableModelType = Union[
    JaxCausalLMProtocol,
    TorchCausalLMProtocol,
    KerasCausalLMProtocol,
    MaxTextCausalLMProtocol,
    MlxCausalLMProtocol,
    QuantizableModelProtocol,
]

ModelType = Union[
    TrainableModelType,
    EmbedderModelProtocol,
    QuantizableModelProtocol,
]


@runtime_checkable
class ToListArrayProtocol(Protocol):
    """Protocol for any array or tensor supporting `.tolist()` and `.shape`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the array into nested Python lists."""
        ...


ImageInput = Union[str, Path, bytes, Sequence[float], NDArrayProtocol, TensorProtocol, ToListArrayProtocol, None]
AudioInput = Union[str, Path, bytes, Sequence[float], NDArrayProtocol, TensorProtocol, ToListArrayProtocol, None]


class MultimodalInput(TypedDict, total=False):
    """Structured multimodal input containing prompt and optional media modalities."""

    prompt: str
    image: ImageInput
    audio: AudioInput
    image_token_mask: list[bool] | None
    audio_token_mask: list[bool] | None
    modality: str


class BaseOperationResultDict(TypedDict):
    """Common base TypedDict for backend and SDK operation results."""

    backend: str
    status: str
    status_code: NotRequired[str]
    error: NotRequired[str]
    error_message: NotRequired[str | None]


class TrainResultDict(BaseOperationResultDict):
    """Structured return dictionary for training, pretraining, SFT, and post-training jobs."""

    action: str
    model: str
    dataset: str
    epochs: int
    learning_rate: NotRequired[float]
    final_loss: float
    distributed_strategy: NotRequired[str]
    checkpoint_dir: NotRequired[str]


class DPOResultDict(BaseOperationResultDict):
    """Structured return dictionary for Direct Preference Optimization (DPO) runs."""

    action: str
    model: str
    dataset: str
    beta: float
    epochs: NotRequired[int]
    final_loss: float


class InferenceResultDict(BaseOperationResultDict):
    """Structured return dictionary for SQL generation and inference."""

    model: str
    prompt: str
    sql: str
    beam_width: int
    confidence_score: float


class EvaluationMetricsDict(TypedDict):
    """Structured dictionary of Text-to-SQL evaluation metrics."""

    exact_match: float
    execution_accuracy: float
    valid_sql: float
    mean_confidence: NotRequired[float]


class EvaluationResultDict(TypedDict):
    """Structured return dictionary for end-to-end model evaluation."""

    backend: str
    model: str
    dataset: str
    metrics: EvaluationMetricsDict
    status: str
    modality: NotRequired[str]
    image_path: NotRequired[str]
    audio_path: NotRequired[str]


class AgentAttemptDict(TypedDict):
    """Record of a single SQL generation and execution attempt within the agentic loop."""

    attempt: int
    prompt: NotRequired[str]
    sql: str
    success: NotRequired[bool]
    error: str | None
    confidence: NotRequired[float]


class AgentLoopResultDict(TypedDict):
    """Structured return dictionary for the self-correcting SQL agent loop."""

    backend: NotRequired[str]
    model: NotRequired[str]
    initial_prompt: NotRequired[str]
    prompt: NotRequired[str]
    sql: NotRequired[str]
    final_sql: str
    results: list[tuple[JSONPrimitive, ...]]
    success: NotRequired[bool]
    attempts: int
    history: list[AgentAttemptDict]
    status: str
    error: NotRequired[str | None]
    confidence_score: NotRequired[float]


AgentResultDict = AgentLoopResultDict


class ChatMessageDict(TypedDict):
    """Single message turn in a multi-turn conversational SQL session."""

    role: str
    content: str


class ChatTurnResultDict(TypedDict):
    """Structured return dictionary for a conversational SQL chat turn."""

    backend: str
    model: str
    response: str
    history: list[ChatMessageDict]
    status: str


class QuantizeResultDict(BaseOperationResultDict):
    """Structured return dictionary for model quantization pipelines."""

    model: str
    method: str
    memory_reduction_factor: float
    export_path: NotRequired[str]
    quantized_layers_count: NotRequired[int]
    metadata: NotRequired[dict[str, JSONValue]]


class PEFTResultDict(BaseOperationResultDict):
    """Structured return dictionary for PEFT / LoRA injection."""

    action: str
    model: str
    target_modules: list[str]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    injected_modules: NotRequired[int]


class ExportResultDict(BaseOperationResultDict):
    """Structured return dictionary for model checkpoint export."""

    model: str
    export_path: str
    file_path: str
    format: str


class LatencyStatsDict(TypedDict):
    """Latency distribution statistics in milliseconds."""

    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    median_ms: NotRequired[float]
    std_ms: NotRequired[float]


class BenchmarkResultDict(BaseOperationResultDict):
    """Structured return dictionary for hardware benchmarking."""

    model: str
    hardware: str
    batch_size: int
    tokens_per_sec: float
    latency_ms: float
    memory_mb: float
    rss_memory_mb: NotRequired[float]
    latency_stats: NotRequired[LatencyStatsDict]


@runtime_checkable
class FastAPIAppProtocol(Protocol):
    """Protocol representing an ASGI / FastAPI application instance."""

    async def __call__(
        self,
        scope: MutableMapping[str, JSONValue],
        receive: Callable[[], Awaitable[MutableMapping[str, JSONValue]]],
        send: Callable[[MutableMapping[str, JSONValue]], Awaitable[None]],
    ) -> None:
        """Handle an ASGI request scope."""
        ...

    def post(
        self,
        path: str,
        *args: str | bool | None,
        **kwargs: str | bool | None,
    ) -> Callable[..., JSONValue]:
        """Register a POST route handler."""
        ...

    def get(
        self,
        path: str,
        *args: str | bool | None,
        **kwargs: str | bool | None,
    ) -> Callable[..., JSONValue]:
        """Register a GET route handler."""
        ...


TBatchItem_co = TypeVar("TBatchItem_co", covariant=True)


@runtime_checkable
class DataLoaderProtocol(Iterable[TBatchItem_co], Protocol[TBatchItem_co]):
    """Protocol representing an iterable batch data loader."""

    def __len__(self) -> int:
        """Return the number of batches in the data loader."""
        ...


class ServeResultDict(BaseOperationResultDict):
    """Structured return dictionary for continuous batching server initialization."""

    model: str
    port: int
    max_batch_size: int
    mode: NotRequired[str]
    app: NotRequired[FastAPIAppProtocol | None]


class ETLResultDict(BaseOperationResultDict):
    """Structured return dictionary for ETL dataloader pipelines."""

    dataset: str
    split: str
    batch_size: int
    distributed: NotRequired[bool]
    loader: NotRequired[DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]] | Iterable[Mapping[str, TensorType | Sequence[int]]] | None]


DataloaderResultDict = ETLResultDict


class LogMetricsResultDict(BaseOperationResultDict):
    """Structured return dictionary for TensorBoard metric logging."""

    logged_step: NotRequired[int]
    step: NotRequired[int]
    log_dir: str
    metrics: NotRequired[dict[str, float]]
    logged_metrics: NotRequired[dict[str, float]]
    timestamp: NotRequired[float]
    step_duration_s: NotRequired[float]
    fallback_file: NotRequired[str]
    action: NotRequired[str]
    run_id: NotRequired[str]
    model: NotRequired[str]
    rank: NotRequired[int]
    world_size: NotRequired[int]
    status_detail: NotRequired[str]


class FewShotPromptResultDict(TypedDict):
    """Structured return dictionary for dynamic few-shot prompt construction."""

    backend: str
    model: str
    few_shot_prompt: str
    status: str
    num_examples: int


class FewShotResultDict(TypedDict):
    """Structured return dictionary for dynamic few-shot prompt construction and SQL generation."""

    backend: str
    model: str
    prompt: NotRequired[str]
    few_shot_prompt: str
    sql: NotRequired[str]
    confidence_score: NotRequired[float]
    status: str
    num_examples: NotRequired[int]


ExtraValue = Union[
    JSONValue,
    Path,
    ModelType,
    TensorType,
    TokenizerProtocol,
    HFTokenizerProtocol,
    DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]],
    Iterable[Mapping[str, TensorType | Sequence[int]]],
    Callable[..., ForwardOutput | JSONValue],
]
type BackendKwargValue = JSONValue | Path | ModelType | TensorType | TokenizerProtocol | HFTokenizerProtocol | DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]] | Iterable[Mapping[str, TensorType | Sequence[int]]] | Sequence[TensorType] | Mapping[str, TensorType]
ExtraKwargsDict = Mapping[str, ExtraValue]


def _empty_extra_kwargs() -> dict[str, ExtraValue]:
    """Return an empty dictionary for configuration extra_kwargs."""
    return {}


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization (DPO) execution."""

    model_name: str
    dataset: str
    beta: float = 0.1
    epochs: int = 1
    learning_rate: float = 1e-05
    batch_size: int = 2
    raise_on_error: bool = False
    extra_kwargs: ExtraKwargsDict = field(default_factory=_empty_extra_kwargs)


@dataclass
class ETLConfig:
    """Configuration for ETL dataset loading and tokenization pipelines."""

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
    stage: str = "sft"
    raise_on_error: bool = False


@dataclass
class TrainingConfig:
    """Configuration for pretraining, SFT, and post-training execution."""

    action: str = ""
    model_name: str = "gemma-4"
    dataset: str = "dummy"
    epochs: int = 1
    learning_rate: float = 0.0001
    batch_size: int = 2
    backend: str = "jax"
    distributed_strategy: str = "none"
    modality: str = "text"
    raise_on_error: bool = False
    extra_kwargs: ExtraKwargsDict = field(default_factory=_empty_extra_kwargs)


@runtime_checkable
class ScalarItemExtractorProtocol(Protocol):
    """Protocol for scalar tensors supporting `.item()` extraction."""

    def item(self) -> float | int | bool:
        """Extract a Python scalar value."""
        ...


@runtime_checkable
class JaxShardingProtocol(Protocol):
    """Protocol for JAX sharding objects such as `NamedSharding`."""

    def __repr__(self) -> str:
        """Return the string representation of the sharding."""
        ...


TModel = TypeVar("TModel")
TOpt = TypeVar("TOpt")
TBatch = TypeVar("TBatch")
TParams = TypeVar("TParams")

type StepResult = TensorProtocol | ScalarItemExtractorProtocol | SupportsFloat | float | str | tuple[TensorProtocol | ScalarItemExtractorProtocol | SupportsFloat | float | Mapping[str, TensorProtocol] | JSONValue, ...]


@dataclass
class TrainingLoopState(Generic[TModel, TOpt, TBatch]):
    """Strongly-typed state container for standard supervised training loops."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    learning_rate: float = 1e-4
    model: TModel | None = None
    optimizer: TOpt | None = None
    train_step: Callable[..., StepResult] | None = None
    sharding: str | Mapping[str, str] | JaxShardingProtocol | None = None


@dataclass
class DPOLoopState(Generic[TModel, TOpt, TBatch]):
    """Strongly-typed state container for Direct Preference Optimization (DPO) loops."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    policy_model: TModel | None = None
    ref_model: TModel | None = None
    optimizer: TOpt | None = None
    beta: float = 0.1
    train_step: Callable[..., StepResult] | None = None


@dataclass
class TrainerState(Generic[TModel, TParams, TOpt, TBatch]):
    """Unified generic state container for training and DPO epoch runners across all backends."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    train_step: Callable[..., StepResult | tuple[TParams, TOpt, StepResult]] | None = None
    loss_and_grad_fn: Callable[..., tuple[TensorProtocol | float, TParams]] | None = None
    params: TParams | None = None
    opt_state: TOpt | None = None
    policy_params: TParams | None = None
    ref_params: TParams | None = None
    model: TModel | ModelType | None = None
    policy_model: TModel | ModelType | None = None
    ref_model: TModel | ModelType | None = None
    optimizer: TOpt | None = None
    criterion: Callable[..., TensorProtocol | float] | None = None
    device: str | None = "cpu"
    beta: float = 0.1


class MaxTextTrainerState(TrainerState[ModelType, TParams, TOpt, TBatch], Generic[TParams, TOpt, TBatch]):
    """Typed trainer state subclass of `TrainerState` for stateless JAX/MaxText parameter and optimizer PyTrees."""


class TorchTrainerState(TrainerState[TModel, Mapping[str, TensorProtocol], TOpt, TBatch], Generic[TModel, TOpt, TBatch]):
    """Typed trainer state subclass of `TrainerState` for PyTorch models, optimizers, and loss criteria."""


class KerasTrainerState(TrainerState[TModel, Mapping[str, TensorProtocol], TOpt, TBatch], Generic[TModel, TOpt, TBatch]):
    """Typed trainer state subclass of `TrainerState` for Keras / TensorFlow compiled step execution."""


@runtime_checkable
class TrainSubmoduleProtocol(Protocol):
    """Protocol for backend `train` submodules."""

    def train_model(self, config: TrainingConfig, **kwargs: JSONValue) -> TrainResultDict:
        """Execute backend training pipeline."""
        ...


@runtime_checkable
class DPOSubmoduleProtocol(Protocol):
    """Protocol for backend `dpo` submodules."""

    def run_dpo(self, config: DPOConfig, **kwargs: JSONValue) -> DPOResultDict:
        """Execute backend DPO pipeline."""
        ...


@runtime_checkable
class ETLSubmoduleProtocol(Protocol):
    """Protocol for backend `etl` submodules."""

    def build_dataloader(self, config: ETLConfig, **kwargs: JSONValue) -> ETLResultDict:
        """Execute backend dataloader construction."""
        ...


@runtime_checkable
class ExportSubmoduleProtocol(Protocol):
    """Protocol for backend `export` submodules."""

    def export_model(self, model_name: str, export_path: str, **kwargs: JSONValue) -> ExportResultDict:
        """Execute backend model export."""
        ...


@runtime_checkable
class LoggingSubmoduleProtocol(Protocol):
    """Protocol for backend `logging` submodules."""

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue) -> LogMetricsResultDict:
        """Execute backend metric logging."""
        ...


@runtime_checkable
class PEFTSubmoduleProtocol(Protocol):
    """Protocol for backend `peft` submodules."""

    def apply_lora(
        self,
        model_name: str,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **kwargs: JSONValue,
    ) -> PEFTResultDict:
        """Execute backend LoRA adapter injection."""
        ...


@runtime_checkable
class QuantizeSubmoduleProtocol(Protocol):
    """Protocol for backend `quantize` submodules."""

    def quantize_model(self, model_name: str, method: str = "int8", **kwargs: JSONValue) -> QuantizeResultDict:
        """Execute backend model quantization."""
        ...


@runtime_checkable
class InferenceSubmoduleProtocol(Protocol):
    """Protocol for backend `inference` submodules."""

    def generate_sql(
        self,
        model_name: str,
        prompt: str,
        beam_width: int = 3,
        max_length: int = 50,
        **kwargs: JSONValue,
    ) -> InferenceResultDict:
        """Execute backend SQL generation."""
        ...


@runtime_checkable
class ServeSubmoduleProtocol(Protocol):
    """Protocol for backend `serve` submodules."""

    def serve_model(
        self,
        model_name: str,
        port: int = 8000,
        max_batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> ServeResultDict:
        """Execute backend continuous batching server initialization."""
        ...


@runtime_checkable
class BenchmarkSubmoduleProtocol(Protocol):
    """Protocol for backend `benchmark` submodules."""

    def benchmark_model(
        self,
        model_name: str,
        hardware: str = "tpu-v5p",
        batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> BenchmarkResultDict:
        """Execute backend hardware benchmark."""
        ...


__all__ = [
    "AgentAttemptDict",
    "AgentLoopResultDict",
    "AgentResultDict",
    "AsyncDatabaseConnectionProtocol",
    "AudioInput",
    "BackendKwargValue",
    "BaseOperationResultDict",
    "BenchmarkResultDict",
    "BenchmarkSubmoduleProtocol",
    "CausalLMProtocol",
    "ChatMessageDict",
    "ChatTurnResultDict",
    "CursorProtocol",
    "DPOConfig",
    "DPOLoopState",
    "DPOResultDict",
    "DPOSubmoduleProtocol",
    "DTypeProtocol",
    "DTypeSpec",
    "DataLoaderProtocol",
    "DatabaseConnectionProtocol",
    "DataloaderResultDict",
    "DuckDBConnectionProtocol",
    "ETLConfig",
    "ETLResultDict",
    "ETLSubmoduleProtocol",
    "EmbedderModelProtocol",
    "EvaluationMetricsDict",
    "EvaluationResultDict",
    "ExportResultDict",
    "ExportSubmoduleProtocol",
    "ExtraKwargsDict",
    "ExtraValue",
    "FastAPIAppProtocol",
    "FewShotPromptResultDict",
    "FewShotResultDict",
    "FlaxNNXModelProtocol",
    "ForwardArg",
    "ForwardOutput",
    "HFTokenizerProtocol",
    "ImageInput",
    "InferenceResultDict",
    "InferenceSubmoduleProtocol",
    "JSONDict",
    "JSONMapping",
    "JSONPrimitive",
    "JSONValue",
    "JaxArrayProtocol",
    "JaxCausalLMProtocol",
    "JaxModelProtocol",
    "KerasBackboneProtocol",
    "KerasCausalLMProtocol",
    "KerasLayerProtocol",
    "KerasModelProtocol",
    "KerasTrainerState",
    "KerasVariableProtocol",
    "KwargScalar",
    "KwargValue",
    "LatencyStatsDict",
    "LogMetricsResultDict",
    "LoggingSubmoduleProtocol",
    "MLXModelProtocol",
    "MaxTextCausalLMProtocol",
    "MaxTextModelProtocol",
    "MaxTextTrainerState",
    "MlxArrayProtocol",
    "MlxCausalLMProtocol",
    "MlxCoreProtocol",
    "MlxModelProtocol",
    "MlxNNLossesProtocol",
    "MlxNNProtocol",
    "ModelConfigProtocol",
    "ModelType",
    "MultimodalInput",
    "NDArrayProtocol",
    "NestedScalarList",
    "OrbaxCheckpointerProtocol",
    "PEFTResultDict",
    "PEFTSubmoduleProtocol",
    "PeftModelProtocol",
    "PyTreeNode",
    "QuantizableModelProtocol",
    "QuantizableModelType",
    "QuantizeResultDict",
    "QuantizeSubmoduleProtocol",
    "QuantizedModelArtifact",
    "ServeResultDict",
    "ServeSubmoduleProtocol",
    "StepResult",
    "TBatch",
    "TBatchItem_co",
    "TModel",
    "TOpt",
    "TParams",
    "TQuantModel",
    "TResult",
    "TensorIndex",
    "TensorProtocol",
    "TensorType",
    "TfModuleProtocol",
    "TfTensorProtocol",
    "TokenizerProtocol",
    "TorchCausalLMProtocol",
    "TorchModelProtocol",
    "TorchOptimizerProtocol",
    "TorchTensorProtocol",
    "TorchTrainerState",
    "TrainResultDict",
    "TrainSubmoduleProtocol",
    "TrainableModelProtocol",
    "TrainableModelType",
    "TrainerState",
    "TrainingConfig",
    "TrainingLoopState",
    "VLLMCompletionOutputProtocol",
    "VLLMEngineProtocol",
    "VLLMRequestOutputProtocol",
    "clear_quantized_artifact_registry",
    "extract_bool_kwarg",
    "extract_float_kwarg",
    "extract_int_kwarg",
    "extract_str_kwarg",
    "get_quantized_artifact",
    "register_quantized_artifact",
    "run_coroutine_sync",
]

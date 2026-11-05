"""Model, optimizer, tokenizer, and training loop state protocols across backends."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Generic,
    Protocol,
    Self,
    SupportsFloat,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

from gemma_4_sql._types_json import JSONValue
from gemma_4_sql._types_tensors import (
    DTypeProtocol,
    DTypeSpec,
    JaxArrayProtocol,
    MlxArrayProtocol,
    NDArrayProtocol,
    NestedScalarList,
    ScalarItemExtractorProtocol,
    TensorProtocol,
    TensorType,
    TfTensorProtocol,
    TorchTensorProtocol,
)

type ForwardArg = TensorType | Sequence[int] | Sequence[Sequence[int]] | str | int | float | bool | None
type ForwardOutput = TensorProtocol | TorchTensorProtocol | MlxArrayProtocol | tuple[TensorProtocol | TorchTensorProtocol | MlxArrayProtocol, ...] | Mapping[str, TensorProtocol | TorchTensorProtocol | MlxArrayProtocol] | float


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol representing a text/SQL tokenizer."""

    @overload
    def encode(self, text: str) -> list[int]:
        """Encode a single string into a list of integer token IDs."""

    @overload
    def encode(self, text: list[str] | tuple[str, ...]) -> list[list[int]]:
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

    @property
    def device(self) -> DTypeProtocol | str | int | None:
        """Provide the device where the parameter tensor resides."""
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

    def eval(self) -> Self:
        """Set the PyTorch model to evaluation mode."""
        ...

    def state_dict(self) -> Mapping[str, TorchTensorProtocol | TensorProtocol]:
        """Return a dictionary containing references to the whole state of the module."""
        ...


TorchCausalLMProtocol = TorchModelProtocol


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
class KerasVariableProtocol(Protocol):
    """Protocol for a Keras 3 / TensorFlow variable supporting `.assign(...)` and `+`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the variable shape tuple."""
        ...

    def assign(self, value: NDArrayProtocol | TfTensorProtocol | TensorProtocol | SupportsFloat) -> None:
        """Assign a new array or tensor value to the variable in-place."""
        ...

    def __add__(self, other: TfTensorProtocol | TensorProtocol | float, /) -> TfTensorProtocol:
        """Add a tensor or scalar to the variable."""
        ...


@runtime_checkable
class KerasLayerProtocol(Protocol):
    """Protocol for a Keras 3 `Layer` exposing `.weights`, `.trainable_weights`, `.trainable_variables`, and `.trainable`."""

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

    @property
    def trainable_variables(self) -> Sequence[KerasVariableProtocol | TfTensorProtocol]:
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
class KerasPreprocessorProtocol(Protocol):
    """Protocol for a KerasNLP model preprocessor exposing mutable `sequence_length`."""

    sequence_length: int

    def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> Mapping[str, TensorType]:
        """Preprocess raw inputs into model tensors."""
        ...


@runtime_checkable
class KerasFitHistoryProtocol(Protocol):
    """Protocol for a Keras `.fit(...)` `History` object exposing `.history`."""

    @property
    def history(self) -> Mapping[str, Sequence[float]]:
        """Provide the dictionary of recorded training metric sequences."""
        ...


@runtime_checkable
class KerasSamplerProtocol(Protocol):
    """Protocol for a KerasNLP token sampler such as `BeamSampler`."""


type KerasScoreType = TfTensorProtocol | NDArrayProtocol | NestedScalarList | float | int | str | Sequence[float | int | str | Sequence[float | int | str | None] | None] | None


@runtime_checkable
class KerasModelProtocol(CausalLMProtocol, Protocol):
    """Focused protocol for Keras 3 / KerasNLP causal language models."""

    sampler: KerasSamplerProtocol | str | Callable[..., TensorProtocol] | None

    def generate(
        self,
        *args: ForwardArg | Sequence[str],
        **kwargs: ForwardArg,
    ) -> TensorProtocol | TfTensorProtocol | Sequence[int] | Sequence[str] | Mapping[str, KerasScoreType] | tuple[str, KerasScoreType] | str:
        """Generate token sequences autoregressively."""
        ...

    def compile(
        self,
        *args: str | Callable[..., TensorProtocol] | None,
        **kwargs: str | float | Sequence[str] | Callable[..., TensorProtocol] | KerasSamplerProtocol | None,
    ) -> None:
        """Compile the model with loss and optimizer."""
        ...

    def fit(
        self,
        *args: Iterable[tuple[TfTensorProtocol, TfTensorProtocol] | Mapping[str, TensorType | Sequence[int]]] | Iterable[Mapping[str, TensorType]] | TensorType,
        **kwargs: float | bool | None,
    ) -> KerasFitHistoryProtocol:
        """Fit the model on a dataset iterator."""
        ...

    @property
    def layers(self) -> Sequence[KerasLayerProtocol]:
        """Provide the sequence of Keras layers."""
        ...

    @property
    def weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide all variables belonging to the model."""
        ...

    @property
    def trainable_weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide trainable variables belonging to the model."""
        ...

    @property
    def trainable_variables(self) -> Sequence[KerasVariableProtocol | TfTensorProtocol]:
        """Provide the sequence of trainable variables."""
        ...

    @property
    def preprocessor(self) -> KerasPreprocessorProtocol:
        """Provide the model's input preprocessor."""
        ...

    @property
    def last_scores(self) -> KerasScoreType:
        """Provide token generation scores from the most recent generation pass."""
        ...

    def save(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Save model weights or checkpoint to disk."""
        ...

    def load_weights(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Load serialized weights into the model."""
        ...


KerasCausalLMProtocol = KerasModelProtocol


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
    ) -> dict[str, PyTreeNode]:
        """Initialize stateless model parameters with a PRNGKey and sample input."""
        ...

    def apply(
        self,
        *args: ForwardArg | Mapping[str, PyTreeNode] | Mapping[str, TensorType] | Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]],
        **kwargs: ForwardArg,
    ) -> JaxArrayProtocol:
        """Apply a stateless transformation or forward pass."""
        ...


@runtime_checkable
class MaxTextModelConstructorProtocol(Protocol):
    """Protocol for `maxtext.models.gemma4.Gemma4Model` constructor."""

    def __call__(
        self,
        model_name: str,
        *,
        dtype: DTypeProtocol | str | None = None,
    ) -> MaxTextModelProtocol:
        """Instantiate a MaxText Gemma 4 model."""
        ...


MaxTextCausalLMProtocol = MaxTextModelProtocol
type PyTreeNode = Mapping[str, "PyTreeNode"] | JaxArrayProtocol | TensorType | SupportsFloat | str


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
class MlxLMLoadProtocol(Protocol):
    """Protocol for `mlx_lm.load(model_name)`."""

    def __call__(
        self,
        model_name: str,
        /,
    ) -> tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None] | MlxCausalLMProtocol:
        """Load an MLX causal language model and optional tokenizer."""
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
    def config(self) -> ModelConfigProtocol | Mapping[str, JSONValue] | None:
        """Provide the model configuration object."""
        ...

    def encode(
        self,
        *args: str | Sequence[str],
        **kwargs: int | bool | str | None,
    ) -> NDArrayProtocol | Sequence[Sequence[float]] | Sequence[float]:
        """Encode text or features into embeddings."""
        ...


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
class JaxShardingProtocol(Protocol):
    """Protocol for JAX sharding objects such as `NamedSharding`."""

    @property
    def device_set(self) -> set[str | int | None] | Sequence[str | int | None] | None:
        """Provide the set of devices covered by this sharding."""
        ...

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
    """Strongly-typed state container for standard stateful supervised training loops."""

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
class MaxTextTrainerState(Generic[TParams, TOpt, TBatch]):
    """Strongly-typed state container for stateless JAX/MaxText parameter and optimizer PyTree loops."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    train_step: Callable[..., StepResult | tuple[TParams, TOpt, StepResult]] | None = None
    params: TParams | None = None
    opt_state: TOpt | None = None
    policy_params: TParams | None = None
    ref_params: TParams | None = None


@dataclass
class TorchTrainerState(Generic[TModel, TOpt, TBatch]):
    """Strongly-typed state container for PyTorch models, optimizers, and loss criteria."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    model: TModel | None = None
    optimizer: TOpt | None = None
    criterion: Callable[..., TensorProtocol | float] | None = None
    device: str | None = "cpu"


@dataclass
class KerasTrainerState(Generic[TModel, TOpt, TBatch]):
    """Strongly-typed state container for Keras / TensorFlow compiled step execution."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    train_step: Callable[..., StepResult] | None = None
    model: TModel | None = None
    optimizer: TOpt | None = None


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
    model: TModel | None = None
    policy_model: TModel | None = None
    ref_model: TModel | None = None
    optimizer: TOpt | None = None
    criterion: Callable[..., TensorProtocol | float] | None = None
    device: str | None = "cpu"
    beta: float = 0.1


__all__ = [
    "CausalLMProtocol",
    "DPOLoopState",
    "EmbedderModelProtocol",
    "FlaxNNXModelProtocol",
    "ForwardArg",
    "ForwardOutput",
    "HFTokenizerProtocol",
    "JaxCausalLMProtocol",
    "JaxModelProtocol",
    "JaxShardingProtocol",
    "KerasBackboneProtocol",
    "KerasCausalLMProtocol",
    "KerasLayerProtocol",
    "KerasModelProtocol",
    "KerasTrainerState",
    "KerasVariableProtocol",
    "MLXModelProtocol",
    "MaxTextCausalLMProtocol",
    "MaxTextModelProtocol",
    "MaxTextTrainerState",
    "MlxCausalLMProtocol",
    "MlxCoreProtocol",
    "MlxModelProtocol",
    "MlxNNLossesProtocol",
    "MlxNNProtocol",
    "ModelConfigProtocol",
    "ModelType",
    "OrbaxCheckpointerProtocol",
    "PeftModelProtocol",
    "PyTreeNode",
    "QuantizableModelProtocol",
    "QuantizableModelType",
    "StepResult",
    "TBatch",
    "TModel",
    "TOpt",
    "TParams",
    "TfModuleProtocol",
    "TokenizerProtocol",
    "TorchCausalLMProtocol",
    "TorchModelProtocol",
    "TorchOptimizerProtocol",
    "TorchParameterProtocol",
    "TorchTrainerState",
    "TrainableModelProtocol",
    "TrainableModelType",
    "TrainerState",
    "TrainingLoopState",
    "VLLMCompletionOutputProtocol",
    "VLLMEngineProtocol",
    "VLLMRequestOutputProtocol",
]

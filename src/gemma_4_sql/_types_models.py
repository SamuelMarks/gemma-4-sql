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

type ForwardArg = (
    TensorType | Sequence[int] | Sequence[Sequence[int]] | NestedScalarList | str | int | float | bool | None
)
type ForwardOutput = (
    TensorProtocol
    | TorchTensorProtocol
    | MlxArrayProtocol
    | tuple[TensorProtocol | TorchTensorProtocol | MlxArrayProtocol, ...]
    | Mapping[str, TensorProtocol | TorchTensorProtocol | MlxArrayProtocol]
    | float
)


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
        """Provide the padding token ID.

        Returns:
            Integer pad token ID or `None` if unset.

        """
        ...

    @property
    def eos_token_id(self) -> int | None:
        """Provide the end-of-sequence token ID.

        Returns:
            Integer EOS token ID or `None` if unset.

        """
        ...

    @property
    def bos_token_id(self) -> int | None:
        """Provide the beginning-of-sequence token ID.

        Returns:
            Integer BOS token ID or `None` if unset.

        """
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
        """Return the vocabulary mapping token strings to integer IDs.

        Returns:
            Mapping from token string to integer ID.

        """
        ...


@runtime_checkable
class CausalLMProtocol(Protocol):
    """Protocol for callable causal language models supporting forward pass invocation."""

    def __call__(
        self,
        *args: ForwardArg,
        **kwargs: ForwardArg,
    ) -> ForwardOutput:
        """Execute the forward pass of the causal language model.

        Args:
            *args: Positional forward pass arguments.
            **kwargs: Keyword forward pass arguments.

        Returns:
            Model forward output logits or tuple.

        """
        ...


@runtime_checkable
class TorchParameterProtocol(Protocol):
    """Protocol representing a trainable PyTorch `nn.Parameter` or `torch.Tensor`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the parameter tensor.

        Returns:
            Tuple of dimension sizes.

        """
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the parameter tensor.

        Returns:
            Integer rank of the parameter tensor.

        """
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the parameter tensor.

        Returns:
            Element data type descriptor.

        """
        ...

    @property
    def device(self) -> DTypeProtocol | str | int | None:
        """Provide the device where the parameter tensor resides.

        Returns:
            Device descriptor or `None`.

        """
        ...


@runtime_checkable
class TorchCausalLMProtocol(Protocol):
    """Focused protocol for PyTorch causal language models."""

    def __call__(
        self,
        *args: ForwardArg,
        **kwargs: ForwardArg,
    ) -> ForwardOutput:
        """Execute the forward pass of the PyTorch module.

        Args:
            *args: Positional input tensors and options.
            **kwargs: Keyword input tensors and options.

        Returns:
            Logits tensor or structured forward output.

        """
        ...

    def parameters(self) -> Iterable[TorchParameterProtocol]:
        """Yield trainable parameter tensors.

        Returns:
            Iterable of parameter tensors.

        """
        ...

    def named_modules(self) -> Iterable[tuple[str, TorchCausalLMProtocol | CausalLMProtocol]]:
        """Yield `(name, module)` pairs across the module hierarchy.

        Returns:
            Iterable of `(qualified_name, submodule)` pairs.

        """
        ...

    def train(self, mode: bool = True) -> Self:
        """Toggle training mode on the PyTorch model.

        Args:
            mode: `True` for training mode, `False` for evaluation mode.

        Returns:
            Self module reference.

        """
        ...

    def eval(self) -> Self:
        """Set the PyTorch model to evaluation mode.

        Returns:
            Self module reference.

        """
        ...

    def state_dict(self) -> Mapping[str, TorchTensorProtocol | TensorProtocol]:
        """Return a dictionary containing references to the whole state of the module.

        Returns:
            Mapping from parameter/buffer names to tensors.

        """
        ...


@runtime_checkable
class TorchOptimizerProtocol(Protocol):
    """Protocol for PyTorch optimizers supporting `.zero_grad()` and `.step()`."""

    def zero_grad(self) -> None:
        """Reset gradients of all optimized `torch.Tensor` parameters."""
        ...

    def step(self) -> TensorProtocol | float | None:
        """Perform a single optimization step.

        Returns:
            Optional loss value evaluated by a closure or `None`.

        """
        ...


@runtime_checkable
class PeftModelProtocol(Protocol):
    """Protocol for a Hugging Face PEFT model.

    Supports `.print_trainable_parameters()`, `.save_pretrained()`, and `.named_modules()`.

    """

    def print_trainable_parameters(self) -> None:
        """Log trainable versus total parameter counts."""
        ...

    def save_pretrained(self, save_directory: str | Path, /) -> None:
        """Persist PEFT adapter weights and config to `save_directory`.

        Args:
            save_directory: Destination directory path.

        """
        ...

    def named_modules(self) -> Iterable[tuple[str, TorchCausalLMProtocol | CausalLMProtocol]]:
        """Yield `(name, module)` pairs across the wrapped PEFT hierarchy.

        Returns:
            Iterable of `(qualified_name, submodule)` pairs.

        """
        ...


@runtime_checkable
class VLLMCompletionOutputProtocol(Protocol):
    """Protocol for a single vLLM completion item exposing `.text`."""

    @property
    def text(self) -> str:
        """Provide the generated completion text.

        Returns:
            Decoded completion string.

        """
        ...


@runtime_checkable
class VLLMRequestOutputProtocol(Protocol):
    """Protocol for a vLLM streaming `RequestOutput` exposing `.outputs`."""

    @property
    def outputs(self) -> Sequence[VLLMCompletionOutputProtocol]:
        """Provide the sequence of candidate completion outputs.

        Returns:
            Sequence of completion candidates.

        """
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
        """Stream `VLLMRequestOutputProtocol` chunks for `prompt`.

        Args:
            prompt: Input prompt string.
            sampling_params: Optional vLLM sampling configuration.
            request_id: Unique request identifier.

        Returns:
            Async iterator of streaming `VLLMRequestOutputProtocol` chunks.

        """
        ...

    async def abort(self, request_id: str, /) -> None:
        """Abort an in-flight vLLM request by `request_id`.

        Args:
            request_id: Unique request identifier to cancel.

        """
        ...


@runtime_checkable
class KerasVariableProtocol(Protocol):
    """Protocol for a Keras 3 / TensorFlow variable supporting `.assign(...)` and `+`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the variable shape tuple.

        Returns:
            Tuple of variable dimensions.

        """
        ...

    def assign(self, value: NDArrayProtocol | TfTensorProtocol | TensorProtocol | SupportsFloat) -> None:
        """Assign a new array or tensor value to the variable in-place.

        Args:
            value: Replacement tensor, array, or scalar value.

        """
        ...

    def __add__(self, other: TfTensorProtocol | TensorProtocol | float, /) -> TfTensorProtocol:
        """Add a tensor or scalar to the variable.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Resulting sum tensor.

        """
        ...


@runtime_checkable
class KerasLayerProtocol(Protocol):
    """Protocol for a Keras 3 `Layer` exposing weight and trainability attributes.

    Exposes `.weights`, `.trainable_weights`, `.trainable_variables`, and `.trainable`.

    Attributes:
        trainable: Boolean flag controlling whether layer variables receive gradient updates.

    """

    @property
    def trainable(self) -> bool:
        """Indicate whether layer variables receive gradient updates.

        Returns:
            Boolean trainability flag.

        """
        ...

    @trainable.setter
    def trainable(self, value: bool) -> None:
        """Update whether layer variables receive gradient updates.

        Args:
            value: Desired trainability state.

        """
        ...

    @property
    def name(self) -> str:
        """Provide the layer name.

        Returns:
            Layer identifier string.

        """
        ...

    @property
    def weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide all variables belonging to the layer.

        Returns:
            Sequence of all layer variables.

        """
        ...

    @property
    def trainable_weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide trainable variables belonging to the layer.

        Returns:
            Sequence of trainable layer variables.

        """
        ...

    @property
    def trainable_variables(self) -> Sequence[KerasVariableProtocol | TfTensorProtocol]:
        """Provide trainable variables belonging to the layer.

        Returns:
            Sequence of trainable variables or tensors.

        """
        ...


@runtime_checkable
class KerasBackboneProtocol(Protocol):
    """Protocol for a KerasNLP `Backbone` supporting `.enable_lora(rank=...)` and `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol]:
        """Provide the sequence of backbone layers.

        Returns:
            Sequence of Keras layers in the backbone.

        """
        ...

    def enable_lora(self, *, rank: int) -> None:
        """Enable native KerasNLP LoRA with the given rank.

        Args:
            rank: Positive low-rank adapter bottleneck dimension.

        """
        ...


@runtime_checkable
class KerasPreprocessorProtocol(Protocol):
    """Protocol for a KerasNLP model preprocessor exposing mutable `sequence_length`.

    Attributes:
        sequence_length: Maximum padded/truncated sequence length.

    """

    sequence_length: int

    def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> Mapping[str, TensorType]:
        """Preprocess raw inputs into model tensors.

        Args:
            *args: Positional raw text or multimodal inputs.
            **kwargs: Keyword preprocessing options.

        Returns:
            Mapping of preprocessed feature tensors.

        """
        ...


@runtime_checkable
class KerasFitHistoryProtocol(Protocol):
    """Protocol for a Keras `.fit(...)` `History` object exposing `.history`."""

    @property
    def history(self) -> Mapping[str, Sequence[float]]:
        """Provide the dictionary of recorded training metric sequences.

        Returns:
            Mapping from metric name to per-epoch float sequence.

        """
        ...


@runtime_checkable
class KerasSamplerProtocol(Protocol):
    """Protocol for a KerasNLP autoregressive token sampler such as `BeamSampler`."""

    def __call__(
        self,
        *args: ForwardArg | Callable[..., TfTensorProtocol | TensorProtocol],
        **kwargs: ForwardArg,
    ) -> TfTensorProtocol | TensorProtocol:
        """Sample token sequences using a step-logit callback and prompt tensor.

        Args:
            *args: Next-token probability callback and prompt tensors.
            **kwargs: Optional decoding state (`cache`, `index`, `mask`, `stop_token_ids`).

        Returns:
            Sampled token ID tensor.

        """
        ...


type KerasScoreType = (
    TfTensorProtocol
    | NDArrayProtocol
    | NestedScalarList
    | float
    | int
    | str
    | Sequence[float | int | str | Sequence[float | int | str | None] | None]
    | None
)


@runtime_checkable
class KerasCompiledOptimizerProtocol(Protocol):
    """Protocol for a Keras optimizer passed to `model.compile(optimizer=...)`."""


@runtime_checkable
class KerasCausalLMProtocol(CausalLMProtocol, Protocol):
    """Focused protocol for Keras 3 / KerasNLP causal language models.

    Attributes:
        sampler: Configured `KerasSamplerProtocol`, preset name string, or `None`.

    """

    sampler: KerasSamplerProtocol | str | Callable[..., TensorProtocol] | None

    def generate(
        self,
        *args: ForwardArg | Sequence[str],
        **kwargs: ForwardArg,
    ) -> (
        TensorProtocol
        | TfTensorProtocol
        | Sequence[int]
        | Sequence[str]
        | Mapping[str, KerasScoreType]
        | tuple[str, KerasScoreType]
        | str
    ):
        """Generate token sequences autoregressively.

        Args:
            *args: Input prompt string, batch of prompts, or token tensor.
            **kwargs: Decoding options such as `max_length`.

        Returns:
            Generated text, token IDs, or structured generation output.

        """
        ...

    def compile(
        self,
        *args: str | Callable[..., TensorProtocol] | None,
        **kwargs: (
            str
            | float
            | Sequence[str]
            | Callable[..., TensorProtocol]
            | KerasSamplerProtocol
            | KerasCompiledOptimizerProtocol
            | None
        ),
    ) -> None:
        """Compile the model with loss, optimizer, and optional token sampler.

        Args:
            *args: Positional compilation options.
            **kwargs: Keyword compilation options (`optimizer`, `loss`, `sampler`).

        """
        ...

    def fit(
        self,
        *args: Iterable[
            tuple[TfTensorProtocol, TfTensorProtocol]
            | Mapping[str, TensorType | Sequence[int] | Sequence[Sequence[int]] | NestedScalarList]
        ]
        | Iterable[Mapping[str, TensorType]]
        | TensorType,
        **kwargs: float | bool | None,
    ) -> KerasFitHistoryProtocol:
        """Fit the model on a dataset iterator.

        Args:
            *args: Dataset iterable or input/target tensors.
            **kwargs: Keyword fit options such as `epochs`.

        Returns:
            `KerasFitHistoryProtocol` containing recorded epoch metrics.

        """
        ...

    @property
    def layers(self) -> Sequence[KerasLayerProtocol]:
        """Provide the sequence of Keras layers.

        Returns:
            Sequence of model layers.

        """
        ...

    @property
    def weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide all variables belonging to the model.

        Returns:
            Sequence of all model variables.

        """
        ...

    @property
    def trainable_weights(self) -> Sequence[KerasVariableProtocol]:
        """Provide trainable variables belonging to the model.

        Returns:
            Sequence of trainable model variables.

        """
        ...

    @property
    def trainable_variables(self) -> Sequence[KerasVariableProtocol | TfTensorProtocol]:
        """Provide the sequence of trainable variables.

        Returns:
            Sequence of trainable variables or tensors.

        """
        ...

    @property
    def preprocessor(self) -> KerasPreprocessorProtocol:
        """Provide the model's input preprocessor.

        Returns:
            Attached `KerasPreprocessorProtocol` instance.

        """
        ...

    @property
    def last_scores(self) -> KerasScoreType:
        """Provide token generation scores from the most recent generation pass.

        Returns:
            Most recent beam or token score structure.

        """
        ...

    def save(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Save model weights or checkpoint to disk.

        Args:
            *args: Destination file or directory path.
            **kwargs: Serialization format options.

        """
        ...

    def load_weights(self, *args: str | Path, **kwargs: bool | str | None) -> None:
        """Load serialized weights into the model.

        Args:
            *args: Source weights file path.
            **kwargs: Weight loading options.

        """
        ...


@runtime_checkable
class TfModuleProtocol(Protocol):
    """Protocol for the `tensorflow` runtime module."""

    @property
    def int32(self) -> DTypeProtocol | str:
        """Provide `tf.int32` dtype.

        Returns:
            32-bit signed integer dtype descriptor.

        """
        ...

    @property
    def float32(self) -> DTypeProtocol | str:
        """Provide `tf.float32` dtype.

        Returns:
            32-bit floating-point dtype descriptor.

        """
        ...

    def cast(
        self,
        x: TfTensorProtocol | TensorProtocol | bool,
        dtype: DTypeProtocol | str,
        /,
    ) -> TfTensorProtocol:
        """Cast tensor `x` to `dtype`.

        Args:
            x: Input tensor or boolean scalar.
            dtype: Target dtype descriptor.

        Returns:
            Tensor cast to `dtype`.

        """
        ...

    def expand_dims(self, input: TfTensorProtocol, axis: int = -1, /) -> TfTensorProtocol:
        """Insert a dimension of length 1 at `axis`.

        Args:
            input: Input tensor.
            axis: Axis index at which to insert singleton dimension.

        Returns:
            Expanded tensor.

        """
        ...

    def gather(
        self,
        params: TfTensorProtocol,
        indices: TfTensorProtocol,
        *,
        batch_dims: int = 0,
    ) -> TfTensorProtocol:
        """Gather slices from `params` according to `indices`.

        Args:
            params: Source parameter/logit tensor.
            indices: Index tensor specifying slices to gather.
            batch_dims: Number of leading batch dimensions.

        Returns:
            Gathered slice tensor.

        """
        ...

    def squeeze(self, input: TfTensorProtocol, axis: int = -1, /) -> TfTensorProtocol:
        """Remove dimensions of size 1 along `axis`.

        Args:
            input: Input tensor.
            axis: Axis index to squeeze.

        Returns:
            Squeezed tensor.

        """
        ...

    def reduce_sum(self, input_tensor: TfTensorProtocol, axis: int = -1, /) -> TfTensorProtocol:
        """Compute the sum of elements across dimensions of a tensor.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Axis along which to sum.

        Returns:
            Reduced sum tensor.

        """
        ...

    def reduce_mean(self, input_tensor: TfTensorProtocol, /) -> TfTensorProtocol:
        """Compute the mean of elements across dimensions of a tensor.

        Args:
            input_tensor: Input tensor to average.

        Returns:
            Scalar mean tensor.

        """
        ...


@runtime_checkable
class ModelConfigProtocol(Protocol):
    """Protocol representing a Gemma 4 model configuration object."""

    @property
    def vocab_size(self) -> int:
        """Provide the vocabulary size.

        Returns:
            Total token vocabulary size.

        """
        ...

    @property
    def hidden_size(self) -> int:
        """Provide the hidden embedding size.

        Returns:
            Transformer hidden dimension.

        """
        ...

    @property
    def num_hidden_layers(self) -> int:
        """Provide the number of transformer layers.

        Returns:
            Number of decoder blocks.

        """
        ...


@runtime_checkable
class FlaxNNXModelProtocol(Protocol):
    """Protocol for stateful JAX/Flax NNX neural network modules exposing `.config` and `__call__`."""

    @property
    def config(self) -> ModelConfigProtocol | None:
        """Provide the architectural configuration of the Flax NNX module.

        Returns:
            `ModelConfigProtocol` instance or `None`.

        """
        ...

    def __call__(
        self,
        *args: ForwardArg,
        **kwargs: ForwardArg,
    ) -> ForwardOutput:
        """Execute the forward pass of the Flax NNX module.

        Args:
            *args: Positional input arrays and cache arguments.
            **kwargs: Keyword input arrays and multimodal arguments.

        Returns:
            Hidden state array, logits array, or `(output, cache)` tuple.

        """
        ...


@runtime_checkable
class JaxCausalLMProtocol(FlaxNNXModelProtocol, Protocol):
    """Focused protocol for JAX/Flax NNX causal language models exposing `.model`, `.config`, and `__call__`."""

    @property
    def model(self) -> FlaxNNXModelProtocol | CausalLMProtocol | None:
        """Provide the inner transformer decoder trunk module.

        Returns:
            Inner `FlaxNNXModelProtocol` trunk or `None`.

        """
        ...


@runtime_checkable
class OrbaxCheckpointerProtocol[StateT](Protocol):
    """Protocol for an Orbax `PyTreeCheckpointer` supporting `.save(path, state)` and `.restore(path)`."""

    def save(self, directory: Path | str, item: StateT) -> None:
        """Persist a PyTree checkpoint to `directory`.

        Args:
            directory: Target checkpoint directory path.
            item: PyTree state object to serialize.

        """
        ...

    def restore(self, directory: Path | str) -> StateT:
        """Restore a PyTree checkpoint from `directory`.

        Args:
            directory: Source checkpoint directory path.

        Returns:
            Restored PyTree state object.

        """
        ...


type PyTreeNode = Mapping[str, "PyTreeNode"] | JaxArrayProtocol | TensorType | SupportsFloat | str


@runtime_checkable
class MaxTextCausalLMProtocol(Protocol):
    """Focused protocol for stateless Flax Linen / MaxText causal language models."""

    def init(
        self,
        *args: ForwardArg | Mapping[str, TensorType],
        **kwargs: ForwardArg,
    ) -> dict[str, PyTreeNode]:
        """Initialize stateless model parameters with a PRNGKey and sample input.

        Args:
            *args: PRNG key and sample input tensors.
            **kwargs: Optional initialization flags.

        Returns:
            Initialized nested parameter PyTree dictionary.

        """
        ...

    def apply(
        self,
        *args: ForwardArg
        | Mapping[str, PyTreeNode]
        | Mapping[str, TensorType]
        | Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]],
        **kwargs: ForwardArg,
    ) -> JaxArrayProtocol:
        """Apply a stateless transformation or forward pass.

        Args:
            *args: Parameter PyTree and input token/feature arrays.
            **kwargs: Optional forward pass flags.

        Returns:
            Output logits `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class MaxTextModelConstructorProtocol(Protocol):
    """Protocol for `maxtext.models.gemma4.Gemma4Model` constructor."""

    def __call__(
        self,
        model_name: str,
        *,
        dtype: DTypeProtocol | str | None = None,
    ) -> MaxTextCausalLMProtocol:
        """Instantiate a MaxText Gemma 4 model.

        Args:
            model_name: Normalized MaxText architecture identifier.
            dtype: Optional compute/parameter dtype specification.

        Returns:
            Instantiated `MaxTextCausalLMProtocol` model.

        """
        ...


@runtime_checkable
class MlxCoreProtocol(Protocol):
    """Protocol for `mlx.core` (`mx`) runtime functions."""

    def array(
        self,
        val: Sequence[int]
        | Sequence[float]
        | Sequence[Sequence[int]]
        | Sequence[Sequence[float]]
        | MlxArrayProtocol
        | NDArrayProtocol,
        *,
        dtype: DTypeProtocol | DTypeSpec = None,
    ) -> MlxArrayProtocol:
        """Construct an MLX array.

        Args:
            val: Input nested sequence or array.
            dtype: Optional target MLX data type.

        Returns:
            Constructed `MlxArrayProtocol`.

        """
        ...

    def logsumexp(
        self,
        a: MlxArrayProtocol,
        axis: int = -1,
        keepdims: bool = False,
    ) -> MlxArrayProtocol:
        """Compute log-sum-exp along `axis`.

        Args:
            a: Input MLX array.
            axis: Reduction axis index.
            keepdims: Whether to retain reduced dimensions of size 1.

        Returns:
            Reduced log-sum-exp MLX array.

        """
        ...

    def take_along_axis(
        self,
        a: MlxArrayProtocol,
        indices: MlxArrayProtocol,
        axis: int = -1,
    ) -> MlxArrayProtocol:
        """Gather elements from `a` along `axis` at `indices`.

        Args:
            a: Source MLX array.
            indices: Integer index array.
            axis: Axis along which to gather.

        Returns:
            Gathered MLX array.

        """
        ...

    def expand_dims(
        self,
        a: MlxArrayProtocol,
        axis: int = -1,
    ) -> MlxArrayProtocol:
        """Insert a singleton dimension at `axis`.

        Args:
            a: Input MLX array.
            axis: Axis index at which to insert dimension of size 1.

        Returns:
            Expanded MLX array.

        """
        ...

    def squeeze(
        self,
        a: MlxArrayProtocol,
        axis: int = -1,
    ) -> MlxArrayProtocol:
        """Remove a singleton dimension at `axis`.

        Args:
            a: Input MLX array.
            axis: Axis index to squeeze.

        Returns:
            Squeezed MLX array.

        """
        ...

    def eval(self, *args: MlxArrayProtocol | Mapping[str, MlxArrayProtocol] | None) -> None:
        """Evaluate lazy MLX arrays and parameter trees.

        Args:
            *args: Lazy MLX arrays or nested parameter trees to materialize.

        """
        ...


@runtime_checkable
class MlxNNLossesProtocol(Protocol):
    """Protocol for `mlx.nn.losses` exposing `log_sigmoid` and `cross_entropy`."""

    def log_sigmoid(self, x: MlxArrayProtocol | float) -> MlxArrayProtocol:
        """Compute elementwise log-sigmoid.

        Args:
            x: Input MLX array or scalar float.

        Returns:
            Elementwise `log(sigmoid(x))` array.

        """
        ...

    def cross_entropy(
        self,
        logits: MlxArrayProtocol,
        targets: MlxArrayProtocol,
        *,
        reduction: str = "mean",
    ) -> MlxArrayProtocol:
        """Compute cross-entropy loss.

        Args:
            logits: Unnormalized vocabulary logits array.
            targets: Ground-truth integer token target array.
            reduction: Loss reduction mode (`'mean'`, `'sum'`, or `'none'`).

        Returns:
            Computed cross-entropy loss array.

        """
        ...


@runtime_checkable
class MlxCausalLMProtocol(Protocol):
    """Focused protocol for Apple Silicon MLX causal language models."""

    def __call__(
        self,
        *args: ForwardArg | MlxArrayProtocol,
        **kwargs: ForwardArg | MlxArrayProtocol,
    ) -> ForwardOutput | MlxArrayProtocol:
        """Execute the forward pass of the MLX model.

        Args:
            *args: Positional input token arrays and cache state.
            **kwargs: Keyword forward arguments.

        Returns:
            Output logits `MlxArrayProtocol` or structured forward output.

        """
        ...

    def parameters(
        self,
    ) -> Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]]:
        """Return the parameter tree of the MLX model.

        Returns:
            Nested mapping of model parameter arrays.

        """
        ...

    def update(
        self,
        parameters: Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]],
    ) -> Self | None:
        """Update the parameters of the MLX model with a parameter tree.

        Args:
            parameters: Nested mapping of replacement parameter arrays.

        Returns:
            Optional self reference or `None`.

        """
        ...

    def freeze(self) -> Self | None:
        """Freeze base model parameters prior to adapter training.

        Returns:
            Optional self reference or `None`.

        """
        ...


@runtime_checkable
class MlxLMLoadProtocol(Protocol):
    """Protocol for `mlx_lm.load(model_name)`."""

    def __call__(
        self,
        model_name: str,
        /,
    ) -> tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None] | MlxCausalLMProtocol:
        """Load an MLX causal language model and optional tokenizer.

        Args:
            model_name: Hugging Face repository ID or local MLX checkpoint directory.

        Returns:
            `(model, tokenizer)` pair or loaded `MlxCausalLMProtocol`.

        """
        ...


@runtime_checkable
class MlxNNProtocol(Protocol):
    """Protocol for `mlx.nn` exposing `losses`, `log_sigmoid`, `value_and_grad`, and `quantize`."""

    @property
    def losses(self) -> MlxNNLossesProtocol:
        """Provide the `mlx.nn.losses` submodule.

        Returns:
            `MlxNNLossesProtocol` loss functions namespace.

        """
        ...

    def log_sigmoid(self, x: MlxArrayProtocol | float) -> MlxArrayProtocol:
        """Compute elementwise log-sigmoid.

        Args:
            x: Input MLX array or scalar.

        Returns:
            Elementwise log-sigmoid array.

        """
        ...

    def value_and_grad(
        self,
        model: MlxCausalLMProtocol,
        fn: Callable[..., MlxArrayProtocol],
    ) -> Callable[..., tuple[MlxArrayProtocol, dict[str, MlxArrayProtocol]]]:
        """Transform `fn` to compute value and gradients with respect to `model`.

        Args:
            model: Target trainable `MlxCausalLMProtocol` module.
            fn: Scalar-loss function to differentiate.

        Returns:
            Callable returning `(loss_array, grad_tree)` for each batch.

        """
        ...

    def quantize(
        self,
        model: MlxCausalLMProtocol,
        *,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        """Quantize linear layers of `model` in-place.

        Args:
            model: Target `MlxCausalLMProtocol` to quantize.
            group_size: Quantization block/group size.
            bits: Bit-width per quantized weight (`4` or `8`).

        """
        ...


@runtime_checkable
class EmbedderModelProtocol(Protocol):
    """Focused protocol for text and schema embedding models."""

    def encode(
        self,
        *args: str | Sequence[str],
        **kwargs: int | bool | str | None,
    ) -> NDArrayProtocol | Sequence[Sequence[float]] | Sequence[float]:
        """Encode text or features into dense vector embeddings.

        Args:
            *args: Single string or sequence of strings to embed.
            **kwargs: Optional batching or normalization flags.

        Returns:
            1D or 2D embedding vector array/sequence.

        """
        ...


TrainableModelType = Union[
    JaxCausalLMProtocol,
    TorchCausalLMProtocol,
    KerasCausalLMProtocol,
    MaxTextCausalLMProtocol,
    MlxCausalLMProtocol,
]

QuantizableModelType = Union[
    JaxCausalLMProtocol,
    TorchCausalLMProtocol,
    KerasCausalLMProtocol,
    MaxTextCausalLMProtocol,
    MlxCausalLMProtocol,
]

ModelType = Union[
    TrainableModelType,
    EmbedderModelProtocol,
]


@runtime_checkable
class JaxShardingProtocol(Protocol):
    """Protocol for JAX sharding objects such as `NamedSharding`."""

    @property
    def device_set(self) -> set[str | int | None] | Sequence[str | int | None] | None:
        """Provide the set of devices covered by this sharding.

        Returns:
            Set or sequence of accelerator device descriptors.

        """
        ...

    def __repr__(self) -> str:
        """Return the string representation of the sharding.

        Returns:
            Human-readable sharding description.

        """
        ...


TModel = TypeVar("TModel")
TOpt = TypeVar("TOpt")
TBatch = TypeVar("TBatch")
TParams = TypeVar("TParams")

type StepResult = (
    TensorProtocol
    | MlxArrayProtocol
    | ScalarItemExtractorProtocol
    | SupportsFloat
    | float
    | str
    | tuple[
        TensorProtocol
        | MlxArrayProtocol
        | ScalarItemExtractorProtocol
        | SupportsFloat
        | float
        | Mapping[str, TensorProtocol]
        | Mapping[str, MlxArrayProtocol]
        | JSONValue,
        ...,
    ]
)


@dataclass
class JaxTrainingState(Generic[TModel, TOpt, TBatch]):
    """Invariant-preserving state container for JAX/Flax NNX supervised training loops.

    Attributes:
        dataloader: Batch iterable supplying tokenized training mappings.
        model: Stateful Flax NNX model under optimization.
        optimizer: NNX optimizer bound to `model`.
        train_step: Compiled or eager step callable `(model, optimizer, batch) -> loss`.
        epochs: Number of training passes over `dataloader`.
        learning_rate: Peak optimizer learning rate.
        sharding: Optional JAX device or mesh sharding specification.

    """

    dataloader: Iterable[TBatch]
    model: TModel
    optimizer: TOpt
    train_step: Callable[..., StepResult]
    epochs: int = 1
    learning_rate: float = 1e-4
    sharding: str | Mapping[str, str] | JaxShardingProtocol | None = None


@dataclass
class JaxDPOState(Generic[TModel, TOpt, TBatch]):
    """Invariant-preserving state container for JAX/Flax NNX Direct Preference Optimization loops.

    Attributes:
        dataloader: Preference batch iterable yielding chosen/rejected sequences.
        policy_model: Trainable policy model updated by DPO gradients.
        ref_model: Frozen reference model providing baseline log-probabilities.
        optimizer: NNX optimizer bound to `policy_model`.
        train_step: Compiled or eager step callable `(policy_model, ref_model, optimizer, batch) -> loss`.
        epochs: Number of training epochs over `dataloader`.
        beta: Positive KL-divergence regularization strength.

    """

    dataloader: Iterable[TBatch]
    policy_model: TModel
    ref_model: TModel
    optimizer: TOpt
    train_step: Callable[..., StepResult]
    epochs: int = 1
    beta: float = 0.1


@dataclass
class DPOLoopState(Generic[TModel, TOpt, TBatch]):
    """Invariant-preserving state container for stateful DPO epoch loops (PyTorch and MLX).

    Attributes:
        dataloader: Preference batch iterable yielding chosen/rejected sequences.
        policy_model: Trainable policy model updated by DPO gradients.
        ref_model: Frozen reference model providing baseline log-probabilities.
        optimizer: Optimizer updating `policy_model` parameters.
        epochs: Number of training epochs over `dataloader`.
        beta: Positive KL-divergence regularization strength.

    """

    dataloader: Iterable[TBatch]
    policy_model: TModel
    ref_model: TModel
    optimizer: TOpt
    epochs: int = 1
    beta: float = 0.1


@dataclass
class MaxTextTrainerState(Generic[TParams, TOpt, TBatch]):
    """Invariant-preserving state container for stateless MaxText supervised training loops.

    Attributes:
        dataloader: Batch iterable yielding tokenized training arrays.
        params: Stateless model parameter PyTree.
        opt_state: Optax optimizer state PyTree.
        train_step: Pure step callable `(params, opt_state, batch) -> (params, opt_state, loss)`.
        epochs: Number of training epochs over `dataloader`.

    """

    dataloader: Iterable[TBatch]
    params: TParams
    opt_state: TOpt
    train_step: Callable[..., tuple[TParams, TOpt, ScalarItemExtractorProtocol | SupportsFloat | float]]
    epochs: int = 1


@dataclass
class MaxTextDPOState(Generic[TParams, TOpt, TBatch]):
    """Invariant-preserving state container for stateless MaxText DPO preference optimization loops.

    Attributes:
        dataloader: Preference batch iterable yielding chosen/rejected sequences.
        policy_params: Trainable policy parameter PyTree.
        ref_params: Frozen reference parameter PyTree.
        opt_state: Optax optimizer state PyTree.
        train_step: Pure DPO step callable
            `(policy_params, ref_params, opt_state, batch) -> (policy_params, opt_state, loss)`.
        epochs: Number of training epochs over `dataloader`.
        beta: Positive KL-divergence regularization strength.

    """

    dataloader: Iterable[TBatch]
    policy_params: TParams
    ref_params: TParams
    opt_state: TOpt
    train_step: Callable[..., tuple[TParams, TOpt, ScalarItemExtractorProtocol | SupportsFloat | float]]
    epochs: int = 1
    beta: float = 0.1


@dataclass
class TorchTrainerState(Generic[TModel, TOpt, TBatch]):
    """Invariant-preserving state container for PyTorch supervised training loops.

    Attributes:
        dataloader: Batch iterable yielding tokenized PyTorch tensors.
        model: Trainable PyTorch causal language model.
        optimizer: PyTorch optimizer bound to `model.parameters()`.
        criterion: Loss callable computing cross-entropy from logits and labels.
        epochs: Number of training epochs over `dataloader`.
        device: Target accelerator device string (`'cpu'`, `'cuda'`, `'mps'`).

    """

    dataloader: Iterable[TBatch]
    model: TModel
    optimizer: TOpt
    criterion: Callable[..., TensorProtocol | float]
    epochs: int = 1
    device: str = "cpu"


@dataclass
class KerasTrainerState(Generic[TModel, TOpt, TBatch]):
    """Invariant-preserving state container for Keras / TensorFlow compiled step execution.

    Attributes:
        dataloader: Batch iterable yielding Keras/TensorFlow input-target tensors.
        model: Trainable Keras causal language model.
        optimizer: Keras optimizer updating `model.trainable_variables`.
        train_step: Step callable executing a single forward/backward update.
        epochs: Number of training epochs over `dataloader`.

    """

    dataloader: Iterable[TBatch]
    model: TModel
    optimizer: TOpt
    train_step: Callable[..., StepResult]
    epochs: int = 1


@dataclass
class MlxTrainerState(Generic[TModel, TOpt, TBatch]):
    """Invariant-preserving state container for Apple Silicon MLX supervised training loops.

    Attributes:
        dataloader: Batch iterable yielding tokenized input/target sequences.
        model: Trainable `MlxCausalLMProtocol` instance.
        optimizer: MLX optimizer updating `model` parameters.
        loss_and_grad_fn: Differentiated `mlx.nn.value_and_grad` step callable.
        epochs: Number of training epochs over `dataloader`.
        learning_rate: Optimizer learning rate.

    """

    dataloader: Iterable[TBatch]
    model: TModel
    optimizer: TOpt
    loss_and_grad_fn: Callable[..., StepResult]
    epochs: int = 1
    learning_rate: float = 1e-4


__all__ = [
    "CausalLMProtocol",
    "DPOLoopState",
    "EmbedderModelProtocol",
    "FlaxNNXModelProtocol",
    "ForwardArg",
    "ForwardOutput",
    "HFTokenizerProtocol",
    "JaxCausalLMProtocol",
    "JaxDPOState",
    "JaxShardingProtocol",
    "JaxTrainingState",
    "KerasBackboneProtocol",
    "KerasCausalLMProtocol",
    "KerasFitHistoryProtocol",
    "KerasLayerProtocol",
    "KerasPreprocessorProtocol",
    "KerasSamplerProtocol",
    "KerasScoreType",
    "KerasTrainerState",
    "KerasVariableProtocol",
    "MaxTextCausalLMProtocol",
    "MaxTextDPOState",
    "MaxTextModelConstructorProtocol",
    "MaxTextTrainerState",
    "MlxCausalLMProtocol",
    "MlxCoreProtocol",
    "MlxLMLoadProtocol",
    "MlxNNLossesProtocol",
    "MlxNNProtocol",
    "MlxTrainerState",
    "ModelConfigProtocol",
    "ModelType",
    "OrbaxCheckpointerProtocol",
    "PeftModelProtocol",
    "PyTreeNode",
    "QuantizableModelType",
    "StepResult",
    "TBatch",
    "TModel",
    "TOpt",
    "TParams",
    "TfModuleProtocol",
    "TokenizerProtocol",
    "TorchCausalLMProtocol",
    "TorchOptimizerProtocol",
    "TorchParameterProtocol",
    "TorchTrainerState",
    "TrainableModelType",
    "VLLMCompletionOutputProtocol",
    "VLLMEngineProtocol",
    "VLLMRequestOutputProtocol",
]

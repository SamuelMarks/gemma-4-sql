"""PyTorch-specific inference logic."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from gemma_4_sql._types_tensors import extract_scalar_float
from gemma_4_sql.backends.common_benchmark import normalize_hardware_family
from gemma_4_sql.backends.common_dpo import ScalarItemProtocol
from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    CacheFingerprintProtocol,
    make_model_cache_key,
    normalize_logprob_confidence,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_multimodal import prepare_multimodal_inputs
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.backends.pytorch.runtime_adapter import TorchRuntimeAdapter
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    DTypeProtocol,
    ForwardArg,
    ForwardOutput,
    InferenceResultDict,
    NDArrayProtocol,
    TorchCausalLMProtocol,
    TorchTensorProtocol,
    extract_status_kwargs,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasInputIdsProtocol(Protocol):
    """Protocol for minimal tokenizer outputs exposing `.input_ids`."""

    @property
    def input_ids(self) -> TorchTensorProtocol:
        """Provide the encoded token ID tensor.

        Returns:
            Encoded prompt `input_ids` tensor.

        """
        ...


@runtime_checkable
class _BatchEncodingProtocol(_HasInputIdsProtocol, Protocol):
    """Protocol for a Hugging Face `BatchEncoding` exposing `.input_ids`, `.keys()`, `__getitem__`, and `.to()`."""

    def keys(self) -> Iterable[str]:
        """Yield mapping keys for `**inputs` unpacking.

        Returns:
            Iterable of string tensor keys.

        """
        ...

    def __getitem__(self, key: str, /) -> TorchTensorProtocol:
        """Look up a tensor by key for `**inputs` unpacking.

        Args:
            key: Batch encoding field name.

        Returns:
            Corresponding tensor value.

        """
        ...

    def to(self, device: str, /) -> _BatchEncodingProtocol:
        """Move encoded tensors to `device`.

        Args:
            device: Target device identifier string.

        Returns:
            Device-placed `_BatchEncodingProtocol`.

        """
        ...


@runtime_checkable
class _HFTokenizerInstanceProtocol(Protocol):
    """Protocol for a loaded Hugging Face tokenizer supporting `__call__` and `.decode`."""

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
    ) -> _BatchEncodingProtocol | _HasInputIdsProtocol | Mapping[str, TorchTensorProtocol]:
        """Encode `text` into a PyTorch `BatchEncoding` or tensor mapping.

        Args:
            text: Input text prompt.
            return_tensors: Tensor framework code (`'pt'`).

        Returns:
            Encoded `_BatchEncodingProtocol` or tensor mapping.

        """
        ...

    def decode(
        self,
        token_ids: TorchTensorProtocol | Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back into a string.

        Args:
            token_ids: Generated token ID sequence.
            skip_special_tokens: Whether to omit special control tokens.

        Returns:
            Decoded text string.

        """
        ...


@runtime_checkable
class _HFTokenizerLoaderProtocol(Protocol):
    """Protocol for `transformers.AutoTokenizer` exposing `.from_pretrained`."""

    def from_pretrained(self, model_name: str, /) -> _HFTokenizerInstanceProtocol:
        """Load a tokenizer from a pretrained identifier or path.

        Args:
            model_name: Pretrained model identifier or local path.

        Returns:
            Loaded `_HFTokenizerInstanceProtocol`.

        """
        ...


@runtime_checkable
class _HFGenerateOutputProtocol(Protocol):
    """Protocol for Hugging Face `.generate(return_dict_in_generate=True, output_scores=True)` output."""

    @property
    def sequences(self) -> Sequence[TorchTensorProtocol]:
        """Provide the generated token ID sequences.

        Returns:
            Sequence of generated token ID tensors.

        """
        ...


@runtime_checkable
class _HFGenerateScoredOutputProtocol(Protocol):
    """Protocol for Hugging Face generate output exposing `.sequences_scores`."""

    @property
    def sequences_scores(self) -> Sequence[ScalarItemProtocol]:
        """Provide beam sequence log-probability scores.

        Returns:
            Sequence of scalar log-probability score tensors.

        """
        ...


@runtime_checkable
class _MovableInferenceModelProtocol(Protocol):
    """Protocol for inference models supporting `.to(device)`."""

    def to[ModelSelfT](self: ModelSelfT, device: str, /) -> ModelSelfT:
        """Move model onto `device`.

        Args:
            device: Target device string.

        Returns:
            Model placed on `device`.

        """
        ...


@runtime_checkable
class _HFModelLoaderProtocol(Protocol):
    """Protocol for `transformers.AutoModelForCausalLM` exposing `.from_pretrained`."""

    def from_pretrained(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
    ) -> _HFGenerateDictModelProtocol:
        """Load a Hugging Face causal LM from a pretrained identifier or path.

        Args:
            model_name: Pretrained model identifier or local path.
            device_map: Device placement strategy.

        Returns:
            Loaded `_HFGenerateDictModelProtocol`.

        """
        ...


@runtime_checkable
class _PeftModelLoaderProtocol(Protocol):
    """Protocol for `peft.PeftModel` exposing `.from_pretrained(model, adapter_path)`."""

    def from_pretrained(
        self,
        model: TorchCausalLMProtocol | _HFGenerateDictModelProtocol,
        adapter_path: str,
        /,
    ) -> _HFGenerateDictModelProtocol:
        """Load and attach a PEFT adapter onto `model`.

        Args:
            model: Base PyTorch causal LM.
            adapter_path: Path to saved PEFT adapter directory.

        Returns:
            Adapter-wrapped `_HFGenerateDictModelProtocol`.

        """
        ...


@runtime_checkable
class _DeviceMovableTensorProtocol(TorchTensorProtocol, Protocol):
    """Protocol for a PyTorch tensor supporting `.device` and `.unsqueeze(dim)`."""

    @property
    def device(self) -> str | int | None:
        """Provide the device where the tensor resides.

        Returns:
            Device identifier or `None`.

        """
        ...

    def unsqueeze(self, dim: int, /) -> _DeviceMovableTensorProtocol:
        """Insert a singleton dimension at `dim`.

        Args:
            dim: Axis index.

        Returns:
            Unsqueezed tensor view.

        """
        ...


@runtime_checkable
class _HFGenerateDictModelProtocol(Protocol):
    """Protocol for a Hugging Face model whose `.generate(...)` returns `_HFGenerateOutputProtocol`."""

    def generate(
        self,
        *args: TorchTensorProtocol | int | bool,
        **kwargs: TorchTensorProtocol | int | bool,
    ) -> _HFGenerateOutputProtocol:
        """Generate token sequences and return a structured `_HFGenerateOutputProtocol`.

        Args:
            *args: Positional generation arguments.
            **kwargs: Keyword generation options.

        Returns:
            Structured `_HFGenerateOutputProtocol` generation output.

        """
        ...


@runtime_checkable
class _NativeGenerateModelProtocol(Protocol):
    """Protocol for a native PyTorch causal LM supporting `.generate(...)` and forward calls."""

    def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> ForwardOutput:
        """Execute the forward pass of the PyTorch module.

        Args:
            *args: Positional forward arguments.
            **kwargs: Keyword forward arguments.

        Returns:
            Forward logits or tuple output.

        """
        ...

    def generate(
        self,
        input_ids: TorchTensorProtocol,
        **kwargs: TorchTensorProtocol | int | bool | None,
    ) -> TorchTensorProtocol:
        """Generate token sequences autoregressively.

        Args:
            input_ids: Prompt token ID tensor.
            **kwargs: Generation keyword arguments.

        Returns:
            Generated token ID tensor.

        """
        ...


@runtime_checkable
class _GatherableLogProbsProtocol(TorchTensorProtocol, Protocol):
    """Protocol for a PyTorch log-probabilities tensor supporting `.gather(dim, index)`."""

    def gather(self, dim: int, index: TorchTensorProtocol, /) -> TorchTensorProtocol:
        """Gather values along `dim` at `index`.

        Args:
            dim: Axis index along which to gather.
            index: Target index tensor.

        Returns:
            Gathered log-probabilities tensor.

        """
        ...


@runtime_checkable
class _TorchInferenceFunctionalProtocol(Protocol):
    """Protocol for `torch.nn.functional` providing `log_softmax`."""

    def log_softmax(self, input_tensor: TorchTensorProtocol, *, dim: int = -1) -> _GatherableLogProbsProtocol:
        """Compute log-softmax along `dim`.

        Args:
            input_tensor: Vocabulary logits tensor.
            dim: Normalization axis index.

        Returns:
            Log-probabilities tensor supporting `.gather(...)`.

        """
        ...


@runtime_checkable
class _TorchInferenceNNProtocol(Protocol):
    """Protocol for `torch.nn` exposing `.functional`."""

    @property
    def functional(self) -> _TorchInferenceFunctionalProtocol:
        """Provide `torch.nn.functional` namespace."""
        ...


@runtime_checkable
class _TorchInferenceModuleProtocol(Protocol):
    """Protocol for `torch` tensor construction, `no_grad`, and `nn.functional.log_softmax` in inference."""

    @property
    def float32(self) -> DTypeProtocol:
        """Provide `torch.float32` dtype."""
        ...

    @property
    def long(self) -> DTypeProtocol:
        """Provide `torch.long` (`int64`) dtype."""
        ...

    @property
    def nn(self) -> _TorchInferenceNNProtocol:
        """Provide `torch.nn` namespace."""
        ...

    def tensor(
        self,
        data: (
            NDArrayProtocol
            | TorchTensorProtocol
            | Sequence[float]
            | Sequence[Sequence[int]]
            | Sequence[Sequence[float]]
            | Sequence[Sequence[Sequence[float]]]
        ),
        *,
        dtype: DTypeProtocol,
    ) -> _DeviceMovableTensorProtocol:
        """Construct a PyTorch tensor from token IDs or multimodal features.

        Args:
            data: Token ID matrix or multimodal feature array.
            dtype: Target PyTorch dtype.

        Returns:
            Constructed `_DeviceMovableTensorProtocol`.

        """
        ...

    def no_grad(self) -> AbstractContextManager[None]:
        """Return a context manager disabling autograd tracking.

        Returns:
            Autograd-disabling context manager.

        """
        ...


torch: _TorchInferenceModuleProtocol | None = optional_import("torch", _TorchInferenceModuleProtocol)
AutoModelForCausalLM: _HFModelLoaderProtocol | None = import_optional_symbol(
    "transformers",
    "AutoModelForCausalLM",
    _HFModelLoaderProtocol,
)
AutoTokenizer: _HFTokenizerLoaderProtocol | None = import_optional_symbol(
    "transformers",
    "AutoTokenizer",
    _HFTokenizerLoaderProtocol,
)

type _CachedNativeModel = TorchCausalLMProtocol | _NativeGenerateModelProtocol
type _CachedPyTorchModel = _CachedNativeModel | _HFGenerateDictModelProtocol
_MODEL_CACHE: BoundedLRUCache[tuple[_CachedNativeModel, _HFTokenizerInstanceProtocol | None]] = BoundedLRUCache(
    maxsize=4
)
_HF_MODEL_CACHE: BoundedLRUCache[tuple[_HFGenerateDictModelProtocol, _HFTokenizerInstanceProtocol]] = BoundedLRUCache(
    maxsize=4
)
MODEL_CACHE = _MODEL_CACHE


def _resolve_model_device(model: _CachedPyTorchModel) -> str:
    """Resolve the device string for a PyTorch model, defaulting to `'cpu'`.

    Args:
        model: Target PyTorch causal language model.

    Returns:
        Device identifier string.

    """
    if isinstance(model, TorchCausalLMProtocol) and callable(model.parameters):
        first_param = next(iter(model.parameters()), None)
        if isinstance(first_param, _DeviceMovableTensorProtocol) and first_param.device is not None:
            return str(first_param.device)
    return "cpu"


def _move_required_tensor(tensor: TorchTensorProtocol, device: str) -> TorchTensorProtocol:
    """Move non-optional `tensor` to `device` if `.to(device)` is supported.

    Args:
        tensor: Tensor to move.
        device: Target device string.

    Returns:
        Moved tensor or original `tensor`.

    """
    try:
        return tensor.to(device)
    except (TypeError, AssertionError, RuntimeError):
        return tensor


def _move_tensor_if_supported(
    tensor: TorchTensorProtocol | None,
    device: str,
) -> TorchTensorProtocol | None:
    """Move optional `tensor` to `device` if `.to(device)` is supported.

    Args:
        tensor: Optional tensor to move.
        device: Target device string.

    Returns:
        Moved tensor or `None`.

    """
    if tensor is None:
        return None
    return _move_required_tensor(tensor, device)


def _run_generation(
    model_name: str,
    prompt: str,
    beam_width: int,
    max_length: int,
    *,
    backend_alias: str = "pytorch",
    **kwargs: BackendKwargValue | Gemma4Config | CacheFingerprintProtocol,
) -> tuple[str, float]:
    """Execute the inference logic across native or Hugging Face PyTorch pipelines.

    Args:
        model_name: The name or path of the target model.
        prompt: The input natural language prompt.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of new tokens to generate.
        backend_alias: The backend alias (`'pytorch'` or `'pytorch_native'`).
        **kwargs: Optional extra arguments including model config and adapter path.

    Returns:
        A tuple of `(generated SQL query string, confidence score)`.

    Raises:
        ImportError: If `peft.PeftModel` is requested via adapter path but unavailable.
        InferenceError: If generation yields an empty sequence or SQL decoding fails.

    """
    assert torch is not None

    def _to_torch_tensor(
        arr: NDArrayProtocol
        | TorchTensorProtocol
        | Sequence[float]
        | Sequence[Sequence[float]]
        | Sequence[Sequence[Sequence[float]]],
    ) -> TorchTensorProtocol:
        """Convert feature array into a batched float32 PyTorch tensor.

        Args:
            arr: Preprocessed image or audio feature array.

        Returns:
            Batched float32 `TorchTensorProtocol`.

        """
        assert torch is not None
        t: _DeviceMovableTensorProtocol = torch.tensor(arr, dtype=torch.float32)
        return t.unsqueeze(0)

    def _passthrough_torch(val: BackendKwargValue | None) -> TorchTensorProtocol | None:
        """Return `val` when it is already a `TorchTensorProtocol`, otherwise `None`.

        Args:
            val: Candidate multimodal keyword argument value.

        Returns:
            `TorchTensorProtocol` instance or `None`.

        """
        return val if isinstance(val, TorchTensorProtocol) else None

    mm_kwargs: dict[str, BackendKwargValue] = {
        k: v for k, v in kwargs.items() if not isinstance(v, (Gemma4Config, CacheFingerprintProtocol))
    }
    hardware = str(kwargs.get("hardware", "cpu"))
    strict_hw = bool(kwargs.get("strict_hardware", False))
    xla_target_dev = (
        TorchRuntimeAdapter(None).get_device(hardware, strict_hardware=strict_hw)
        if normalize_hardware_family(hardware) == "tpu"
        else ""
    )
    use_xla = xla_target_dev.startswith("xla")
    (prompt, pixel_values, audio_values) = prepare_multimodal_inputs(
        prompt,
        mm_kwargs,
        _to_torch_tensor,
        _passthrough_torch,
    )

    if backend_alias == "pytorch_native":
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4
        from gemma_4_sql.tokenization import SQLTokenizer

        tokenizer: _HFTokenizerInstanceProtocol | None = None
        input_ids: TorchTensorProtocol | None = None
        if AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                inputs_nat = tokenizer(prompt, return_tensors="pt")
                input_ids = (
                    inputs_nat.input_ids
                    if isinstance(inputs_nat, _HasInputIdsProtocol)
                    else inputs_nat.get("input_ids")
                )
            except (OSError, ValueError, RuntimeError, KeyError, AttributeError):
                tokenizer = None

        if tokenizer is None or input_ids is None:
            sql_tok = SQLTokenizer()
            tokens = sql_tok.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long)
        assert input_ids is not None

        raw_cfg = kwargs.get("config")
        cfg_in = raw_cfg if isinstance(raw_cfg, Gemma4Config) else None
        direct_cached = _MODEL_CACHE.get_model(model_name) if cfg_in is None else None
        native_cache_key = make_model_cache_key(
            model_name,
            backend_name="native",
            config=cfg_in,
            factories=(NativeGemma4.from_pretrained,),
        )
        cached_native = direct_cached if direct_cached is not None else _MODEL_CACHE.get_model(native_cache_key)
        if cached_native is not None:
            model: _CachedNativeModel = cached_native[0]
        else:
            loaded_native: _CachedNativeModel = NativeGemma4.from_pretrained(model_name, config=cfg_in)
            model = loaded_native
        if use_xla:
            if isinstance(model, _MovableInferenceModelProtocol):
                model = model.to(xla_target_dev)
            input_ids = _move_required_tensor(input_ids, xla_target_dev)
            pixel_values = _move_tensor_if_supported(pixel_values, xla_target_dev)
            audio_values = _move_tensor_if_supported(audio_values, xla_target_dev)
        if isinstance(model, TorchCausalLMProtocol) and callable(model.train):
            model.train(False)
        _MODEL_CACHE.put_model(native_cache_key, (model, tokenizer))
        assert isinstance(model, _NativeGenerateModelProtocol)
        native_gen_model: _NativeGenerateModelProtocol = model
        with torch.no_grad():
            output_ids = native_gen_model.generate(
                input_ids,
                max_new_tokens=max_length,
                pixel_values=pixel_values,
                audio_values=audio_values,
            )
        if use_xla:
            TorchRuntimeAdapter(None).synchronize(xla_target_dev)

        gen_tokens = output_ids[0][input_ids.shape[-1] :]
        if len(gen_tokens) == 0:
            raise InferenceError("PyTorch native generation yielded an empty sequence.")

        if tokenizer is not None:
            sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        else:
            raw_list = gen_tokens.tolist()
            tok_list: list[int] = (
                [int(t) for t in raw_list if isinstance(t, (int, float, bool))]
                if isinstance(raw_list, list)
                else [int(raw_list)]
            )
            sql = SQLTokenizer().decode(tok_list).strip()

        if not sql:
            raise InferenceError("PyTorch native generation decoded into an empty SQL query string.")

        with torch.no_grad():
            fwd_out = native_gen_model(output_ids)
            raw_logits = fwd_out[0] if isinstance(fwd_out, tuple) else fwd_out
            assert isinstance(raw_logits, TorchTensorProtocol)
            logits: TorchTensorProtocol = raw_logits
            prompt_len = int(input_ids.shape[-1])
            step_logits = logits[0, prompt_len - 1 : -1, :]
            log_probs = torch.nn.functional.log_softmax(step_logits, dim=-1)
            assert isinstance(gen_tokens, _DeviceMovableTensorProtocol)
            selected_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            logprob_sum = float(selected_log_probs.sum().item())
        confidence_score = normalize_logprob_confidence(logprob_sum, len(gen_tokens))
        return (sql, confidence_score)

    adapter_key = str(kwargs.get("adapter_path") or kwargs.get("lora_path") or "")
    peft_cls_ref: _PeftModelLoaderProtocol | None = (
        import_optional_symbol("peft", "PeftModel", _PeftModelLoaderProtocol) if adapter_key else None
    )
    tok_factory: Callable[..., _HFTokenizerInstanceProtocol] | None = (
        AutoTokenizer.from_pretrained if AutoTokenizer is not None else None
    )
    model_factory: Callable[..., _CachedPyTorchModel] | None = (
        AutoModelForCausalLM.from_pretrained if AutoModelForCausalLM is not None else None
    )
    peft_factory: Callable[..., _CachedPyTorchModel] | None = (
        peft_cls_ref.from_pretrained if peft_cls_ref is not None else None
    )
    hf_cache_key = make_model_cache_key(
        model_name,
        backend_name=f"hf:{xla_target_dev}" if use_xla else "hf",
        adapter_path=adapter_key,
        factories=(tok_factory, model_factory, peft_factory),
    )
    cached_hf = _HF_MODEL_CACHE.get_model(hf_cache_key)
    if cached_hf is not None:
        hf_model, hf_tokenizer = cached_hf
    else:
        assert AutoTokenizer is not None
        assert AutoModelForCausalLM is not None
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_xla:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            if isinstance(hf_model, _MovableInferenceModelProtocol):
                hf_model = hf_model.to(xla_target_dev)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        if "adapter_path" in kwargs or "lora_path" in kwargs:
            adapter_path = str(kwargs.get("adapter_path") or kwargs.get("lora_path"))
            try:
                if peft_cls_ref is None:
                    raise ImportError("peft.PeftModel is not available.")
                hf_model = peft_cls_ref.from_pretrained(hf_model, adapter_path)
                if use_xla and isinstance(hf_model, _MovableInferenceModelProtocol):
                    hf_model = hf_model.to(xla_target_dev)
            except (ImportError, ValueError, RuntimeError, OSError) as e:
                logger.warning("Could not load adapter from %s: %s", adapter_path, e)
        _HF_MODEL_CACHE.put_model(hf_cache_key, (hf_model, hf_tokenizer))

    device = xla_target_dev if use_xla else _resolve_model_device(hf_model)
    inputs = hf_tokenizer(prompt, return_tensors="pt")
    if isinstance(inputs, _BatchEncodingProtocol) and callable(inputs.to):
        inputs = inputs.to(device)
    extra_gen: dict[str, TorchTensorProtocol] = {}
    moved_pv = _move_tensor_if_supported(pixel_values, device)
    moved_av = _move_tensor_if_supported(audio_values, device)
    if moved_pv is not None:
        extra_gen["pixel_values"] = moved_pv
    if moved_av is not None:
        extra_gen["audio_values"] = moved_av

    unpackable_inputs: _BatchEncodingProtocol | Mapping[str, TorchTensorProtocol] = (
        inputs if isinstance(inputs, (_BatchEncodingProtocol, Mapping)) else {"input_ids": inputs.input_ids}
    )
    outputs = hf_model.generate(
        **unpackable_inputs,
        **extra_gen,
        max_new_tokens=max_length,
        num_beams=beam_width,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if use_xla:
        TorchRuntimeAdapter(None).synchronize(xla_target_dev)
    sequences = outputs.sequences
    input_length = 0
    in_ids: TorchTensorProtocol | None = (
        inputs.input_ids if isinstance(inputs, _HasInputIdsProtocol) else inputs.get("input_ids")
    )
    if in_ids is not None and len(in_ids.shape) > 0:
        input_length = int(in_ids.shape[-1])
    if input_length > 0 and len(sequences[0]) >= input_length:
        generated_tokens = sequences[0][input_length:]
        sql = hf_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    else:
        generated_text = hf_tokenizer.decode(sequences[0], skip_special_tokens=True)
        sql = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text.strip()

    if not sql:
        raise InferenceError("PyTorch generation yielded an empty SQL sequence.")

    raw_conf = 0.8
    if isinstance(outputs, _HFGenerateScoredOutputProtocol):
        scores_seq = outputs.sequences_scores
        if len(scores_seq) > 0:
            raw_conf = extract_scalar_float(scores_seq[0])
    confidence_score = max(0.0, min(1.0, math.exp(raw_conf))) if raw_conf <= 0.0 else max(0.0, min(1.0, raw_conf))

    return (sql, confidence_score)


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: BackendKwargValue | Gemma4Config | CacheFingerprintProtocol,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using PyTorch.

    Args:
        model_name: The name or path of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Additional parameters.

    Returns:
        A dictionary containing the generated SQL and metadata.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.

    """
    backend_alias = str(kwargs.get("backend_alias", kwargs.get("backend", "pytorch")))
    if torch is None or (backend_alias != "pytorch_native" and (AutoModelForCausalLM is None or AutoTokenizer is None)):
        raise DependencyMissingError("PyTorch dependencies are missing.")

    gen_kwargs = dict(kwargs)
    gen_kwargs.pop("backend_alias", None)
    gen_kwargs.pop("backend", None)

    def _execute() -> tuple[str, float]:
        """Run PyTorch native or HuggingFace generation.

        Returns:
            Tuple of `(generated SQL query string, confidence score)`.

        """
        logger.info("Generating with %s", model_name)
        return _run_generation(
            model_name,
            prompt,
            beam_width,
            max_length,
            backend_alias=backend_alias,
            **gen_kwargs,
        )

    wrapper_kwargs = extract_status_kwargs(kwargs)
    return run_inference_wrapper(backend_alias, model_name, prompt, beam_width, _execute, **wrapper_kwargs)

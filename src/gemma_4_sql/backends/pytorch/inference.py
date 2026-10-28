"""PyTorch-specific inference logic."""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    CacheFingerprintProtocol,
    make_model_cache_key,
    normalize_logprob_confidence,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_multimodal import prepare_multimodal_inputs
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError, StatusValue
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    InferenceResultDict,
    NDArrayProtocol,
    TorchCausalLMProtocol,
    TorchTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _BatchEncodingProtocol(Protocol):
    """Protocol for a Hugging Face `BatchEncoding` exposing `.input_ids`, `.keys()`, `__getitem__`, and `.to(device)`."""

    @property
    def input_ids(self) -> TorchTensorProtocol:
        """Provide the encoded token ID tensor."""
        ...

    def keys(self) -> Iterable[str]:
        """Yield mapping keys for `**inputs` unpacking."""
        ...

    def __getitem__(self, key: str, /) -> TorchTensorProtocol:
        """Look up a tensor by key for `**inputs` unpacking."""
        ...

    def to(self, device: str, /) -> _BatchEncodingProtocol:
        """Move encoded tensors to `device`."""
        ...


@runtime_checkable
class _HFTokenizerInstanceProtocol(Protocol):
    """Protocol for a loaded Hugging Face tokenizer supporting `__call__` and `.decode`."""

    def __call__(self, text: str, *, return_tensors: str = "pt") -> _BatchEncodingProtocol:
        """Encode `text` into a PyTorch `BatchEncoding`."""
        ...

    def decode(self, token_ids: TorchTensorProtocol | Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into a string."""
        ...


@runtime_checkable
class _HFTokenizerLoaderProtocol(Protocol):
    """Protocol for `transformers.AutoTokenizer` exposing `.from_pretrained`."""

    def from_pretrained(self, model_name: str, /) -> _HFTokenizerInstanceProtocol:
        """Load a tokenizer from a pretrained identifier or path."""
        ...


@runtime_checkable
class _HFGenerateOutputProtocol(Protocol):
    """Protocol for Hugging Face `.generate(return_dict_in_generate=True, output_scores=True)` output."""

    @property
    def sequences(self) -> Sequence[TorchTensorProtocol]:
        """Provide the generated token ID sequences."""
        ...

    @property
    def sequences_scores(self) -> Sequence[TorchTensorProtocol]:
        """Provide the beam sequence log-probability scores."""
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
        """Load a Hugging Face causal LM from a pretrained identifier or path."""
        ...


@runtime_checkable
class _PeftModelLoaderProtocol(Protocol):
    """Protocol for `peft.PeftModel` exposing `.from_pretrained(model, adapter_path)`."""

    def from_pretrained(
        self,
        model: TorchCausalLMProtocol,
        adapter_path: str,
        /,
    ) -> _HFGenerateDictModelProtocol:
        """Load and attach a PEFT adapter onto `model`."""
        ...


@runtime_checkable
class _DeviceMovableTensorProtocol(TorchTensorProtocol, Protocol):
    """Protocol for a PyTorch tensor supporting `.device`, `.to(device)`, and `.unsqueeze(dim)`."""

    @property
    def device(self) -> str | int | None:
        """Provide the device where the tensor resides."""
        ...

    def to(self, device: str, /) -> _DeviceMovableTensorProtocol:
        """Move tensor to `device`."""
        ...

    def unsqueeze(self, dim: int, /) -> _DeviceMovableTensorProtocol:
        """Insert a singleton dimension at `dim`."""
        ...


@runtime_checkable
class _HFGenerateDictModelProtocol(TorchCausalLMProtocol, Protocol):
    """Protocol for a Hugging Face model whose `.generate(...)` returns `_HFGenerateOutputProtocol`."""

    def generate(
        self,
        *args: TorchTensorProtocol | int | bool,
        **kwargs: TorchTensorProtocol | int | bool,
    ) -> _HFGenerateOutputProtocol:
        """Generate token sequences and return a structured `_HFGenerateOutputProtocol`."""
        ...


torch = optional_import("torch")
AutoModelForCausalLM: _HFModelLoaderProtocol | None = optional_attr(
    "transformers",
    "AutoModelForCausalLM",
    _HFModelLoaderProtocol,
)
AutoTokenizer: _HFTokenizerLoaderProtocol | None = optional_attr(
    "transformers",
    "AutoTokenizer",
    _HFTokenizerLoaderProtocol,
)

_MODEL_CACHE: BoundedLRUCache[tuple[TorchCausalLMProtocol, _HFTokenizerInstanceProtocol | None]] = BoundedLRUCache(maxsize=4)


def _resolve_model_device(model: TorchCausalLMProtocol) -> str:
    """Resolve the device string for a PyTorch model, defaulting to `'cpu'`.

    Args:
        model: Target PyTorch causal language model.

    Returns:
        Device identifier string.

    """
    with contextlib.suppress(AttributeError, StopIteration, TypeError):
        first_param = next(iter(model.parameters()))
        return str(first_param.device)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    return "cpu"


def _move_tensor_if_supported(tensor: TorchTensorProtocol | None, device: str) -> TorchTensorProtocol | None:
    """Move `tensor` to `device` if `.to(device)` is supported.

    Args:
        tensor: Optional tensor to move.
        device: Target device string.

    Returns:
        Moved tensor or original `tensor`.

    """
    if tensor is not None and isinstance(tensor, _DeviceMovableTensorProtocol):
        with contextlib.suppress(AttributeError, TypeError):
            return tensor.to(device)
    return tensor


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
        InferenceError: If generation yields an empty sequence or SQL decoding fails.

    """
    assert torch is not None

    def _to_torch_tensor(
        arr: NDArrayProtocol | TorchTensorProtocol | Sequence[float] | Sequence[Sequence[float]],
    ) -> TorchTensorProtocol:
        """Convert feature array into a batched float32 PyTorch tensor."""
        assert torch is not None
        t: _DeviceMovableTensorProtocol = torch.tensor(arr, dtype=torch.float32)
        return t.unsqueeze(0)

    mm_kwargs: dict[str, BackendKwargValue] = {k: v for k, v in kwargs.items() if not isinstance(v, (Gemma4Config, CacheFingerprintProtocol))}
    (prompt, pixel_values, audio_values) = prepare_multimodal_inputs(
        prompt,
        mm_kwargs,
        _to_torch_tensor,
    )

    if backend_alias == "pytorch_native":
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4
        from gemma_4_sql.tokenization import SQLTokenizer

        tokenizer: _HFTokenizerInstanceProtocol | None = None
        input_ids: TorchTensorProtocol | None = None
        if AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids
            except (OSError, ValueError, RuntimeError, KeyError, AttributeError):
                tokenizer = None

        if tokenizer is None or input_ids is None:
            sql_tok = SQLTokenizer()
            tokens = sql_tok.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long)
        assert input_ids is not None

        raw_cfg = kwargs.get("config")
        cfg_in = raw_cfg if isinstance(raw_cfg, Gemma4Config) else None
        native_cache_key = make_model_cache_key(
            model_name,
            backend_name="native",
            config=cfg_in,
            factories=(NativeGemma4.from_pretrained,),
        )
        cached_native = _MODEL_CACHE.get_model(native_cache_key)
        model: TorchCausalLMProtocol = cached_native[0] if cached_native is not None else NativeGemma4.from_pretrained(model_name, config=cfg_in)
        with contextlib.suppress(AttributeError):
            model.train(False)
        _MODEL_CACHE.put_model(native_cache_key, (model, tokenizer))
        gen_kwargs: dict[str, int | TorchTensorProtocol] = {"max_new_tokens": max_length}
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if audio_values is not None:
            gen_kwargs["audio_values"] = audio_values

        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[unresolved-attribute]

        assert isinstance(output_ids, TorchTensorProtocol)
        gen_tokens = output_ids[0][input_ids.shape[-1] :]
        if len(gen_tokens) == 0:
            raise InferenceError("PyTorch native generation yielded an empty sequence.")

        if tokenizer is not None:
            sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        else:
            raw_list = gen_tokens.tolist()
            tok_list: list[int] = [int(t) for t in raw_list if isinstance(t, (int, float, bool))] if isinstance(raw_list, list) else [int(raw_list)]
            sql = SQLTokenizer().decode(tok_list).strip()

        if not sql:
            raise InferenceError("PyTorch native generation decoded into an empty SQL query string.")

        with torch.no_grad():
            fwd_out = model(output_ids)
            logits: TorchTensorProtocol = fwd_out[0] if isinstance(fwd_out, tuple) else fwd_out  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
            prompt_len = int(input_ids.shape[-1])
            step_logits = logits[0, prompt_len - 1 : -1, :]
            log_probs = torch.nn.functional.log_softmax(step_logits, dim=-1)
            assert isinstance(gen_tokens, _DeviceMovableTensorProtocol)
            selected_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            logprob_sum = float(selected_log_probs.sum().item())
        confidence_score = normalize_logprob_confidence(logprob_sum, len(gen_tokens))
        return (sql, confidence_score)

    adapter_key = str(kwargs.get("adapter_path") or kwargs.get("lora_path") or "")
    peft_cls_ref: _PeftModelLoaderProtocol | None = optional_attr("peft", "PeftModel", _PeftModelLoaderProtocol) if adapter_key else None
    tok_factory: Callable[..., _HFTokenizerInstanceProtocol] | None = AutoTokenizer.from_pretrained if AutoTokenizer is not None else None
    model_factory: Callable[..., TorchCausalLMProtocol] | None = AutoModelForCausalLM.from_pretrained if AutoModelForCausalLM is not None else None
    peft_factory: Callable[..., TorchCausalLMProtocol] | None = peft_cls_ref.from_pretrained if peft_cls_ref is not None else None
    hf_cache_key = make_model_cache_key(
        model_name,
        backend_name="hf",
        adapter_path=adapter_key,
        factories=(tok_factory, model_factory, peft_factory),
    )
    cached_hf = _MODEL_CACHE.get_model(hf_cache_key)
    if cached_hf is not None and cached_hf[1] is not None:
        hf_model: _HFGenerateDictModelProtocol = cached_hf[0]  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
        hf_tokenizer = cached_hf[1]
    else:
        assert AutoTokenizer is not None
        assert AutoModelForCausalLM is not None
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        if "adapter_path" in kwargs or "lora_path" in kwargs:
            adapter_path = str(kwargs.get("adapter_path") or kwargs.get("lora_path"))
            try:
                if peft_cls_ref is None:
                    raise ImportError("peft.PeftModel is not available.")
                hf_model = peft_cls_ref.from_pretrained(hf_model, adapter_path)
            except (ImportError, ValueError, RuntimeError, OSError) as e:
                logger.warning("Could not load adapter from %s: %s", adapter_path, e)
        _MODEL_CACHE.put_model(hf_cache_key, (hf_model, hf_tokenizer))

    device = _resolve_model_device(hf_model)
    inputs = hf_tokenizer(prompt, return_tensors="pt")
    with contextlib.suppress(AttributeError, TypeError):
        inputs = inputs.to(device)
    extra_gen: dict[str, TorchTensorProtocol] = {}
    moved_pv = _move_tensor_if_supported(pixel_values, device)
    moved_av = _move_tensor_if_supported(audio_values, device)
    if moved_pv is not None:
        extra_gen["pixel_values"] = moved_pv
    if moved_av is not None:
        extra_gen["audio_values"] = moved_av

    outputs = hf_model.generate(
        **inputs,
        **extra_gen,
        max_new_tokens=max_length,
        num_beams=beam_width,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    sequences = outputs.sequences
    input_length = 0
    with contextlib.suppress(AttributeError, IndexError, TypeError):
        input_length = int(inputs.input_ids.shape[-1])
    if input_length > 0 and len(sequences[0]) >= input_length:
        generated_tokens = sequences[0][input_length:]
        sql = hf_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    else:
        generated_text = hf_tokenizer.decode(sequences[0], skip_special_tokens=True)
        sql = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text.strip()

    if not sql:
        raise InferenceError("PyTorch generation yielded an empty SQL sequence.")

    raw_conf = 0.8
    with contextlib.suppress(AttributeError, IndexError, TypeError):
        raw_conf = float(outputs.sequences_scores[0].item())
    if raw_conf <= 0.0:
        confidence_score = max(0.0, min(1.0, math.exp(raw_conf)))
    else:
        confidence_score = max(0.0, min(1.0, raw_conf))

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
        """Run PyTorch native or HuggingFace generation."""
        logger.info("Generating with %s", model_name)
        return _run_generation(
            model_name,
            prompt,
            beam_width,
            max_length,
            backend_alias=backend_alias,
            **gen_kwargs,
        )

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return run_inference_wrapper(backend_alias, model_name, prompt, beam_width, _execute, **wrapper_kwargs)

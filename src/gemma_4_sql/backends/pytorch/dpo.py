"""PyTorch-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_dpo import (
    ArithmeticTensorProtocol,
    generic_dpo_loss,
    run_dpo_wrapper,
)
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.pytorch.distributed import TorchDistModuleProtocol, cleanup_distributed
from gemma_4_sql.backends.pytorch.etl import build_dataloader
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.exceptions import DependencyMissingError, ExecutionStatus, StatusValue, TrainingError
from gemma_4_sql.type_hints import (
    DPOConfig,
    DPOLoopState,
    DPOResultDict,
    ETLConfig,
    ForwardArg,
    ForwardOutput,
    TorchCausalLMProtocol,
    TorchOptimizerProtocol,
    TorchTensorProtocol,
    TrainerState,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _PretrainedCausalLMLoaderProtocol(Protocol):
    """Protocol for a Hugging Face model class exposing `.from_pretrained(model_name)`."""

    def from_pretrained(self, model_name: str, /) -> TorchCausalLMProtocol:
        """Load a causal language model from a pretrained identifier or path."""
        ...


torch = optional_import("torch")
nn = optional_import("torch.nn")
optim = optional_import("torch.optim")
functional = optional_import("torch.nn.functional")
AutoModelForCausalLM: _PretrainedCausalLMLoaderProtocol | None = optional_attr(
    "transformers",
    "AutoModelForCausalLM",
    _PretrainedCausalLMLoaderProtocol,
)


def dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float = 0.1,
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
    """Compute the Direct Preference Optimization (DPO) loss for PyTorch tensors.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    Raises:
        DependencyMissingError: If PyTorch DPO dependencies are missing.

    """
    if torch is None or functional is None:
        raise DependencyMissingError("PyTorch DPO dependencies are missing.", backend="pytorch")
    return generic_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        functional.logsigmoid,
    )


def _extract_logits(
    raw_out: ForwardOutput,
) -> TorchTensorProtocol | ForwardOutput:
    """Extract the logits tensor from a tuple or direct tensor output.

    Args:
        raw_out: Model forward output.

    Returns:
        Extracted logits tensor or original output.

    """
    return raw_out[0] if isinstance(raw_out, tuple) else raw_out


type TorchDPOModel = Callable[..., ForwardOutput | SupportsFloat | str] | TorchCausalLMProtocol
type TorchDPOBatchValue = ForwardArg | SupportsFloat


def _compute_pytorch_logps(
    model: TorchDPOModel | None,
    inputs: TorchDPOBatchValue,
    labels: TorchDPOBatchValue = None,
) -> TorchTensorProtocol:
    """Compute per-sequence summed token log-probabilities for PyTorch models.

    Applies `log_softmax` over the vocabulary axis, gathers log-probabilities at target
    token indices, masks out padding tokens (`labels != 0`), and sums along the sequence dimension.

    Args:
        model: Policy or reference PyTorch model callable.
        inputs: Input token ID tensor.
        labels: Optional target token ID tensor of shape `(batch_size, seq_len)`.

    Returns:
        Per-sequence log-probability tensor of shape `(batch_size,)`.

    """
    assert model is not None
    raw_out = model(inputs)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    logits = _extract_logits(raw_out)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    target = labels if labels is not None else inputs
    if torch is not None and functional is not None and isinstance(logits, TorchTensorProtocol) and isinstance(target, TorchTensorProtocol) and logits.ndim >= 2 and target.dtype in (torch.long, torch.int64, torch.int32):
        if labels is None and target.ndim == 2 and int(target.shape[1]) > 1 and int(logits.shape[1]) > 1:
            logits = logits[:, :-1, :]
            target = target[:, 1:]
        log_probs = functional.log_softmax(logits, dim=-1)
        gathered: TorchTensorProtocol = torch.gather(
            log_probs,
            dim=-1,
            index=target.unsqueeze(-1),
        ).squeeze(-1)
        mask: TorchTensorProtocol = torch.ne(target, 0)
        summed: TorchTensorProtocol = (gathered * mask).sum(dim=-1)
        return summed
    with contextlib.suppress(AttributeError, TypeError):
        res_mean: TorchTensorProtocol = logits.mean(dim=-1)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue,reportAssignmentType]  # ty: ignore[invalid-assignment, unresolved-attribute, unknown-argument]
        return res_mean
    res_fallback: TorchTensorProtocol = logits  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
    return res_fallback


def _run_dpo_step(
    policy_model: TorchDPOModel | None,
    ref_model: TorchDPOModel | None,
    optimizer: TorchOptimizerProtocol | None,
    batch: Mapping[str, TorchDPOBatchValue],
    beta: float,
) -> TorchTensorProtocol | ArithmeticTensorProtocol | float:
    """Run a single PyTorch DPO optimization step.

    Args:
        policy_model: Trainable PyTorch policy model.
        ref_model: Frozen PyTorch reference model.
        optimizer: PyTorch optimizer instance.
        batch: Preference batch dictionary containing chosen and rejected inputs/labels.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO step loss tensor.

    """
    assert torch is not None
    with contextlib.suppress(AttributeError, TypeError):
        optimizer.zero_grad()  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
    ch_inputs = batch.get("chosen_inputs", batch.get("inputs"))
    re_inputs = batch.get("rejected_inputs", batch.get("inputs"))
    ch_labels = batch.get("chosen_labels")
    re_labels = batch.get("rejected_labels")

    pi_ch_logps = _compute_pytorch_logps(policy_model, ch_inputs, ch_labels)
    pi_re_logps = _compute_pytorch_logps(policy_model, re_inputs, re_labels)
    with torch.no_grad():
        ref_ch_logps = _compute_pytorch_logps(ref_model, ch_inputs, ch_labels)
        ref_re_logps = _compute_pytorch_logps(ref_model, re_inputs, re_labels)

    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    with contextlib.suppress(AttributeError, TypeError):
        loss.backward()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    with contextlib.suppress(AttributeError, TypeError):
        optimizer.step()  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
    return loss


def _run_training_epochs[ModelT, OptT, BatchT: Mapping[str, TorchDPOBatchValue], ParamsT](
    state: DPOLoopState[ModelT, OptT, BatchT] | TrainerState[ModelT, ParamsT, OptT, BatchT],
) -> float:
    """Run PyTorch DPO training epochs over the preference dataloader.

    Args:
        state: `DPOLoopState` holding models, optimizer, dataloader, and beta.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader or ()
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    beta = state.beta

    def _step(batch: BatchT) -> float:
        """Execute a single PyTorch DPO step and extract scalar float loss."""
        loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        try:
            return float(loss.item())  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        except (AttributeError, TypeError, ValueError):
            return float(loss)  # pyright: ignore[reportArgumentType]

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _load_dpo_models(
    model_name: str,
    backend_alias: str = "pytorch",
    model_config: Gemma4Config | None = None,
) -> tuple[TorchCausalLMProtocol, TorchCausalLMProtocol]:
    """Load policy and reference models for PyTorch DPO.

    Args:
        model_name: Hugging Face model ID or local path.
        backend_alias: Backend variant (`'pytorch'` or `'pytorch_native'`).
        model_config: Optional explicit `Gemma4Config` for native initialization.

    Returns:
        A 2-tuple of `(policy_model, ref_model)`.

    Raises:
        TrainingError: If model loading fails.

    """
    try:
        if backend_alias == "pytorch_native":
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4ForCausalLM

            policy_model: TorchCausalLMProtocol = NativeGemma4ForCausalLM.from_pretrained(model_name, config=model_config)
            ref_model: TorchCausalLMProtocol = NativeGemma4ForCausalLM.from_pretrained(model_name, config=model_config)
            return (policy_model, ref_model)

        gemma4_cls = optional_attr(
            "transformers.models.gemma4",
            "Gemma4ForCausalLM",
            _PretrainedCausalLMLoaderProtocol,
        )
        loader_cls = gemma4_cls if gemma4_cls is not None else AutoModelForCausalLM
        if loader_cls is None:
            raise ImportError("Neither transformers.models.gemma4.Gemma4ForCausalLM nor AutoModelForCausalLM is available.")
        return (loader_cls.from_pretrained(model_name), loader_cls.from_pretrained(model_name))
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        msg = f"Failed to load model {model_name}"
        raise TrainingError(msg, backend=backend_alias, model_name=model_name, operation="dpo", cause=e) from e


def run_dpo(
    config: DPOConfig,
    **kwargs: StatusValue | Gemma4Config | TorchDistModuleProtocol,
) -> DPOResultDict:
    """Run a DPO training loop for PyTorch.

    Args:
        config: The DPO configuration.
        **kwargs: Hyperparameters for DPO (e.g., `beta`, `learning_rate`, `backend_alias`, `raise_on_error`).

    Returns:
        Structured `DPOResultDict` with execution status and final loss.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    extra = config.extra_kwargs
    backend_alias = str(kwargs.get("backend_alias", extra.get("backend_alias", "pytorch")))
    raw_cfg = kwargs.get("model_config") or extra.get("model_config")
    model_config = raw_cfg if isinstance(raw_cfg, Gemma4Config) else None
    raw_dist = kwargs.get("dist_module") or extra.get("dist_module")
    dist_module = raw_dist if isinstance(raw_dist, TorchDistModuleProtocol) else None

    if torch is None or nn is None or optim is None:
        raise DependencyMissingError("PyTorch dependencies are missing.", backend=backend_alias)

    def _execute(
        model_name: str,
        dataset: str,
        beta: float,
        epochs: int,
        learning_rate: float,
        batch_size: int,
    ) -> tuple[str, float]:
        """Execute PyTorch DPO and guarantee distributed process group cleanup."""
        try:
            assert optim is not None
            (policy_model, ref_model) = _load_dpo_models(
                model_name,
                backend_alias=backend_alias,
                model_config=model_config,
            )
            optimizer: TorchOptimizerProtocol = optim.AdamW(policy_model.parameters(), lr=learning_rate)
            data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
            dataloader = data_dict.get("loader", None)
            if not isinstance(dataloader, Iterable):
                raise TrainingError(
                    f"Invalid dataloader for dataset: {dataset}",
                    backend=backend_alias,
                    model_name=model_name,
                    operation="dpo",
                )
            final_loss = _run_training_epochs(
                DPOLoopState(
                    dataloader=dataloader,
                    epochs=epochs,
                    policy_model=policy_model,
                    ref_model=ref_model,
                    optimizer=optimizer,
                    beta=beta,
                )
            )
            return (ExecutionStatus.COMPLETED.value, float(final_loss))
        finally:
            cleanup_distributed(dist_module)

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return run_dpo_wrapper(
        backend_alias,
        config,
        _execute,
        **wrapper_kwargs,
    )

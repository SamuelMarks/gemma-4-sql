"""PyTorch-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_dpo import generic_dpo_loss, run_dpo_wrapper
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.pytorch.etl import build_dataloader
from gemma_4_sql.backends.pytorch.train import _cleanup_distributed
from gemma_4_sql.exceptions import DependencyMissingError, ExecutionStatus, TrainingError
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState, ETLConfig, TensorType, TorchCausalLMProtocol, TrainerState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from gemma_4_sql.type_hints import DPOResultDict
logger = logging.getLogger(__name__)

torch = optional_import("torch")
nn = optional_import("torch.nn")
optim = optional_import("torch.optim")
functional = optional_import("torch.nn.functional")
AutoModelForCausalLM = optional_attr("transformers", "AutoModelForCausalLM")


def dpo_loss(
    policy_chosen_logps: TensorType,
    policy_rejected_logps: TensorType,
    ref_chosen_logps: TensorType,
    ref_rejected_logps: TensorType,
    beta: float = 0.1,
) -> tuple[Any, Any, Any]:
    """Compute the Direct Preference Optimization (DPO) loss for PyTorch tensors.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    """
    if torch is None or functional is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        functional.logsigmoid,
    )


def _compute_pytorch_logps(model: Any, inputs: Any, labels: Any = None) -> Any:
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
    raw_out = model(inputs)
    logits = raw_out[0] if isinstance(raw_out, tuple) else getattr(raw_out, "logits", raw_out)
    target = labels if labels is not None else inputs
    ndim = getattr(logits, "ndim", 0)
    if torch is not None and functional is not None and isinstance(ndim, int) and ndim >= 2 and hasattr(target, "unsqueeze") and getattr(target, "dtype", None) in (getattr(torch, "long", None), getattr(torch, "int64", None), getattr(torch, "int32", None)):
        log_probs = functional.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        mask = target != 0
        return (gathered * mask).sum(dim=-1)
    return logits.mean(dim=-1) if hasattr(logits, "mean") else logits


def _run_dpo_step(policy_model: TorchCausalLMProtocol, ref_model: TorchCausalLMProtocol, optimizer: Any, batch: Mapping[str, Any], beta: float) -> Any:
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
    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()
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
    if hasattr(loss, "backward"):
        loss.backward()
    if hasattr(optimizer, "step"):
        optimizer.step()
    return loss


def _run_training_epochs(state: DPOLoopState[Any, Any, Any] | TrainerState[Any, Any, Any, Any]) -> float:
    """Run PyTorch DPO training epochs over the preference dataloader.

    Args:
        state: `DPOLoopState` holding models, optimizer, dataloader, and beta.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[dict[str, Any]] = state.dataloader or ()
    policy_model: Any = state.policy_model
    ref_model: Any = state.ref_model
    optimizer = state.optimizer
    beta = state.beta

    def _step(batch: dict[str, Any]) -> float:
        """Execute a single PyTorch DPO step and extract scalar float loss."""
        loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)
        return float(loss.item() if hasattr(loss, "item") else loss)

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _load_dpo_models(model_name: str, backend_alias: str = "pytorch", model_config: Any = None) -> tuple[Any, Any]:
    """Load policy and reference models for PyTorch DPO.

    Args:
        model_name: Hugging Face model ID or local path.
        backend_alias: Backend variant (`'pytorch'` or `'pytorch_native'`).
        model_config: Optional explicit `Gemma4Config` for native initialization.

    Returns:
        A 2-tuple of `(policy_model, ref_model)`.

    Raises:
        ValueError: If model loading fails.

    """
    try:
        if backend_alias == "pytorch_native":
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4ForCausalLM

            policy_model = NativeGemma4ForCausalLM.from_pretrained(model_name, config=model_config)
            ref_model = NativeGemma4ForCausalLM.from_pretrained(model_name, config=model_config)
            return (policy_model, ref_model)

        gemma4_cls = optional_attr("transformers.models.gemma4", "Gemma4ForCausalLM")
        loader_cls = gemma4_cls if gemma4_cls is not None else AutoModelForCausalLM
        if loader_cls is None:
            raise ImportError("Neither transformers.models.gemma4.Gemma4ForCausalLM nor AutoModelForCausalLM is available.")
        return (loader_cls.from_pretrained(model_name), loader_cls.from_pretrained(model_name))
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        msg = f"Failed to load model {model_name}"
        raise TrainingError(msg, backend=backend_alias, model_name=model_name, operation="dpo", cause=e) from e


def run_dpo(config: DPOConfig, **kwargs: object) -> DPOResultDict:
    """Run a DPO training loop for PyTorch.

    Args:
        config: The DPO configuration.
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate, backend_alias, raise_on_error).

    Returns:
        Structured `DPOResultDict` with execution status and final loss.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    extra = getattr(config, "extra_kwargs", {}) or {}
    backend_alias = str(kwargs.get("backend_alias", extra.get("backend_alias", "pytorch")))
    model_config = kwargs.get("model_config") or extra.get("model_config")
    dist_module = kwargs.get("dist_module") or extra.get("dist_module")

    if torch is None or nn is None or optim is None:
        raise DependencyMissingError("PyTorch dependencies are missing.", backend=backend_alias)

    def _execute(model_name: str, dataset: str, beta: float, epochs: int, learning_rate: float, batch_size: int) -> tuple[str, float]:
        """Execute PyTorch DPO and guarantee distributed process group cleanup."""
        try:
            (policy_model, ref_model) = _load_dpo_models(model_name, backend_alias=backend_alias, model_config=model_config)
            optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)
            data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
            dataloader = data_dict.get("loader", None)
            if dataloader is None or not hasattr(dataloader, "__iter__"):
                raise TrainingError(f"Invalid dataloader for dataset: {dataset}", backend=backend_alias, model_name=model_name, operation="dpo")
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
            _cleanup_distributed(dist_module)

    return run_dpo_wrapper(
        backend_alias,
        config,
        _execute,
        **kwargs,
    )

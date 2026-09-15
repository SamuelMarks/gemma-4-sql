"""PyTorch-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.pytorch.etl import build_dataloader
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import torch as _torch
    from torch import nn as _nn
    from torch import optim as _optim
    from torch.nn import functional as _functional

    torch: Any = _torch
    nn: Any = _nn
    optim: Any = _optim
    functional: Any = _functional
except (ImportError, AttributeError):
    torch = None
    nn = None
    optim = None
    functional = None


def dpo_loss(policy_chosen_logps: Any, policy_rejected_logps: Any, ref_chosen_logps: Any, ref_rejected_logps: Any, beta: float = 0.1) -> tuple[Any, Any, Any]:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A tuple containing the results.
    """
    if torch is None or functional is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, functional.logsigmoid)


def _run_dpo_step(policy_model: Any, ref_model: Any, optimizer: Any, batch: JSONDict, beta: float) -> Any:
    """Run a single DPO training step.

    Returns:
        object: The resulting output from the operation.

    """
    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()
    pi_ch = policy_model(batch["chosen_inputs"])
    pi_re = policy_model(batch["rejected_inputs"])
    with torch.no_grad():
        ref_ch = ref_model(batch["chosen_inputs"])
        ref_re = ref_model(batch["rejected_inputs"])
    pi_ch_logps = pi_ch.mean(dim=-1) if hasattr(pi_ch, "mean") else pi_ch
    pi_re_logps = pi_re.mean(dim=-1) if hasattr(pi_re, "mean") else pi_re
    ref_ch_logps = ref_ch.mean(dim=-1) if hasattr(ref_ch, "mean") else ref_ch
    ref_re_logps = ref_re.mean(dim=-1) if hasattr(ref_re, "mean") else ref_re
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    if hasattr(loss, "backward"):
        loss.backward()
    if hasattr(optimizer, "step"):
        optimizer.step()
    return loss


def _run_training_epochs(state: TrainerState) -> float:
    """Execute function.

    Returns:
        The execution result.

    """
    dataloader = state.dataloader
    epochs = state.epochs
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    beta = state.beta
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)
            loss_val = float(loss.item() if hasattr(loss, "item") else loss)
            epoch_loss += loss_val
        final_loss = epoch_loss / max(1, len(dataloader))
    return float(final_loss)


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    """Run a DPO training loop for PyTorch.

    Args:
        config: The DPO configuration.
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate).

    Returns:
        A dict with the execution status and metrics.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.
        ValueError: If model loading fails.
    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)

    if torch is None or nn is None or optim is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch dependencies are missing.")

    try:
        try:
            gemma4_for_causal_lm_cls = __import__("transformers.models.gemma4", fromlist=["Gemma4ForCausalLM"]).Gemma4ForCausalLM
            policy_model = gemma4_for_causal_lm_cls.from_pretrained(model_name)
            ref_model = gemma4_for_causal_lm_cls.from_pretrained(model_name)
        except (ImportError, ValueError) as e:
            msg = f"Failed to load model {model_name}"
            raise ValueError(msg) from e

        optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)
        batch_size = getattr(config, "batch_size", 2)
        data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
        dataloader = data_dict.get("loader", None)
        if dataloader is None or not hasattr(dataloader, "__iter__"):
            raise ValueError(f"Invalid dataloader for dataset: {dataset}")
        final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=policy_model, ref_model=ref_model, optimizer=optimizer, beta=beta))
        status = "completed"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("DPO failed: ")
        status = f"failed: {e!s}"
        final_loss = 0.0

    return {"backend": "pytorch", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": float(final_loss)}

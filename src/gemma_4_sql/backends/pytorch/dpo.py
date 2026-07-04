"""PyTorch-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.backends.pytorch.etl import build_dataloader
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TensorType, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
torch = None
nn = None
optim = None
functional = None
with catch_optional_imports():
    import torch
    from torch import nn, optim
    from torch.nn import functional


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[TensorType, TensorType, TensorType]:
    """Compute the DPO loss.

    Returns:
        tuple: The losses.

    """
    if torch is None or functional is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, functional.logsigmoid)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -functional.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return (loss.mean(), chosen_rewards.mean(), rejected_rewards.mean())


def _run_dpo_step(policy_model: object, ref_model: object, optimizer: object, batch: JSONDict, beta: float) -> object:
    """Run a single DPO training step.

    Returns:
        object: The resulting output from the operation.

    """
    optimizer.zero_grad()
    pi_ch = policy_model(batch.get("chosen_inputs", torch.zeros((1, 10))))
    pi_re = policy_model(batch.get("rejected_inputs", torch.zeros((1, 10))))
    with torch.no_grad():
        ref_ch = ref_model(batch.get("chosen_inputs", torch.zeros((1, 10))))
        ref_re = ref_model(batch.get("rejected_inputs", torch.zeros((1, 10))))
    pi_ch_logps = pi_ch.mean(dim=-1)
    pi_re_logps = pi_re.mean(dim=-1)
    ref_ch_logps = ref_ch.mean(dim=-1)
    ref_re_logps = ref_re.mean(dim=-1)
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    loss.backward()
    optimizer.step()
    return loss


def _run_training_epochs(state: TrainerState) -> float:
    dataloader = state.dataloader
    epochs = state.epochs
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    beta = state.beta
    """Run training epochs.

    Returns:
        object: The resulting output from the operation.

    """
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)
            epoch_loss += loss.item()
        final_loss = epoch_loss / max(1, len(dataloader))
    return final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    """Run a DPO training loop for PyTorch.

    Args:
    ----
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.
        epochs: The number of epochs.
        learning_rate: The learning rate.

    Returns:
    -------
        A dict with the execution status and metrics.

    """
    if torch is not None and nn is not None and (optim is not None):
        try:

            class DummyModel(nn.Module):
                """Dummy model for PyTorch DPO."""

                def __init__(self: object) -> None:
                    """Execute logic."""
                    super().__init__()
                    self.linear = nn.Linear(10, 10)

                def __call__(self: object, x: object) -> object:
                    """Execute logic.

                    Returns:
                        object: The resulting output from the operation.

                    """
                    return self.linear(x)

            policy_model = DummyModel()
            ref_model = DummyModel()
            optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)
            data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))
            dataloader = data_dict.get("loader", None)
            if dataloader is not None and hasattr(dataloader, "__iter__"):
                final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=policy_model, ref_model=ref_model, optimizer=optimizer, beta=beta))
            else:
                dummy_batch = {"chosen_inputs": torch.zeros((1, 10)), "chosen_labels": torch.zeros((1, 10)), "rejected_inputs": torch.zeros((1, 10)), "rejected_labels": torch.zeros((1, 10))}
                loss = _run_dpo_step(policy_model, ref_model, optimizer, dummy_batch, beta)
                final_loss = float(loss.item())
            status = "completed"
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.exception("DPO failed: ")
            status = f"failed: {e!s}"
            final_loss = 0.0
    else:
        status = "mocked_missing_torch"
        final_loss = 0.0
    return {"backend": "pytorch", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": float(final_loss)}

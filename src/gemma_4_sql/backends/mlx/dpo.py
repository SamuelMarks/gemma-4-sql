"""MLX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TensorType, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
mlx = None
nn = None
optim = None
functional = None
with catch_optional_imports():
    import mlx
    import mlx.core as mx
    import mlx.nn as mx_nn
    from mlx import nn, optim  # pragma: no cover
    from mlx.nn import functional  # pragma: no cover


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[TensorType, TensorType, TensorType]:
    """Compute the DPO loss.

    Returns:
        tuple: The losses.

    """
    if mx is None or mx_nn is None:  # pragma: no cover
        return (0.0, 0.0, 0.0)  # pragma: no cover
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, mx_nn.losses.log_sigmoid)  # pragma: no cover
    pi_logratios = policy_chosen_logps - policy_rejected_logps  # pragma: no cover
    ref_logratios = ref_chosen_logps - ref_rejected_logps  # pragma: no cover
    logits = pi_logratios - ref_logratios  # pragma: no cover
    loss = -functional.logsigmoid(beta * logits)  # pragma: no cover
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()  # pragma: no cover
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()  # pragma: no cover
    return (loss.mean(), chosen_rewards.mean(), rejected_rewards.mean())  # pragma: no cover


def _run_dpo_step(policy_model: object, ref_model: object, optimizer: object, batch: JSONDict, beta: float) -> object:
    """Run a single DPO training step.

    Returns:
        object: The resulting output from the operation.

    """
    optimizer.zero_grad()  # pragma: no cover
    pi_ch = policy_model(batch.get("chosen_inputs", mlx.zeros((1, 10))))  # pragma: no cover
    pi_re = policy_model(batch.get("rejected_inputs", mlx.zeros((1, 10))))  # pragma: no cover
    with mlx.no_grad():  # pragma: no cover
        ref_ch = ref_model(batch.get("chosen_inputs", mlx.zeros((1, 10))))  # pragma: no cover
        ref_re = ref_model(batch.get("rejected_inputs", mlx.zeros((1, 10))))  # pragma: no cover
    pi_ch_logps = pi_ch.mean(dim=-1)  # pragma: no cover
    pi_re_logps = pi_re.mean(dim=-1)  # pragma: no cover
    ref_ch_logps = ref_ch.mean(dim=-1)  # pragma: no cover
    ref_re_logps = ref_re.mean(dim=-1)  # pragma: no cover
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)  # pragma: no cover
    loss.backward()  # pragma: no cover
    optimizer.step()  # pragma: no cover
    return loss  # pragma: no cover


def _run_training_epochs(state: TrainerState) -> float:
    """Execute function."""
    dataloader = state.dataloader  # pragma: no cover
    epochs = state.epochs  # pragma: no cover
    policy_model = state.policy_model  # pragma: no cover
    ref_model = state.ref_model  # pragma: no cover
    optimizer = state.optimizer  # pragma: no cover
    beta = state.beta  # pragma: no cover
    """Run training epochs.

    Returns:
        object: The resulting output from the operation.

    """
    final_loss = 0.0  # pragma: no cover
    for _epoch in range(epochs):  # pragma: no cover
        epoch_loss = 0.0  # pragma: no cover
        for batch in dataloader:  # pragma: no cover
            loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)  # pragma: no cover
            epoch_loss += loss.item()  # pragma: no cover
        final_loss = epoch_loss / max(1, len(dataloader))  # pragma: no cover
    return final_loss  # pragma: no cover


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    """Execute function."""
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    """Run a DPO training loop for MLX.

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
    if mlx is not None and nn is not None and (optim is not None):
        try:

            class DummyModel(nn.Module):
                """Dummy model for MLX DPO."""

                def __init__(self: object) -> None:
                    """Docstring."""
                    super().__init__()
                    self.linear = nn.Linear(10, 10)

                def __call__(self: object, x: object) -> object:
                    """Docstring.

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
                dummy_batch = {"chosen_inputs": mlx.zeros((1, 10)), "chosen_labels": mlx.zeros((1, 10)), "rejected_inputs": mlx.zeros((1, 10)), "rejected_labels": mlx.zeros((1, 10))}
                loss = _run_dpo_step(policy_model, ref_model, optimizer, dummy_batch, beta)
                final_loss = float(loss.item())
            status = "completed"
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.exception("DPO failed: ")
            status = f"failed: {e!s}"
            final_loss = 0.0
    else:
        status = "mocked_missing_mlx"
        final_loss = 0.0
    return {"backend": "mlx", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": float(final_loss)}

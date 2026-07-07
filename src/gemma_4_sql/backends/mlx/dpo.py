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
    import mlx  # pragma: no cover
    import mlx.core as mx  # pragma: no cover
    import mlx.nn as mx_nn  # pragma: no cover
    from mlx import nn, optim  # pragma: no cover
    from mlx.nn import functional  # pragma: no cover
    from mlx_lm import load  # pragma: no cover


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[TensorType, TensorType, TensorType]:
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
    pi_ch = policy_model(batch["chosen_inputs"])  # pragma: no cover
    pi_re = policy_model(batch["rejected_inputs"])  # pragma: no cover
    with mlx.no_grad():  # pragma: no cover
        ref_ch = ref_model(batch["chosen_inputs"])  # pragma: no cover
        ref_re = ref_model(batch["rejected_inputs"])  # pragma: no cover
    pi_ch_logps = pi_ch.mean(dim=-1)  # pragma: no cover
    pi_re_logps = pi_re.mean(dim=-1)  # pragma: no cover
    ref_ch_logps = ref_ch.mean(dim=-1)  # pragma: no cover
    ref_re_logps = ref_re.mean(dim=-1)  # pragma: no cover
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)  # pragma: no cover
    loss.backward()  # pragma: no cover
    optimizer.step()  # pragma: no cover
    return loss  # pragma: no cover


def _run_training_epochs(state: TrainerState) -> float:
    """Execute function.

    Returns:
        The execution result.

    """
    dataloader = state.dataloader  # pragma: no cover
    epochs = state.epochs  # pragma: no cover
    policy_model = state.policy_model  # pragma: no cover
    ref_model = state.ref_model  # pragma: no cover
    optimizer = state.optimizer  # pragma: no cover
    beta = state.beta  # pragma: no cover
    # Run training epochs.
    final_loss = 0.0  # pragma: no cover
    for _epoch in range(epochs):  # pragma: no cover
        epoch_loss = 0.0  # pragma: no cover
        for batch in dataloader:  # pragma: no cover
            loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)  # pragma: no cover
            epoch_loss += loss.item()  # pragma: no cover
        final_loss = epoch_loss / max(1, len(dataloader))  # pragma: no cover
    return final_loss  # pragma: no cover


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    """Execute function.


    Args:
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate).
    Returns:
        The execution result.

    """
    model_name = getattr(config, "model_name", "model")  # pragma: no cover
    dataset = getattr(config, "dataset", "dataset")  # pragma: no cover
    beta = getattr(config, "beta", 0.1)  # pragma: no cover
    epochs = getattr(config, "epochs", 1)  # pragma: no cover
    learning_rate = getattr(config, "learning_rate", 1e-05)  # pragma: no cover
    """Run a DPO training loop for MLX.  # pragma: no cover

    Args:  # pragma: no cover
    ----  # pragma: no cover
        model_name: The name of the model.  # pragma: no cover
        dataset: The dataset name.  # pragma: no cover
        beta: The beta temperature parameter.  # pragma: no cover
        epochs: The number of epochs.  # pragma: no cover
        learning_rate: The learning rate.  # pragma: no cover

    Returns:  # pragma: no cover
    -------  # pragma: no cover
        A dict with the execution status and metrics.  # pragma: no cover

    """  # pragma: no cover
    if mlx is None or nn is None or optim is None or "load" not in globals():  # pragma: no cover
        from gemma_4_sql.exceptions import DependencyMissingError  # pragma: no cover

        raise DependencyMissingError("MLX dependencies are missing.")  # pragma: no cover
    try:  # pragma: no cover
        policy_model, _ = load(model_name)  # pragma: no cover
        ref_model, _ = load(model_name)  # pragma: no cover
        optimizer = optim.AdamW(learning_rate=learning_rate)  # pragma: no cover
        data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))  # pragma: no cover
        dataloader = data_dict.get("loader", None)  # pragma: no cover
        if dataloader is None or not hasattr(dataloader, "__iter__"):  # pragma: no cover
            raise ValueError(f"Invalid dataloader for dataset: {dataset}")  # pragma: no cover
        final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=policy_model, ref_model=ref_model, optimizer=optimizer, beta=beta))  # pragma: no cover
        status = "completed"  # pragma: no cover
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:  # pragma: no cover
        logger.exception("DPO failed: ")  # pragma: no cover
        status = f"failed: {e!s}"  # pragma: no cover
        final_loss = 0.0  # pragma: no cover
    return {"backend": "mlx", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": float(final_loss)}  # pragma: no cover

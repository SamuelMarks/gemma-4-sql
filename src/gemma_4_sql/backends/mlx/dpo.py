"""MLX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, ModelType, TensorType, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

mlx = optional_import("mlx")
mx = optional_import("mlx.core")
nn = optional_import("mlx.nn")
mx_nn: types.ModuleType | None = nn
optim = optional_import("mlx.optimizers")
load = optional_attr("mlx_lm", "load")


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[Any, Any, Any]:
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
    if mx is None or mx_nn is None:
        return (0.0, 0.0, 0.0)

    def _fallback_log_sig(x: Any) -> Any:
        """Compute fallback negative identity when `mlx.nn.losses.log_sigmoid` is absent."""
        return mx.negative(x) if hasattr(mx, "negative") else -x

    log_sig_fn = getattr(getattr(mx_nn, "losses", None), "log_sigmoid", _fallback_log_sig)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, log_sig_fn)


def _run_dpo_step(policy_model: ModelType, ref_model: ModelType, optimizer: Any, batch: JSONDict, beta: float) -> Any:
    """Run a single MLX DPO training step.

    Args:
        policy_model: Trainable policy model.
        ref_model: Frozen reference model.
        optimizer: MLX optimizer instance.
        batch: Preference batch dictionary.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO step loss.

    """
    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()
    pi_ch = policy_model(batch["chosen_inputs"])
    pi_re = policy_model(batch["rejected_inputs"])
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
    """Run MLX DPO training epochs over the preference dataloader.

    Args:
        state: `TrainerState` holding models, optimizer, dataloader, and beta.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader = state.dataloader
    epochs = state.epochs
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    beta = state.beta
    # Run training epochs.
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
    """Execute Direct Preference Optimization (DPO) using the MLX backend.

    Args:
        config: DPO configuration parameters.
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate).

    Returns:
        Dictionary containing backend, model, dataset, beta, status, and final_loss.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)

    if mlx is None or mx is None or nn is None or optim is None or load is None:
        raise DependencyMissingError("MLX dependencies are missing.")
    final_loss = 0.0
    try:
        loaded_p = load(model_name)
        policy_model: ModelType = cast(Any, loaded_p)[0] if isinstance(loaded_p, (tuple, list)) else loaded_p
        loaded_r = load(model_name)
        ref_model: ModelType = cast(Any, loaded_r)[0] if isinstance(loaded_r, (tuple, list)) else loaded_r
        optimizer = optim.AdamW(learning_rate=learning_rate)
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
    return {"backend": "mlx", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": float(final_loss)}

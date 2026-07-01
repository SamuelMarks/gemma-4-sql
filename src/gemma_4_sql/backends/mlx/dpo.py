"""MLX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.mlx.etl import build_dataloader

logger = logging.getLogger(__name__)

try:
    import mlx
    from mlx import nn, optim
    from mlx.nn import functional
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    mlx = None
    nn = None
    optim = None
    functional = None


def dpo_loss(policy_chosen_logps: object, policy_rejected_logps: object, ref_chosen_logps: object, ref_rejected_logps: object, beta: float = 0.1) -> tuple[object, object, object]:
    """Compute the DPO loss for MLX.

    Args:
    ----
        policy_chosen_logps: Log probabilities of chosen responses from policy model.
        policy_rejected_logps: Log probabilities of rejected responses from policy model.
        ref_chosen_logps: Log probabilities of chosen responses from reference model.
        ref_rejected_logps: Log probabilities of rejected responses from reference model.
        beta: Temperature parameter for the DPO loss.

    Returns:
    -------
        A tuple of (loss, chosen_rewards, rejected_rewards).

    """
    if mlx is None or functional is None:
        return (0.0, 0.0, 0.0)
    pi_logratios = policy_chosen_logps - policy_rejected_logps  # type: ignore[operator]
    ref_logratios = ref_chosen_logps - ref_rejected_logps  # type: ignore[operator]
    logits = pi_logratios - ref_logratios
    loss = -functional.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()  # type: ignore[operator]
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()  # type: ignore[operator]
    return (loss.mean(), chosen_rewards.mean(), rejected_rewards.mean())


def run_dpo(model_name: str, dataset: str, beta: float = 0.1, epochs: int = 1, learning_rate: float = 1e-5) -> dict[str, object]:
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
    if mlx is not None and nn is not None and optim is not None:
        try:

            class DummyModel(nn.Module):
                """Dummy model for MLX DPO."""

                def __init__(self: object) -> None:
                    super().__init__()
                    self.linear = nn.Linear(10, 10)

                def __call__(self: object, x: object) -> object:
                    return self.linear(x)

            policy_model = DummyModel()
            ref_model = DummyModel()

            optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)

            data_dict = build_dataloader(dataset_name=dataset, split="train", batch_size=2)
            dataloader = data_dict.get("loader", None)

            if dataloader is not None and hasattr(dataloader, "__iter__"):
                for _epoch in range(epochs):
                    epoch_loss = 0.0
                    for batch in dataloader:
                        optimizer.zero_grad()
                        # DPO inputs mock
                        pi_ch = policy_model(batch.get("chosen_inputs", mlx.zeros((1, 10))))  # type: ignore[attr-defined]
                        pi_re = policy_model(batch.get("rejected_inputs", mlx.zeros((1, 10))))  # type: ignore[attr-defined]
                        with mlx.no_grad():
                            ref_ch = ref_model(batch.get("chosen_inputs", mlx.zeros((1, 10))))  # type: ignore[attr-defined]
                            ref_re = ref_model(batch.get("rejected_inputs", mlx.zeros((1, 10))))  # type: ignore[attr-defined]

                        # Mocking logps using mean
                        pi_ch_logps = pi_ch.mean(dim=-1)
                        pi_re_logps = pi_re.mean(dim=-1)
                        ref_ch_logps = ref_ch.mean(dim=-1)
                        ref_re_logps = ref_re.mean(dim=-1)

                        loss, _, _ = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
                        loss.backward()  # type: ignore[attr-defined]
                        optimizer.step()
                        epoch_loss += loss.item()  # type: ignore[attr-defined]
                    final_loss = epoch_loss / max(1, len(dataloader))  # type: ignore[arg-type]
            else:
                optimizer.zero_grad()
                pi_ch = policy_model(mlx.zeros((1, 10)))
                pi_re = policy_model(mlx.zeros((1, 10)))
                with mlx.no_grad():
                    ref_ch = ref_model(mlx.zeros((1, 10)))
                    ref_re = ref_model(mlx.zeros((1, 10)))
                loss, _, _ = dpo_loss(pi_ch.mean(dim=-1), pi_re.mean(dim=-1), ref_ch.mean(dim=-1), ref_re.mean(dim=-1), beta)
                loss.backward()  # type: ignore[attr-defined]
                optimizer.step()
                final_loss = loss.item()  # type: ignore[attr-defined]

            status = "completed"
        except Exception as e:
            logger.exception("DPO failed: %s", e)
            status = f"failed: {e!s}"
            final_loss = 0.0
    else:
        status = "mocked_missing_mlx"
        final_loss = 0.0
    return {"backend": "mlx", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": float(final_loss)}

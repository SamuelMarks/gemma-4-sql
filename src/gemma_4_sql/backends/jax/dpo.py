"""JAX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

from gemma_4_sql.backends.jax.etl import build_dataloader

try:
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp
    import optax
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    jax = None
    jnp = None
    jnn = None
    optax = None
try:
    from flax import nnx

    from .gemma4 import Gemma4Config, Gemma4ForCausalLM
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    Gemma4ForCausalLM = None  # type: ignore[misc]
    Gemma4Config = None
    nnx = None


def dpo_loss(policy_chosen_logps: object, policy_rejected_logps: object, ref_chosen_logps: object, ref_rejected_logps: object, beta: float = 0.1) -> tuple[object, object, object]:
    """Compute the DPO loss for JAX.

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
    if jnp is None or jnn is None:
        return (0.0, 0.0, 0.0)
    pi_logratios = policy_chosen_logps - policy_rejected_logps  # type: ignore[operator]
    ref_logratios = ref_chosen_logps - ref_rejected_logps  # type: ignore[operator]
    logits = pi_logratios - ref_logratios
    loss = -jnn.log_sigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)  # type: ignore[operator]
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)  # type: ignore[operator]
    return (jnp.mean(loss), jnp.mean(chosen_rewards), jnp.mean(rejected_rewards))


def run_dpo(model_name: str, dataset: str, beta: float = 0.1, epochs: int = 1, learning_rate: float = 1e-5) -> dict[str, object]:
    """Run a DPO training loop for JAX.

    Args:
    ----
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
    -------
        A dict with the execution status and metrics.

    """
    final_loss = 0.0
    status = "completed"
    if jax is not None and jnp is not None and jnn is not None and (optax is not None) and (Gemma4ForCausalLM is not None):
        try:
            policy_model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))  # type: ignore[arg-type]
            ref_model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(1))  # type: ignore[arg-type]

            optimizer = nnx.Optimizer(policy_model, optax.adamw(learning_rate))

            def compute_logps(model: object, inputs: object, labels: object) -> object:
                """Mock logp computation."""
                logits = model(inputs)  # type: ignore[operator]
                return jnp.sum(logits * labels, axis=-1)

            def dpo_step_loss(policy_model: object, ref_model: object, batch: dict[str, object]) -> object:
                """Compute DPO loss for a step."""
                pi_ch_logps = compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])
                pi_re_logps = compute_logps(policy_model, batch["rejected_inputs"], batch["rejected_labels"])
                ref_ch_logps = compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"])
                ref_re_logps = compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"])
                loss, _, _ = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
                return loss

            @nnx.jit  # type: ignore[misc]
            def train_step(policy_model: object, ref_model: object, optimizer: nnx.Optimizer, batch: dict[str, object]) -> object:
                """Execute a single JAX-compiled DPO training step."""
                loss, grads = nnx.value_and_grad(dpo_step_loss)(policy_model, ref_model, batch)
                optimizer.update(grads)
                return loss

            data_dict = build_dataloader(dataset_name=dataset, split="train", batch_size=2)
            dataloader = data_dict.get("loader", None)

            if dataloader is not None and hasattr(dataloader, "__iter__"):
                for _epoch in range(epochs):
                    epoch_loss = 0.0
                    for batch in dataloader:
                        loss = train_step(policy_model, ref_model, optimizer, batch)
                        epoch_loss += loss.item()  # type: ignore[attr-defined]
                    final_loss = epoch_loss / max(1, len(dataloader))  # type: ignore[arg-type, operator]
            else:
                dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
                dummy_batch = {"chosen_inputs": dummy_input, "chosen_labels": dummy_input, "rejected_inputs": dummy_input, "rejected_labels": dummy_input}
                loss = train_step(policy_model, ref_model, optimizer, dummy_batch)
                final_loss = loss.item()  # type: ignore[attr-defined]

        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_jax"
        final_loss = 0.0
    return {"backend": "jax", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

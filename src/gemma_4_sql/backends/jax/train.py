"""JAX-specific training pipeline with masked next-token causal LM cross-entropy."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

import jax
import optax
from flax import nnx
from jax import Array

from gemma_4_sql._types_tensors import extract_scalar_float
from gemma_4_sql.backends._jax_common import (
    NNXOptimizerProtocol,
    build_tpu_mesh_and_sharding,
    call_jax_causal_lm,
    initialize_jax_distributed_if_configured,
)
from gemma_4_sql.backends._jax_common import (
    apply_nnx_optimizer_update as _apply_nnx_optimizer_update,
)
from gemma_4_sql.backends.common_dpo import ScalarItemProtocol
from gemma_4_sql.backends.common_train import (
    generic_run_training_epochs,
    resolve_training_dataloader,
    run_train_wrapper,
)
from gemma_4_sql.backends.jax._protocols import TypedJaxDeviceProtocol, is_jnp_missing
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.jax.inference import MODEL_CACHE as _MODEL_CACHE
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    ETLConfig,
    JaxCausalLMProtocol,
    JaxShardingProtocol,
    JaxTrainingState,
    TrainingConfig,
    TrainResultDict,
)

Gemma4Config = ModelConfig
IGNORE_INDEX = -100
type JaxTrainStepFn = Callable[..., Array | ScalarItemProtocol | SupportsFloat | float]


@runtime_checkable
class _NNXValueAndGradModuleProtocol(Protocol):
    """Protocol for a `flax.nnx` module exposing `value_and_grad` and `jit`."""

    def value_and_grad(self, fn: Callable[..., Array], /) -> Callable[..., tuple[Array, Array]]:
        """Create a value-and-gradient function for an NNX module.

        Args:
            fn: Loss callable accepting `(model, batch)`.

        Returns:
            Callable returning `(loss, grads)`.

        """
        ...

    def jit(self, fn: JaxTrainStepFn, /) -> JaxTrainStepFn:
        """JIT-compile an NNX function.

        Args:
            fn: Training step callable.

        Returns:
            Compiled training step callable.

        """
        ...


def _create_nnx_optimizer[ModelT: JaxCausalLMProtocol | Gemma4ForCausalLM | nnx.Module](
    model: ModelT,
    tx: optax.GradientTransformation,
) -> nnx.Optimizer[ModelT]:
    """Create a Flax NNX Optimizer (`nnx.Optimizer(model, tx, wrt=nnx.Param)`).

    Args:
        model: Flax NNX module to optimize.
        tx: Optax gradient transformation.

    Returns:
        Configured `nnx.Optimizer` instance.

    """
    opt_cls: Callable[..., nnx.Optimizer[ModelT]] = nnx.Optimizer
    return opt_cls(model, tx, wrt=nnx.Param)


def _loss_fn(
    model: Callable[..., Array] | JaxCausalLMProtocol | Gemma4ForCausalLM,
    batch: Mapping[str, Array],
) -> Array:
    """Compute shifted, `loss_mask`-weighted next-token causal cross-entropy loss.

    Supports both pre-shifted ETL batches (`inputs = full[:-1]`, `targets = full[1:]`,
    `loss_mask`) and raw unshifted `input_ids`-only batches (`logits[:, :-1, :]` vs
    `input_ids[:, 1:]`), masking out prompt tokens (`loss_mask == 0`) and `IGNORE_INDEX`.

    Args:
        model: Trainable JAX/Flax NNX causal language model.
        batch: Batch mapping containing `'inputs'`/`'input_ids'`, optional `'targets'`/`'labels'`,
            and optional `'loss_mask'`.

    Returns:
        Scalar mean masked cross-entropy loss array.

    """
    inputs = batch["inputs"] if "inputs" in batch else batch["input_ids"]
    seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
    positions = jnp.arange(seq_len)[None, :]
    raw_out = call_jax_causal_lm(model, inputs, positions)
    first_out = raw_out[0] if isinstance(raw_out, tuple) else raw_out
    logits: Array = first_out if isinstance(first_out, Array) else jnp.asarray(first_out)

    raw_targets = batch["targets"] if "targets" in batch else batch.get("labels")
    if raw_targets is None:
        shift_logits = logits[:, :-1, :]
        shift_targets = inputs[:, 1:]
        raw_mask = batch["loss_mask"][:, 1:] if "loss_mask" in batch else None
    else:
        min_len = min(int(logits.shape[1]), int(raw_targets.shape[-1]))
        shift_logits = logits[:, :min_len, :]
        shift_targets = raw_targets[:, :min_len]
        raw_mask = batch["loss_mask"][:, :min_len] if "loss_mask" in batch else None

    valid_mask = (shift_targets != IGNORE_INDEX).astype(jnp.float32)
    if raw_mask is not None:
        valid_mask *= raw_mask.astype(jnp.float32)

    safe_targets = jnp.where(shift_targets >= 0, shift_targets, 0)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, safe_targets)
    masked_loss = per_token_loss * valid_mask
    denom = jnp.maximum(jnp.sum(valid_mask), 1.0)
    res: Array = jnp.sum(masked_loss) / denom
    return res


def _get_train_step_fn() -> JaxTrainStepFn:
    """Return a JIT-compiled Flax NNX training step function.

    Returns:
        Compiled training step callable `(model, optimizer, batch) -> loss`.

    Raises:
        DependencyMissingError: If `flax.nnx.value_and_grad` is unavailable.

    """
    if nnx is None or not isinstance(nnx, _NNXValueAndGradModuleProtocol):
        raise DependencyMissingError(
            "Flax NNX value_and_grad is required for JAX training.",
            dependency_name="flax.nnx",
            backend="jax",
        )

    def train_step[OptOutT](
        model: JaxCausalLMProtocol,
        optimizer: NNXOptimizerProtocol[JaxCausalLMProtocol, Array, OptOutT],
        batch: Mapping[str, Array],
    ) -> Array:
        """Execute a single JAX-compiled training step.

        Args:
            model: Trainable Flax NNX causal language model.
            optimizer: Flax NNX optimizer instance.
            batch: Token batch dictionary with `'inputs'`, `'targets'`, and optional `'loss_mask'`.

        Returns:
            Scalar batch loss tensor.

        """
        (loss, grads) = nnx.value_and_grad(_loss_fn)(model, batch)
        _apply_nnx_optimizer_update(optimizer, model, grads)
        return loss

    return nnx.jit(train_step)


def _run_training_epochs[ModelT, OptT, BatchValT](
    state: JaxTrainingState[ModelT, OptT, Mapping[str, BatchValT]]
    | JaxTrainingState[ModelT, OptT, dict[str, BatchValT]],
) -> float:
    """Run supervised training epochs over the JAX dataloader.

    Args:
        state: `JaxTrainingState` holding dataloader, model, optimizer, sharding, and step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[Mapping[str, BatchValT]] = state.dataloader
    train_step = state.train_step
    model = state.model
    sharding = state.sharding

    def process_batch(batch: Mapping[str, BatchValT]) -> float:
        """Place batch tensors onto target sharding and execute one training step.

        Args:
            batch: Mapping with `inputs`, `targets`, and optional `loss_mask` arrays.

        Returns:
            Scalar batch loss as a float.

        """
        device_put_fn: Callable[[BatchValT, str | Mapping[str, str] | JaxShardingProtocol | None], BatchValT] = vars(
            jax
        )["device_put"]
        sharded_batch: dict[str, BatchValT] = {
            **batch,
            "inputs": device_put_fn(batch["inputs"], sharding),
            "targets": device_put_fn(batch["targets"], sharding),
        }
        if "loss_mask" in batch:
            sharded_batch["loss_mask"] = device_put_fn(batch["loss_mask"], sharding)
        loss = train_step(model, state.optimizer, sharded_batch)
        if isinstance(loss, (ScalarItemProtocol, int, float, SupportsFloat)):
            return extract_scalar_float(loss)
        return 0.0

    return generic_run_training_epochs(state.epochs, dataloader, process_batch)


def _resolve_gemma4_config(model_name: str) -> ModelConfig:
    """Resolve a `Gemma4Config` preset for the given `model_name`.

    Args:
        model_name: Model name or preset identifier.

    Returns:
        Resolved `ModelConfig` object.

    """
    return resolve_gemma4_config(Gemma4Config, model_name)


def _execute_train(
    dataset: str,
    epochs: int,
    learning_rate: float,
    batch_size: int = 2,
    model_name: str = "gemma-4",
) -> tuple[str, float]:
    """Execute the core training loop for JAX.

    Args:
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        learning_rate: Training learning rate.
        batch_size: Training batch size.
        model_name: Target model identifier used to select architectural preset.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    if jax is None or is_jnp_missing(jnp) or optax is None or not callable(Gemma4ForCausalLM) or nnx is None:
        raise DependencyMissingError("JAX dependencies are missing for training.")

    cached_model = _MODEL_CACHE.get_model(model_name)
    if isinstance(cached_model, nnx.Module):
        model: Gemma4ForCausalLM | nnx.Module = cached_model
    else:
        model = Gemma4ForCausalLM(_resolve_gemma4_config(model_name), rngs=nnx.Rngs(0))
    devices_fn: Callable[[], Sequence[TypedJaxDeviceProtocol]] = vars(jax)["devices"]
    mesh = jax.sharding.Mesh(devices_fn(), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=100,
        decay_steps=max(1, epochs * 1000),
        end_value=learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule),
    )
    optimizer = _create_nnx_optimizer(model, tx)
    train_step = _get_train_step_fn()
    dataloader, _ = resolve_training_dataloader(
        dataset,
        lambda: build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size)),
        lambda: [
            {
                "inputs": jnp.array([[2, 10, 11, 12]], dtype=jnp.int32),
                "targets": jnp.array([[10, 11, 12, 1]], dtype=jnp.int32),
                "loss_mask": jnp.array([[0, 1, 1, 1]], dtype=jnp.int32),
            }
        ],
    )

    final_loss = _run_training_epochs(
        JaxTrainingState(
            dataloader=dataloader,
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            train_step=train_step,
            learning_rate=learning_rate,
            sharding=sharding,
        )
    )
    _MODEL_CACHE.put_model(model_name, model)
    return "completed", final_loss


def train_model(config: TrainingConfig, **kwargs: StatusValue) -> TrainResultDict:
    """Train a Text-to-SQL model using JAX and Flax NNX.

    Args:
        config: Training configuration parameters.
        **kwargs: Extra runtime options (such as `raise_on_error=True`).

    Returns:
        Structured `TrainResultDict` containing JAX training status and final loss.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    raw_alias = (
        kwargs.get("backend_alias")
        or kwargs.get("backend")
        or (config.backend if config.backend == "jax_native" else "jax")
    )
    backend_alias = str(raw_alias)
    hardware = str(kwargs.get("hardware", config.extra_kwargs.get("hardware", "cpu")))
    strict_hw = bool(kwargs.get("strict_hardware", config.extra_kwargs.get("strict_hardware", False)))

    if jax is None or is_jnp_missing(jnp) or nnx is None or optax is None or not callable(Gemma4ForCausalLM):
        raise DependencyMissingError("JAX dependencies are missing for training.", backend=backend_alias)

    _ = initialize_jax_distributed_if_configured(jax)
    if strict_hw:
        _ = build_tpu_mesh_and_sharding(jax, hardware, strict_hardware=True, backend_label=backend_alias)

    model_name = config.model_name
    dataset = config.dataset
    epochs = config.epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    return run_train_wrapper(
        backend_alias,
        config,
        lambda: _execute_train(dataset, epochs, learning_rate, batch_size=batch_size, model_name=model_name),
        **kwargs,
    )

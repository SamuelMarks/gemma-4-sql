"""MLX-specific DPO (Direct Preference Optimization) logic without fake TypeGuards."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql._types_models import MlxLMLoadProtocol
from gemma_4_sql.backends.common_dpo import (
    ArithmeticTensorProtocol,
    generic_dpo_loss,
    run_dpo_wrapper,
)
from gemma_4_sql.backends.common_train import (
    generic_run_training_epochs,
    resolve_training_dataloader,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    DPOConfig,
    DPOLoopState,
    DPOResultDict,
    ETLConfig,
    ForwardOutput,
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    MlxCoreProtocol,
    MlxNNProtocol,
    NDArrayProtocol,
    TensorProtocol,
    TensorType,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _MlxOptimizerProtocol[ModelT](Protocol):
    """Protocol for an `mlx.optimizers` optimizer supporting `.update(model, grads)`."""

    def update(
        self,
        model: ModelT,
        gradients: dict[str, MlxArrayProtocol],
    ) -> None:
        """Apply gradients to `model` parameters.

        Args:
            model: Target MLX model.
            gradients: Nested gradient dictionary.

        """
        ...


@runtime_checkable
class _MlxOptimizerStateProtocol(Protocol):
    """Protocol for an optimizer exposing `.state`."""

    @property
    def state(self) -> Mapping[str, MlxArrayProtocol] | Sequence[MlxArrayProtocol]:
        """Provide the optimizer state structure."""
        ...


type MlxDPOBatchValue = (
    TensorType
    | JSONValue
    | MlxArrayProtocol
    | NDArrayProtocol
    | SupportsFloat
    | Sequence[int]
    | Sequence[Sequence[int]]
    | int
    | float
    | str
    | None
)


@runtime_checkable
class _MlxDPOCallableProtocol(Protocol):
    """Protocol for a callable MLX DPO model."""

    def __call__(self, *args: MlxDPOBatchValue) -> MlxDPOCallReturn:
        """Evaluate model forward pass."""
        ...


type MlxDPOBatch = Mapping[str, MlxDPOBatchValue]
type MlxDPOCallReturn = MlxArrayProtocol | ForwardOutput | ArithmeticTensorProtocol | SupportsFloat | str
type MlxDPOModel = _MlxDPOCallableProtocol | Callable[..., MlxDPOCallReturn] | MlxCausalLMProtocol


type _MlxDPOEvalArg = (
    MlxArrayProtocol
    | ArithmeticTensorProtocol
    | float
    | Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]]
    | Mapping[str, MlxArrayProtocol]
    | Sequence[MlxArrayProtocol]
)


@runtime_checkable
class _MlxDPOCoreProtocol(Protocol):
    """Protocol for `mlx.core` operations required by MLX DPO."""

    def array(self, val: MlxDPOBatchValue, /) -> MlxArrayProtocol:
        """Convert `val` into an `MlxArrayProtocol`.

        Args:
            val: Token ID sequence, scalar, or array.

        Returns:
            Constructed `MlxArrayProtocol`.

        """
        ...

    def logsumexp(
        self,
        a: MlxArrayProtocol,
        *,
        axis: int = -1,
        keepdims: bool = False,
    ) -> MlxArrayProtocol:
        """Compute log-sum-exp along `axis`.

        Args:
            a: Vocabulary logits array.
            axis: Normalization axis index.
            keepdims: Whether to keep reduced dimensions of size 1.

        Returns:
            Log-partition `MlxArrayProtocol`.

        """
        ...

    def take_along_axis(
        self,
        a: MlxArrayProtocol,
        indices: MlxArrayProtocol,
        *,
        axis: int = -1,
    ) -> MlxArrayProtocol:
        """Gather elements from `a` along `axis` at `indices`.

        Args:
            a: Log-probabilities array.
            indices: Expanded target token ID array.
            axis: Gather axis index.

        Returns:
            Gathered log-probabilities `MlxArrayProtocol`.

        """
        ...

    def mean[TensorT: ArithmeticTensorProtocol | float](self, a: TensorT, /) -> TensorT:
        """Compute scalar mean over all elements of `a`.

        Args:
            a: Per-example DPO loss array.

        Returns:
            Mean scalar loss.

        """
        ...

    def sum(self, a: MlxArrayProtocol, *, axis: int = -1) -> MlxArrayProtocol:
        """Sum array elements along `axis`.

        Args:
            a: Masked token log-probabilities array.
            axis: Sequence axis index.

        Returns:
            Per-sequence summed log-probabilities `MlxArrayProtocol`.

        """
        ...

    def expand_dims(self, a: MlxArrayProtocol, *, axis: int) -> MlxArrayProtocol:
        """Insert a length-1 axis at `axis`.

        Args:
            a: Input token ID array.
            axis: Dimension index to insert.

        Returns:
            Expanded `MlxArrayProtocol`.

        """
        ...

    def squeeze(self, a: MlxArrayProtocol, *, axis: int) -> MlxArrayProtocol:
        """Remove length-1 axis at `axis`.

        Args:
            a: Gathered log-probability array.
            axis: Singleton dimension index.

        Returns:
            Squeezed `MlxArrayProtocol`.

        """
        ...

    def not_equal(self, a: MlxArrayProtocol, b: int, /) -> MlxArrayProtocol:
        """Compute elementwise `a != b` mask.

        Args:
            a: Target token ID array.
            b: Padding token ID integer.

        Returns:
            Boolean mask `MlxArrayProtocol`.

        """
        ...

    def eval(self, *args: _MlxDPOEvalArg | None) -> None:
        """Evaluate lazy MLX parameter, optimizer state, and loss arrays.

        Args:
            *args: Parameter mappings, optimizer state, and loss values.

        """
        ...


@runtime_checkable
class _MlxDPONNProtocol(Protocol):
    """Protocol for `mlx.nn` exposing `log_sigmoid` and `value_and_grad`."""

    def log_sigmoid[TensorT: ArithmeticTensorProtocol | float](self, x: TensorT, /) -> TensorT:
        """Compute elementwise `log(sigmoid(x))`.

        Args:
            x: Preference margin array or scalar.

        Returns:
            Log-sigmoid array or scalar.

        """
        ...

    def value_and_grad[LossReturnT](
        self,
        model: MlxDPOModel,
        fn: Callable[..., LossReturnT],
        /,
    ) -> Callable[..., tuple[LossReturnT, dict[str, MlxArrayProtocol]]]:
        """Create a value-and-gradient callable with respect to `model`.

        Args:
            model: Trainable MLX policy model.
            fn: Scalar DPO loss callable.

        Returns:
            Callable returning `(loss, grads)`.

        """
        ...


@runtime_checkable
class _MlxDPOOptimModuleProtocol(Protocol):
    """Protocol for `mlx.optimizers` exposing `AdamW`."""

    def AdamW(self, *, learning_rate: float) -> _MlxOptimizerProtocol[MlxCausalLMProtocol]:
        """Construct an `AdamW` MLX optimizer.

        Args:
            learning_rate: Optimizer learning rate.

        Returns:
            Configured `_MlxOptimizerProtocol`.

        """
        ...


mlx = optional_import("mlx")
mx: _MlxDPOCoreProtocol | None = optional_import("mlx.core", _MlxDPOCoreProtocol)
nn: _MlxDPONNProtocol | None = optional_import("mlx.nn", _MlxDPONNProtocol)
mx_nn = nn
optim: _MlxDPOOptimModuleProtocol | None = optional_import("mlx.optimizers", _MlxDPOOptimModuleProtocol)
load: MlxLMLoadProtocol | None = import_optional_symbol(
    "mlx_lm",
    "load",
    MlxLMLoadProtocol,
)


def dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float = 0.1,
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
    """Compute the Direct Preference Optimization (DPO) loss for MLX arrays.

    Args:
        policy_chosen_logps: Log probabilities of chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of rejected completions from the reference model.
        beta: Temperature parameter controlling deviation from the reference model.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    Raises:
        DependencyMissingError: If MLX dependencies (`mx` or `mx_nn`) are missing.

    """
    if mx is None or mx_nn is None:
        raise DependencyMissingError("MLX DPO dependencies are missing.", backend="mlx")
    return generic_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        mx_nn.log_sigmoid,
    )


def _to_mlx_array(
    val: MlxDPOBatchValue,
) -> MlxArrayProtocol | None:
    """Convert `val` to an `MlxArrayProtocol` when `mx` is available and `val` is array-convertible.

    Args:
        val: Candidate token ID tensor or sequence.

    Returns:
        Converted `MlxArrayProtocol` or `None`.

    """
    if isinstance(val, MlxArrayProtocol):
        return val
    if mx is not None and isinstance(val, (list, tuple, int, float)):
        arr: MlxArrayProtocol = mx.array(val)
        return arr
    return None


def _compute_mlx_logps(
    model: MlxDPOModel,
    inputs: MlxDPOBatchValue,
    labels: MlxDPOBatchValue = None,
) -> MlxArrayProtocol | ArithmeticTensorProtocol | float:
    """Compute per-sequence summed token log-probabilities for MLX models.

    Args:
        model: Policy or reference MLX model callable.
        inputs: Input token ID array of shape `(batch_size, seq_len)`.
        labels: Optional target token ID array of shape `(batch_size, seq_len)`.

    Returns:
        Per-sequence log-probability tensor of shape `(batch_size,)` or scalar fallback.

    """
    if isinstance(model, MlxCausalLMProtocol):
        arr_in = _to_mlx_array(inputs)
        raw_logits: MlxDPOCallReturn = model(arr_in) if arr_in is not None else 0.0
    else:
        raw_logits = model(inputs)

    target = labels if labels is not None else inputs
    target_arr = _to_mlx_array(target)
    if mx is not None and isinstance(raw_logits, MlxArrayProtocol) and raw_logits.ndim >= 2 and target_arr is not None:
        logits: MlxArrayProtocol = raw_logits
        if labels is None and target_arr.ndim == 2 and int(target_arr.shape[1]) > 1 and int(logits.shape[1]) > 1:
            logits = logits[:, :-1, :]
            target_arr = target_arr[:, 1:]
        log_probs: MlxArrayProtocol = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        indices: MlxArrayProtocol = mx.expand_dims(target_arr, axis=-1)
        gathered: MlxArrayProtocol = mx.take_along_axis(log_probs, indices, axis=-1)
        squeezed: MlxArrayProtocol = mx.squeeze(gathered, axis=-1)
        mask: MlxArrayProtocol = mx.not_equal(target_arr, 0)
        masked = squeezed * mask
        res: MlxArrayProtocol = masked.sum(axis=-1)
        return res

    if isinstance(raw_logits, MlxArrayProtocol):
        return raw_logits.mean(axis=-1)
    if isinstance(raw_logits, ArithmeticTensorProtocol):
        return raw_logits
    if isinstance(raw_logits, SupportsFloat):
        return float(raw_logits)
    return 0.0


def _loss_fn(
    policy_model: MlxDPOModel,
    ref_model: MlxDPOModel,
    batch: MlxDPOBatch,
    beta: float,
) -> MlxArrayProtocol | ArithmeticTensorProtocol | float:
    """Evaluate scalar DPO loss for `mlx.nn.value_and_grad` differentiation.

    Args:
        policy_model: Trainable MLX policy model.
        ref_model: Frozen MLX reference model.
        batch: Preference batch mapping containing chosen and rejected inputs/labels.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO step loss.

    """
    ch_inputs = batch.get("chosen_inputs", batch.get("inputs"))
    re_inputs = batch.get("rejected_inputs", batch.get("inputs"))
    ch_labels = batch.get("chosen_labels", batch.get("targets"))
    re_labels = batch.get("rejected_labels", batch.get("targets"))
    pi_ch_logps = _compute_mlx_logps(policy_model, ch_inputs, ch_labels)
    pi_re_logps = _compute_mlx_logps(policy_model, re_inputs, re_labels)
    ref_ch_logps = _compute_mlx_logps(ref_model, ch_inputs, ch_labels)
    ref_re_logps = _compute_mlx_logps(ref_model, re_inputs, re_labels)
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    return loss


def _run_dpo_step[ModelT](
    policy_model: ModelT,
    ref_model: ModelT,
    optimizer: _MlxOptimizerProtocol[ModelT] | str | None,
    batch: MlxDPOBatch,
    beta: float,
) -> MlxArrayProtocol | ArithmeticTensorProtocol | float:
    """Run a single MLX DPO training step using functional `nn.value_and_grad`.

    Args:
        policy_model: Trainable MLX policy model.
        ref_model: Frozen MLX reference model.
        optimizer: MLX optimizer instance supporting `update(model, grads)`.
        batch: Preference batch dictionary.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO step loss.

    """
    assert isinstance(policy_model, (MlxCausalLMProtocol, _MlxDPOCallableProtocol))
    assert isinstance(ref_model, (MlxCausalLMProtocol, _MlxDPOCallableProtocol))
    if nn is not None:
        loss_and_grad_fn = nn.value_and_grad(policy_model, _loss_fn)
        (loss, grads) = loss_and_grad_fn(policy_model, ref_model, batch, beta)
        if isinstance(optimizer, _MlxOptimizerProtocol) and callable(optimizer.update):
            optimizer.update(policy_model, grads)
        if (
            mx is not None
            and isinstance(policy_model, MlxCausalLMProtocol)
            and isinstance(optimizer, _MlxOptimizerStateProtocol)
        ):
            mx.eval(policy_model.parameters(), optimizer.state, loss)
        return loss
    return _loss_fn(policy_model, ref_model, batch, beta)


class _MlxDPOLoopStateProtocol[ModelT, OptT, BatchT](Protocol):
    """Covariant read-only protocol for `DPOLoopState` in MLX DPO training."""

    @property
    def dataloader(self) -> Iterable[BatchT]:
        """Provide preference batch iterable."""
        ...

    @property
    def epochs(self) -> int:
        """Provide epoch count."""
        ...

    @property
    def policy_model(self) -> ModelT:
        """Provide policy model."""
        ...

    @property
    def ref_model(self) -> ModelT:
        """Provide reference model."""
        ...

    @property
    def optimizer(self) -> OptT:
        """Provide optimizer instance or identifier."""
        ...

    @property
    def beta(self) -> float:
        """Provide KL penalty coefficient."""
        ...


def _run_training_epochs[
    ModelT,
    BatchT: MlxDPOBatch,
](
    state: _MlxDPOLoopStateProtocol[ModelT, _MlxOptimizerProtocol[ModelT] | str | None, BatchT],
) -> float:
    """Run MLX DPO training epochs over the preference dataloader.

    Args:
        state: `DPOLoopState` holding models, optimizer, dataloader, and beta.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    beta = state.beta

    def _step(batch: BatchT) -> float:
        """Process a single preference batch and return scalar float loss.

        Args:
            batch: Preference batch mapping.

        Returns:
            Scalar float DPO loss for the batch.

        """
        loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)
        if isinstance(loss, MlxArrayProtocol):
            raw_item = loss.item()
            return float(raw_item.real) if isinstance(raw_item, complex) else float(raw_item)
        try:
            return float(loss)
        except (TypeError, ValueError):
            return 0.0

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _execute_dpo(
    model_name: str,
    dataset: str,
    beta: float,
    epochs: int,
    learning_rate: float,
    batch_size: int = 2,
) -> tuple[str, float]:
    """Execute the core DPO loop for MLX.

    Args:
        model_name: Target model name.
        dataset: Dataset name.
        beta: DPO beta temperature.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.

    Returns:
        Tuple of `(status, final_loss)`.

    Raises:
        ValueError: If dataloader is invalid.

    """
    assert load is not None
    assert optim is not None
    loaded_p = load(model_name)
    policy_model: MlxCausalLMProtocol = loaded_p[0] if isinstance(loaded_p, (tuple, list)) else loaded_p
    loaded_r = load(model_name)
    ref_model: MlxCausalLMProtocol = loaded_r[0] if isinstance(loaded_r, (tuple, list)) else loaded_r
    optimizer: _MlxOptimizerProtocol[MlxCausalLMProtocol] = optim.AdamW(learning_rate=learning_rate)
    dataloader, _ = resolve_training_dataloader(
        dataset,
        lambda: build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size)),
        lambda: [
            {
                "chosen_inputs": [[1, 2, 3]],
                "chosen_labels": [[2, 3, 4]],
                "rejected_inputs": [[1, 5, 6]],
                "rejected_labels": [[5, 6, 7]],
            }
        ],
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
    return "completed", float(final_loss)


def run_dpo(config: DPOConfig, **kwargs: StatusValue) -> DPOResultDict:
    """Execute Direct Preference Optimization (DPO) using the MLX backend.

    Args:
        config: DPO configuration parameters.
        **kwargs: Hyperparameters for DPO (e.g., `beta`, `learning_rate`, `raise_on_error`).

    Returns:
        Structured `DPOResultDict` containing backend, model, dataset, beta, status, and final_loss.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    if mlx is None or mx is None or nn is None or optim is None or load is None:
        raise DependencyMissingError("MLX dependencies are missing.", backend="mlx")
    return run_dpo_wrapper(
        "mlx",
        config,
        lambda m, d, b, ep, lr, bs: _execute_dpo(m, d, b, ep, lr, batch_size=bs),
        **kwargs,
    )


__all__ = [
    "MlxArrayProtocol",
    "MlxCausalLMProtocol",
    "MlxCoreProtocol",
    "MlxNNProtocol",
    "dpo_loss",
    "run_dpo",
]

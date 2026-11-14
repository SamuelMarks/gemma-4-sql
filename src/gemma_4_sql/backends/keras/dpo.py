"""Keras-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, Self, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_dpo import ArithmeticTensorProtocol, generic_dpo_loss, run_dpo_wrapper
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.keras._env import resolve_and_validate_keras_backend
from gemma_4_sql.backends.keras._runtime import (
    KerasWithDistributionProtocol,
    enter_keras_distribution_scope,
    load_keras_gemma_model,
    load_preset_with_hf_fallback,
    synchronize_keras_outputs,
)
from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol, import_optional_symbol, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    DPOConfig,
    DPOResultDict,
    DTypeProtocol,
    ETLConfig,
    ForwardOutput,
    KerasCausalLMProtocol,
    KerasTrainerState,
    KerasVariableProtocol,
    TfTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier.

        Args:
            model_name: Preset identifier string.

        Returns:
            Loaded `KerasCausalLMProtocol`.

        """
        ...


@runtime_checkable
class _KerasOptimizerProtocol[TensorT, VarT](Protocol):
    """Protocol for a Keras optimizer supporting `.apply_gradients(...)`."""

    def apply_gradients(
        self,
        grads_and_vars: Iterable[tuple[TensorT, VarT]],
        /,
    ) -> None:
        """Apply gradients to variables.

        Args:
            grads_and_vars: Iterable of `(gradient, variable)` pairs.

        """
        ...


type KerasDPOVar = KerasVariableProtocol | TfTensorProtocol | ArithmeticTensorProtocol | SupportsFloat | float | str


@runtime_checkable
class _TrainableVarsModelProtocol(Protocol):
    """Protocol for a Keras model exposing `.trainable_variables`."""

    @property
    def trainable_variables(self) -> Sequence[KerasDPOVar]:
        """Provide the model's trainable variables.

        Returns:
            Sequence of trainable variables.

        """
        ...


@runtime_checkable
class _TFMathNamespaceProtocol(Protocol):
    """Protocol for `tf.math` exposing `log_sigmoid` and `multiply`."""

    def log_sigmoid[TensorT: ArithmeticTensorProtocol | float](self, x: TensorT, /) -> TensorT:
        """Compute elementwise `log(sigmoid(x))`.

        Args:
            x: Logits margin tensor or float.

        Returns:
            Log-sigmoid output.

        """
        ...

    def multiply[TensorT: ArithmeticTensorProtocol | float](self, x: TensorT, y: TensorT, /) -> TensorT:
        """Compute elementwise product `x * y`.

        Args:
            x: Left-hand tensor operand.
            y: Right-hand tensor operand.

        Returns:
            Elementwise product tensor.

        """
        ...


@runtime_checkable
class _TFDTypesNamespaceProtocol(Protocol):
    """Protocol for `tf.dtypes` exposing `cast`."""

    def cast[TensorT](self, x: TensorT | bool, dtype: DTypeProtocol, /) -> TensorT:
        """Cast tensor `x` to `dtype`.

        Args:
            x: Input tensor or boolean mask.
            dtype: Target TensorFlow dtype.

        Returns:
            Cast tensor.

        """
        ...


@runtime_checkable
class _TFNNNamespaceProtocol(Protocol):
    """Protocol for `tf.nn` exposing `log_softmax`."""

    def log_softmax[TensorT](self, logits: TensorT, *, axis: int = -1) -> TensorT:
        """Compute log-softmax along `axis`.

        Args:
            logits: Unnormalized logits tensor.
            axis: Vocabulary dimension index.

        Returns:
            Log-probability tensor.

        """
        ...


@runtime_checkable
class _TFGradientTapeProtocol(Protocol):
    """Protocol for `tf.GradientTape` context manager and `.gradient(...)`."""

    def __enter__(self) -> Self:
        """Enter gradient recording context.

        Returns:
            Active tape instance.

        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
        /,
    ) -> bool | None:
        """Exit gradient recording context.

        Args:
            exc_type: Exception type if raised, else `None`.
            exc_val: Exception instance if raised, else `None`.
            exc_tb: Traceback if raised, else `None`.

        Returns:
            Suppression flag or `None`.

        """
        ...

    def gradient[TensorT, VarT](
        self,
        target: TensorT | ArithmeticTensorProtocol | float,
        sources: Sequence[VarT],
        /,
    ) -> Sequence[TensorT]:
        """Compute gradients of `target` with respect to `sources`.

        Args:
            target: Differentiable scalar loss.
            sources: Sequence of trainable variables.

        Returns:
            Sequence of gradient tensors.

        """
        ...


@runtime_checkable
class _TFDPOModuleProtocol(Protocol):
    """Protocol for `tensorflow` math, gather, tape, and `tf.function` ops in Keras DPO."""

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `tf.int32` dtype."""
        ...

    @property
    def float32(self) -> DTypeProtocol:
        """Provide `tf.float32` dtype."""
        ...

    @property
    def math(self) -> _TFMathNamespaceProtocol:
        """Provide `tf.math` namespace."""
        ...

    @property
    def dtypes(self) -> _TFDTypesNamespaceProtocol:
        """Provide `tf.dtypes` namespace."""
        ...

    @property
    def nn(self) -> _TFNNNamespaceProtocol:
        """Provide `tf.nn` namespace."""
        ...

    def reduce_mean[TensorT: ArithmeticTensorProtocol | float](self, input_tensor: TensorT, /) -> TensorT:
        """Compute scalar mean across all elements of `input_tensor`.

        Args:
            input_tensor: Per-example DPO loss tensor.

        Returns:
            Mean scalar loss.

        """
        ...

    def expand_dims[TensorT](self, input_tensor: TensorT, *, axis: int) -> TensorT:
        """Insert a length-1 dimension at `axis`.

        Args:
            input_tensor: Input tensor.
            axis: Dimension index to insert.

        Returns:
            Expanded tensor.

        """
        ...

    def gather[TensorT](self, params: TensorT | ForwardOutput, indices: TensorT, *, batch_dims: int = 0) -> TensorT:
        """Gather slices from `params` at `indices`.

        Args:
            params: Source log-probability tensor.
            indices: Expanded target token ID tensor.
            batch_dims: Number of leading batch dimensions.

        Returns:
            Gathered log-probabilities tensor.

        """
        ...

    def squeeze[TensorT](self, input_tensor: TensorT, *, axis: int) -> TensorT:
        """Remove length-1 dimension at `axis`.

        Args:
            input_tensor: Input tensor.
            axis: Singleton dimension index.

        Returns:
            Squeezed tensor.

        """
        ...

    def reduce_sum[TensorT](self, input_tensor: TensorT, *, axis: int) -> TensorT:
        """Sum tensor elements along `axis`.

        Args:
            input_tensor: Masked token log-probabilities tensor.
            axis: Sequence axis index.

        Returns:
            Summed per-sequence log-probabilities tensor.

        """
        ...

    def GradientTape(self) -> _TFGradientTapeProtocol:
        """Construct a `GradientTape` differentiation context.

        Returns:
            New `_TFGradientTapeProtocol` instance.

        """
        ...

    def function[FnT](self, func: FnT, /) -> FnT:
        """Compile `func` into a TensorFlow graph callable.

        Args:
            func: Python step function.

        Returns:
            Compiled graph callable with matching signature.

        """
        ...


@runtime_checkable
class _KerasDPOOptimizersProtocol(Protocol):
    """Protocol for `keras.optimizers` exposing `AdamW`."""

    def AdamW(
        self,
        *,
        learning_rate: float,
    ) -> _KerasOptimizerProtocol[ArithmeticTensorProtocol, KerasDPOVar]:
        """Construct an `AdamW` Keras optimizer.

        Args:
            learning_rate: Optimizer learning rate.

        Returns:
            Configured `_KerasOptimizerProtocol`.

        """
        ...


@runtime_checkable
class _KerasDPOModuleProtocol(Protocol):
    """Protocol for `keras` module exposing `.optimizers`."""

    @property
    def optimizers(self) -> _KerasDPOOptimizersProtocol:
        """Provide `keras.optimizers` namespace."""
        ...


@runtime_checkable
class _KerasDPOOpsProtocol(Protocol):
    """Protocol for portable `keras.ops` functions used in multi-backend DPO."""

    def cast[TensorT](self, x: TensorT | ForwardOutput | bool, dtype: str) -> TensorT:
        """Cast `x` to `dtype` (`'float32'` or `'int32'`)."""
        ...

    def log_softmax[TensorT](self, x: TensorT, *, axis: int = -1) -> TensorT:
        """Compute numerically stable log-softmax along `axis`."""
        ...

    def expand_dims[TensorT](self, x: TensorT, *, axis: int = -1) -> TensorT:
        """Insert singleton dimension at `axis`."""
        ...

    def take_along_axis[TensorT](self, x: TensorT, indices: TensorT, *, axis: int = -1) -> TensorT:
        """Gather token log-probabilities along `axis`."""
        ...

    def squeeze[TensorT](self, x: TensorT, *, axis: int = -1) -> TensorT:
        """Squeeze singleton dimension at `axis`."""
        ...

    def multiply[TensorT](self, x1: TensorT, x2: TensorT) -> TensorT:
        """Elementwise multiply `x1` and `x2`."""
        ...

    def sum[TensorT](self, x: TensorT, *, axis: int = -1) -> TensorT:
        """Sum tensor elements along `axis`."""
        ...

    def mean[TensorT](self, x: TensorT) -> TensorT:
        """Compute scalar arithmetic mean of `x`."""
        ...

    def softplus[TensorT](self, x: TensorT) -> TensorT:
        """Compute `log(1 + exp(x))` elementwise."""
        ...


@runtime_checkable
class _KerasWithDPOOpsProtocol(Protocol):
    """Protocol for `keras` module exposing `.ops`."""

    @property
    def ops(self) -> _KerasDPOOpsProtocol:
        """Provide `keras.ops` namespace."""
        ...


@runtime_checkable
class _KerasValueVarProtocol(Protocol):
    """Protocol for a Keras 3 `Variable` exposing `.value`."""

    @property
    def value(self) -> ArithmeticTensorProtocol:
        """Provide underlying backend tensor value."""
        ...


@runtime_checkable
class _StatelessCallModelProtocol(Protocol):
    """Protocol for Keras 3 models supporting `stateless_call` for JAX functional autograd."""

    @property
    def non_trainable_variables(self) -> Sequence[_KerasValueVarProtocol]:
        """Provide non-trainable Keras variables."""
        ...

    def stateless_call[InputT: ArithmeticTensorProtocol](
        self,
        trainable_variables: Sequence[ArithmeticTensorProtocol],
        non_trainable_variables: Sequence[ArithmeticTensorProtocol],
        inputs: InputT,
        /,
    ) -> tuple[InputT | ForwardOutput, Sequence[ArithmeticTensorProtocol]]:
        """Run stateless forward pass returning `(outputs, updated_non_trainable_vars)`."""
        ...


@runtime_checkable
class _JaxValueAndGradProtocol(Protocol):
    """Protocol for `jax` exposing `value_and_grad` and `jit`."""

    def value_and_grad[ArgT, BatchT, LossT](
        self,
        fun: Callable[[ArgT, BatchT], LossT],
        argnums: int = 0,
    ) -> Callable[[ArgT, BatchT], tuple[LossT, ArgT]]:
        """Differentiate `fun` returning `(value, grad)`."""
        ...

    def jit[FnT](self, fun: FnT, /) -> FnT:
        """Compile `fun` with XLA JIT."""
        ...


@runtime_checkable
class _TorchGradTensorProtocol(Protocol):
    """Protocol for a PyTorch tensor exposing `.grad`."""

    @property
    def grad(self) -> ArithmeticTensorProtocol | None:
        """Provide accumulated gradient tensor or `None`."""
        ...

    @grad.setter
    def grad(self, val: ArithmeticTensorProtocol | None) -> None:
        """Assign or clear `.grad`."""
        ...


@runtime_checkable
class _TorchBackwardScalarProtocol(Protocol):
    """Protocol for a scalar PyTorch loss supporting `.backward()`."""

    def backward(self) -> None:
        """Backpropagate gradients to leaf tensors."""
        ...


tf: _TFDPOModuleProtocol | ModuleNamespaceProtocol | None = optional_import("tensorflow", _TFDPOModuleProtocol)
keras: _KerasDPOModuleProtocol | None = optional_import("keras", _KerasDPOModuleProtocol)
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = import_optional_symbol(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)


def dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float = 0.1,
    *,
    active_backend: str = "tensorflow",
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
    """Compute Direct Preference Optimization (DPO) loss using `keras.ops` or TensorFlow.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    Raises:
        DependencyMissingError: If neither TensorFlow nor `keras.ops` is available.

    """
    if active_backend in {"jax", "torch"} and isinstance(keras, _KerasWithDPOOpsProtocol):
        k_ops = keras.ops

        def _keras_log_sigmoid(
            x: ArithmeticTensorProtocol | float,
        ) -> ArithmeticTensorProtocol | float:
            """Compute `log(sigmoid(x)) = -softplus(-x)` via `keras.ops`.

            Args:
                x: Input margin tensor or float.

            Returns:
                Log-sigmoid tensor or float.

            """
            return -k_ops.softplus(-x)

        (losses_k, ch_rew_k, re_rew_k) = generic_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            _keras_log_sigmoid,
        )
        mean_k: TensorT | ArithmeticTensorProtocol | float = k_ops.mean(losses_k)
        return (mean_k, ch_rew_k, re_rew_k)

    if tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")
    assert isinstance(tf, _TFDPOModuleProtocol)
    (losses, chosen_rewards, rejected_rewards) = generic_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        tf.math.log_sigmoid,
    )
    mean_loss: TensorT | ArithmeticTensorProtocol | float = tf.reduce_mean(losses)
    return (mean_loss, chosen_rewards, rejected_rewards)


@runtime_checkable
class _KerasDPOCallableProtocol[TensorT](Protocol):
    """Protocol for any callable Keras DPO model."""

    def __call__(self, inputs: TensorT, /) -> TensorT | ForwardOutput:
        """Evaluate forward logits on `inputs`.

        Args:
            inputs: Input token ID tensor.

        Returns:
            Output vocabulary logits tensor.

        """
        ...


type KerasDPOModel[TensorT] = _KerasDPOCallableProtocol[TensorT] | Callable[[TensorT], TensorT | ForwardOutput]


def _compute_logps[TensorT: ArithmeticTensorProtocol | float](
    model: KerasDPOModel[TensorT],
    inputs: TensorT,
    labels: TensorT,
    *,
    active_backend: str = "tensorflow",
) -> TensorT:
    """Compute sequence log probabilities for DPO using `keras.ops` (JAX/PyTorch) or TensorFlow.

    Args:
        model: Keras causal LM callable.
        inputs: Input token ID tensor of shape `(batch_size, seq_len)`.
        labels: Target token ID tensor of shape `(batch_size, seq_len)`.
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).

    Returns:
        Summed non-padding token log probabilities per sequence.

    Raises:
        DependencyMissingError: If neither TensorFlow nor `keras.ops` is available.

    """
    if active_backend in {"jax", "torch"} and isinstance(keras, _KerasWithDPOOpsProtocol):
        k_ops = keras.ops
        raw_out = model(inputs)
        assert isinstance(raw_out, (TfTensorProtocol, ArithmeticTensorProtocol))
        logits_k: TensorT = k_ops.cast(raw_out, "float32")
        log_probs_k: TensorT = k_ops.log_softmax(logits_k, axis=-1)
        idx_k: TensorT = k_ops.expand_dims(k_ops.cast(labels, "int32"), axis=-1)
        gathered_k: TensorT = k_ops.squeeze(k_ops.take_along_axis(log_probs_k, idx_k, axis=-1), axis=-1)
        ne_mask_k: TensorT | bool = labels != 0
        mask_k: TensorT = k_ops.cast(ne_mask_k, "float32")
        return k_ops.sum(k_ops.multiply(gathered_k, mask_k), axis=-1)

    if tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")
    assert isinstance(tf, _TFDPOModuleProtocol)
    tf_cast = tf.dtypes.cast
    raw_logits = model(inputs)
    assert isinstance(raw_logits, (TfTensorProtocol, ArithmeticTensorProtocol))
    logits = tf_cast(raw_logits, tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels_expanded: TensorT = tf.expand_dims(tf_cast(labels, tf.int32), axis=-1)
    selected_log_probs: TensorT = tf.gather(log_probs, labels_expanded, batch_dims=2)
    selected_log_probs = tf.squeeze(selected_log_probs, axis=-1)
    ne_mask: TensorT | bool = labels != 0
    mask: TensorT = tf_cast(ne_mask, tf.float32)
    res: TensorT = tf.reduce_sum(tf.math.multiply(selected_log_probs, mask), axis=-1)
    return res


def _build_jax_dpo_step_fn[TensorT: ArithmeticTensorProtocol](
    policy_model: _TrainableVarsModelProtocol | KerasDPOModel[TensorT],
    ref_model: KerasDPOModel[TensorT],
    optimizer: _KerasOptimizerProtocol[ArithmeticTensorProtocol, KerasDPOVar],
    beta: float,
    *,
    device: str = "cpu",
) -> Callable[[Mapping[str, TensorT]], TensorT | ArithmeticTensorProtocol | float]:
    """Build a JAX stateless `value_and_grad` DPO train step for `keras_jax` without TensorFlow.

    Args:
        policy_model: Trainable Keras 3 policy model supporting `stateless_call`.
        ref_model: Frozen reference Keras 3 model.
        optimizer: Keras 3 optimizer instance.
        beta: DPO KL penalty coefficient.
        device: Resolved JAX device string (`'tpu:0'`, `'gpu:0'`, `'cpu'`).

    Returns:
        Callable step function executing stateless JAX DPO optimization.

    Raises:
        DependencyMissingError: If `jax` is not installed.

    """
    jax_mod = optional_import("jax", _JaxValueAndGradProtocol)
    if jax_mod is None:
        raise DependencyMissingError("JAX dependencies are missing for keras_jax DPO.", backend="keras_jax")
    assert isinstance(policy_model, _StatelessCallModelProtocol)
    stateless_policy: _StatelessCallModelProtocol = policy_model

    def _loss_for_vars(
        tv_values: list[ArithmeticTensorProtocol],
        batch: Mapping[str, TensorT],
    ) -> ArithmeticTensorProtocol | float:
        """Compute scalar DPO loss functionally with respect to `tv_values`.

        Args:
            tv_values: Differentiable trainable variable arrays.
            batch: Preference batch mapping.

        Returns:
            Scalar DPO loss value.

        """
        ntv_values: list[ArithmeticTensorProtocol] = [v.value for v in stateless_policy.non_trainable_variables]

        def _call_stateless(inp: TensorT, /) -> TensorT | ForwardOutput:
            """Evaluate `stateless_policy.stateless_call` returning forward logits.

            Args:
                inp: Input token ID array.

            Returns:
                Forward logits array.

            """
            out, _ = stateless_policy.stateless_call(tv_values, ntv_values, inp)
            return out

        pi_ch = _compute_logps(_call_stateless, batch["chosen_inputs"], batch["chosen_labels"], active_backend="jax")
        pi_re = _compute_logps(
            _call_stateless, batch["rejected_inputs"], batch["rejected_labels"], active_backend="jax"
        )
        ref_ch = _compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"], active_backend="jax")
        ref_re = _compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"], active_backend="jax")
        (loss_val, _, _) = dpo_loss(pi_ch, pi_re, ref_ch, ref_re, beta, active_backend="jax")
        return loss_val

    vg_fn = jax_mod.jit(jax_mod.value_and_grad(_loss_for_vars, 0))

    def _jax_step(batch: Mapping[str, TensorT]) -> TensorT | ArithmeticTensorProtocol | float:
        """Execute one JIT-compiled stateless JAX DPO step and apply gradients.

        Args:
            batch: Preference batch mapping.

        Returns:
            Scalar step loss value.

        """
        t_vars_raw: Sequence[KerasDPOVar] = (
            policy_model.trainable_variables if isinstance(policy_model, _TrainableVarsModelProtocol) else ()
        )
        tv_vals: list[ArithmeticTensorProtocol] = [v.value for v in t_vars_raw if isinstance(v, _KerasValueVarProtocol)]
        loss_out, grads = vg_fn(tv_vals, batch)
        optimizer.apply_gradients(list(zip(grads, t_vars_raw, strict=False)))
        synchronize_keras_outputs("jax", loss_out, device=device)
        return loss_out

    return _jax_step


def _build_torch_dpo_step_fn[TensorT: ArithmeticTensorProtocol](
    policy_model: _TrainableVarsModelProtocol | KerasDPOModel[TensorT],
    ref_model: KerasDPOModel[TensorT],
    optimizer: _KerasOptimizerProtocol[ArithmeticTensorProtocol, KerasDPOVar],
    beta: float,
    *,
    device: str = "cpu",
) -> Callable[[Mapping[str, TensorT]], TensorT | ArithmeticTensorProtocol | float]:
    """Build a PyTorch / `torch_xla` autograd DPO train step for `keras_pytorch` without TensorFlow.

    Args:
        policy_model: Trainable Keras 3 policy model backed by PyTorch variables.
        ref_model: Frozen reference Keras 3 model.
        optimizer: Keras 3 optimizer instance.
        beta: DPO KL penalty coefficient.
        device: Resolved PyTorch/XLA device string (`'xla:0'`, `'cuda'`, `'cpu'`).

    Returns:
        Callable step function executing PyTorch/XLA DPO optimization.

    """

    def _torch_step(batch: Mapping[str, TensorT]) -> TensorT | ArithmeticTensorProtocol | float:
        """Zero variable `.grad` buffers, backpropagate DPO loss, and synchronize XLA.

        Args:
            batch: Preference batch mapping.

        Returns:
            Scalar step loss value.

        """
        trainable_vars: Sequence[KerasDPOVar] = (
            policy_model.trainable_variables if isinstance(policy_model, _TrainableVarsModelProtocol) else ()
        )
        for var in trainable_vars:
            if isinstance(var, _KerasValueVarProtocol) and isinstance(var.value, _TorchGradTensorProtocol):
                var.value.grad = None
        assert isinstance(policy_model, _KerasDPOCallableProtocol)
        pi_ch = _compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"], active_backend="torch")
        pi_re = _compute_logps(policy_model, batch["rejected_inputs"], batch["rejected_labels"], active_backend="torch")
        ref_ch = _compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"], active_backend="torch")
        ref_re = _compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"], active_backend="torch")
        (loss_val, _, _) = dpo_loss(pi_ch, pi_re, ref_ch, ref_re, beta, active_backend="torch")
        if isinstance(loss_val, _TorchBackwardScalarProtocol):
            loss_val.backward()
        grad_pairs: list[tuple[ArithmeticTensorProtocol, KerasDPOVar]] = []
        for var in trainable_vars:
            if isinstance(var, _KerasValueVarProtocol) and isinstance(var.value, _TorchGradTensorProtocol):
                g = var.value.grad
                if g is not None:
                    grad_pairs.append((g, var))
        if grad_pairs:
            optimizer.apply_gradients(grad_pairs)
        synchronize_keras_outputs("torch", loss_val, device=device)
        return loss_val

    return _torch_step


def _get_train_step_fn[TensorT: ArithmeticTensorProtocol](
    policy_model: _TrainableVarsModelProtocol | KerasDPOModel[TensorT],
    ref_model: KerasDPOModel[TensorT],
    optimizer: _KerasOptimizerProtocol[TensorT, KerasDPOVar],
    beta: float,
    *,
    active_backend: str = "tensorflow",
    device: str = "cpu",
    arith_optimizer: _KerasOptimizerProtocol[ArithmeticTensorProtocol, KerasDPOVar] | None = None,
) -> Callable[[Mapping[str, TensorT]], TensorT | ArithmeticTensorProtocol | float]:
    """Return a backend-native DPO train step function across JAX, TensorFlow, or PyTorch.

    Args:
        policy_model: Trainable policy model.
        ref_model: Frozen reference model.
        optimizer: Keras optimizer instance.
        beta: KL penalty temperature coefficient.
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).
        device: Resolved accelerator device string.
        arith_optimizer: Optional arithmetic-typed optimizer handle used for JAX/PyTorch steps.

    Returns:
        Compiled or backend-native DPO step function.

    Raises:
        DependencyMissingError: If required backend runtime (`jax` or `tensorflow`) is missing.

    """
    if active_backend == "jax":
        assert arith_optimizer is not None
        return _build_jax_dpo_step_fn(policy_model, ref_model, arith_optimizer, beta, device=device)
    if active_backend == "torch":
        assert arith_optimizer is not None
        return _build_torch_dpo_step_fn(policy_model, ref_model, arith_optimizer, beta, device=device)

    if tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")
    assert isinstance(tf, _TFDPOModuleProtocol)

    def train_step(
        batch: Mapping[str, TensorT],
    ) -> TensorT | ArithmeticTensorProtocol | float:
        """Compute DPO loss and apply gradients under a TensorFlow GradientTape.

        Args:
            batch: Preference batch dictionary.

        Returns:
            Scalar step loss tensor.

        """
        assert isinstance(tf, _TFDPOModuleProtocol)
        trainable_vars: Sequence[KerasDPOVar] = (
            policy_model.trainable_variables if isinstance(policy_model, _TrainableVarsModelProtocol) else ()
        )
        assert isinstance(policy_model, _KerasDPOCallableProtocol)
        with tf.GradientTape() as tape:
            pi_ch_logps = _compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])
            pi_re_logps = _compute_logps(policy_model, batch["rejected_inputs"], batch["rejected_labels"])
            ref_ch_logps = _compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"])
            ref_re_logps = _compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"])
            (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
        grads: Sequence[TensorT] = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars, strict=False))
        return loss

    compiled_fn: Callable[[Mapping[str, TensorT]], TensorT | ArithmeticTensorProtocol | float] = tf.function(train_step)
    return compiled_fn


def _run_training_epochs[ModelT, OptT, BatchT](
    state: KerasTrainerState[ModelT, OptT, BatchT],
) -> float:
    """Run Keras DPO training epochs over the preference dataloader.

    Args:
        state: `KerasTrainerState` holding dataloader, epoch count, and compiled step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader or ()
    train_step = state.train_step

    def _step(batch: BatchT) -> float:
        """Execute a single Keras DPO step and extract scalar float loss.

        Args:
            batch: Preference batch item.

        Returns:
            Scalar step loss as a Python `float`.

        """
        assert train_step is not None
        loss = train_step(batch)
        if isinstance(loss, (int, float, SupportsFloat)):
            return float(loss)
        return 0.0

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _execute_dpo(
    model_name: str,
    dataset: str,
    beta: float,
    epochs: int,
    learning_rate: float,
    batch_size: int = 2,
    *,
    active_backend: str = "tensorflow",
    hardware: str = "cpu",
    strict_hardware: bool = False,
) -> tuple[str, float]:
    """Execute the core Keras 3 DPO loop across JAX, TensorFlow, or PyTorch backends.

    Args:
        model_name: Target model name.
        dataset: Target dataset name.
        beta: DPO beta temperature.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        batch_size: Training batch size.
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).
        hardware: Target hardware identifier (`'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).
        strict_hardware: Whether to fail fast when requested TPU/GPU hardware is unavailable.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        ImportError: Internally raised and wrapped if KerasHub/KerasNLP `GemmaCausalLM` is unavailable.
        ValueError: If model loading or dataloader fails.

    """
    if keras is None or (active_backend == "tensorflow" and tf is None):
        raise DependencyMissingError("Keras dependencies are missing.")

    tf_mod_ns: ModuleNamespaceProtocol | None = tf if isinstance(tf, ModuleNamespaceProtocol) else None
    dist_keras = keras if isinstance(keras, KerasWithDistributionProtocol) else None

    with enter_keras_distribution_scope(
        active_backend,
        hardware,
        tf_mod=tf_mod_ns,
        keras_mod=dist_keras,
        strict_hardware=strict_hardware,
        backend_label=f"keras_{active_backend}",
    ) as (resolved_dev, _):
        try:
            gemma_causal_lm_cls = load_keras_gemma_model(
                model_name,
                _GemmaCausalLMLoaderProtocol,
                explicit_cls=GemmaCausalLM,
                symbol_importer=import_optional_symbol,
            )
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_hub / keras_nlp GemmaCausalLM is not available.")
            policy_model: KerasCausalLMProtocol = load_preset_with_hf_fallback(gemma_causal_lm_cls, model_name)
            ref_model: KerasCausalLMProtocol = load_preset_with_hf_fallback(gemma_causal_lm_cls, model_name)
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load Keras model {model_name}") from e

        optimizer: _KerasOptimizerProtocol[ArithmeticTensorProtocol, KerasDPOVar] = keras.optimizers.AdamW(
            learning_rate=learning_rate
        )
        if active_backend == "jax":
            train_step = _build_jax_dpo_step_fn(
                policy_model,
                ref_model,
                optimizer,
                beta,
                device=resolved_dev,
            )
        elif active_backend == "torch":
            train_step = _build_torch_dpo_step_fn(
                policy_model,
                ref_model,
                optimizer,
                beta,
                device=resolved_dev,
            )
        else:
            train_step = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
        data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
        dataloader = data_dict.get("loader", None)
        has_loader = isinstance(dataloader, Iterable)

        if not has_loader:
            raise ValueError(f"Invalid dataloader for dataset: {dataset}")

        final_loss = _run_training_epochs(
            KerasTrainerState(
                dataloader=dataloader,
                model=policy_model,
                optimizer=optimizer,
                epochs=epochs,
                train_step=train_step,
            )
        )
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: StatusValue) -> DPOResultDict:
    """Run a DPO training loop for Keras 3 (`keras_jax`, `keras_tensorflow`, or `keras_pytorch`).

    Args:
        config: DPO configuration object.
        **kwargs: Hyperparameters and runtime options (`backend_alias`, `hardware`, `strict_hardware`).

    Returns:
        Structured `DPOResultDict` containing execution status and loss.

    Raises:
        DependencyMissingError: If Keras DPO dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    extra = config.extra_kwargs
    backend_alias = str(kwargs.get("backend_alias", extra.get("backend_alias", "keras")))
    hardware = str(kwargs.get("hardware", extra.get("hardware", "cpu")))
    strict_hw = bool(kwargs.get("strict_hardware", extra.get("strict_hardware", False)))
    raw_kb = kwargs.get("keras_backend", extra.get("keras_backend"))
    explicit_kb = str(raw_kb) if isinstance(raw_kb, str) and raw_kb else None

    if backend_alias == "keras" and explicit_kb is None and tf is not None:
        active_kb = "tensorflow"
    elif backend_alias == "keras" and explicit_kb is None and tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")
    else:
        active_kb = resolve_and_validate_keras_backend(backend_alias, explicit_kb)

    if keras is None or (active_kb == "tensorflow" and tf is None):
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend=backend_alias)

    res = run_dpo_wrapper(
        backend_alias,
        config,
        lambda m, d, b, ep, lr, bs: _execute_dpo(
            m,
            d,
            b,
            ep,
            lr,
            batch_size=bs,
            active_backend=active_kb,
            hardware=hardware,
            strict_hardware=strict_hw,
        ),
        **kwargs,
    )
    res["keras_backend"] = active_kb
    return res

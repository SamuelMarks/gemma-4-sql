"""Multi-backend Keras 3 TPU unit and behavioral tests (`keras_jax`, `keras_tensorflow`, `keras_pytorch`)."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import jax
import pytest
import torch

import gemma_4_sql.backends.keras as keras_pkg
import gemma_4_sql.backends.keras._env as keras_env
import gemma_4_sql.backends.keras._runtime as keras_rt
import gemma_4_sql.backends.keras.benchmark as keras_bm
import gemma_4_sql.backends.keras.dpo as keras_dpo
import gemma_4_sql.backends.keras.export as keras_exp
import gemma_4_sql.backends.keras.inference as keras_inf
import gemma_4_sql.backends.keras.logging as keras_log
import gemma_4_sql.backends.keras.quantize as keras_quant
import gemma_4_sql.backends.keras.serve as keras_serve
import gemma_4_sql.backends.keras.train as keras_train
from gemma_4_sql._types_models import KerasCompiledOptimizerProtocol, KerasPreprocessorProtocol
from gemma_4_sql._types_results import ETLBatchValue
from gemma_4_sql.backends.common_dpo import ArithmeticTensorProtocol
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.keras.adapter import KerasBackend
from gemma_4_sql.backends.keras.dpo import KerasDPOVar
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError, HardwareUnavailableError
from gemma_4_sql.type_hints import (
    DPOConfig,
    ETLConfig,
    ForwardArg,
    ForwardOutput,
    TensorProtocol,
    TensorType,
    TfTensorProtocol,
    TrainingConfig,
)
from tests.conftest import requires_keras
from tests.utils.tensor_helpers import get_typed_keras_ops
from tests.utils.tpu_doubles import (
    FakeJaxTPUDevice,
    FakeKerasDistributionNamespace,
    FakeKerasGemma4CausalLM,
    FakeMultiHostJaxModule,
    FakeTFDistributeNamespace,
    FakeTorchXLACoreModel,
    FakeTorchXLAModule,
    FakeTPUClusterResolver,
)

pytestmark = requires_keras
typed_keras_ops = get_typed_keras_ops()


class _FakeBackendAttr:
    """Typed `keras.backend` double returning a configured backend string."""

    def __init__(self, name: str) -> None:
        """Store `name`."""
        self._name = name

    def backend(self) -> str:
        """Return `self._name`."""
        return self._name


class _JaxBackedKerasOps:
    """Typed `keras.ops` double delegating to `jnp` and `jax.nn` for stateless JAX DPO."""

    def cast(self, x: jax.Array | bool, dtype: str) -> jax.Array:
        """Cast `x` to `dtype` via `jnp.array`.

        Returns:
            Casted `jax.Array`.

        """
        target_dt = jnp.float32 if dtype == "float32" else jnp.int32
        return jnp.array(x, dtype=target_dt)

    def log_softmax(self, x: jax.Array, axis: int = -1) -> jax.Array:
        """Compute `jax.nn.log_softmax(x, axis=axis)`.

        Returns:
            Log-probabilities `jax.Array`.

        """
        return jax.nn.log_softmax(x, axis=axis)

    def expand_dims(self, x: jax.Array, axis: int = -1) -> jax.Array:
        """Expand `axis` via `jnp.expand_dims`.

        Returns:
            Expanded `jax.Array`.

        """
        return jnp.expand_dims(x, axis=axis)

    def take_along_axis(self, arr: jax.Array, indices: jax.Array, axis: int = -1) -> jax.Array:
        """Gather along `axis` via `jnp.take_along_axis`.

        Returns:
            Gathered `jax.Array`.

        """
        return jnp.take_along_axis(arr, indices, axis=axis)

    def squeeze(self, x: jax.Array, axis: int = -1) -> jax.Array:
        """Squeeze `axis` via `jnp.squeeze`.

        Returns:
            Squeezed `jax.Array`.

        """
        return jnp.squeeze(x, axis=axis)

    def multiply(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        """Multiply elementwise.

        Returns:
            Product `jax.Array`.

        """
        return x1 * x2

    def sum(self, x: jax.Array, axis: int | None = None) -> jax.Array:
        """Sum along `axis`.

        Returns:
            Summed `jax.Array`.

        """
        return jnp.sum(x, axis=axis)

    def softplus(self, x: jax.Array) -> jax.Array:
        """Compute `jax.nn.softplus(x)`.

        Returns:
            Softplus `jax.Array`.

        """
        return jax.nn.softplus(x)

    def mean(self, x: jax.Array, axis: int | None = None) -> jax.Array:
        """Mean along `axis`.

        Returns:
            Mean `jax.Array`.

        """
        return jnp.mean(x, axis=axis)

    def ones(self, shape: Sequence[int], dtype: str = "int32") -> jax.Array:
        """Create int32 ones array.

        Returns:
            Ones `jax.Array`.

        """
        _ = dtype
        return jnp.ones(tuple(shape), dtype=jnp.int32)


class _FakeTFExperimentalMem:
    """Typed `tf.config.experimental` memory info double."""

    def __init__(self, *, peak_bytes: int = 256 * 1024 * 1024, should_raise: bool = False) -> None:
        """Configure memory stats or `ValueError`."""
        self.peak_bytes = peak_bytes
        self.should_raise = should_raise

    def get_memory_info(self, dev: str, /) -> Mapping[str, int | float]:
        """Return memory mapping or raise `ValueError`.

        Returns:
            Memory info mapping.

        Raises:
            ValueError: When `self.should_raise` is `True`.

        """
        _ = dev
        if self.should_raise:
            raise ValueError("mem info unsupported")
        return {"peak": self.peak_bytes, "current": self.peak_bytes // 2}


class _FakeTFConfig:
    """Typed `tf.config` double supporting cluster connect, logical/physical devices, and memory info."""

    def __init__(
        self,
        *,
        logical_tpus: Sequence[str] = ("TPU:0", "TPU:1", "TPU:2", "TPU:3"),
        physical_tpus: Sequence[str] = (),
        physical_gpus: Sequence[str] = (),
        peak_bytes: int = 256 * 1024 * 1024,
        mem_raises: bool = False,
    ) -> None:
        """Initialize simulated device lists and experimental memory namespace."""
        self._logical_tpus = tuple(logical_tpus)
        self._physical_tpus = tuple(physical_tpus)
        self._physical_gpus = tuple(physical_gpus)
        self.connected: list[str] = []
        self.experimental = _FakeTFExperimentalMem(peak_bytes=peak_bytes, should_raise=mem_raises)

    def experimental_connect_to_cluster(self, resolver: FakeTPUClusterResolver, /) -> None:
        """Record cluster connection master."""
        self.connected.append(resolver.master())

    def list_logical_devices(self, device_type: str | None = None, /) -> Sequence[str]:
        """Return logical TPU devices when requested.

        Returns:
            Tuple of logical device strings.

        """
        return self._logical_tpus if device_type in {None, "TPU"} else ()

    def list_physical_devices(self, device_type: str | None = None, /) -> Sequence[str]:
        """Return physical TPU or GPU devices.

        Returns:
            Tuple of physical device strings.

        """
        return self._physical_tpus if device_type == "TPU" else self._physical_gpus


class _FakeTFTpuExperimental:
    """Typed `tf.tpu.experimental` double."""

    def __init__(self) -> None:
        """Initialize `init_calls` counter."""
        self.init_calls = 0

    def initialize_tpu_system(self, resolver: FakeTPUClusterResolver, /) -> None:
        """Increment `init_calls`."""
        _ = resolver
        self.init_calls += 1


class _FakeTFTpuNamespace:
    """Typed `tf.tpu` namespace double."""

    def __init__(self) -> None:
        """Attach `experimental`."""
        self.experimental = _FakeTFTpuExperimental()


class _FakeFullTFModule:
    """Typed `tensorflow` module double for `initialize_tf_tpu_cluster` and `TPUStrategy` tests."""

    __name__: str = "tensorflow"

    def __init__(
        self,
        *,
        resolver_raises: bool = False,
        physical_tpus: Sequence[str] = (),
        physical_gpus: Sequence[str] = (),
        peak_bytes: int = 256 * 1024 * 1024,
        mem_raises: bool = False,
    ) -> None:
        """Initialize `distribute`, `config`, and `tpu` namespaces."""
        self.distribute = FakeTFDistributeNamespace(resolver_raises=resolver_raises)
        self.config = _FakeTFConfig(
            physical_tpus=physical_tpus,
            physical_gpus=physical_gpus,
            peak_bytes=peak_bytes,
            mem_raises=mem_raises,
        )
        self.tpu = _FakeTFTpuNamespace()


def test_keras_env_and_adapter_aliases() -> None:
    """Verify `keras/_env.py` conflict checks, unloaded-keras env resolution, and `KerasBackend` aliases."""
    assert keras_pkg.get_trainer() == "keras_trainer"
    for alias in ("keras_jax", "keras_tensorflow", "keras_tf", "keras_pytorch", "keras_torch"):
        inst = KerasBackend.from_alias(alias)
        assert inst.backend_alias == alias
        assert inst.get_trainer() == "keras_trainer"

    # Non-torch branch in ensure_keras_backend_configured
    with patch.dict(os.environ, {"KERAS_BACKEND": "jax"}):
        assert keras_env.ensure_keras_backend_configured("jax") == "jax"

    # Loaded keras mid-process conflict raises ConfigurationError
    class _TorchLoadedKeras:
        """Keras stub reporting `'torch'` backend."""

        backend = _FakeBackendAttr("torch")

    with patch.dict(sys.modules, {"keras": _TorchLoadedKeras()}):
        assert keras_env.resolve_and_validate_keras_backend("keras") == "torch"
        assert keras_env.resolve_and_validate_keras_backend("keras_pytorch") == "torch"
        with pytest.raises(ConfigurationError, match="Keras backend mismatch"):
            keras_env.resolve_and_validate_keras_backend("keras_jax")

    # Unloaded keras in sys.modules sets KERAS_BACKEND or falls back to env
    with patch.dict(sys.modules, {"keras": None}), patch.dict(os.environ, {"KERAS_BACKEND": "tensorflow"}):
        assert keras_env.resolve_and_validate_keras_backend(None, None) == "tensorflow"
        assert keras_env.resolve_and_validate_keras_backend("keras_jax") == "jax"
        assert os.environ["KERAS_BACKEND"] == "jax"


def test_keras_runtime_loader_tpu_cluster_scopes_sync_and_hbm() -> None:
    """Verify all multi-backend helpers in `src/gemma_4_sql/backends/keras/_runtime.py`."""
    keras_rt.reset_tf_tpu_init_state()

    # 1. load_keras_gemma_model & load_preset_with_hf_fallback
    loader = FakeKerasGemma4CausalLM(require_hf_scheme=True)

    def _none_importer(
        _mod: str,
        _sym: str,
        _proto: type[FakeKerasGemma4CausalLM],
    ) -> FakeKerasGemma4CausalLM | None:
        """Return `None` for all symbol candidates.

        Returns:
            `None`.

        """
        return None

    def _hub_importer(
        mod: str,
        sym: str,
        _proto: type[FakeKerasGemma4CausalLM],
    ) -> FakeKerasGemma4CausalLM | None:
        """Return `loader` for `keras_hub.models.Gemma4CausalLM`.

        Returns:
            `loader` or `None`.

        """
        return loader if (mod, sym) == ("keras_hub.models", "Gemma4CausalLM") else None

    assert (
        keras_rt.load_keras_gemma_model(
            "google/gemma-4-E4B",
            FakeKerasGemma4CausalLM,
            symbol_importer=_none_importer,
        )
        is None
    )
    found = keras_rt.load_keras_gemma_model(
        "google/gemma-4-E4B",
        FakeKerasGemma4CausalLM,
        symbol_importer=_hub_importer,
    )
    assert found is loader
    loaded_m = keras_rt.load_preset_with_hf_fallback(loader, "google/gemma-4-E4B")
    assert loaded_m.preset == "hf://google/gemma-4-E4B"
    assert loaded_m.generate("SELECT", max_length=16) == "SELECT SELECT 16;"
    with pytest.raises(ValueError, match="requires hf://"):
        keras_rt.load_preset_with_hf_fallback(loader, "plain_preset_without_slash")

    # 2. initialize_tf_tpu_cluster: None, incomplete ModuleNamespaceProtocol, idempotent success, and error handling
    assert keras_rt.initialize_tf_tpu_cluster(None) == (None, 0)

    class _EmptyNS:
        """Minimal `ModuleNamespaceProtocol` double."""

        __name__: str = "empty_ns"

    class _IncompleteTFNamespace:
        """`ModuleNamespaceProtocol` double whose `distribute`/`config`/`tpu` fail inner checks."""

        __name__: str = "incomplete_tf"
        distribute = _EmptyNS()
        config = _EmptyNS()
        tpu = _EmptyNS()

    assert keras_rt.initialize_tf_tpu_cluster(_IncompleteTFNamespace()) == (None, 0)

    tf_ok = _FakeFullTFModule()
    res1, cnt1 = keras_rt.initialize_tf_tpu_cluster(tf_ok)
    res2, cnt2 = keras_rt.initialize_tf_tpu_cluster(tf_ok)
    assert res1 is not None
    assert res2 is not None
    assert (cnt1, cnt2) == (4, 4)
    assert tf_ok.tpu.experimental.init_calls == 1

    keras_rt.reset_tf_tpu_init_state()
    tf_err = _FakeFullTFModule(resolver_raises=True)
    assert keras_rt.initialize_tf_tpu_cluster(tf_err) == (None, 0)

    # 3. enter_keras_distribution_scope across jax, torch, and tensorflow
    tpu_devs = (
        FakeJaxTPUDevice("TPU_0", peak_bytes=512 * 1024 * 1024),
        FakeJaxTPUDevice("TPU_1", peak_bytes=256 * 1024 * 1024),
    )
    fake_jax = FakeMultiHostJaxModule(tpu_devices=tpu_devs)
    dist_ns = FakeKerasDistributionNamespace()
    mp_strat = dist_ns.ModelParallel(layout_map={"w": "model"})
    assert dist_ns.model_parallel_calls == 1
    with mp_strat.scope():
        pass

    class _KerasWithDist:
        """Typed `KerasWithDistributionProtocol` holder."""

        distribution = dist_ns

    with patch.dict(sys.modules, {"jax": fake_jax}):
        with keras_rt.enter_keras_distribution_scope("jax", "tpu-v6e", keras_mod=_KerasWithDist()) as (dev, cnt):
            assert (dev, cnt) == ("tpu:0", 2)
        assert dist_ns.data_parallel_devices == [("TPU_0", "TPU_1")]

        single_jax = FakeMultiHostJaxModule(tpu_devices=(FakeJaxTPUDevice("TPU_0"),))
        with patch.dict(sys.modules, {"jax": single_jax}):
            with keras_rt.enter_keras_distribution_scope("jax", "tpu", keras_mod=_KerasWithDist()) as (d1, c1):
                assert (d1, c1) == ("tpu:0", 1)
            with keras_rt.enter_keras_distribution_scope("jax", "gpu", strict_hardware=False) as (dg, cg):
                assert (dg, cg) == ("cpu", 1)
            err_jax_gpu = keras_rt.enter_keras_distribution_scope("jax", "gpu", strict_hardware=True)
            with pytest.raises(HardwareUnavailableError, match="Keras JAX backend"):
                contextlib.ExitStack().enter_context(err_jax_gpu)

    # Torch scope on XLA vs CPU
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel()
    with patch.dict(sys.modules, {"torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod}):
        with keras_rt.enter_keras_distribution_scope("torch", "tpu-v6e") as (dt, ct):
            assert (dt, ct) == ("xla:0", 4)
        with keras_rt.enter_keras_distribution_scope("torch", "cpu") as (dc, cc):
            assert (dc, cc) == ("cpu", 1)

    # TensorFlow scopes: TPUStrategy, physical TPU fallback, strict/non-strict TPU fallback, GPU, CPU
    keras_rt.reset_tf_tpu_init_state()
    with keras_rt.enter_keras_distribution_scope("tensorflow", "tpu-v6e", tf_mod=tf_ok) as (dtf, ctf):
        assert (dtf, ctf) == ("/TPU:0", 4)
    assert tf_ok.distribute.last_tpu_strategy is not None
    assert tf_ok.distribute.last_tpu_strategy.scope_ctx.entered == 1

    tf_phys_tpu = _FakeFullTFModule(resolver_raises=True, physical_tpus=("TPU:0", "TPU:1"))
    with keras_rt.enter_keras_distribution_scope("tensorflow", "tpu", tf_mod=tf_phys_tpu) as (dp, cp):
        assert (dp, cp) == ("/TPU:0", 2)

    err_tf_tpu = keras_rt.enter_keras_distribution_scope("tensorflow", "tpu-v5p", tf_mod=tf_err, strict_hardware=True)
    with pytest.raises(HardwareUnavailableError, match="could not initialize TPU"):
        contextlib.ExitStack().enter_context(err_tf_tpu)

    # Non-strict TPU with tf_mod=None covers _list_tf_physical_devices(None) -> () and 564->568
    with keras_rt.enter_keras_distribution_scope("tensorflow", "tpu", tf_mod=None, strict_hardware=False) as (dn, cn):
        assert (dn, cn) == ("/CPU:0", 1)

    tf_gpu = _FakeFullTFModule(physical_gpus=("GPU:0", "GPU:1"))
    with keras_rt.enter_keras_distribution_scope("tensorflow", "gpu", tf_mod=tf_gpu) as (dg1, cg1):
        assert (dg1, cg1) == ("/GPU:0", 2)

    class _GPUOnlyTF:
        """TF stub exposing `config.list_physical_devices('GPU')` without `MirroredStrategy`."""

        config = _FakeTFConfig(physical_gpus=("GPU:0",))

    with keras_rt.enter_keras_distribution_scope("tensorflow", "gpu", tf_mod=_GPUOnlyTF()) as (dg2, cg2):
        assert (dg2, cg2) == ("/GPU:0", 1)

    err_tf_gpu = keras_rt.enter_keras_distribution_scope("tensorflow", "gpu", tf_mod=tf_err, strict_hardware=True)
    with pytest.raises(HardwareUnavailableError, match="zero GPUs"):
        contextlib.ExitStack().enter_context(err_tf_gpu)

    with keras_rt.enter_keras_distribution_scope("tensorflow", "gpu", tf_mod=tf_err, strict_hardware=False) as (
        dcpu1,
        ccpu1,
    ):
        assert (dcpu1, ccpu1) == ("/CPU:0", 1)
    with keras_rt.enter_keras_distribution_scope("tensorflow", "cpu", tf_mod=None) as (dcpu2, ccpu2):
        assert (dcpu2, ccpu2) == ("/CPU:0", 1)

    # 4. synchronize_keras_outputs & query_keras_tpu_hbm_mb
    class _RaisingBlockJax(FakeMultiHostJaxModule):
        """JAX double whose `block_until_ready` raises `TypeError`."""

        def block_until_ready[OutT](self, x: OutT, /) -> OutT:
            """Raise `TypeError` to exercise safe exception handling.

            Raises:
                TypeError: Always raised.

            """
            _ = x
            raise TypeError("non-jax value")

    with patch.dict(sys.modules, {"jax": _RaisingBlockJax(tpu_devices=tpu_devs)}):
        keras_rt.synchronize_keras_outputs("jax", 1.0, device="tpu:0")
        assert keras_rt.query_keras_tpu_hbm_mb("jax", device="tpu:0") == pytest.approx(512.0)
        assert keras_rt.query_keras_tpu_hbm_mb("jax", device="gpu:0") is None

    assert keras_rt._extract_jax_device_peak_mb("raw_str") is None
    assert keras_rt._extract_jax_device_peak_mb(FakeJaxTPUDevice("TPU_0", peak_bytes=None)) is None
    assert keras_rt._extract_jax_device_peak_mb(FakeJaxTPUDevice("TPU_0", peak_bytes=0)) is None

    with patch.dict(sys.modules, {"jax": None}):
        keras_rt.synchronize_keras_outputs("jax", 1.0)
        assert keras_rt.query_keras_tpu_hbm_mb("jax", device="tpu:0") is None

    with patch.dict(sys.modules, {"torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod}):
        keras_rt.synchronize_keras_outputs("torch", 1.0, device="xla:0")
        assert keras_rt.query_keras_tpu_hbm_mb("torch", device="xla:0") == pytest.approx(512.0)
    with patch.dict(sys.modules, {"torch_xla": None, "torch_xla.core.xla_model": None}):
        assert keras_rt.query_keras_tpu_hbm_mb("torch", device="xla:0") is None

    assert keras_rt.query_keras_tpu_hbm_mb("tensorflow", device="/TPU:0", tf_mod=tf_ok) == pytest.approx(256.0)
    assert (
        keras_rt.query_keras_tpu_hbm_mb("tensorflow", device="/TPU:0", tf_mod=_FakeFullTFModule(peak_bytes=0)) is None
    )
    assert (
        keras_rt.query_keras_tpu_hbm_mb("tensorflow", device="/TPU:0", tf_mod=_FakeFullTFModule(mem_raises=True))
        is None
    )
    assert keras_rt.query_keras_tpu_hbm_mb("tensorflow", device="/CPU:0", tf_mod=tf_ok) is None


def test_keras_benchmark_jax_torch_and_tf_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `keras/benchmark.py` across `keras_jax`, `keras_pytorch`, and `keras_tensorflow`."""
    gen_lengths: list[int] = []

    class _BenchKerasModel:
        """Callable and generative Keras model double."""

        def __call__(self, inputs: ForwardOutput) -> ForwardOutput:
            """Return `inputs` unchanged."""
            return inputs

        def generate(self, inputs: ForwardOutput, max_length: int = 40) -> str:
            """Record `max_length` and return SQL string.

            Returns:
                SQL string.

            """
            _ = inputs
            gen_lengths.append(max_length)
            return "SELECT 1;"

    def _load_bench_model(_m: str, _d: str) -> _BenchKerasModel:
        """Return `_BenchKerasModel`.

        Returns:
            `_BenchKerasModel` instance.

        """
        return _BenchKerasModel()

    # 1. keras_jax prefill (with TypeError fallback on jax.jit) and ops.ones / missing-ops branches
    class _OpsOnlyKeras:
        """Keras double exposing `.ops.ones` without `.random`."""

        backend = _FakeBackendAttr("jax")
        ops = _JaxBackedKerasOps()

    class _BareKeras:
        """Keras double exposing neither `.random` nor `.ops`."""

        backend = _FakeBackendAttr("jax")

    tpu_devs = (FakeJaxTPUDevice("TPU_0", peak_bytes=384 * 1024 * 1024),)

    class _TypeErrJitJax(FakeMultiHostJaxModule):
        """JAX double whose `jit` wrapper raises `TypeError` to test eager fallback."""

        def jit(
            self,
            fn: Callable[[TensorType], ForwardOutput | str],
            /,
        ) -> Callable[[TensorType], ForwardOutput | str]:
            """Return wrapper raising `TypeError`.

            Returns:
                Callable raising `TypeError`.

            """
            _ = fn

            def _raise_te(_inp: TensorType) -> ForwardOutput | str:
                """Raise `TypeError`.

                Raises:
                    TypeError: Always raised.

                """
                raise TypeError("stub non-jax output")

            return _raise_te

    with (
        monkeypatch.context() as mp,
        patch.dict(sys.modules, {"keras": _OpsOnlyKeras(), "jax": _TypeErrJitJax(tpu_devices=tpu_devs)}),
    ):
        mp.setattr(keras_bm, "keras", _OpsOnlyKeras())
        mp.setattr(keras_bm, "tf", None)
        res_jax_pf = keras_bm.benchmark_keras_model(
            "gemma-4",
            hardware="tpu-v6e",
            batch_size=2,
            options={"backend_alias": "keras_jax", "mode": "prefill", "warmup_steps": 1, "num_runs": 2},
            model_loader=_load_bench_model,
        )
        assert res_jax_pf["status"] == "success"
        assert res_jax_pf["backend"] == "keras_jax"
        assert res_jax_pf.get("keras_backend") == "jax"
        assert res_jax_pf.get("resolved_device") == "tpu:0"
        assert res_jax_pf.get("peak_hbm_mb") == pytest.approx(384.0)

        # keras_jax with jax=None in sys.modules covers 530->550
        with patch.dict(sys.modules, {"jax": None}):
            res_jax_no_jit = keras_bm.benchmark_keras_model(
                "gemma-4",
                hardware="cpu",
                batch_size=1,
                options={"backend_alias": "keras_jax", "mode": "prefill", "warmup_steps": 1, "num_runs": 1},
                model_loader=_load_bench_model,
            )
            assert res_jax_no_jit["status"] == "success"

        # Bare keras without random or ops raises DependencyMissingError
        mp.setattr(keras_bm, "keras", _BareKeras())
        with pytest.raises(DependencyMissingError, match=r"keras\.random or keras\.ops"):
            keras_bm.benchmark_keras_model(
                "gemma-4",
                hardware="cpu",
                batch_size=1,
                options={"backend_alias": "keras_jax", "raise_on_error": True},
                model_loader=_load_bench_model,
            )

    # 2. Real keras.random.randint + decode mode on keras_pytorch XLA TPU (verifying warmup parity!)
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel(peak_bytes=640 * 1024 * 1024)

    class _TorchBenchKeras:
        """Keras double reporting `'torch'` backend."""

        backend = _FakeBackendAttr("torch")

    with (
        monkeypatch.context() as mp,
        patch.dict(
            sys.modules,
            {"keras": _TorchBenchKeras(), "torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod},
        ),
    ):
        mp.setattr(keras_bm, "tf", None)
        res_pt_dec = keras_bm.benchmark_keras_model(
            "gemma-4",
            hardware="tpu-v6e",
            batch_size=1,
            options={
                "backend_alias": "keras_pytorch",
                "mode": "decode",
                "max_new_tokens": 10,
                "warmup_steps": 1,
                "num_runs": 2,
            },
            model_loader=_load_bench_model,
        )
        assert res_pt_dec["status"] == "success"
        assert res_pt_dec["backend"] == "keras_pytorch"
        assert res_pt_dec.get("resolved_device") == "xla:0"
        assert res_pt_dec.get("peak_hbm_mb") == pytest.approx(640.0)
        assert gen_lengths == [42, 42, 42]

        # Non-callable generate model on decode mode covers False branch of _KerasGenerateModelProtocol
        class _CallOnlyModel:
            """Callable-only model without `.generate`."""

            def __call__(self, x: ForwardOutput) -> ForwardOutput:
                """Return `x`."""
                return x

        def _load_call_only(_m: str, _d: str) -> _CallOnlyModel:
            """Return `_CallOnlyModel`.

            Returns:
                `_CallOnlyModel` instance.

            """
            return _CallOnlyModel()

        assert _CallOnlyModel()(1) == 1
        res_call_only = keras_bm.benchmark_keras_model(
            "gemma-4",
            hardware="cpu",
            batch_size=1,
            options={"backend_alias": "keras_pytorch", "mode": "decode", "warmup_steps": 1, "num_runs": 1},
            model_loader=_load_call_only,
        )
        assert res_call_only["status"] == "success"

        # Unrecorded _run_benchmark_pass fallback branch (line 787)
        def _unrecorded_pass(
            _model: keras_bm.KerasBenchmarkModel,
            _bs: int,
            _runs: int,
            _wu: int,
            _mode: str,
            _mnt: int,
            _hw: str,
            *,
            active_backend: str = "torch",
            strict_hardware: bool = False,
        ) -> tuple[float, float, float]:
            """Return static tuple without setting `_LAST_KERAS_METRICS`.

            Returns:
                Static `(tps, latency_ms, memory_mb)` tuple.

            """
            _ = (active_backend, strict_hardware)
            return (100.0, 5.0, 64.0)

        mp.setattr(keras_bm, "_run_benchmark_pass", _unrecorded_pass)
        res_unrec = keras_bm.benchmark_keras_model(
            "gemma-4",
            hardware="cpu",
            batch_size=1,
            options={"backend_alias": "keras_pytorch"},
            model_loader=_load_bench_model,
        )
        assert res_unrec["tokens_per_sec"] == 100.0
        assert res_unrec.get("keras_backend") == "torch"

    # 3. TensorFlow _get_device_str TPU resolution and strict_hardware CPU fallback error
    keras_rt.reset_tf_tpu_init_state()
    tf_tpu = _FakeFullTFModule()
    with monkeypatch.context() as mp:
        mp.setattr(keras_bm, "tf", tf_tpu)
        assert keras_bm._get_device_str("tpu-v6e") == "/TPU:0"
        assert keras_bm._get_device_str("mps") == "/CPU:0"

        class _TFCpuStub(_FakeFullTFModule):
            """TF stub adding `random.set_seed` for `_run_benchmark_pass` strict hardware check."""

            class random:
                """Random seed holder."""

                @staticmethod
                def set_seed(_s: int) -> None:
                    """No-op seed."""

        keras_rt.reset_tf_tpu_init_state()
        mp.setattr(keras_bm, "tf", _TFCpuStub(resolver_raises=True))
        assert keras_bm._get_device_str("tpu-v6e") == "/CPU:0"
        with pytest.raises(HardwareUnavailableError, match="fell back to '/CPU:0'"):
            keras_bm._run_benchmark_pass(
                _BenchKerasModel(),
                1,
                1,
                1,
                "prefill",
                8,
                "tpu-v6e",
                active_backend="tensorflow",
                strict_hardware=True,
            )


def test_keras_train_dpo_inference_serve_logging_across_all_three_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Keras `train`, `dpo` (JAX, PyTorch, TF engines), `inference`, `serve`, `logging`, `export`, `quantize`."""
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel()

    # 1. keras/train.py on keras_pytorch (XLA TPU per-batch sync + non-dict batch + missing ops error)
    def _default_loss_hist() -> dict[str, list[float]]:
        """Return default loss history dictionary.

        Returns:
            Loss history dictionary.

        """
        return {"loss": [0.25]}

    @dataclass
    class _FakeHistory:
        """Fit history double."""

        history: dict[str, list[float]] = field(default_factory=_default_loss_hist)

    class _TinyPreproc(KerasPreprocessorProtocol):
        """Preprocessor stub."""

        sequence_length: int = 128

        def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> dict[str, TensorType]:
            """Return empty dict."""
            _ = (args, kwargs)
            return {}

    assert _TinyPreproc()() == {}

    class _StrictTrainModel:
        """Strictly typed Keras model double consuming `fit` iterator."""

        preprocessor = _TinyPreproc()

        def compile(
            self,
            *,
            loss: Callable[..., TensorProtocol],
            optimizer: KerasCompiledOptimizerProtocol,
            metrics: list[str],
        ) -> None:
            """Accept optimizer, loss, and metrics."""
            _ = (loss, optimizer, metrics)

        def fit(
            self,
            batches: Iterator[tuple[TfTensorProtocol, TfTensorProtocol] | Mapping[str, ETLBatchValue]],
            epochs: int = 1,
        ) -> _FakeHistory:
            """Consume all batches to trigger XLA sync per batch.

            Returns:
                `_FakeHistory` instance.

            """
            _ = epochs
            consumed = list(batches)
            assert len(consumed) == 2
            return _FakeHistory()

    class _FakeTrainLoader:
        """`GemmaCausalLM` loader double for `keras/train.py`."""

        @staticmethod
        def from_preset(_name: str) -> _StrictTrainModel:
            """Return `_StrictTrainModel`."""
            return _StrictTrainModel()

    def _fake_train_dataloader(_cfg: ETLConfig) -> dict[str, list[dict[str, list[list[int]]]]]:
        """Yield one `inputs`/`targets` dict and one raw dict without `targets`.

        Returns:
            Loader mapping.

        """
        return {
            "loader": [
                {"inputs": [[1, 2, 3]], "targets": [[2, 3, 4]]},
                {"raw_only": [[1, 2, 3]]},
            ]
        }

    assert keras_train.keras is not None
    orig_keras_train_mod = keras_train.keras

    class _TorchBackendHolder:
        """Keras stub reporting `'torch'` backend in `sys.modules`."""

        backend = _FakeBackendAttr("torch")

    with (
        monkeypatch.context() as mp,
        patch.dict(
            sys.modules,
            {"keras": _TorchBackendHolder(), "torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod},
        ),
    ):
        mp.setattr(keras_train, "tf", None)
        mp.setattr(keras_train, "GemmaCausalLM", _FakeTrainLoader)
        mp.setattr(keras_train, "build_dataloader", _fake_train_dataloader)
        tr_res = keras_train.train_model(
            TrainingConfig(
                action="sft",
                model_name="gemma-4",
                dataset="d",
                epochs=1,
                batch_size=1,
                backend="keras_pytorch",
            ),
            backend_alias="keras_pytorch",
            hardware="tpu-v6e",
        )
        assert tr_res["status"] == "completed"
        assert tr_res.get("keras_backend") == "torch"
        assert xla_mod.sync_calls >= 2

        # Missing keras.ops error in _to_int32_tensor when tf is None
        class _KerasNoOps:
            """Keras stub with optimizers and losses but no `.ops`."""

            backend = _FakeBackendAttr("torch")
            optimizers = orig_keras_train_mod.optimizers
            losses = orig_keras_train_mod.losses

        mp.setattr(keras_train, "keras", _KerasNoOps())
        with pytest.raises(DependencyMissingError, match=r"keras\.ops or tensorflow"):
            keras_train._execute_train(
                "gemma-4",
                "d",
                1,
                1,
                1e-4,
                active_backend="torch",
                hardware="cpu",
            )

    # Missing tf for keras_tensorflow train_model
    class _TFKerasNoOps:
        """Keras stub reporting `'tensorflow'` backend."""

        backend = _FakeBackendAttr("tensorflow")

    with monkeypatch.context() as mp, patch.dict(sys.modules, {"keras": _TFKerasNoOps()}):
        mp.setattr(keras_train, "tf", None)
        with pytest.raises(DependencyMissingError, match="Keras training dependencies"):
            keras_train.train_model(
                TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1, backend="keras_tensorflow"),
                backend_alias="keras_tensorflow",
            )

    # 2. keras/dpo.py — Engine A: JAX stateless `jax.jit(jax.value_and_grad(...))` without TensorFlow!
    @dataclass
    class _JaxVar:
        """Keras 3 JAX variable double wrapping a `jax.Array`."""

        value: jax.Array

    class _JaxStatelessDPOModel:
        """Keras 3 model supporting `stateless_call` and forward `__call__` on JAX arrays."""

        def __init__(self) -> None:
            """Initialize trainable embedding/logit projection array `(8, 8)`."""
            self.trainable_variables = (_JaxVar(jnp.ones((8, 8), dtype=jnp.float32) * 0.1),)
            self.non_trainable_variables = (_JaxVar(jnp.zeros((8,), dtype=jnp.float32)),)

        def stateless_call(
            self,
            trainable_values: Sequence[ArithmeticTensorProtocol],
            non_trainable_values: Sequence[ArithmeticTensorProtocol],
            inputs: jax.Array,
            /,
        ) -> tuple[jax.Array, Sequence[ArithmeticTensorProtocol]]:
            """Compute differentiable vocabulary logits `(batch, seq, 8)` from `trainable_values[0]`.

            Returns:
                Tuple of `(logits, non_trainable_values)`.

            """
            w = trainable_values[0]
            assert isinstance(w, jax.Array)
            logits = jnp.take(w, inputs % 8, axis=0)
            return (logits, non_trainable_values)

        def __call__(self, inputs: jax.Array, /) -> jax.Array:
            """Evaluate forward logits using current `trainable_variables`.

            Returns:
                Vocabulary logits `jax.Array`.

            """
            out, _ = self.stateless_call([v.value for v in self.trainable_variables], (), inputs)
            return out

    class _JaxOptimizer:
        """Keras 3 optimizer double recording `apply_gradients` on JAX arrays."""

        def __init__(self, learning_rate: float = 1e-4) -> None:
            """Initialize `applied` counter."""
            _ = learning_rate
            self.applied = 0

        def apply_gradients(
            self,
            grads_and_vars: Iterable[tuple[ArithmeticTensorProtocol, KerasDPOVar]],
            /,
        ) -> None:
            """Increment `self.applied`."""
            assert len(list(grads_and_vars)) == 1
            self.applied += 1

    class _JaxOptimizersNS:
        """Namespace exposing `AdamW`."""

        AdamW = _JaxOptimizer

    class _JaxKerasModule:
        """Full `keras` module double configured for `KERAS_BACKEND='jax'`."""

        backend = _FakeBackendAttr("jax")
        ops = _JaxBackedKerasOps()
        optimizers = _JaxOptimizersNS()

    class _JaxDPOLoader:
        """Preset loader returning `_JaxStatelessDPOModel`."""

        @staticmethod
        def from_preset(_name: str) -> _JaxStatelessDPOModel:
            """Return `_JaxStatelessDPOModel`."""
            return _JaxStatelessDPOModel()

    def _jax_dpo_batches(_cfg: ETLConfig) -> dict[str, list[dict[str, jax.Array]]]:
        """Return JAX preference batch.

        Returns:
            Loader mapping.

        """
        return {
            "loader": [
                {
                    "chosen_inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
                    "chosen_labels": jnp.array([[2, 3, 4]], dtype=jnp.int32),
                    "rejected_inputs": jnp.array([[1, 5, 6]], dtype=jnp.int32),
                    "rejected_labels": jnp.array([[5, 6, 7]], dtype=jnp.int32),
                }
            ]
        }

    jax_keras_mod = _JaxKerasModule()
    with monkeypatch.context() as mp, patch.dict(sys.modules, {"keras": jax_keras_mod}):
        mp.setattr(keras_dpo, "keras", jax_keras_mod)
        mp.setattr(keras_dpo, "tf", None)
        mp.setattr(keras_dpo, "GemmaCausalLM", _JaxDPOLoader)
        mp.setattr(keras_dpo, "build_dataloader", _jax_dpo_batches)

        res_dpo_jax = keras_dpo.run_dpo(
            DPOConfig(model_name="gemma-4", dataset="dpo_ds", epochs=1, batch_size=1),
            backend_alias="keras_jax",
            hardware="cpu",
        )
        assert res_dpo_jax["status"] == "completed"
        assert res_dpo_jax["backend"] == "keras_jax"
        assert res_dpo_jax.get("keras_backend") == "jax"

        # Cover _get_train_step_fn(active_backend="jax") and missing jax error
        jax_opt = _JaxOptimizer()
        step_jax: Callable[[Mapping[str, jax.Array]], jax.Array | ArithmeticTensorProtocol | float] = (
            keras_dpo._get_train_step_fn(
                _JaxStatelessDPOModel(),
                _JaxStatelessDPOModel(),
                jax_opt,
                0.1,
                active_backend="jax",
                arith_optimizer=jax_opt,
            )
        )
        assert callable(step_jax)

        with patch.dict(sys.modules, {"jax": None}), pytest.raises(DependencyMissingError, match="keras_jax DPO"):
            keras_dpo._build_jax_dpo_step_fn(_JaxStatelessDPOModel(), _JaxStatelessDPOModel(), jax_opt, 0.1)

        # Missing both keras.ops and tf in _compute_logps & dpo_loss
        mp.setattr(keras_dpo, "keras", None)
        with pytest.raises(DependencyMissingError, match="Keras DPO dependencies"):
            keras_dpo._compute_logps(
                _JaxStatelessDPOModel(),
                jnp.ones((1, 2), dtype=jnp.int32),
                jnp.ones((1, 2), dtype=jnp.int32),
            )
        with pytest.raises(DependencyMissingError, match="Keras DPO dependencies"):
            keras_dpo.dpo_loss(1.0, 0.5, 1.0, 0.5, 0.1)
        with pytest.raises(DependencyMissingError, match="Keras DPO dependencies"):
            keras_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"), backend_alias="keras")
        with pytest.raises(DependencyMissingError, match="Keras DPO dependencies"):
            keras_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"), backend_alias="keras_jax")

    # 3. keras/dpo.py — Engine B: PyTorch `.backward()` + `.grad` + `torch_xla.sync()` without TensorFlow!
    @dataclass
    class _TorchVar:
        """Keras 3 PyTorch variable double wrapping a leaf `torch.Tensor`."""

        value: torch.Tensor

    class _TorchDPOModel:
        """Keras 3 PyTorch model computing differentiable logits from leaf `torch.Tensor`."""

        def __init__(self) -> None:
            """Initialize leaf parameter `(16, 16)` with `requires_grad=True`."""
            w = torch.ones((16, 16), dtype=torch.float32, requires_grad=True)
            self.trainable_variables = (_TorchVar(w),)

        def __call__(self, inputs: torch.Tensor, /) -> torch.Tensor:
            """Index `w` by `inputs % 16`.

            Returns:
                Vocabulary logits `torch.Tensor`.

            """
            return self.trainable_variables[0].value[inputs % 16]

    class _TorchKerasOpt:
        """Keras 3 optimizer double recording `apply_gradients` on PyTorch tensors."""

        def __init__(self, learning_rate: float = 1e-4) -> None:
            """Initialize `applied` counter."""
            _ = learning_rate
            self.applied = 0

        def apply_gradients(
            self,
            grads_and_vars: Iterable[tuple[ArithmeticTensorProtocol, KerasDPOVar]],
            /,
        ) -> None:
            """Increment `self.applied`."""
            assert len(list(grads_and_vars)) == 1
            self.applied += 1

    class _TorchDPOLoader:
        """Preset loader returning `_TorchDPOModel`."""

        @staticmethod
        def from_preset(_name: str) -> _TorchDPOModel:
            """Return `_TorchDPOModel`."""
            return _TorchDPOModel()

    def _torch_dpo_batches(_cfg: ETLConfig) -> dict[str, list[dict[str, torch.Tensor]]]:
        """Return PyTorch preference batch.

        Returns:
            Loader mapping.

        """
        return {
            "loader": [
                {
                    "chosen_inputs": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "chosen_labels": torch.tensor([[2, 3, 4]], dtype=torch.long),
                    "rejected_inputs": torch.tensor([[1, 5, 6]], dtype=torch.long),
                    "rejected_labels": torch.tensor([[5, 6, 7]], dtype=torch.long),
                }
            ]
        }

    class _TorchOptNS:
        """Namespace exposing `AdamW` for PyTorch Keras DPO."""

        AdamW = _TorchKerasOpt

    class _TorchKerasModule:
        """Full `keras` module double configured for `KERAS_BACKEND='torch'`."""

        backend = _FakeBackendAttr("torch")
        ops = typed_keras_ops
        optimizers = _TorchOptNS()

    torch_keras_mod = _TorchKerasModule()
    with (
        monkeypatch.context() as mp,
        patch.dict(
            sys.modules,
            {"keras": torch_keras_mod, "torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod},
        ),
    ):
        mp.setattr(keras_dpo, "keras", torch_keras_mod)
        mp.setattr(keras_dpo, "tf", None)
        mp.setattr(keras_dpo, "GemmaCausalLM", _TorchDPOLoader)
        mp.setattr(keras_dpo, "build_dataloader", _torch_dpo_batches)

        sync_before = xla_mod.sync_calls
        res_dpo_pt = keras_dpo.run_dpo(
            DPOConfig(model_name="gemma-4", dataset="dpo_ds", epochs=1, batch_size=1),
            backend_alias="keras_pytorch",
            hardware="tpu-v6e",
        )
        assert res_dpo_pt["status"] == "completed"
        assert res_dpo_pt["backend"] == "keras_pytorch"
        assert res_dpo_pt.get("keras_backend") == "torch"
        assert xla_mod.sync_calls > sync_before

        torch_opt = _TorchKerasOpt()
        step_pt: Callable[[Mapping[str, torch.Tensor]], torch.Tensor | ArithmeticTensorProtocol | float] = (
            keras_dpo._get_train_step_fn(
                _TorchDPOModel(),
                _TorchDPOModel(),
                torch_opt,
                0.1,
                active_backend="torch",
                arith_optimizer=torch_opt,
            )
        )
        assert callable(step_pt)

        # Cover non-value var, None grad, non-backward loss_val, and empty grad_pairs in _build_torch_dpo_step_fn
        class _MixedNoGradModel:
            """Callable model with both string sentinel and un-differentiated `_TorchVar`."""

            def __init__(self) -> None:
                """Initialize mixed `trainable_variables` tuple."""
                unused_t = torch.ones((8, 8), dtype=torch.float32, requires_grad=True)
                self.trainable_variables = ("non_var_str", _TorchVar(unused_t))
                self._jax_m = _JaxStatelessDPOModel()

            def __call__(self, inputs: jax.Array, /) -> jax.Array:
                """Return JAX logits so `dpo_loss` returns a `jax.Array` without `.backward`."""
                return self._jax_m(inputs)

        with monkeypatch.context() as mp2:
            mp2.setattr(keras_dpo, "keras", jax_keras_mod)
            no_grad_step = keras_dpo._build_torch_dpo_step_fn(
                _MixedNoGradModel(),
                _MixedNoGradModel(),
                torch_opt,
                0.1,
                device="cpu",
            )
            _ = no_grad_step(_jax_dpo_batches(ETLConfig(dataset_name="d", split="train"))["loader"][0])

    # Engine C: TensorFlow `TPUStrategy` scope + `@tf.function(jit_compile=True)` `GradientTape` DPO
    keras_rt.reset_tf_tpu_init_state()
    tf_tpu_dpo = _FakeFullTFModule()

    class _TFKerasModForDPO:
        """Keras module double reporting `'tensorflow'` backend for `keras_tensorflow` DPO."""

        backend = _FakeBackendAttr("tensorflow")
        optimizers = _JaxOptimizersNS()

    def _tf_step_stub(
        _pm: _JaxStatelessDPOModel,
        _rm: _JaxStatelessDPOModel,
        _opt: _JaxOptimizer,
        _beta: float,
    ) -> Callable[[Mapping[str, jax.Array]], float]:
        """Return compiled TF step callable.

        Returns:
            Step callable.

        """

        def _s(_b: Mapping[str, jax.Array]) -> float:
            """Return scalar loss `0.4`."""
            return 0.4

        return _s

    with monkeypatch.context() as mp, patch.dict(sys.modules, {"keras": _TFKerasModForDPO()}):
        mp.setattr(keras_dpo, "keras", _TFKerasModForDPO())
        mp.setattr(keras_dpo, "tf", tf_tpu_dpo)
        mp.setattr(keras_dpo, "GemmaCausalLM", _JaxDPOLoader)
        mp.setattr(keras_dpo, "build_dataloader", _jax_dpo_batches)
        mp.setattr(keras_dpo, "_get_train_step_fn", _tf_step_stub)
        res_dpo_tf = keras_dpo.run_dpo(
            DPOConfig(model_name="gemma-4", dataset="dpo_ds", epochs=1, batch_size=1),
            backend_alias="keras_tensorflow",
            hardware="tpu-v6e",
            strict_hardware=True,
        )
        assert res_dpo_tf["status"] == "completed"
        assert res_dpo_tf["backend"] == "keras_tensorflow"
        assert res_dpo_tf.get("keras_backend") == "tensorflow"
        assert tf_tpu_dpo.distribute.last_tpu_strategy is not None
        assert tf_tpu_dpo.distribute.last_tpu_strategy.scope_ctx.entered == 1

    # 4. Inference, Serve, Logging, Export, Quantize without TensorFlow
    with monkeypatch.context() as mp, patch.dict(sys.modules, {"keras": jax_keras_mod, "jax": fake_jax_for_inf()}):
        mp.setattr(keras_inf, "keras", jax_keras_mod)
        mp.setattr(keras_inf, "tf", None)
        mp.setattr(keras_inf, "GemmaCausalLM", FakeKerasGemma4CausalLM())
        keras_inf._MODEL_CACHE.clear()
        inf_res = keras_inf.generate_sql(
            "google/gemma-4-E4B",
            "SELECT",
            backend_alias="keras_jax",
            hardware="tpu-v6e",
        )
        assert inf_res["status"] == "success"
        assert inf_res["backend"] == "keras_jax"
        assert inf_res.get("keras_backend") == "jax"
        keras_inf._MODEL_CACHE.clear()

        # Missing tf for keras_tensorflow and canonical 'keras' in generate_sql (lines 319 and 321)
        with patch.dict(sys.modules, {"keras": _TFKerasNoOps()}):
            mp.setattr(keras_inf, "keras", _TFKerasNoOps())
            with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
                keras_inf.generate_sql("m", "p", backend_alias="keras_tensorflow")
            with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
                keras_inf.generate_sql("m", "p", backend_alias="keras")

        # Serve on keras_jax without TensorFlow (line 192: backend_variant + keras_backend)
        mp.setattr(keras_serve, "keras", jax_keras_mod)
        mp.setattr(keras_serve, "tf", None)
        srv_res = keras_serve.serve_model("salem-local-keras-stub", backend_alias="keras_jax")
        assert srv_res["backend"] == "keras"
        assert srv_res.get("backend_variant") == "keras_jax"
        assert srv_res.get("keras_backend") == "jax"

        # Logging on keras_jax without TensorFlow (lines 175-184)
        mp.setattr(keras_log, "tf", None)
        log_res = keras_log.log_metrics(
            {"loss": 0.12},
            step=1,
            log_dir=str(tmp_path / "kjax_logs"),
            backend_alias="keras_jax",
        )
        assert log_res["backend"] == "keras_jax"
        assert log_res.get("keras_backend") == "jax"

        # Export & Quantize keras_backend provenance
        class _SavableKerasPreset:
            """Preset loader returning a savable & quantizable model."""

            class _Model:
                """Model with `.save` and `.quantize`."""

                layers: tuple[()] = ()

                def save(self, path: Path | str) -> None:
                    """Write placeholder file."""
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_bytes(b"keras")

                def quantize(self, mode: str) -> None:
                    """Accept quantization mode."""
                    assert mode == "int8"

            @classmethod
            def from_preset(cls, _name: str) -> _SavableKerasPreset._Model:
                """Return `_Model`."""
                return cls._Model()

        mp.setattr(keras_exp, "keras", jax_keras_mod)
        mp.setattr(keras_exp, "GemmaCausalLM", _SavableKerasPreset)
        exp_res = keras_exp.export_model("gemma-4", str(tmp_path / "kexp"), backend_alias="keras_jax")
        assert exp_res["backend"] == "keras_jax"
        assert exp_res.get("keras_backend") == "jax"

        mp.setattr(keras_quant, "keras", jax_keras_mod)
        q_res = keras_quant.quantize_model(
            "gemma-4",
            "int8",
            backend_alias="keras_jax",
            model=_SavableKerasPreset._Model(),
        )
        assert q_res["backend"] == "keras_jax"
        assert q_res.get("keras_backend") == "jax"


def fake_jax_for_inf() -> FakeMultiHostJaxModule:
    """Construct a 2-chip `FakeMultiHostJaxModule` for `keras_jax` TPU inference tests.

    Returns:
        Configured `FakeMultiHostJaxModule`.

    """
    return FakeMultiHostJaxModule(tpu_devices=(FakeJaxTPUDevice("TPU_0"), FakeJaxTPUDevice("TPU_1")))

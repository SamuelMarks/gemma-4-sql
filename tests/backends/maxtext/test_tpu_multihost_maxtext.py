"""Multi-chip and multi-host TPU unit and behavioral tests for MaxText (`v4`/`v5e`/`v5p`/`v6e`)."""

from __future__ import annotations

import jax
import pytest

import gemma_4_sql.backends._jax_common as jax_common
import gemma_4_sql.backends.maxtext as mt_pkg
import gemma_4_sql.backends.maxtext.benchmark as mt_bm
import gemma_4_sql.backends.maxtext.config_generator as mt_cfg
import gemma_4_sql.backends.maxtext.dpo as mt_dpo
import gemma_4_sql.backends.maxtext.train as mt_train
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.maxtext.adapter import MaxTextBackend
from gemma_4_sql.exceptions import HardwareUnavailableError
from gemma_4_sql.type_hints import DPOConfig, DTypeProtocol, PyTreeNode, TensorType, TrainingConfig
from tests.conftest import requires_jax
from tests.utils.tpu_doubles import FakeJaxTPUDevice, FakeMultiHostJaxModule

pytestmark = requires_jax


def test_maxtext_tpu_slice_presets_and_gin_topology_lines() -> None:
    """Verify `resolve_tpu_slice_parallelism` and Gin config generation for `v4`, `v5e`, `v5p`, and `v6e` slices."""
    assert mt_pkg.get_trainer() == "maxtext_trainer"
    mt_inst = MaxTextBackend.from_alias("maxtext")
    assert mt_inst.get_trainer() == "maxtext_trainer"

    assert mt_cfg.resolve_tpu_slice_parallelism(None) is None
    assert mt_cfg.resolve_tpu_slice_parallelism("") is None
    assert mt_cfg.resolve_tpu_slice_parallelism("cpu") is None

    v4_32 = mt_cfg.resolve_tpu_slice_parallelism("tpu-v4-32")
    assert v4_32 is not None
    assert v4_32["ici_fsdp_parallelism"] == 8
    assert v4_32["ici_tensor_parallelism"] == 2

    assert mt_cfg.resolve_tpu_slice_parallelism("tpu-v6e") == mt_cfg.TPU_SLICE_TOPOLOGY_PRESETS["v6e-8"]
    assert mt_cfg.resolve_tpu_slice_parallelism("tpu-v5e") == mt_cfg.TPU_SLICE_TOPOLOGY_PRESETS["v5litepod-8"]
    assert mt_cfg.resolve_tpu_slice_parallelism("v5litepod-custom") == mt_cfg.TPU_SLICE_TOPOLOGY_PRESETS["v5litepod-8"]
    assert mt_cfg.resolve_tpu_slice_parallelism("tpu-v5p") == mt_cfg.TPU_SLICE_TOPOLOGY_PRESETS["v5p-8"]
    assert mt_cfg.resolve_tpu_slice_parallelism("tpu-v4-custom") == mt_cfg.TPU_SLICE_TOPOLOGY_PRESETS["v4-8"]

    gin_text = mt_cfg.generate_maxtext_gin_config(
        TrainingConfig(
            action="pretrain",
            model_name="google/gemma-4-E4B",
            dataset="sql_ds",
            epochs=1,
            batch_size=8,
            backend="maxtext",
        ),
        hardware="tpu-v6e-16",
        dcn_data_parallelism=2,
    )
    assert "ici_fsdp_parallelism = 8" in gin_text
    assert "ici_tensor_parallelism = 2" in gin_text
    assert "dcn_data_parallelism = 2" in gin_text


def test_maxtext_multichip_tpu_benchmark_and_strict_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MaxText multi-chip TPU `device_put` sharding on `tpu-v5p`/`tpu-v6e` and `strict_hardware=True`."""
    jax_common.reset_jax_distributed_state()

    class _TinyMaxTextModel:
        """Stateless JAX model double for MaxText multi-chip TPU benchmark."""

        def __init__(self, model_name: str, *, dtype: DTypeProtocol | str | None = None) -> None:
            """Record `model_name` and `dtype`."""
            self.model_name = model_name
            self.dtype = dtype

        def init(self, rng: TensorType, dummy_inputs: TensorType) -> dict[str, PyTreeNode]:
            """Return initialized parameter dict."""
            _ = (rng, dummy_inputs)
            return {"w": jnp.ones((8, 8), dtype=jnp.float32)}

        def apply(self, params: mt_bm._MaxTextBenchmarkParams, inputs: TensorType) -> jax.Array:
            """Return deterministic float32 vocabulary logits."""
            _ = params
            return jnp.ones((int(inputs.shape[0]), int(inputs.shape[1]), 32), dtype=jnp.float32)

    tpu_pair = (
        FakeJaxTPUDevice("TPU_0", peak_bytes=512 * 1024 * 1024),
        FakeJaxTPUDevice("TPU_1", peak_bytes=1024 * 1024 * 1024),
    )

    class _MaxTextMultiChipJax(FakeMultiHostJaxModule):
        """Multi-chip JAX double delegating `random` and `default_device` to real `jax`."""

        random = jax.random
        default_device = staticmethod(jax.default_device)

        def jit[FnT](self, fn: FnT, /) -> FnT:
            """Return `fn` unchanged.

            Returns:
                Unchanged `fn`.

            """
            return fn

    mc_jax = _MaxTextMultiChipJax(tpu_devices=tpu_pair)
    with monkeypatch.context() as mp:
        mp.setattr(mt_bm, "jax", mc_jax)
        mp.setattr(mt_bm, "Gemma4Model", _TinyMaxTextModel)
        res_bm = mt_bm.benchmark_model(
            "google/gemma-4-E4B",
            hardware="tpu-v5p",
            batch_size=2,
            warmup_steps=1,
            num_runs=2,
            strict_hardware=True,
        )
    assert res_bm["status"] == "success"
    assert res_bm["backend"] == "maxtext"
    assert res_bm.get("resolved_device") == "tpu:0"
    assert res_bm.get("device_count") == 2
    assert res_bm.get("peak_hbm_mb") == pytest.approx(1024.0)
    assert res_bm.get("compile_latency_ms") is not None
    assert mc_jax.device_put_calls == 1

    # strict_hardware=True on CPU-only host raises HardwareUnavailableError in benchmark, train, and dpo
    with monkeypatch.context() as mp:
        mp.setattr(mt_bm, "Gemma4Model", _TinyMaxTextModel)
        mp.setattr(mt_train, "Gemma4Model", _TinyMaxTextModel)
        mp.setattr(mt_dpo, "Gemma4Model", _TinyMaxTextModel)
        with pytest.raises(HardwareUnavailableError):
            mt_bm.benchmark_model("google/gemma-4-E4B", hardware="tpu-v6e", batch_size=1, strict_hardware=True)
        with pytest.raises(HardwareUnavailableError):
            mt_train.train_model(
                TrainingConfig(
                    action="sft",
                    model_name="google/gemma-4-E4B",
                    dataset="d",
                    epochs=1,
                    backend="maxtext",
                ),
                hardware="tpu-v6e",
                strict_hardware=True,
            )
        with pytest.raises(HardwareUnavailableError):
            mt_dpo.run_dpo(
                DPOConfig(model_name="google/gemma-4-E4B", dataset="d", epochs=1, batch_size=1),
                hardware="tpu-v6e",
                strict_hardware=True,
            )
    jax_common.reset_jax_distributed_state()

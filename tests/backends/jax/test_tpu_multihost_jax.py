"""Multi-chip and multi-host TPU unit and behavioral tests for `_jax_common` and `jax` (`jax_native`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import jax
import pytest

import gemma_4_sql.backends._jax_common as jax_common
import gemma_4_sql.backends.jax as jax_pkg
import gemma_4_sql.backends.jax.benchmark as jax_bm
import gemma_4_sql.backends.jax.dpo as jax_dpo
import gemma_4_sql.backends.jax.export as jax_exp
import gemma_4_sql.backends.jax.inference as jax_inf
import gemma_4_sql.backends.jax.logging as jax_log
import gemma_4_sql.backends.jax.peft as jax_peft
import gemma_4_sql.backends.jax.train as jax_train
from gemma_4_sql.backends.jax.adapter import JaxBackend
from gemma_4_sql.exceptions import HardwareUnavailableError
from gemma_4_sql.type_hints import DPOConfig, TrainingConfig
from tests.conftest import requires_jax
from tests.utils.tpu_doubles import (
    FakeJaxTPUDevice,
    FakeMultiHostJaxModule,
    FakeNamedSharding,
)

pytestmark = requires_jax


def test_jax_common_distributed_bootstrap_mesh_sharding_and_hbm() -> None:
    """Verify `_jax_common` distributed bootstrap, TPU `Mesh` sharding, and HBM aggregation."""
    assert jax_pkg.get_trainer() == "jax_trainer"
    jax_nat = JaxBackend.from_alias("jax_native")
    assert jax_nat.backend_alias == "jax_native"
    assert jax_nat.get_trainer() == "jax_trainer"

    jax_common.reset_jax_distributed_state()

    # 1. None module & empty force_attempt + Megascale/TPU env idempotency + raising distributed
    topo_none = jax_common.initialize_jax_distributed_if_configured(None)
    assert topo_none.is_distributed is False
    assert topo_none.device_kind == "cpu"

    tpu_chips = (
        FakeJaxTPUDevice("TPU_0", peak_bytes=512 * 1024 * 1024),
        FakeJaxTPUDevice("TPU_1", peak_bytes=768 * 1024 * 1024),
        FakeJaxTPUDevice("TPU_2", peak_bytes=256 * 1024 * 1024),
        FakeJaxTPUDevice("TPU_3", peak_bytes=256 * 1024 * 1024),
    )
    assert str(tpu_chips[0]) == "TPU_0"

    mh_jax = FakeMultiHostJaxModule(tpu_devices=tpu_chips, proc_index=1, proc_count=2)
    with patch.dict(
        os.environ,
        {
            "MEGASCALE_COORDINATOR_ADDRESS": "10.0.0.1:8476",
            "TPU_NUM_WORKERS": "2",
            "TPU_WORKER_ID": "1",
        },
        clear=False,
    ):
        topo1 = jax_common.initialize_jax_distributed_if_configured(mh_jax)
        topo2 = jax_common.initialize_jax_distributed_if_configured(mh_jax)
        assert topo1.is_distributed is True
        assert topo1.process_index == 1
        assert topo1.process_count == 2
        assert topo1.local_device_count == 4
        assert topo1.global_device_count == 8
        assert topo1.device_kind == "tpu"
        assert topo2.is_distributed is True
        assert len(mh_jax.distributed.calls) == 1
        assert mh_jax.distributed.calls[0] == {
            "coordinator_address": "10.0.0.1:8476",
            "num_processes": 2,
            "process_id": 1,
        }

    # Explicit force_attempt with empty env (covers False branches of coord/parsed_np/parsed_pid)
    with patch.dict(
        os.environ,
        {
            "JAX_COORDINATOR_ADDRESS": "",
            "MEGASCALE_COORDINATOR_ADDRESS": "",
            "NUM_PROCESSES": "",
            "TPU_NUM_WORKERS": "",
            "PROCESS_ID": "",
            "TPU_WORKER_ID": "",
        },
        clear=False,
    ):
        _ = jax_common.initialize_jax_distributed_if_configured(mh_jax, force_attempt=True)
        assert len(mh_jax.distributed.calls) == 2
        assert mh_jax.distributed.calls[1] == {}

    # Distributed initialize raising RuntimeError is logged and handled cleanly
    err_dist_jax = FakeMultiHostJaxModule(tpu_devices=(), dist_raises=True)
    topo_cpu = jax_common.initialize_jax_distributed_if_configured(
        err_dist_jax,
        coordinator_address="127.0.0.1:1234",
        force_attempt=True,
    )
    assert topo_cpu.device_kind == "cpu"

    # Topology RuntimeError on process_index() + GPU device_kind discovery
    class _BrokenTopoGpuJax:
        """JAX double raising `RuntimeError` in `process_index` and exposing `'gpu'` devices."""

        def process_index(self) -> int:
            """Raise `RuntimeError`.

            Raises:
                RuntimeError: Always raised.

            """
            raise RuntimeError("not ready")

        def process_count(self) -> int:
            """Return `1`."""
            return 1

        def local_device_count(self) -> int:
            """Return `1`."""
            return 1

        def device_count(self) -> int:
            """Return `1`."""
            return 1

        def devices(self, backend: str | None = None, /) -> list[str]:
            """Return `['GPU_0']` for `'gpu'`, else `[]`."""
            return ["GPU_0"] if backend == "gpu" else []

    probe_gpu = _BrokenTopoGpuJax()
    assert (probe_gpu.process_count(), probe_gpu.local_device_count(), probe_gpu.device_count()) == (1, 1, 1)
    topo_gpu = jax_common.initialize_jax_distributed_if_configured(probe_gpu)
    assert topo_gpu.device_kind == "gpu"
    jax_common.reset_jax_distributed_state()

    # 2. build_tpu_mesh_and_sharding across None, multi-chip data-parallel vs replicated, strict errors, and fallbacks
    with pytest.raises(HardwareUnavailableError, match="JAX module unavailable"):
        jax_common.build_tpu_mesh_and_sharding(None, "tpu-v6e", strict_hardware=True)
    assert jax_common.build_tpu_mesh_and_sharding(None, "cpu", fallback_device="fb") == ("fb", [], "cpu", None)

    prim, devs, dev_str, shard_dp = jax_common.build_tpu_mesh_and_sharding(mh_jax, "tpu-v6e", batch_size=8)
    assert prim == tpu_chips[0]
    assert len(devs) == 4
    assert dev_str == "tpu:0"
    assert isinstance(shard_dp, FakeNamedSharding)
    assert shard_dp.spec == ("data", None)
    assert shard_dp.device_set == ("TPU_0", "TPU_1")
    assert shard_dp.is_fully_addressable() is True

    _, _, _, shard_rep = jax_common.build_tpu_mesh_and_sharding(mh_jax, "tpu-v5p", batch_size=1)
    assert isinstance(shard_rep, FakeNamedSharding)
    assert shard_rep.spec == ()

    with pytest.raises(HardwareUnavailableError, match="could not resolve 'tpu-v5p'"):
        jax_common.build_tpu_mesh_and_sharding(err_dist_jax, "tpu-v5p", strict_hardware=True)

    # Non-strict GPU fallback to CPU when GPU raises RuntimeError
    _, cpu_devs, cpu_str, cpu_shard = jax_common.build_tpu_mesh_and_sharding(
        err_dist_jax,
        "gpu",
        strict_hardware=False,
    )
    assert len(cpu_devs) == 1
    assert cpu_str == "cpu"
    assert cpu_shard is None

    # Failing CPU devices + failing Mesh construction fallback
    class _BrokenShardingNS:
        """Sharding namespace raising `TypeError` on `Mesh`."""

        def Mesh(self, _d: list[FakeJaxTPUDevice | str], _a: tuple[str, ...], /) -> str:
            """Raise `TypeError`.

            Raises:
                TypeError: Always raised.

            """
            raise TypeError("bad mesh")

        def PartitionSpec(self, *axes: str | None) -> tuple[str | None, ...]:
            """Return `axes`."""
            return axes

        def NamedSharding(self, mesh: str, spec: tuple[str | None, ...], /) -> FakeNamedSharding:
            """Return `FakeNamedSharding`."""
            return FakeNamedSharding(mesh, spec)

    broken_ns = _BrokenShardingNS()
    assert broken_ns.PartitionSpec("data") == ("data",)
    assert isinstance(broken_ns.NamedSharding("m", ()), FakeNamedSharding)

    class _CpuErrAndBadMeshJax:
        """JAX double testing `devices('cpu')` RuntimeError and `Mesh` TypeError."""

        sharding = broken_ns

        def __init__(self, *, multi_tpu: bool) -> None:
            """Configure mode."""
            self.multi_tpu = multi_tpu

        def devices(self, backend: str | None = None, /) -> list[str]:
            """Return 2 TPUs when `multi_tpu=True`, else raise `RuntimeError` on CPU.

            Returns:
                List of device strings.

            Raises:
                RuntimeError: When `multi_tpu` is `False` or `backend != 'tpu'`.

            """
            if self.multi_tpu and backend == "tpu":
                return ["TPU_0", "TPU_1"]
            raise RuntimeError("no devices")

    p_fb, d_fb, s_fb, sh_fb = jax_common.build_tpu_mesh_and_sharding(
        _CpuErrAndBadMeshJax(multi_tpu=False),
        "cpu",
        fallback_device="cpu_fb",
    )
    assert (p_fb, d_fb, s_fb, sh_fb) == ("cpu_fb", [], "cpu", None)
    _, _, s_bad_mesh, sh_bad_mesh = jax_common.build_tpu_mesh_and_sharding(
        _CpuErrAndBadMeshJax(multi_tpu=True),
        "tpu",
        batch_size=2,
    )
    assert s_bad_mesh == "tpu:0"
    assert sh_bad_mesh is None

    # 3. aggregate_jax_peak_hbm_mb across valid, None, raising, 0, and string devices
    peak = jax_common.aggregate_jax_peak_hbm_mb([
        FakeJaxTPUDevice("TPU_0", peak_bytes=512 * 1024 * 1024),
        FakeJaxTPUDevice("TPU_1", peak_bytes=None),
        FakeJaxTPUDevice("TPU_2", raise_on_stats=True),
        FakeJaxTPUDevice("TPU_3", peak_bytes=0),
        "raw_str",
    ])
    assert peak == pytest.approx(512.0)
    assert jax_common.aggregate_jax_peak_hbm_mb([FakeJaxTPUDevice("TPU_1", peak_bytes=None), "raw_str"]) is None


def test_jax_native_multichip_benchmark_strict_hardware_and_multihost_rank_gating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-chip `device_put` benchmark, `strict_hardware=True` errors, and rank-0 export/logging gating."""
    # 1. Multi-chip TPU sharded benchmark on jax_native (covering line 220 `jax.device_put`)
    tpu_pair = (
        FakeJaxTPUDevice("TPU_0", peak_bytes=320 * 1024 * 1024),
        FakeJaxTPUDevice("TPU_1", peak_bytes=640 * 1024 * 1024),
    )

    class _BenchMultiChipJax(FakeMultiHostJaxModule):
        """Multi-chip JAX double delegating `random` and `default_device` to real `jax`."""

        random = jax.random
        default_device = staticmethod(jax.default_device)

    mc_jax = _BenchMultiChipJax(tpu_devices=tpu_pair)
    with monkeypatch.context() as mp:
        mp.setattr(jax_bm, "jax", mc_jax)
        res_bm = jax_bm.benchmark_model(
            "gemma-4-tiny",
            hardware="tpu-v6e",
            batch_size=2,
            backend_alias="jax_native",
            warmup_steps=1,
            num_runs=2,
            strict_hardware=True,
        )
    assert res_bm["status"] == "success"
    assert res_bm["backend"] == "jax_native"
    assert res_bm.get("resolved_device") == "tpu:0"
    assert res_bm.get("device_count") == 2
    assert res_bm.get("peak_hbm_mb") == pytest.approx(640.0)
    assert res_bm.get("compile_latency_ms") is not None
    assert mc_jax.device_put_calls == 1

    # 2. strict_hardware=True raises HardwareUnavailableError across train, dpo, peft, and inference on CPU-only host
    with pytest.raises(HardwareUnavailableError):
        jax_train.train_model(
            TrainingConfig(action="sft", model_name="gemma-4-tiny", dataset="d", epochs=1, backend="jax_native"),
            backend_alias="jax_native",
            hardware="tpu-v6e",
            strict_hardware=True,
        )
    with pytest.raises(HardwareUnavailableError):
        jax_dpo.run_dpo(
            DPOConfig(model_name="gemma-4-tiny", dataset="d", epochs=1, batch_size=1),
            backend_alias="jax_native",
            hardware="tpu-v6e",
            strict_hardware=True,
        )
    with pytest.raises(HardwareUnavailableError):
        jax_peft.apply_lora(
            "gemma-4-tiny",
            ["q_proj"],
            backend_alias="jax_native",
            hardware="tpu-v6e",
            strict_hardware=True,
        )
    with pytest.raises(HardwareUnavailableError):
        jax_inf.generate_sql(
            "gemma-4-tiny",
            "SELECT",
            backend_alias="jax_native",
            hardware="tpu-v6e",
            strict_hardware=True,
        )

    # 3. Multi-host rank-1 gating in jax/export.py (line 159) and jax/logging.py (line 38)
    worker1_jax = FakeMultiHostJaxModule(tpu_devices=tpu_pair, proc_index=1, proc_count=2)
    with monkeypatch.context() as mp:
        mp.setattr(jax_exp, "jax", worker1_jax)
        exp_w1 = jax_exp.export_model("gemma-4-tiny", str(tmp_path / "w1_exp"), backend_alias="jax_native")
        assert exp_w1["backend"] == "jax_native"
        assert exp_w1["status"] == "skipped_non_zero_rank"
        assert not (tmp_path / "w1_exp" / "model.safetensors").exists()

        with patch.dict(sys.modules, {"jax": worker1_jax}):
            log_w1 = jax_log.log_metrics(
                {"loss": 0.5},
                step=1,
                log_dir=str(tmp_path / "w1_logs"),
                backend_alias="jax_native",
            )
        assert log_w1["backend"] == "jax_native"
        assert log_w1["status"] == "skipped_non_zero_rank"
        assert not (tmp_path / "w1_logs").exists()
    jax_common.reset_jax_distributed_state()

"""Behavioral and unit tests for PyTorch model quantization and GGUF export without MagicMock."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from gemma_4_sql.exceptions import DependencyMissingError, QuantizationError
from tests.conftest import requires_torch

pytestmark = requires_torch


class _FakeAWQInstance:
    """Typed `AutoAWQForCausalLM` model instance stub."""

    def __init__(self) -> None:
        """Initialize stub tracking flags."""
        self.quantized = False
        self.saved_dir: str | None = None

    def quantize(
        self,
        _tok: _FakeTokenizerInstance,
        *,
        quant_config: dict[str, str | int | bool],
        calib_data: list[str],
    ) -> None:
        """Record AWQ quantization call."""
        _ = (quant_config, calib_data)
        self.quantized = True

    def save_quantized(self, out_dir: str) -> None:
        """Record saved directory."""
        self.saved_dir = out_dir


class _FakeAWQClass:
    """Typed `AutoAWQForCausalLM` loader stub."""

    @staticmethod
    def from_pretrained(_model_name: str) -> _FakeAWQInstance:
        """Return `_FakeAWQInstance`."""
        return _FakeAWQInstance()


class _FakeAWQModule:
    """Typed `awq` module stub."""

    AutoAWQForCausalLM = _FakeAWQClass


class _FakeTokenizerInstance:
    """Typed HuggingFace tokenizer stub."""

    def save_pretrained(self, _out_dir: str) -> None:
        """No-op save_pretrained."""


class _FakeAutoTokenizerClass:
    """Typed `AutoTokenizer` loader stub."""

    @staticmethod
    def from_pretrained(_model_name: str) -> _FakeTokenizerInstance:
        """Return `_FakeTokenizerInstance`."""
        return _FakeTokenizerInstance()


class _FakeAutoModelClass:
    """Typed `AutoModelForCausalLM` loader stub."""

    @staticmethod
    def from_pretrained(_model_name: str, **kwargs: torch.dtype | str) -> torch.nn.Linear:
        """Return a tiny `torch.nn.Linear` module."""
        _ = kwargs
        return torch.nn.Linear(4, 4)


class _FakeTransformersModule:
    """Typed `transformers` module stub for external AWQ/GPTQ tests."""

    AutoTokenizer = _FakeAutoTokenizerClass
    AutoModelForCausalLM = _FakeAutoModelClass


class _FakeGPTQQuantizerInstance:
    """Typed `GPTQQuantizer` instance stub."""

    def __init__(
        self,
        *,
        bits: int,
        dataset: str | list[str],
        group_size: int,
        damp_percent: float,
    ) -> None:
        """Store GPTQ quantizer settings."""
        self.bits = bits
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent

    def quantize_model(self, model: torch.nn.Linear, _tok: _FakeTokenizerInstance) -> torch.nn.Linear:
        """Return `model` as quantized stub."""
        return model

    def save(self, _model: torch.nn.Linear, _out_dir: str) -> None:
        """No-op save."""


class _FakeOptimumGPTQNamespace:
    """Typed `optimum.gptq` namespace stub."""

    GPTQQuantizer = _FakeGPTQQuantizerInstance


class _FakeOptimumModule:
    """Typed `optimum` module stub."""

    gptq = _FakeOptimumGPTQNamespace


def test_pytorch_quantize_imports() -> None:
    """Verify pytorch.quantize gracefully sets optional imports to None when unavailable."""
    import importlib

    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch.dict(sys.modules, {"torch": None, "transformers": None}):
        importlib.reload(quantize_module)
        assert quantize_module.torch is None
        assert quantize_module.BitsAndBytesConfig is None
        assert quantize_module.AutoModelForCausalLM is None
        assert quantize_module.AutoTokenizer is None
    importlib.reload(quantize_module)


def test_validate_gguf_file(tmp_path: Path) -> None:
    """Verify validate_gguf_file rejects missing files, corrupt magic bytes, unsupported versions, and bad offsets."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    file_path = tmp_path / "test.gguf"

    with pytest.raises(FileNotFoundError):
        quantize_module.validate_gguf_file(file_path)

    file_path.write_bytes(b"BADH")
    with pytest.raises(ValueError, match="Corrupt GGUF magic"):
        quantize_module.validate_gguf_file(file_path)

    file_path.write_bytes(b"GGUF" + struct.pack("<I", 1))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        quantize_module.validate_gguf_file(file_path)

    with open(file_path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 0))
        f.write(
            struct.pack("<Q", 2)
            + b"t1"
            + struct.pack("<I", 1)
            + struct.pack("<Q", 4)
            + struct.pack("<I", 0)
            + struct.pack("<Q", 7)
        )
    with pytest.raises(ValueError, match="Unaligned GGUF tensor offset"):
        quantize_module.validate_gguf_file(file_path)

    with open(file_path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 0))
        f.write(
            struct.pack("<Q", 2)
            + b"t1"
            + struct.pack("<I", 1)
            + struct.pack("<Q", 4)
            + struct.pack("<I", 0)
            + struct.pack("<Q", 6400)
        )
        f.write(b"\x00" * 64)
    with pytest.raises(ValueError, match="exceeds file size"):
        quantize_module.validate_gguf_file(file_path)


def test_validate_gguf_file_valid(tmp_path: Path) -> None:
    """Verify validate_gguf_file parses all GGUF v3 metadata types and tensor headers."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    file_path = tmp_path / "test.gguf"

    with open(file_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 10))

        # KV 1: String
        f.write(struct.pack("<Q", 2) + b"k1" + struct.pack("<I", 8) + struct.pack("<Q", 2) + b"v1")
        # KV 2: Bool
        f.write(struct.pack("<Q", 2) + b"k2" + struct.pack("<I", 7) + struct.pack("<B", 1))
        # KV 3: Uint32
        f.write(struct.pack("<Q", 2) + b"k3" + struct.pack("<I", 4) + struct.pack("<I", 42))
        # KV 4: Int32
        f.write(struct.pack("<Q", 2) + b"k4" + struct.pack("<I", 5) + struct.pack("<i", -42))
        # KV 5: Float32
        f.write(struct.pack("<Q", 2) + b"k5" + struct.pack("<I", 6) + struct.pack("<f", math.pi))
        # KV 6: Uint64
        f.write(struct.pack("<Q", 2) + b"k6" + struct.pack("<I", 10) + struct.pack("<Q", 100))
        # KV 7: Int64
        f.write(struct.pack("<Q", 2) + b"k7" + struct.pack("<I", 11) + struct.pack("<q", -100))
        # KV 8: Float64
        f.write(struct.pack("<Q", 2) + b"k8" + struct.pack("<I", 12) + struct.pack("<d", 2.0))
        # KV 9: Array of uint32
        f.write(
            struct.pack("<Q", 2)
            + b"k9"
            + struct.pack("<I", 9)
            + struct.pack("<I", 4)
            + struct.pack("<Q", 1)
            + struct.pack("<I", 5)
        )
        # KV 10: Array of strings
        f.write(
            struct.pack("<Q", 3)
            + b"k10"
            + struct.pack("<I", 9)
            + struct.pack("<I", 8)
            + struct.pack("<Q", 1)
            + struct.pack("<Q", 1)
            + b"a"
        )

        # Tensor descriptor
        f.write(
            struct.pack("<Q", 2)
            + b"t1"
            + struct.pack("<I", 1)
            + struct.pack("<Q", 10)
            + struct.pack("<I", 1)
            + struct.pack("<Q", 0)
        )

    res = quantize_module.validate_gguf_file(file_path)
    assert res["version"] == 3
    assert res["tensor_count"] == 1
    meta = res.get("metadata")
    assert isinstance(meta, dict)
    assert meta["k1"] == "v1"
    assert meta["k2"] is True
    assert meta["k3"] == 42
    assert meta["k4"] == -42
    assert "tensors" in res


def test_write_gguf_file_and_tensor_to_2d_list_edge(tmp_path: Path) -> None:
    """Verify _write_gguf_file and _tensor_to_2d_list edge cases."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    class _ScalarTolistStub:
        """Minimal `ToListArrayProtocol` stub returning a scalar float from `tolist()`."""

        def tolist(self) -> float:
            """Return scalar float to test non-list branch."""
            return 1.0

    assert quantize_module._tensor_to_2d_list(_ScalarTolistStub()) == []

    file_path = tmp_path / "test.gguf"
    quantize_module._write_gguf_file(file_path, "m")
    parsed = quantize_module.validate_gguf_file(file_path)
    assert parsed["version"] == 3

    file_path_custom = tmp_path / "custom.gguf"
    quantize_module._write_gguf_file(
        file_path_custom,
        "m",
        {"blk.0.attn_q.weight": torch.ones((4, 8), dtype=torch.float32)},
        {"general.architecture": "gemma4"},
    )
    parsed_custom = quantize_module.validate_gguf_file(file_path_custom)
    assert parsed_custom["tensor_count"] == 1
    assert parsed_custom["tensors"][0]["shape"] == [8, 4]


def test_apply_awq_and_gptq_quantization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _apply_awq_quantization and _apply_gptq_quantization run typed stubs and handle missing packages."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch.dict(sys.modules, {"awq": _FakeAWQModule, "transformers": _FakeTransformersModule}):
        res = quantize_module._apply_awq_quantization("m", str(tmp_path), ["data"])
        assert res[1] == "quantized_awq"

        res2 = quantize_module._apply_awq_quantization("m", str(tmp_path / "sub"), None)
        assert res2[1] == "quantized_awq"

    with patch.dict(sys.modules, {"awq": None}), pytest.raises(DependencyMissingError):
        quantize_module._apply_awq_quantization("m")

    with patch.dict(sys.modules, {"optimum": _FakeOptimumModule, "transformers": _FakeTransformersModule}):
        res_g1 = quantize_module._apply_gptq_quantization("m", str(tmp_path))
        assert res_g1[1] == "quantized_gptq"

        # Also exercise branch when torch lacks float16 attribute
        class _TorchWithoutF16:
            """Minimal torch stub without `.float16`."""

        monkeypatch.setattr(quantize_module, "torch", _TorchWithoutF16())
        res_g2 = quantize_module._apply_gptq_quantization("m", str(tmp_path / "sub"))
        assert res_g2[1] == "quantized_gptq"
        monkeypatch.undo()

    with patch.dict(sys.modules, {"optimum": None}), pytest.raises(DependencyMissingError):
        quantize_module._apply_gptq_quantization("m")


def test_export_gguf(tmp_path: Path) -> None:
    """Verify _export_gguf creates and validates GGUF files, reuses existing files, and fails on missing models."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    calls: list[str] = []

    def _fake_extract(
        model_or_name: str | torch.nn.Module, *, allow_dummy_fallback: bool = False
    ) -> dict[str, torch.Tensor]:
        """Record call and return synthetic embedding tensor.

        Returns:
            Mapping with synthetic `token_embd.weight` tensor.
        """
        _ = allow_dummy_fallback
        calls.append(str(model_or_name))
        return {"token_embd.weight": torch.randn(4, 32, dtype=torch.float32)}

    with patch("gemma_4_sql.backends.pytorch.gguf.extract_pytorch_state_dict", side_effect=_fake_extract):
        res = quantize_module._export_gguf("m", str(tmp_path))
        assert res[1] == "quantized_gguf"
        assert (tmp_path / "m.gguf").exists()
        assert len(calls) == 1

        # Re-exporting when file already exists skips extraction
        quantize_module._export_gguf("m", str(tmp_path))
        assert len(calls) == 1

    # Missing model with default allow_dummy_fallback=False raises QuantizationError
    with pytest.raises(QuantizationError, match="Failed to extract PyTorch weights"):
        quantize_module._export_gguf("missing-model-id", str(tmp_path / "missing_dir"))


def test_quantize_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify quantize_model runs end-to-end across GGUF, INT8, INT4, AWQ, and GPTQ."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM
    from gemma_4_sql.type_hints import get_quantized_artifact

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
        layer_types=["sliding_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    tiny_model = Gemma4ForCausalLM(cfg)

    # 1. End-to-end GGUF quantization against in-memory tiny PyTorch model and failure when missing
    gguf_dir = tmp_path / "gguf_out"
    res_gguf = quantize_module.quantize_model(
        "tiny-gemma4", "gguf", export_path=str(gguf_dir), out_type="q8_0", model=tiny_model
    )
    assert res_gguf["status"] == "quantized_gguf"
    assert res_gguf["memory_reduction_factor"] == 0.6
    gguf_file = gguf_dir / "tiny-gemma4.gguf"
    assert gguf_file.exists()
    validated = quantize_module.validate_gguf_file(gguf_file)
    assert validated["version"] == 3
    assert int(str(validated["tensor_count"])) > 0

    res_gguf_missing = quantize_module.quantize_model(
        "nonexistent-model", "gguf", export_path=str(tmp_path / "fail_gguf")
    )
    assert str(res_gguf_missing["status"]).startswith("failed:")
    with pytest.raises(QuantizationError):
        quantize_module.quantize_model(
            "nonexistent-model",
            "gguf",
            export_path=str(tmp_path / "fail_gguf2"),
            raise_on_error=True,
        )

    # 2. End-to-end INT8 and INT4 quantization on tiny PyTorch model
    def _bnb_cfg(**kw: bool) -> dict[str, bool]:
        """Return kwargs dict."""
        return kw

    monkeypatch.setattr(quantize_module, "BitsAndBytesConfig", _bnb_cfg)
    res_int8 = quantize_module.quantize_model("tiny-gemma4", "int8", model=tiny_model)
    assert res_int8["status"] == "quantized_int8"
    assert res_int8["memory_reduction_factor"] == 0.5
    art_int8 = get_quantized_artifact(tiny_model)
    assert art_int8 is not None and art_int8.method == "int8"

    tiny_model_int4 = Gemma4ForCausalLM(cfg)
    res_int4 = quantize_module.quantize_model("tiny-gemma4", "int4", model=tiny_model_int4)
    assert res_int4["status"] == "quantized_int4"
    assert res_int4["memory_reduction_factor"] == 0.75
    art_int4 = get_quantized_artifact(tiny_model_int4)
    assert art_int4 is not None and art_int4.method == "int4"

    # 3. End-to-end native AWQ and GPTQ tensor quantization (including pytorch_native auto-instantiation)
    tiny_awq_model = Gemma4ForCausalLM(cfg)
    res_native_awq = quantize_module.quantize_model("tiny-gemma4", "awq", model=tiny_awq_model)
    assert res_native_awq["status"] == "quantized_awq"
    assert get_quantized_artifact(tiny_awq_model) is not None

    res_auto_native = quantize_module.quantize_model("gemma-4-tiny", "awq", backend_alias="pytorch_native")
    assert res_auto_native["status"] == "quantized_awq"

    tiny_gptq_model = Gemma4ForCausalLM(cfg)
    res_native_gptq = quantize_module.quantize_model("tiny-gemma4", "gptq", model=tiny_gptq_model)
    assert res_native_gptq["status"] == "quantized_gptq"
    assert get_quantized_artifact(tiny_gptq_model) is not None

    w_test = torch.randn(4, 8, dtype=torch.float32)
    act_test = torch.randn(6, 8, dtype=torch.float32)
    awq_scales = quantize_module.calibrate_pytorch_awq_scales(w_test, activations=act_test, n_grid=5)
    assert awq_scales.shape == (1, 8)
    gptq_w = quantize_module.calibrate_pytorch_gptq_weights(w_test, activations=act_test, bits=4)
    assert gptq_w.shape == (4, 8)

    # 4. External AutoAWQ and Optimum GPTQ dispatch through quantize_model
    with patch.dict(sys.modules, {"awq": _FakeAWQModule, "transformers": _FakeTransformersModule}):
        res_awq = quantize_module.quantize_model(
            "tiny-gemma4",
            "awq",
            export_path=str(tmp_path / "awq_out"),
            calib_data=["SELECT 1;"],
        )
        assert res_awq["status"] == "quantized_awq"

    with patch.dict(sys.modules, {"optimum": _FakeOptimumModule, "transformers": _FakeTransformersModule}):
        res_gptq = quantize_module.quantize_model(
            "tiny-gemma4",
            "gptq",
            export_path=str(tmp_path / "gptq_out"),
        )
        assert res_gptq["status"] == "quantized_gptq"

    monkeypatch.setattr(quantize_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("m", "int8")
    with pytest.raises(DependencyMissingError):
        quantize_module.calibrate_pytorch_awq_scales(w_test)
    with pytest.raises(DependencyMissingError):
        quantize_module.calibrate_pytorch_gptq_weights(w_test)

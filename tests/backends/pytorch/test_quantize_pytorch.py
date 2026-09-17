"""Tests for PyTorch quantization logic (BitsAndBytes, AWQ, GPTQ, GGUF)."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import pytest

import gemma_4_sql.backends.pytorch.quantize as pt_quantize
from gemma_4_sql.backends.pytorch.quantize import (
    _apply_awq_quantization,
    _apply_gptq_quantization,
    _export_gguf,
    _write_gguf_file,
    quantize_model,
    validate_gguf_file,
)
from gemma_4_sql.exceptions import DependencyMissingError


class MockTorch:
    """Mock PyTorch module for testing."""

    float16 = "float16"


class MockBitsAndBytesConfig:
    """Mock BitsAndBytesConfig."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize mock config."""


class MockAutoModelForCausalLM:
    """Mock AutoModelForCausalLM."""

    @staticmethod
    def from_pretrained(_model_name: str, **_kwargs: object) -> object:
        """Return dummy model."""
        return object()


def test_quantize_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch quantize raises DependencyMissingError when PyTorch is absent."""
    monkeypatch.setattr(pt_quantize, "torch", None)
    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", None)
    monkeypatch.setattr(pt_quantize, "AutoModelForCausalLM", None)
    with pytest.raises(DependencyMissingError, match=r"PyTorch quantization dependencies are missing\."):
        quantize_model("model", "int8")


def test_quantize_pytorch_bnb_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test standard BitsAndBytes int8 and int4 quantization methods."""
    monkeypatch.setattr(pt_quantize, "torch", MockTorch())
    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", MockBitsAndBytesConfig)
    monkeypatch.setattr(pt_quantize, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    res_int8 = quantize_model("model", "int8")
    assert res_int8["backend"] == "pytorch"
    assert res_int8["status"] == "quantized_int8"

    res_int4 = quantize_model("model", "int4")
    assert res_int4["status"] == "quantized_int4"

    res_unknown = quantize_model("model", "unknown_method")
    assert "unsupported" in str(res_unknown["status"])


def test_quantize_pytorch_bnb_in_memory_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test BitsAndBytes quantization attached directly to an in-memory model instance."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(pt_quantize, "torch", MockTorch())
    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", MockBitsAndBytesConfig)

    mock_model = MagicMock()
    mock_model.config = MagicMock()

    res = quantize_model("dummy_name", "int8", model=mock_model)
    assert res["status"] == "quantized_int8"
    assert getattr(mock_model, "_is_quantized", False) is True
    assert getattr(mock_model, "_quant_method", "") == "int8"
    assert hasattr(mock_model.config, "quantization_config")


def test_quantize_pytorch_error_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error branch during quantization wrapper execution."""
    monkeypatch.setattr(pt_quantize, "torch", MockTorch())
    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", Exception)
    res = quantize_model("model", "int8")
    assert "failed" in str(res["status"])


def test_validate_gguf_file_valid_and_invalid(tmp_path: Path) -> None:
    """Test validate_gguf_file with valid, non-existent, and corrupt files."""
    gguf_path = tmp_path / "valid.gguf"
    _write_gguf_file(gguf_path, model_name="test_model")

    info = validate_gguf_file(gguf_path)
    assert info["version"] == 3
    assert info["tensor_count"] == 1
    assert "general.architecture" in info["metadata"]
    assert info["tensors"][0]["name"] == "token_embd.weight"

    # Non-existent file
    with pytest.raises(FileNotFoundError, match="GGUF file not found"):
        validate_gguf_file(tmp_path / "missing.gguf")

    # Corrupt magic
    corrupt_magic = tmp_path / "corrupt_magic.gguf"
    with open(corrupt_magic, "wb") as f:
        f.write(b"BADM\x03\x00\x00\x00")
    with pytest.raises(ValueError, match="Corrupt GGUF magic header"):
        validate_gguf_file(corrupt_magic)

    # Unsupported version
    bad_version = tmp_path / "bad_version.gguf"
    with open(bad_version, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 99))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        validate_gguf_file(bad_version)


def test_gguf_metadata_types_and_tensor_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GGUF parsing with uint32, float32, and unknown metadata types, and raw bytes tensors."""
    custom_gguf = tmp_path / "custom_meta.gguf"
    with open(custom_gguf, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))  # 1 tensor
        f.write(struct.pack("<Q", 8))  # 8 metadata keys

        # KV 1: UINT32 (5)
        k1 = b"custom.uint"
        f.write(struct.pack("<Q", len(k1)) + k1)
        f.write(struct.pack("<I", 5))
        f.write(struct.pack("<I", 42))

        # KV 2: FLOAT32 (6)
        k2 = b"custom.float"
        f.write(struct.pack("<Q", len(k2)) + k2)
        f.write(struct.pack("<I", 6))
        f.write(struct.pack("<f", math.pi))

        # KV 3: UNKNOWN (99)
        k3 = b"custom.unknown"
        f.write(struct.pack("<Q", len(k3)) + k3)
        f.write(struct.pack("<I", 99))

        # KV 4: INT64 (11)
        k4 = b"custom.int64"
        f.write(struct.pack("<Q", len(k4)) + k4)
        f.write(struct.pack("<I", 11))
        f.write(struct.pack("<q", 1234567890123))

        # KV 5: FLOAT64 (12)
        k5 = b"custom.float64"
        f.write(struct.pack("<Q", len(k5)) + k5)
        f.write(struct.pack("<I", 12))
        f.write(struct.pack("<d", math.e))

        # KV 6: ARRAY of INT32 (arr_type 4)
        k6 = b"custom.arr_int"
        f.write(struct.pack("<Q", len(k6)) + k6)
        f.write(struct.pack("<I", 9))  # ARRAY
        f.write(struct.pack("<I", 4))  # arr_type = INT32
        f.write(struct.pack("<Q", 2))  # arr_len = 2
        f.write(struct.pack("<ii", 10, 20))

        # KV 7: ARRAY of FLOAT32 (arr_type 6)
        k7 = b"custom.arr_float"
        f.write(struct.pack("<Q", len(k7)) + k7)
        f.write(struct.pack("<I", 9))  # ARRAY
        f.write(struct.pack("<I", 6))  # arr_type = FLOAT32
        f.write(struct.pack("<Q", 1))  # arr_len = 1
        f.write(struct.pack("<f", 1.5))

        # KV 8: ARRAY of OTHER (arr_type 99)
        k8 = b"custom.arr_other"
        f.write(struct.pack("<Q", len(k8)) + k8)
        f.write(struct.pack("<I", 9))  # ARRAY
        f.write(struct.pack("<I", 99))  # arr_type = OTHER
        f.write(struct.pack("<Q", 1))  # arr_len = 1

        # Tensor descriptor
        t_name = b"raw_tensor"
        f.write(struct.pack("<Q", len(t_name)) + t_name)
        f.write(struct.pack("<I", 1))  # 1 dim
        f.write(struct.pack("<Q", 16))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<Q", 0))

        # Align to 32 bytes
        pad = (32 - (f.tell() % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(b"\x00" * 64)

    parsed = validate_gguf_file(custom_gguf)
    assert parsed["metadata"]["custom.uint"] == 42
    assert abs(parsed["metadata"]["custom.float"] - math.pi) < 1e-4
    assert parsed["metadata"]["custom.unknown"] == "unknown"

    # Test _write_gguf_file with np is None and unaligned raw bytes
    monkeypatch.setattr(pt_quantize, "np", None)
    np_none_gguf = tmp_path / "np_none.gguf"
    _write_gguf_file(np_none_gguf, model_name="fallback", tensors={"raw": bytes(35)})
    validate_gguf_file(np_none_gguf)


def test_export_gguf_end_to_end(tmp_path: Path) -> None:
    """Test _export_gguf creation and re-validation when file exists."""
    reduction, status = _export_gguf("gemma4_sql", str(tmp_path))
    assert reduction == 0.6
    assert status == "quantized_gguf"

    # Call again to exercise already-exists branch
    red2, stat2 = _export_gguf("gemma4_sql", str(tmp_path))
    assert red2 == 0.6
    assert stat2 == "quantized_gguf"

    # Test through top-level quantize_model
    res = quantize_model("gemma4_sql", method="gguf", export_path=str(tmp_path))
    assert res["status"] == "quantized_gguf"


def test_export_gguf_extract_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _export_gguf handles extract_pytorch_state_dict exception gracefully.

    Args:
        tmp_path: Temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def mock_fail_extract(model_name: str) -> dict[str, object]:
        raise RuntimeError("State dict extraction failed")

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gguf.extract_pytorch_state_dict", mock_fail_extract)
    fail_dir = tmp_path / "gguf_fail_dir"
    reduction, status = _export_gguf("model_extract_fail", str(fail_dir))
    assert reduction == 0.6
    assert status == "quantized_gguf"


def test_apply_awq_quantization_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test _apply_awq_quantization execution, calibration, and serialization."""
    import sys

    class MockAWQModel:
        """Mock AutoAWQForCausalLM instance."""

        quantized: bool = False
        saved_path: str | None = None

        def quantize(self, _tokenizer: Any, quant_config: Any = None, calib_data: Any = None) -> None:
            """Simulate model quantization with calibration data."""
            self.quantized = True
            assert quant_config["w_bit"] == 4
            assert calib_data is not None

        def save_quantized(self, save_path: str) -> None:
            """Simulate saving quantized model."""
            self.saved_path = save_path

        @classmethod
        def from_pretrained(cls, _name: str) -> MockAWQModel:
            """Simulate loading model."""
            return cls()

    class MockTokenizer:
        """Mock Hugging Face AutoTokenizer."""

        saved_path: str | None = None

        def save_pretrained(self, save_path: str) -> None:
            """Simulate saving tokenizer."""
            self.saved_path = save_path

        @classmethod
        def from_pretrained(cls, _name: str) -> MockTokenizer:
            """Simulate loading tokenizer."""
            return cls()

    mock_awq_module = type("MockAWQModule", (), {"AutoAWQForCausalLM": MockAWQModel})
    mock_transformers_module = type("MockTransformersModule", (), {"AutoTokenizer": MockTokenizer})
    monkeypatch.setitem(sys.modules, "awq", mock_awq_module)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers_module)

    out_dir = tmp_path / "awq_export"
    reduction, status = _apply_awq_quantization(
        model_name="gemma-4-sql",
        export_path=str(out_dir),
        calib_data=["SELECT 1;"],
        w_bit=4,
        q_group_size=128,
    )
    assert reduction == 0.75
    assert status == "quantized_awq"

    # Test with default calib_data and default export path
    _apply_awq_quantization(model_name="gemma-4-sql")

    # Test through top-level quantize_model
    res = quantize_model("gemma-4-sql", method="awq", export_path=str(out_dir))
    assert res["status"] == "quantized_awq"


def test_apply_awq_quantization_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _apply_awq_quantization raises DependencyMissingError when autoawq is absent."""
    import sys

    monkeypatch.setitem(sys.modules, "awq", None)
    with pytest.raises(DependencyMissingError, match="AutoAWQ is required for AWQ quantization"):
        _apply_awq_quantization("gemma-model")


def test_apply_gptq_quantization_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test _apply_gptq_quantization execution, quantization, and artifact serialization."""
    import sys

    class MockGPTQQuantizer:
        """Mock GPTQQuantizer from Optimum."""

        bits: int
        dataset: str

        def __init__(self, bits: int = 4, dataset: str = "c4", **kwargs: object) -> None:
            """Initialize mock quantizer."""
            self.bits = bits
            self.dataset = dataset

        def quantize_model(self, model: Any, _tokenizer: Any) -> Any:
            """Simulate model quantization."""
            return model

        def save(self, _model: Any, save_dir: str) -> None:
            """Simulate artifact serialization."""
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / "model.safetensors").write_bytes(b"dummy_weights")

    class MockTokenizer:
        """Mock AutoTokenizer."""

        def save_pretrained(self, _save_path: str) -> None:
            """Simulate saving tokenizer."""

        @classmethod
        def from_pretrained(cls, _name: str) -> MockTokenizer:
            """Simulate loading tokenizer."""
            return cls()

    class MockModel:
        """Mock AutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, _name: str, **_kwargs: object) -> MockModel:
            """Simulate loading causal LM."""
            return cls()

    mock_optimum_gptq = type("MockOptimumGPTQ", (), {"GPTQQuantizer": MockGPTQQuantizer})
    mock_optimum = type("MockOptimum", (), {"gptq": mock_optimum_gptq})
    monkeypatch.setitem(sys.modules, "optimum", mock_optimum)
    monkeypatch.setitem(sys.modules, "optimum.gptq", mock_optimum_gptq)

    mock_transformers = type(
        "MockTransformers",
        (),
        {"AutoModelForCausalLM": MockModel, "AutoTokenizer": MockTokenizer},
    )
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    out_dir = tmp_path / "gptq_export"
    reduction, status = _apply_gptq_quantization(
        model_name="gemma-4-sql",
        export_path=str(out_dir),
        bits=4,
        dataset="c4",
        group_size=128,
        damp_percent=0.01,
    )
    assert reduction == 0.75
    assert status == "quantized_gptq"
    assert (out_dir / "model.safetensors").exists()

    # Test with default export_path
    _apply_gptq_quantization(model_name="gemma-4-sql")

    # Test through top-level quantize_model
    res = quantize_model("gemma-4-sql", method="gptq", export_path=str(out_dir))
    assert res["status"] == "quantized_gptq"


def test_apply_gptq_quantization_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _apply_gptq_quantization raises DependencyMissingError when optimum is absent."""
    import sys

    monkeypatch.setitem(sys.modules, "optimum", None)
    monkeypatch.setitem(sys.modules, "optimum.gptq", None)
    with pytest.raises(DependencyMissingError, match="Optimum is required for GPTQ quantization"):
        _apply_gptq_quantization("gemma-model")

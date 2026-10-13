"""Unit and integration tests for PyTorch multimodal ETL and inference pipelines."""

from __future__ import annotations

from tests.conftest import requires_torch

pytestmark = requires_torch

from collections import UserDict
from pathlib import Path
from typing import Any

import pytest
import torch

from gemma_4_sql.backends.pytorch.inference import generate_sql
from gemma_4_sql.tokenization import SQLTokenizer


def test_pytorch_etl_multimodal_extraction(tmp_path: Path) -> None:
    """Docstring for test_pytorch_etl_multimodal_extraction."""
    import importlib

    from gemma_4_sql.backends.pytorch import etl

    importlib.reload(etl)
    _get_pytorch_classes = etl._get_pytorch_classes

    import torch

    etl.torch = torch
    """Test PyTorch dataset extraction of image and audio items."""

    import pytest

    monkeypatch_fix = pytest.MonkeyPatch()

    class _MockOpenedImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode):
            return self

        def resize(self, size, resample):
            return self

        def getpixel(self, coords):
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        class Resampling:
            BILINEAR = 2

        @staticmethod
        def open(buf):
            return _MockOpenedImage()

    monkeypatch_fix.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)

    import torch

    dummy_img = tmp_path / "diagram.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x8c!\x00\x00\x00\x00IEND\xaeB`\x82")

    class _MockPILResampling:
        BILINEAR = 2

    class _MockOpenedImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode):
            return self

        def resize(self, size, resample):
            return self

        def getpixel(self, coords):
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        Resampling = _MockPILResampling

        @staticmethod
        def open(buf):
            return _MockOpenedImage()

    monkeypatch_fix = pytest.MonkeyPatch()
    monkeypatch_fix.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)
    dummy_audio = tmp_path / "query.wav"
    dummy_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    mock_records = [
        {
            "question": "What is the average price?",
            "query": "SELECT AVG(price) FROM products",
            "image_url": str(dummy_img),
            "audio_clip": str(dummy_audio),
        },
        {
            "question": "Count orders",
            "query": "SELECT COUNT(*) FROM orders",
        },
    ]

    tokenizer = SQLTokenizer(model_name=None)
    pt_dataset_cls = _get_pytorch_classes()
    ds = pt_dataset_cls(list(mock_records), tokenizer)

    assert len(ds) == 2
    item0 = ds[0]
    assert "inputs" in item0
    assert "targets" in item0
    assert "pixel_values" in item0
    assert "audio_values" in item0
    assert isinstance(item0["pixel_values"], torch.Tensor)
    assert isinstance(item0["audio_values"], torch.Tensor)

    # Item without multimodal attachments
    item1 = ds[1]
    assert "inputs" in item1
    assert "pixel_values" not in item1
    assert "audio_values" not in item1


def test_pytorch_collate_multimodal() -> None:
    """Docstring for test_pytorch_collate_multimodal."""
    import importlib

    from gemma_4_sql.backends.pytorch import etl

    importlib.reload(etl)
    _collate_fn = etl._collate_fn

    import torch

    etl.torch = torch
    """Test _collate_fn stacking pixel and audio tensors when present in batch."""
    import torch

    batch: list[dict[str, Any]] = [
        {
            "inputs": torch.tensor([1, 2, 3], dtype=torch.long),
            "targets": torch.tensor([4, 5], dtype=torch.long),
            "pixel_values": torch.zeros((3, 224, 224), dtype=torch.float32),
            "audio_values": torch.zeros((1600,), dtype=torch.float32),
        },
        {
            "inputs": torch.tensor([1, 2], dtype=torch.long),
            "targets": torch.tensor([4], dtype=torch.long),
            "pixel_values": torch.ones((3, 224, 224), dtype=torch.float32),
            "audio_values": torch.ones((1600,), dtype=torch.float32),
        },
    ]

    collated = _collate_fn(batch)
    assert collated["inputs"].shape == (2, 3)
    assert collated["targets"].shape == (2, 2)
    assert "pixel_values" in collated
    assert collated["pixel_values"].shape == (2, 3, 224, 224)
    assert "audio_values" in collated
    assert collated["audio_values"].shape == (2, 1600)


def test_pytorch_inference_multimodal_native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_pytorch_inference_multimodal_native."""
    import importlib

    import gemma_4_sql.backends.pytorch.inference as inf

    importlib.reload(inf)
    generate_sql = inf.generate_sql

    import torch

    """Test PyTorch native inference accepting multimodal image and audio tensors."""

    import pytest

    monkeypatch_fix = pytest.MonkeyPatch()

    class _MockOpenedImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode):
            return self

        def resize(self, size, resample):
            return self

        def getpixel(self, coords):
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        class Resampling:
            BILINEAR = 2

        @staticmethod
        def open(buf):
            return _MockOpenedImage()

    monkeypatch_fix.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)

    dummy_img = tmp_path / "schema.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x8c!\x00\x00\x00\x00IEND\xaeB`\x82")

    class _MockPILResampling:
        BILINEAR = 2

    class _MockOpenedImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode):
            return self

        def resize(self, size, resample):
            return self

        def getpixel(self, coords):
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        Resampling = _MockPILResampling

        @staticmethod
        def open(buf):
            return _MockOpenedImage()

    monkeypatch_fix = pytest.MonkeyPatch()
    monkeypatch_fix.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)
    dummy_wav = tmp_path / "voice.wav"
    dummy_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    passed_kwargs: dict[str, Any] = {}

    class MockNativeModel:
        """Mock native Gemma4 causal model."""

        def eval(self) -> None:
            """Set eval mode."""

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> MockNativeModel:
            """Instantiate mock model."""
            return cls()

        def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Simulate generation and capture kwargs."""
            passed_kwargs.update(kwargs)
            # Return input_ids followed by generated token IDs [10, 20, 30]
            gen = torch.tensor([[10, 20, 30]], dtype=torch.long)
            return torch.cat([input_ids, gen], dim=-1)

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", MockNativeModel)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", None)

    res = generate_sql(
        "gemma-4",
        "List all customers",
        backend_alias="pytorch_native",
        image_path=str(dummy_img),
        modality="vision",
    )
    assert res["status"] == "success"
    assert "pixel_values" in passed_kwargs

    assert isinstance(passed_kwargs["pixel_values"], torch.Tensor)


def test_pytorch_inference_multimodal_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_pytorch_inference_multimodal_hf."""
    import torch

    """Test PyTorch Hugging Face inference forwarding multimodal tensors to generate."""

    import pytest

    monkeypatch_fix = pytest.MonkeyPatch()

    class _MockOpenedImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode):
            return self

        def resize(self, size, resample):
            return self

        def getpixel(self, coords):
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        class Resampling:
            BILINEAR = 2

        @staticmethod
        def open(buf):
            return _MockOpenedImage()

    monkeypatch_fix.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)

    dummy_img = tmp_path / "schema.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x8c!\x00\x00\x00\x00IEND\xaeB`\x82")

    class _MockPILResampling:
        BILINEAR = 2

    class _MockOpenedImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode):
            return self

        def resize(self, size, resample):
            return self

        def getpixel(self, coords):
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        Resampling = _MockPILResampling

        @staticmethod
        def open(buf):
            return _MockOpenedImage()

    monkeypatch_fix = pytest.MonkeyPatch()
    monkeypatch_fix.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)

    passed_extra: dict[str, Any] = {}

    class MockHFModel:
        """Mock Hugging Face model."""

        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> MockHFModel:
            """Instantiate mock model."""
            return cls()

        def generate(self, **kwargs: Any) -> Any:
            """Capture generate call."""
            passed_extra.update(kwargs)

            class MockOutput:
                """Docstring for MockOutput."""

                def __init__(self) -> None:
                    """Docstring for __init__."""
                    self.sequences = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
                    self.sequences_scores = [torch.tensor(0.9)]

            return MockOutput()

    class MockHFTokenizer:
        """Mock Hugging Face tokenizer."""

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> MockHFTokenizer:
            """Instantiate mock tokenizer."""
            return cls()

        def __call__(self, text: str, **kwargs: Any) -> Any:
            """Simulate tokenization."""

            class MockInputs(UserDict):
                """Docstring for MockInputs."""

                def __init__(self) -> None:
                    """Docstring for __init__."""
                    super().__init__({"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})
                    self.input_ids = self["input_ids"]

                def to(self, device: Any) -> MockInputs:
                    """Docstring for to."""
                    return self

            return MockInputs()

        def decode(self, tokens: Any, **kwargs: Any) -> str:
            """Simulate decoding."""
            return "SELECT * FROM hf_table"

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockHFModel)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockHFTokenizer)

    res = generate_sql(
        "mock-hf-model",
        "Select records",
        backend_alias="pytorch",
        image_path=str(dummy_img),
        modality="vision",
    )
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM hf_table"
    assert "pixel_values" in passed_extra

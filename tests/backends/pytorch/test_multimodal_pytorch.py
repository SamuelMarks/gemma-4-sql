"""Unit and integration tests for PyTorch multimodal ETL and inference pipelines."""

from __future__ import annotations

from collections import UserDict
from pathlib import Path
from typing import Any

import pytest
import torch

from gemma_4_sql.backends.pytorch.inference import generate_sql
from gemma_4_sql.tokenization import SQLTokenizer
from tests.conftest import MockPILImage, requires_torch

pytestmark = requires_torch


def test_pytorch_etl_multimodal_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
    synthetic_wav_bytes: bytes,
) -> None:
    """Test PyTorch dataset extraction of image and audio items."""
    from gemma_4_sql.backends.pytorch import etl

    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", MockPILImage)

    dummy_img = tmp_path / "diagram.png"
    dummy_img.write_bytes(synthetic_image_bytes)
    dummy_audio = tmp_path / "query.wav"
    dummy_audio.write_bytes(synthetic_wav_bytes)

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
    pt_dataset_cls = etl._get_pytorch_classes()
    ds = pt_dataset_cls(list(mock_records), tokenizer)

    assert len(ds) == 2
    item0 = ds[0]
    assert "inputs" in item0
    assert "targets" in item0
    assert "pixel_values" in item0
    assert "audio_values" in item0
    assert isinstance(item0["pixel_values"], torch.Tensor)
    assert isinstance(item0["audio_values"], torch.Tensor)

    item1 = ds[1]
    assert "inputs" in item1
    assert "pixel_values" not in item1
    assert "audio_values" not in item1


def test_pytorch_collate_multimodal() -> None:
    """Test _collate_fn stacking pixel and audio tensors when present in batch."""
    from gemma_4_sql.backends.pytorch.etl import _collate_fn

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
    assert collated["pixel_values"].shape == (2, 3, 224, 224)
    assert collated["audio_values"].shape == (2, 1600)


def test_pytorch_inference_multimodal_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
    synthetic_wav_bytes: bytes,
) -> None:
    """Test PyTorch native inference accepting multimodal image and audio tensors."""
    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", MockPILImage)

    dummy_img = tmp_path / "schema.png"
    dummy_img.write_bytes(synthetic_image_bytes)
    dummy_wav = tmp_path / "voice.wav"
    dummy_wav.write_bytes(synthetic_wav_bytes)

    passed_kwargs: dict[str, Any] = {}

    class MockNativeModel:
        """Mock native Gemma4 causal model."""

        def eval(self) -> None:
            """Set eval mode."""

        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> MockNativeModel:
            """Instantiate mock model."""
            return cls()

        def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Simulate generation and capture kwargs."""
            passed_kwargs.update(kwargs)
            gen = torch.tensor([[65, 66, 67]], dtype=torch.long)
            return torch.cat([input_ids, gen], dim=-1)

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Return vocabulary logits for confidence score computation."""
            return torch.ones((1, input_ids.shape[-1], 256), dtype=torch.float32)

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", MockNativeModel)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", None)

    res = generate_sql(
        "gemma-4",
        "List all customers",
        backend_alias="pytorch_native",
        image_path=str(dummy_img),
        audio_path=str(dummy_wav),
        modality="multimodal",
    )
    assert res["status"] == "success"
    assert isinstance(passed_kwargs["pixel_values"], torch.Tensor)
    assert isinstance(passed_kwargs["audio_values"], torch.Tensor)
    assert 0.0 < res["confidence_score"] <= 1.0


def test_pytorch_inference_multimodal_hf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
    synthetic_wav_bytes: bytes,
) -> None:
    """Test PyTorch Hugging Face inference forwarding multimodal tensors to generate."""
    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", MockPILImage)

    dummy_img = tmp_path / "schema.png"
    dummy_img.write_bytes(synthetic_image_bytes)
    dummy_wav = tmp_path / "voice.wav"
    dummy_wav.write_bytes(synthetic_wav_bytes)

    class MockTokOut(UserDict[str, Any]):
        """Mock tokenizer output supporting .to() and .input_ids."""

        def __init__(self) -> None:
            """Initialize MockTokOut."""
            super().__init__({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})
            self.input_ids = self["input_ids"]

        def to(self, _device: Any) -> MockTokOut:
            """Return self."""
            return self

    class MockTokenizer:
        """Mock AutoTokenizer."""

        @classmethod
        def from_pretrained(cls, _name: str) -> MockTokenizer:
            """Return MockTokenizer."""
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> MockTokOut:
            """Tokenize prompt."""
            _ = return_tensors
            return MockTokOut()

        def decode(self, _ids: Any, skip_special_tokens: bool = True) -> str:
            """Decode token IDs."""
            _ = skip_special_tokens
            return "SELECT * FROM customers;"

    captured_gen_kwargs: dict[str, Any] = {}

    class MockHFModel:
        """Mock HuggingFace AutoModelForCausalLM."""

        device = "cpu"

        @classmethod
        def from_pretrained(cls, _name: str, **_kwargs: Any) -> MockHFModel:
            """Return MockHFModel."""
            return cls()

        def generate(self, **kwargs: Any) -> Any:
            """Record kwargs and return sequences and scores."""
            captured_gen_kwargs.update(kwargs)

            class GenOut:
                """Mock generation output."""

                sequences = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
                scores = (torch.ones((1, 10)), torch.ones((1, 10)))

            return GenOut()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockHFModel)

    res = generate_sql(
        "gemma-4",
        "List all customers",
        backend_alias="pytorch",
        image_path=str(dummy_img),
        audio_path=str(dummy_wav),
        modality="multimodal",
    )
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM customers;"
    assert "pixel_values" in captured_gen_kwargs
    assert "audio_values" in captured_gen_kwargs

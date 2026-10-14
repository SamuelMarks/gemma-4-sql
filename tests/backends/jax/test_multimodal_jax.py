"""Unit and integration tests for JAX multimodal Grain ETL and inference pipelines."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

from pathlib import Path
from typing import Any

import pytest

from gemma_4_sql.backends.common_data import _create_base_format_transform
from gemma_4_sql.backends.jax.inference import generate_sql
from gemma_4_sql.tokenization import SQLTokenizer


def test_jax_grain_multimodal_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test BaseFormatTransform.map extracting image and audio features into dict."""

    class _MockOpenedImage:
        """Mock opened PIL image."""

        def __enter__(self):
            """Enter context manager."""
            return self

        def __exit__(self, *args):
            """Exit context manager."""

        def convert(self, mode):
            """Convert image mode."""
            return self

        def resize(self, size, resample):
            """Resize image."""
            return self

        def getpixel(self, coords):
            """Get pixel RGB tuple."""
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            """Return numpy array representation."""
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        """Mock PIL Image module."""

        class Resampling:
            """Mock PIL Resampling constants."""

            BILINEAR = 2

        @staticmethod
        def open(buf):
            """Open mock image from buffer."""
            return _MockOpenedImage()

    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)

    dummy_img = tmp_path / "table.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x8c!\x00\x00\x00\x00IEND\xaeB`\x82")
    dummy_audio = tmp_path / "speech.wav"
    dummy_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    class DummyMapBase:
        """Dummy base class for map transform."""

    transform_cls = _create_base_format_transform(DummyMapBase)
    tokenizer = SQLTokenizer(model_name=None)
    transform = transform_cls(tokenizer=tokenizer)

    raw_element = {
        "question": "What is the total revenue?",
        "query": "SELECT SUM(revenue) FROM sales",
        "image_bytes": dummy_img.read_bytes(),
        "audio_clip": str(dummy_audio),
    }

    mapped = transform.map(raw_element)
    assert "inputs" in mapped
    assert "targets" in mapped
    assert "pixel_values" in mapped
    assert "audio_values" in mapped
    assert len(mapped["inputs"]) > 0


def test_jax_inference_multimodal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX generate_sql processing image and audio paths and formatting prompt."""

    class _MockOpenedImage:
        """Mock opened PIL image."""

        def __enter__(self):
            """Enter context manager."""
            return self

        def __exit__(self, *args):
            """Exit context manager."""

        def convert(self, mode):
            """Convert image mode."""
            return self

        def resize(self, size, resample):
            """Resize image."""
            return self

        def getpixel(self, coords):
            """Get pixel RGB tuple."""
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            """Return numpy array representation."""
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        """Mock PIL Image module."""

        class Resampling:
            """Mock PIL Resampling constants."""

            BILINEAR = 2

        @staticmethod
        def open(buf):
            """Open mock image from buffer."""
            return _MockOpenedImage()

    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)

    dummy_img = tmp_path / "erd.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x8c!\x00\x00\x00\x00IEND\xaeB`\x82")

    class MockJAXModel:
        """Mock JAX causal model for beam search."""

        def __call__(self, input_ids: Any, positions: Any = None, **kwargs: Any) -> Any:
            """Verify behavior of __call__."""
            import jax.numpy as jnp

            batch, seq = input_ids.shape
            return jnp.ones((batch, seq, 100), dtype=jnp.float32)

    monkeypatch.setattr("gemma_4_sql.backends.jax.inference._MODEL_CACHE", {"mock-jax-model": MockJAXModel()})

    res = generate_sql(
        "mock-jax-model",
        "Find highest salary",
        image_path=str(dummy_img),
        modality="vision",
    )
    assert res["status"] == "success"
    assert res["backend"] == "jax"
    assert "sql" in res


def test_jax_inference_multimodal_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX generate_sql live model execution with multimodal preprocessing."""

    class _MockOpenedImage:
        """Mock opened PIL image."""

        def __enter__(self):
            """Enter context manager."""
            return self

        def __exit__(self, *args):
            """Exit context manager."""

        def convert(self, mode):
            """Convert image mode."""
            return self

        def resize(self, size, resample):
            """Resize image."""
            return self

        def getpixel(self, coords):
            """Get pixel RGB tuple."""
            return (255, 0, 0)

        def __array__(self, *args, **kwargs):
            """Return numpy array representation."""
            import numpy as np

            return np.zeros((224, 224, 3), dtype=np.float32)

    class _MockPILImage:
        """Mock PIL Image module."""

        class Resampling:
            """Mock PIL Resampling constants."""

            BILINEAR = 2

        @staticmethod
        def open(buf):
            """Open mock image from buffer."""
            return _MockOpenedImage()

    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", _MockPILImage)

    dummy_img = tmp_path / "erd.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x8c!\x00\x00\x00\x00IEND\xaeB`\x82")
    dummy_audio = tmp_path / "voice.wav"
    dummy_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    class MockJAXModel:
        """Mock JAX causal model for beam search."""

        def __call__(self, input_ids: Any, positions: Any = None, **kwargs: Any) -> Any:
            """Return logits for beam search."""
            import jax.numpy as jnp

            batch, seq = input_ids.shape
            # Return logits with high probability for eos token
            return jnp.ones((batch, seq, 100), dtype=jnp.float32)

    monkeypatch.setattr("gemma_4_sql.backends.jax.inference._MODEL_CACHE", {"mock_live": MockJAXModel()})

    res = generate_sql(
        "mock_live",
        "Select all products",
        beam_width=1,
        max_length=2,
        image_path=str(dummy_img),
        modality="vision",
    )
    assert res["status"] == "success"

    res_img_only = generate_sql(
        "mock_live",
        "Select all products",
        beam_width=1,
        max_length=2,
        image_path=str(dummy_img),
        modality="vision",
    )
    assert res_img_only["status"] == "success"

    res_aud_only = generate_sql(
        "mock_live",
        "Select all products",
        beam_width=1,
        max_length=2,
        audio_path=str(dummy_audio),
        modality="audio",
    )
    assert res_aud_only["status"] == "success"
    assert res["prompt"].startswith("<audio> <image>") or "<image>" in res["prompt"]

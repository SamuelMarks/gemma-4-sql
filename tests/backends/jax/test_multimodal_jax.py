"""Unit and integration tests for JAX multimodal Grain ETL and inference pipelines."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.common_data import _create_base_format_transform
from gemma_4_sql.backends.jax.inference import generate_sql
from gemma_4_sql.tokenization import SQLTokenizer
from tests.conftest import MockPILImage, requires_jax

pytestmark = requires_jax


def test_jax_grain_multimodal_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
    synthetic_wav_bytes: bytes,
) -> None:
    """Test BaseFormatTransform.map extracting image and audio features into dict."""
    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", MockPILImage)

    dummy_img = tmp_path / "table.png"
    dummy_img.write_bytes(synthetic_image_bytes)
    dummy_audio = tmp_path / "speech.wav"
    dummy_audio.write_bytes(synthetic_wav_bytes)

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
    inputs_seq = mapped["inputs"]
    assert isinstance(inputs_seq, list) and len(inputs_seq) > 0


def test_jax_inference_multimodal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
) -> None:
    """Test JAX generate_sql processing image and audio paths and formatting prompt."""
    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", MockPILImage)

    dummy_img = tmp_path / "erd.png"
    dummy_img.write_bytes(synthetic_image_bytes)

    class MockJAXModel:
        """Mock JAX causal model for beam search."""

        def __call__(
            self,
            input_ids: jax.Array,
            _positions: jax.Array | None = None,
            **_kwargs: jax.Array | str | float | bool | None,
        ) -> jax.Array:
            """Execute callable test stub."""
            batch, seq = input_ids.shape
            return jnp.ones((batch, seq, 100), dtype=jnp.float32)

    import gemma_4_sql.backends.jax.inference as jax_inf

    monkeypatch.setitem(jax_inf._MODEL_CACHE, "mock-jax-model", MockJAXModel())

    res = generate_sql(
        "mock-jax-model",
        "Find highest salary",
        image_path=str(dummy_img),
        modality="vision",
    )
    assert res["status"] == "success"
    assert res["backend"] == "jax"
    assert "sql" in res


def test_jax_inference_multimodal_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
    synthetic_wav_bytes: bytes,
) -> None:
    """Test JAX generate_sql live model execution with multimodal preprocessing."""
    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.Image", MockPILImage)

    dummy_img = tmp_path / "erd.png"
    dummy_img.write_bytes(synthetic_image_bytes)
    dummy_audio = tmp_path / "voice.wav"
    dummy_audio.write_bytes(synthetic_wav_bytes)

    class MockJAXModel:
        """Mock JAX causal model for beam search."""

        def __call__(
            self,
            input_ids: jax.Array,
            _positions: jax.Array | None = None,
            **_kwargs: jax.Array | str | float | bool | None,
        ) -> jax.Array:
            """Return logits for beam search."""
            batch, seq = input_ids.shape
            return jnp.ones((batch, seq, 100), dtype=jnp.float32)

    import gemma_4_sql.backends.jax.inference as jax_inf

    monkeypatch.setitem(jax_inf._MODEL_CACHE, "mock_live", MockJAXModel())

    res = generate_sql(
        "mock_live",
        "Select all products",
        beam_width=1,
        max_length=2,
        image_path=str(dummy_img),
        modality="vision",
    )
    assert res["status"] == "success"

    res_aud_only = generate_sql(
        "mock_live",
        "Select all products",
        beam_width=1,
        max_length=2,
        audio_path=str(dummy_audio),
        modality="audio",
    )
    assert res_aud_only["status"] == "success"
    assert "<image>" in res["prompt"]

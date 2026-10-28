"""Behavioral unit tests for the SDK Inference module across backends and modalities."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_4_sql.exceptions import ConfigurationError
from gemma_4_sql.sdk.inference import generate, generate_sql
from gemma_4_sql.sdk.registry import get_backend
from tests.utils.test_utils import RecordingBackendDouble


@pytest.mark.parametrize("backend_name", ["jax", "keras", "maxtext", "pytorch", "mlx"])
def test_generate_routes_to_real_registered_backend(
    backend_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `generate` resolves the real registered backend adapter and forwards all arguments."""
    adapter = get_backend(backend_name)
    double = RecordingBackendDouble(backend_name)
    monkeypatch.setattr(adapter, "generate_sql", double.generate_sql)

    res = generate(
        "gemma-4-E2B",
        "Find all active users",
        backend=backend_name,
        beam_width=5,
        max_length=128,
        image_path=Path("/tmp/erd.png"),
        audio_path=Path("/tmp/voice.wav"),
        modality="multimodal",
        temperature=0.2,
    )

    assert res["backend"] == backend_name
    assert res["sql"] == f"SELECT * FROM {backend_name}_table;"
    assert len(double.generate_calls) == 1
    model_name, prompt, beam_width, max_length, kw = double.generate_calls[0]
    assert model_name == "gemma-4-E2B"
    assert prompt == "Find all active users"
    assert beam_width == 5
    assert max_length == 128
    assert kw["image_path"] == "/tmp/erd.png"
    assert kw["audio_path"] == "/tmp/voice.wav"
    assert kw["modality"] == "multimodal"
    assert kw["temperature"] == 0.2
    assert generate_sql is generate


def test_generate_invalid() -> None:
    """Verify `generate` raises `ConfigurationError` (a `ValueError` subclass) for an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: invalid"):
        generate("model1", "Find all users", "invalid")

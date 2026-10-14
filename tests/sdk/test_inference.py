"""Behavioral unit tests for the SDK Inference module across backends and modalities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gemma_4_sql.exceptions import ConfigurationError
from gemma_4_sql.sdk.inference import generate, generate_sql
from gemma_4_sql.sdk.registry import get_backend


@pytest.mark.parametrize("backend_name", ["jax", "keras", "maxtext", "pytorch", "mlx"])
def test_generate_routes_to_real_registered_backend(
    backend_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `generate` resolves the real registered backend adapter and forwards all arguments."""
    adapter = get_backend(backend_name)
    captured: dict[str, Any] = {}

    def _spy_generate_sql(
        model_name: str,
        prompt: str,
        beam_width: int = 3,
        max_length: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record arguments passed to adapter.generate_sql."""
        captured["model_name"] = model_name
        captured["prompt"] = prompt
        captured["beam_width"] = beam_width
        captured["max_length"] = max_length
        captured["kwargs"] = kwargs
        return {
            "backend": backend_name,
            "model": model_name,
            "prompt": prompt,
            "sql": f"SELECT * FROM {backend_name}_table",
            "status": "success",
            "beam_width": beam_width,
            "confidence_score": 0.95,
        }

    monkeypatch.setattr(adapter, "generate_sql", _spy_generate_sql)

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
    assert res["sql"] == f"SELECT * FROM {backend_name}_table"
    assert captured["model_name"] == "gemma-4-E2B"
    assert captured["prompt"] == "Find all active users"
    assert captured["beam_width"] == 5
    assert captured["max_length"] == 128
    assert captured["kwargs"]["image_path"] == "/tmp/erd.png"
    assert captured["kwargs"]["audio_path"] == "/tmp/voice.wav"
    assert captured["kwargs"]["modality"] == "multimodal"
    assert captured["kwargs"]["temperature"] == 0.2
    assert generate_sql is generate


def test_generate_invalid() -> None:
    """Verify `generate` raises `ConfigurationError` (a `ValueError` subclass) for an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: invalid"):
        generate("model1", "Find all users", "invalid")

"""Tests for maxtext serve."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.maxtext.serve as serve_module
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_maxtext_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext serve missing deps."""
    monkeypatch.setattr(serve_module, "jax", None)
    monkeypatch.setattr(serve_module, "gemma4", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        serve_module.serve_model("model")


def test_maxtext_serve_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext serve success."""
    mock_jax = MagicMock()
    monkeypatch.setattr(serve_module, "jax", mock_jax)
    monkeypatch.setattr(serve_module, "gemma4", MagicMock())

    res = serve_module.serve_model("model", port=8090, max_batch_size=64)
    assert res["backend"] == "maxtext"
    assert res["port"] == 8090
    assert res["max_batch_size"] == 64
    assert res["status"] == "running_maxtext_serve"


def test_maxtext_create_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext create app."""
    mock_jax = MagicMock()
    monkeypatch.setattr(serve_module, "jax", mock_jax)
    monkeypatch.setattr(serve_module, "gemma4", MagicMock())

    with patch("gemma_4_sql.backends.common_serve.create_common_app") as mock_create:
        serve_module._create_app("model")
        mock_create.assert_called_once()
        kwargs = mock_create.call_args[1]

        startup = kwargs["startup_callback"]
        startup()
        mock_jax.distributed.initialize.assert_called_once()

        mock_jax.distributed.initialize.side_effect = RuntimeError("error")
        startup()

        generate = kwargs["generate_logic"]

        with patch("gemma_4_sql.backends.maxtext.inference.generate_sql") as mock_generate_sql:
            mock_generate_sql.return_value = {"sql": "SELECT 1"}
            assert generate("prompt") == "SELECT 1"

            mock_generate_sql.return_value = {"sql": ""}
            with pytest.raises(InferenceError, match="returned empty SQL"):
                generate("prompt")

            mock_generate_sql.side_effect = RuntimeError("test error")
            with pytest.raises(InferenceError, match="MaxText generation failed"):
                generate("prompt")

            mock_generate_sql.side_effect = InferenceError("test error")
            with pytest.raises(InferenceError, match="test error"):
                generate("prompt")

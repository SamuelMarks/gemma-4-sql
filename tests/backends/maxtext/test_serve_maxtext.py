"""Tests for maxtext serve."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_maxtext_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext serve missing deps."""
    import gemma_4_sql.backends.maxtext.serve as serve_module

    monkeypatch.setattr(serve_module, "jax", None)
    monkeypatch.setattr(serve_module, "gemma4", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        serve_module.serve_model("model")


def test_maxtext_serve_success():
    """Test maxtext serve success."""
    import gemma_4_sql.backends.maxtext.serve as serve_module

    mock_jax = MagicMock()
    serve_module.jax = mock_jax
    serve_module.gemma4 = MagicMock()

    with patch("gemma_4_sql.backends.maxtext.serve.serve_model_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "ok"}
        result = serve_module.serve_model("model")
        assert result == {"status": "ok"}

        # Test app factory
        app_factory = mock_wrapper.call_args[1]["app_factory"]
        app_factory()


def test_maxtext_create_app():
    """Test maxtext create app."""
    import gemma_4_sql.backends.maxtext.serve as serve_module

    mock_jax = MagicMock()
    serve_module.jax = mock_jax
    serve_module.gemma4 = MagicMock()

    with patch("gemma_4_sql.backends.maxtext.serve.create_common_app") as mock_create:
        serve_module._create_app("model")
        mock_create.assert_called_once()
        kwargs = mock_create.call_args[1]

        # Test startup
        startup = kwargs["startup_callback"]
        startup()
        mock_jax.distributed.initialize.assert_called_once()

        # Test startup exception
        mock_jax.distributed.initialize.side_effect = RuntimeError("error")
        startup()  # Should handle exception and log warning

        # Test generate
        generate = kwargs["generate_logic"]

        with patch("gemma_4_sql.backends.maxtext.inference.generate_sql") as mock_generate_sql:
            mock_generate_sql.return_value = {"sql": "SELECT 1"}
            assert generate("prompt") == "SELECT 1"

            # Test empty SQL
            mock_generate_sql.return_value = {"sql": ""}
            with pytest.raises(InferenceError, match="returned empty SQL"):
                generate("prompt")

            # Test exception
            mock_generate_sql.side_effect = Exception("test error")
            with pytest.raises(InferenceError, match="MaxText generation failed"):
                generate("prompt")

            # Test raise InferenceError
            mock_generate_sql.side_effect = InferenceError("test error")
            with pytest.raises(InferenceError, match="test error"):
                generate("prompt")

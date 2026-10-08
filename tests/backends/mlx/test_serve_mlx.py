"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.mlx.serve import (
    _app_factory,
    _batch_generate_queries,
    _generate_query,
    _load_mlx_model,
    serve_model,
)
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_load_mlx_model_cache():
    """Docstring for test_load_mlx_model_cache."""
    with patch("gemma_4_sql.backends.mlx.serve._mlx_model_cache", {"test_model": ("model", "tok")}):
        model, tok = _load_mlx_model("test_model")
        assert model == "model"
        assert tok == "tok"


def test_load_mlx_model_success_tuple():
    """Docstring for test_load_mlx_model_success_tuple."""
    mock_load = MagicMock()
    mock_load.return_value = ["model", "tok"]
    with patch.dict("sys.modules", {"mlx_lm": MagicMock(load=mock_load)}):
        model, tok = _load_mlx_model("new_model")
        assert model == "model"
        assert tok == "tok"


def test_load_mlx_model_success_single():
    """Docstring for test_load_mlx_model_success_single."""
    mock_load = MagicMock()
    mock_load.return_value = "model"
    with patch.dict("sys.modules", {"mlx_lm": MagicMock(load=mock_load)}):
        model, tok = _load_mlx_model("single_model")
        assert model == "model"
        assert tok is None


def test_load_mlx_model_error():
    """Docstring for test_load_mlx_model_error."""
    mock_load = MagicMock()
    mock_load.side_effect = RuntimeError("Failed")
    with patch.dict("sys.modules", {"mlx_lm": MagicMock(load=mock_load)}), pytest.raises(RuntimeError):
        _load_mlx_model("error_model")


def test_generate_query_success():
    """Docstring for test_generate_query_success."""
    with patch("gemma_4_sql.backends.mlx.inference.generate_sql", return_value={"sql": "SELECT 1;"}):
        assert _generate_query("test", "model") == "SELECT 1;"


def test_generate_query_empty():
    """Docstring for test_generate_query_empty."""
    with patch("gemma_4_sql.backends.mlx.inference.generate_sql", return_value={"sql": ""}), pytest.raises(InferenceError, match="returned empty SQL"):
        _generate_query("test", "model")


def test_generate_query_error():
    """Docstring for test_generate_query_error."""
    with patch("gemma_4_sql.backends.mlx.inference.generate_sql", side_effect=ValueError("Error")), pytest.raises(InferenceError, match="MLX generation failed"):
        _generate_query("test", "model")


def test_generate_query_inference_error_propagated():
    """Docstring for test_generate_query_inference_error_propagated."""
    with patch("gemma_4_sql.backends.mlx.inference.generate_sql", side_effect=InferenceError("Direct error")), pytest.raises(InferenceError, match="Direct error"):
        _generate_query("test", "model")


def test_batch_generate_queries():
    """Docstring for test_batch_generate_queries."""
    with patch("gemma_4_sql.backends.mlx.serve._generate_query", side_effect=["SELECT 1;", "SELECT 2;"]):
        assert _batch_generate_queries(["a", "b"], "model") == ["SELECT 1;", "SELECT 2;"]


def test_app_factory():
    """Docstring for test_app_factory."""
    with patch("gemma_4_sql.backends.mlx.serve.create_common_app") as mock_create:
        _app_factory("model")
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs

        # test startup callback
        startup = kwargs["startup_callback"]
        with patch("gemma_4_sql.backends.mlx.serve._load_mlx_model") as mock_load:
            # mx is None case
            with patch("gemma_4_sql.backends.mlx.serve.mx", None):
                startup()
                mock_load.assert_not_called()

            # mx is not None case
            with patch("gemma_4_sql.backends.mlx.serve.mx", MagicMock()):
                startup()
                mock_load.assert_called_once_with("model")

                # error during preload
                mock_load.side_effect = RuntimeError("error")
                startup()  # shouldn't raise

        # test generate logic
        gen = kwargs["generate_logic"]
        with patch("gemma_4_sql.backends.mlx.serve._generate_query") as mock_gen:
            gen("prompt")
            mock_gen.assert_called_once_with("prompt", model_name="model")

        # test batch generate logic
        bgen = kwargs["batch_generate_logic"]
        with patch("gemma_4_sql.backends.mlx.serve._batch_generate_queries") as mock_bgen:
            bgen(["prompt"])
            mock_bgen.assert_called_once_with(["prompt"], model_name="model")


def test_serve_model_missing_deps():
    """Docstring for test_serve_model_missing_deps."""
    with patch("gemma_4_sql.backends.mlx.serve.mx", None), pytest.raises(DependencyMissingError):
        serve_model("model")


def test_serve_model_success():
    """Docstring for test_serve_model_success."""
    with patch("gemma_4_sql.backends.mlx.serve.mx", MagicMock()), patch("gemma_4_sql.backends.mlx.serve.serve_model_wrapper", return_value={"status": "running_mlx_serve"}):
        res = serve_model("model")
        assert res["status"] == "running_mlx_serve"


def test_serve_model_not_running():
    """Docstring for test_serve_model_not_running."""
    with patch("gemma_4_sql.backends.mlx.serve.mx", MagicMock()), patch("gemma_4_sql.backends.mlx.serve.serve_model_wrapper", return_value={"status": "mocked"}):
        res = serve_model("model")
        assert res["status"] == "mocked"

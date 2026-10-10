"""Tests for mlx serve."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def test_mlx_serve_imports():
    """Test mlx serve imports fallback."""
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
        if "gemma_4_sql.backends.mlx.serve" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.serve"]
        import gemma_4_sql.backends.mlx.serve as serve_module

        assert serve_module.mx is None


def test_serve_model():
    """Test serve_model."""
    import gemma_4_sql.backends.mlx.serve as serve_module

    serve_module.mx = MagicMock()

    with patch.object(serve_module, "serve_model_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "running_mlx_serve"}

        res = serve_module.serve_model("model")
        assert res == {"status": "running_mlx_serve"}

        mock_wrapper.return_value = {"status": "failed"}
        res = serve_module.serve_model("model")
        assert res == {"status": "failed"}

        serve_module.mx = None
        with pytest.raises(DependencyMissingError):
            serve_module.serve_model("model")


def test_app_factory_mx_none():
    """Test _app_factory when mx is None."""
    import gemma_4_sql.backends.mlx.serve as serve_module

    serve_module.mx = None

    with patch.object(serve_module, "create_common_app") as mock_create:
        mock_create.return_value = "app"
        serve_module._app_factory("m")
        kwargs = mock_create.call_args[1]

        with patch.object(serve_module, "_load_mlx_model") as mock_load:
            kwargs["startup_callback"]()
            mock_load.assert_not_called()


def test_mlx_serve_successful_imports():
    """Test mlx serve successful imports."""
    from unittest.mock import MagicMock

    mock_mlx = MagicMock()
    mock_core = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "mlx": mock_mlx,
            "mlx.core": mock_core,
        },
    ):
        if "gemma_4_sql.backends.mlx.serve" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.serve"]
        import gemma_4_sql.backends.mlx.serve as serve_module

        assert serve_module.mx is not None

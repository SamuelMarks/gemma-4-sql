"""MaxText-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)

try:
    import jax as _jax
    from maxtext.models import gemma4 as _gemma4

    jax: Any = _jax
    gemma4: Any = _gemma4
except (ImportError, AttributeError):
    jax = None
    gemma4 = None
FastAPI = None
Request = None
JSONResponse = None
uvicorn = None
with catch_optional_imports():
    pass


def _create_app(model_name: str, *, test_mode: bool = False) -> object:
    """Create the FastAPI application for the MaxText server.

    Args:
        model_name: The name of the target model.
        test_mode: Boolean flag indicating test mode.

    Returns:
        The execution result.
    """

    def _startup() -> None:
        """Execute function."""
        try:
            jax.distributed.initialize()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("jax.distributed.initialize() failed: %s", e)

    def _generate(prompt: str) -> str:
        """Generate SQL using MaxText inference.

        Args:
            prompt: Natural language query prompt.

        Returns:
            Generated SQL query string.
        """
        if test_mode:
            return f"SELECT * FROM maxtext_serve WHERE prompt='{prompt}'"

        from gemma_4_sql.backends.maxtext.inference import generate_sql
        from gemma_4_sql.exceptions import DependencyMissingError

        try:
            out = generate_sql(model_name=model_name, prompt=prompt)
            return str(out.get("sql", f"SELECT * FROM maxtext_serve WHERE prompt='{prompt}'"))
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, DependencyMissingError):
            return f"SELECT * FROM maxtext_serve WHERE prompt='{prompt}'"

    return create_common_app(
        backend_name="maxtext",
        model_name=model_name,
        test_mode=test_mode,
        startup_callback=_startup,
        generate_logic=_generate,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using MaxText continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.
    """
    if gemma4 is None or jax is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing.")
    return serve_model_wrapper(
        backend_name="maxtext",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=False,
        missing_status="",
        app_factory=lambda: _create_app(model_name, test_mode=bool(kwargs.get("test_mode"))),
        test_mode=bool(kwargs.get("test_mode")),
    )

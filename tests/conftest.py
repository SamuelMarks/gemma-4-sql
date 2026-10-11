"""Global pytest fixtures for gemma-4-sql tests."""

import os
import sys
import warnings
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from gemma_4_sql.backends.lazy_loader import LazyLoader

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("KERAS_BACKEND", "torch")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import torch  # Preload to avoid PyTorch/coverage C-tracer segfault in Python 3.12
except (ImportError, RuntimeError):
    torch = None

HAS_TORCH: bool = LazyLoader("torch").is_available
HAS_JAX: bool = LazyLoader("jax").is_available and LazyLoader("flax").is_available
HAS_KERAS: bool = LazyLoader("keras").is_available
HAS_MAXTEXT: bool = LazyLoader("maxtext").is_available
HAS_MLX: bool = LazyLoader("mlx").is_available
HAS_DUCKDB: bool = LazyLoader("duckdb").is_available

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed in this environment")
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax is not installed in this environment")
requires_keras = pytest.mark.skipif(not HAS_KERAS, reason="Keras is not installed in this environment")
requires_maxtext = pytest.mark.skipif(not HAS_MAXTEXT, reason="MaxText is not installed in this environment")
requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed in this environment")
requires_duckdb = pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is not installed in this environment")


@pytest.fixture
def in_memory_duckdb() -> Generator[Any, None, None]:
    """Provide an isolated in-memory DuckDB instance with transaction auto-cleanup.

    Yields:
        DuckDB connection object.
    """
    import duckdb

    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
async def in_memory_sqlite() -> AsyncGenerator[Any, None]:
    """Provide an isolated in-memory SQLite / aiosqlite database.

    Yields:
        aiosqlite connection object.
    """
    import aiosqlite

    async with aiosqlite.connect(":memory:") as db:
        yield db


@pytest.fixture
def synthetic_dataset() -> list[dict[str, str]]:
    """Supply valid text-to-SQL pairs without depending on network calls.

    Returns:
        List of dictionaries with dataset items.
    """
    return [{"question": "What is the total?", "query": "SELECT sum(val) FROM table;", "sql": "SELECT sum(val) FROM table;", "nl": "What is the total?"}, {"question": "Count users", "query": "SELECT count(*) FROM users;", "sql": "SELECT count(*) FROM users;", "nl": "Count users"}]


@pytest.fixture
def missing_dependency_context() -> Any:
    """Opt-in context manager fixture for testing missing optional packages.

    Provides a scoped unittest.mock.patch.dict(sys.modules, ...) with automatic restoration.

    Returns:
        A context manager function to mock missing modules.
    """

    @contextmanager
    def _mock_missing(modules: list[str]) -> Generator[None, None, None]:
        mocks = dict.fromkeys(modules)
        with patch.dict(sys.modules, mocks):
            yield

    return _mock_missing


@pytest.fixture(autouse=True)
def _suppress_expected_errors(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Suppress expected errors during test execution."""
    yield

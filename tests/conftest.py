"""Global pytest fixtures and environment markers for gemma-4-sql tests."""

from __future__ import annotations

import io
import os
import sys
import types
import warnings
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Self
from unittest.mock import patch

import aiosqlite
import duckdb
import pytest
import pytest_asyncio

from gemma_4_sql.backends.lazy_loader import LazyLoader, optional_import

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("KERAS_BACKEND", "torch")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

torch = optional_import("torch")

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
def in_memory_duckdb() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Provide an isolated in-memory DuckDB instance with automatic cleanup.

    Yields:
        DuckDB connection object.

    """
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest_asyncio.fixture
async def in_memory_sqlite() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Provide an isolated in-memory SQLite / aiosqlite database.

    Yields:
        Active `aiosqlite` connection object.

    """
    async with aiosqlite.connect(":memory:") as db:
        yield db


@pytest.fixture
def synthetic_dataset() -> list[dict[str, str]]:
    """Supply valid text-to-SQL pairs without depending on network calls.

    Returns:
        List of dictionaries with dataset items.

    """
    return [
        {
            "question": "What is the total?",
            "query": "SELECT sum(val) FROM table;",
            "sql": "SELECT sum(val) FROM table;",
            "nl": "What is the total?",
        },
        {
            "question": "Count users",
            "query": "SELECT count(*) FROM users;",
            "sql": "SELECT count(*) FROM users;",
            "nl": "Count users",
        },
    ]


@pytest.fixture
def missing_dependency_context() -> Callable[[list[str]], AbstractContextManager[None]]:
    """Provide an opt-in context manager fixture for testing missing optional packages.

    Returns:
        A context manager callable that masks specified module names in `sys.modules`.

    """

    @contextmanager
    def _mock_missing(modules: list[str]) -> Generator[None, None, None]:
        """Temporarily set specified module names to None in sys.modules."""
        mocks = dict.fromkeys(modules)
        with patch.dict(sys.modules, mocks):
            yield

    return _mock_missing


class MockOpenedImage:
    """Shared mock opened PIL image for multimodal tests."""

    def __init__(self) -> None:
        """Initialize MockOpenedImage with default 224x224 dimensions."""
        self._size: tuple[int, int] = (224, 224)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_val: BaseException | None = None,
        _exc_tb: types.TracebackType | None = None,
    ) -> None:
        """Exit context manager."""

    def convert(self, _mode: str) -> MockOpenedImage:
        """Return self on mode conversion."""
        return self

    def resize(self, size: tuple[int, int] | Sequence[int], _resample: int | None = None) -> MockOpenedImage:
        """Record resized dimensions and return self."""
        self._size = (int(size[0]), int(size[1]))
        return self

    def tobytes(self) -> bytes:
        """Return raw RGB pixel bytes of length `height * width * 3`."""
        w, h = self._size
        return b"\x00" * (h * w * 3)


class MockPILImage:
    """Shared mock PIL.Image module for multimodal tests."""

    class Resampling:
        """Mock PIL Resampling constants."""

        BILINEAR = 2

    @staticmethod
    def open(_buf: str | Path | bytes | io.BytesIO) -> MockOpenedImage:
        """Return a new MockOpenedImage."""
        return MockOpenedImage()


@pytest.fixture
def synthetic_image_bytes() -> bytes:
    """Provide a minimal valid 1x1 PNG byte sequence."""
    return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63fccfc0500f000485018084a98c210000000049454e44ae426082")


@pytest.fixture
def synthetic_wav_bytes() -> bytes:
    """Provide a minimal valid 16-bit PCM RIFF WAVE byte sequence."""
    return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

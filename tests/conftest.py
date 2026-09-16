"""Module docstring."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import torch  # Preload to avoid PyTorch/coverage C-tracer segfault in Python 3.12
except (ImportError, RuntimeError):
    torch = None

import pytest

"""Global pytest fixtures for gemma-4-sql tests."""

import importlib.machinery
import json
import sys
import types
import typing
from unittest.mock import MagicMock

try:
    import duckdb as _real_duckdb
except ImportError:
    _real_duckdb = None

try:
    import aiosqlite as _real_aiosqlite
except ImportError:
    _real_aiosqlite = None

pytest._real_duckdb = _real_duckdb


class MockDatasets(types.ModuleType):
    """Mock for datasets module."""

    def __init__(self: object) -> None:
        """Initialize MockDatasets as a valid module with spec."""
        super().__init__("datasets")
        self.__spec__ = importlib.machinery.ModuleSpec("datasets", None)

    def load_dataset(self: object, *_args: object, **_kwargs: object) -> list[dict[str, str]]:
        """Mock load_dataset.

        Returns:
            object: Description of return.

        """
        return [{"query": "SELECT 1", "sql": "SELECT 1", "question": "test", "nl": "test"}]


class MockConftestGrain(types.ModuleType):
    """Mock grain for test environment."""

    def __init__(self) -> None:
        """Initialize MockConftestGrain as a valid module with spec."""
        super().__init__("grain")
        self.__spec__ = importlib.machinery.ModuleSpec("grain", None)
        self.python = self

    class RandomAccessDataSource:
        """Mock RandomAccessDataSource."""

    class MapTransform:
        """Mock MapTransform."""

    class DataLoader:
        """Mock DataLoader."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize Mock DataLoader."""
            self.data_source = args[0] if args else None

        def __iter__(self) -> typing.Iterator[dict[str, typing.Any]]:
            """Iterate mock batches."""
            try:
                import jax.numpy as jnp

                yield {
                    "inputs": jnp.ones((2, 10), dtype=jnp.int32),
                    "targets": jnp.ones((2, 10), dtype=jnp.int32),
                    "chosen_inputs": jnp.ones((2, 10), dtype=jnp.int32),
                    "chosen_input_ids": jnp.ones((2, 10), dtype=jnp.int32),
                    "chosen_labels": jnp.ones((2, 10), dtype=jnp.int32),
                    "rejected_inputs": jnp.ones((2, 10), dtype=jnp.int32),
                    "rejected_input_ids": jnp.ones((2, 10), dtype=jnp.int32),
                    "rejected_labels": jnp.ones((2, 10), dtype=jnp.int32),
                }
            except (ImportError, AttributeError):
                yield {"inputs": [1, 2], "targets": [2, 3]}

    @staticmethod
    def IndexSampler(*args: object, **kwargs: object) -> object:
        """Mock IndexSampler."""
        return object()

    @staticmethod
    def Batch(*args: object, **kwargs: object) -> object:
        """Mock Batch."""
        return object()

    @staticmethod
    def NoSharding() -> object:
        """Mock NoSharding."""
        return object()

    @staticmethod
    def JAXDistributedSharding(*args: object, **kwargs: object) -> object:
        """Mock JAXDistributedSharding."""
        return object()


sys.modules["datasets"] = MockDatasets()
_conftest_grain = MockConftestGrain()
sys.modules["grain"] = _conftest_grain
sys.modules["grain.python"] = _conftest_grain


class MockConn:
    """Mock for DuckDB connection."""

    def execute(self: object, *_args: object, **_kwargs: object) -> object:
        """Mock execute.

        Returns:
            object: Description of return.

        """
        return self

    def fetchall(self: typing.Any) -> list:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [[json.dumps({"success": True, "generated_sql": "SELECT COUNT(*) FROM test", "results": [[1]]})]]

    def fetchdf(self: object) -> object:
        """Mock fetchdf.

        Returns:
            object: Description of return.

        """

        class MockDF:
            """Provide class docstring."""

            def to_dict(self: typing.Any, orient: str = "records") -> list[dict[str, str]]:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return [{"query": "SELECT 1", "nl": "Get 1", "sql": "SELECT 1", "sql_prompt": "Get 1"}]

        return MockDF()

    def create_function(self, name: object, func: object, args: object, ret: object) -> None:
        """Execute function."""

    def cursor(self: typing.Any) -> typing.Any:
        """Return self as DB-API 2.0 cursor."""
        return self

    def close(self: typing.Any) -> None:
        """Mock close."""


class MockDuckDB:
    """Mock for DuckDB module."""

    def connect(self: object, *_args: object, **_kwargs: object) -> object:
        """Mock connect.

        Returns:
            object: Description of return.

        """
        return MockConn()


sys.modules["duckdb"] = MockDuckDB()


class MockHFTokenizer:
    """Mock HF Tokenizer."""

    def encode(self: object, _text: str, **_kwargs: object) -> list[int]:
        """Mock encode.

        Returns:
            object: Description of return.

        """
        return [99, 100]

    def decode(self: object, _tokens: list[int], **_kwargs: object) -> str:
        """Mock decode.

        Returns:
            object: Description of return.

        """
        return "hf_decoded"


class MockAutoTokenizer:
    """Mock AutoTokenizer."""

    @classmethod
    def from_pretrained(cls: type, _model_name: str) -> object:
        """Mock from_pretrained.

        Returns:
            object: Description of return.

        """
        return MockHFTokenizer()


class MockModel:
    """Provide class docstring."""

    def save_pretrained(self: object, *_a: object, **_k: object) -> None:
        """Execute function."""

    def parameters(self: object) -> dict:
        """Execute function.

        Returns:
            dict: Empty parameter dictionary.
        """
        return {}


class MockGemma4ForCausalLM:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls: type, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockModel()


@pytest.fixture(autouse=True)
def _mock_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock external network calls like datasets.load_dataset and duckdb.connect."""
    monkeypatch.setitem(sys.modules, "datasets", MockDatasets())
    monkeypatch.setitem(sys.modules, "grain", MockConftestGrain())
    monkeypatch.setitem(sys.modules, "grain.python", MockConftestGrain())
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.grain", MockConftestGrain(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.grain", MockConftestGrain(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.grain", MockConftestGrain(), raising=False)
    monkeypatch.setitem(sys.modules, "duckdb", MockDuckDB())
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.tokenization.AutoTokenizer", MockAutoTokenizer, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.export.gemma4_for_causal_lm_cls", MockGemma4ForCausalLM, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.export.save_file", None, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.peft.peft", None, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.mlx.peft.load", lambda m: (MockModel(), None), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.mlx.inference.load", lambda m: (MockModel(), MockHFTokenizer()), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.mlx.train.load", lambda m: (MockModel(), MockHFTokenizer()), raising=False)


class MockDBModule:
    """Implementation of MockDBModule."""

    def __init__(self, name: str) -> None:
        """Initialize the instance."""
        self.name = name

    def connect(self, *args: object, **kwargs: object) -> None:
        """Execute the connect operation."""

    class Error(Exception):
        """Implementation of Error."""


from unittest.mock import AsyncMock

sys.modules["psycopg2"] = MagicMock(Error=Exception)
sys.modules["asyncpg"] = MagicMock(Error=Exception)
sys.modules["snowflake"] = MagicMock(Error=Exception)
sys.modules["snowflake.connector"] = MagicMock(Error=Exception)

if _real_aiosqlite is None:
    mock_aiosqlite = MagicMock(Error=Exception)
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_aiosqlite.connect = AsyncMock(return_value=mock_conn)
    sys.modules["aiosqlite"] = mock_aiosqlite

sys.modules["sentence_transformers"] = MagicMock(Error=Exception)


import pytest


@pytest.fixture(autouse=True)
def _suppress_expected_errors(request):
    """Docstring."""
    yield

"""Global pytest fixtures for gemma-4-sql tests."""

import json
import sys

import pytest


class MockDatasets:
    """Mock for datasets module."""

    def load_dataset(self: object, *_args: object, **_kwargs: object) -> list[dict[str, str]]:
        """Mock load_dataset."""
        return [{"query": "SELECT 1", "sql": "SELECT 1", "question": "test", "nl": "test"}]


sys.modules["datasets"] = MockDatasets()


class MockConn:
    """Mock for DuckDB connection."""

    def execute(self: object, *_args: object, **_kwargs: object) -> object:
        """Mock execute."""
        return self

    def fetchall(self: object) -> list:
        return [[json.dumps({"success": True, "generated_sql": "SELECT COUNT(*) FROM test", "results": [[1]]})]]

    def fetchdf(self: object) -> object:
        """Mock fetchdf."""

        class MockDF:
            def to_dict(self: object, orient: str = "records") -> list[dict[str, str]]:
                return [{"query": "SELECT 1", "nl": "Get 1", "sql": "SELECT 1", "sql_prompt": "Get 1"}]

        return MockDF()

    def create_function(self, name, func, args, ret) -> None:
        pass

    def close(self: object) -> None:
        """Mock close."""


class MockDuckDB:
    """Mock for DuckDB module."""

    def connect(self: object, *_args: object, **_kwargs: object) -> object:
        """Mock connect."""
        return MockConn()


sys.modules["duckdb"] = MockDuckDB()


@pytest.fixture(autouse=True)
def _mock_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock external network calls like datasets.load_dataset and duckdb.connect."""
    monkeypatch.setitem(sys.modules, "datasets", MockDatasets())

    # Force import before mocking

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.datasets", MockDatasets(), raising=False)

    monkeypatch.setitem(sys.modules, "duckdb", MockDuckDB())
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.duckdb", MockDuckDB(), raising=False)

    class MockHFTokenizer:
        """Mock HF Tokenizer."""

        def encode(self: object, _text: str, **_kwargs: object) -> list[int]:
            """Mock encode."""
            return [99, 100]

        def decode(self: object, _tokens: list[int], **_kwargs: object) -> str:
            """Mock decode."""
            return "hf_decoded"

    class MockAutoTokenizer:
        """Mock AutoTokenizer."""

        @classmethod
        def from_pretrained(cls: type, _model_name: str) -> object:
            """Mock from_pretrained."""
            return MockHFTokenizer()

    # Mock AutoTokenizer in the tokenization module
    monkeypatch.setattr("gemma_4_sql.tokenization.AutoTokenizer", MockAutoTokenizer, raising=False)

    class MockModel:
        def save_pretrained(self: object, *_a: object, **_k: object) -> None:
            pass

    class MockGemma4ForCausalLM:
        @classmethod
        def from_pretrained(cls: type, *_args: object, **_kwargs: object) -> object:
            return MockModel()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.export.gemma4_for_causal_lm_cls", MockGemma4ForCausalLM, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.export.save_file", None, raising=False)

    monkeypatch.setattr("gemma_4_sql.backends.keras.few_shot.tf", True, raising=False)

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.peft.peft", None, raising=False)

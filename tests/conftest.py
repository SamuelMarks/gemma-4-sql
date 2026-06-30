"""Global pytest fixtures for gemma-4-sql tests."""

import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _mock_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock external network calls like datasets.load_dataset and duckdb.connect."""

    class MockDatasets:
        def load_dataset(self: object, *_args: object, **_kwargs: object) -> list[dict[str, str]]:
            return [{"query": "SELECT 1", "sql": "SELECT 1", "question": "test", "nl": "test"}]

    monkeypatch.setitem(sys.modules, "datasets", MockDatasets())

    # Force import before mocking

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.datasets", MockDatasets(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.datasets", MockDatasets(), raising=False)

    class MockConn:
        def execute(self: object, *_args: object, **_kwargs: object) -> object:
            return self

        def fetchdf(self: object) -> object:
            class MockDF:
                def to_dict(self: object, _orient: str) -> list[dict[str, str]]:
                    return [{"query": "SELECT 1", "nl": "Get 1"}]

            return MockDF()

        def close(self: object) -> None:
            pass

    class MockDuckDB:
        def connect(self: object, *_args: object, **_kwargs: object) -> object:
            return MockConn()

    monkeypatch.setitem(sys.modules, "duckdb", MockDuckDB())
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.duckdb", MockDuckDB(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.duckdb", MockDuckDB(), raising=False)

    monkeypatch.setattr("gemma_4_sql.tokenization.SQLTokenizer.__init__", lambda _self, *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr("gemma_4_sql.tokenization.SQLTokenizer.encode", lambda _self, *_args, **_kwargs: [1, 2, 3], raising=False)
    # mock decode and vocab_size so inference doesn't fail
    monkeypatch.setattr("gemma_4_sql.tokenization.SQLTokenizer.decode", lambda _self, *_args, **_kwargs: "SELECT 1", raising=False)
    monkeypatch.setattr("gemma_4_sql.tokenization.SQLTokenizer.vocab_size", 100, raising=False)
    # mock hf_tokenizer property
    monkeypatch.setattr("gemma_4_sql.tokenization.SQLTokenizer.hf_tokenizer", mock.MagicMock(), raising=False)

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

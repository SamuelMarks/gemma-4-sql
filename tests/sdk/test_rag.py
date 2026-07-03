"""Tests for RAG-based schema contextualization module."""

import pytest

from gemma_4_sql.sdk import rag
from gemma_4_sql.sdk.rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema


def test_extract_schema_entities() -> None:
    """Execute function."""
    ddl = "\n\n    CREATE TABLE users (\n        id INT,\n        name VARCHAR,\n        PRIMARY KEY (id)\n    );\n    CREATE TABLE orders (\n        order_id INT,\n        user_id INT,\n        amount DECIMAL,\n        FOREIGN KEY (user_id) REFERENCES users(id)\n    );\n    "
    schema = extract_schema_entities(ddl)
    if "users" not in schema:
        raise AssertionError
    if not schema["users"] == ["id", "name"]:
        raise AssertionError
    if "orders" not in schema:
        raise AssertionError
    if not schema["orders"] == ["order_id", "user_id", "amount"]:
        raise AssertionError


def test_extract_schema_entities_ignore_comments() -> None:
    """Execute function."""
    ddl = "\n\n    -- This is a comment\n    CREATE TABLE test (\n        col1 INT\n    );\n    "
    schema = extract_schema_entities(ddl)
    if "test" not in schema:
        raise AssertionError
    if not schema["test"] == ["col1"]:
        raise AssertionError


def test_retrieve_relevant_schema() -> None:
    """Execute function."""
    schema = {"users": ["id", "name"], "orders": ["order_id", "user_id", "amount"], "products": ["prod_id", "name", "price"]}
    context = retrieve_relevant_schema("Find all users names", schema)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError
    if not "Table: orders" not in context:
        raise AssertionError
    if not "Table: products" not in context:
        raise AssertionError
    context2 = retrieve_relevant_schema("What is the total amount for orders?", schema)
    if "Table: orders | Columns: order_id, user_id, amount" not in context2:
        raise AssertionError


def test_retrieve_relevant_schema_fallback() -> None:
    """Execute function."""
    schema = {"users": ["id", "name"], "orders": ["order_id", "user_id", "amount"]}
    context = retrieve_relevant_schema("Show everything", schema, top_k_tables=1)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError


def test_retrieve_relevant_schema_empty() -> None:
    """Execute function."""
    context = retrieve_relevant_schema("Show everything", {})
    if context != "":
        raise AssertionError


class MockSentenceTransformer:
    """Provide class docstring."""

    def __init__(self, name: str) -> None:
        """Execute function."""

    def encode(self, docs: list[str]) -> list[list[float]]:
        """Execute function."""
        return [[1.0] for _ in docs]


class MockSimilarities:
    """Provide class docstring."""

    def argsort(self) -> object:
        """Execute function."""
        return [0]

    def __getitem__(self, idx: object) -> object:
        """Execute function."""
        if isinstance(idx, int):
            return 1.0
        return self


def mock_cosine_similarity(_a: object, _b: object) -> list:
    """Execute function."""
    return [MockSimilarities()]


def test_retrieve_relevant_schema_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(rag, "SentenceTransformer", MockSentenceTransformer)
    monkeypatch.setattr(rag, "cosine_similarity", mock_cosine_similarity)
    schema = {"users": ["id", "name"]}
    context = retrieve_relevant_schema("Show everything", schema, top_k_tables=1)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError


def test_retrieve_relevant_schema_semantic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(rag, "SentenceTransformer", MockSentenceTransformer)
    __import__("typing")

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(rag, "cosine_similarity", raise_err)
    schema = {"users": ["id", "name"]}
    context = retrieve_relevant_schema("users", schema, top_k_tables=1)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError


def test_build_rag_prompt_no_ddl() -> None:
    """Execute function."""
    if not build_rag_prompt("Find users") == "Find users":
        raise AssertionError


def test_build_rag_prompt_with_ddl() -> None:
    """Execute function."""
    ddl = "CREATE TABLE users (id INT, name VARCHAR);"
    prompt = "Find all users"
    rag_prompt = build_rag_prompt(prompt, ddl)
    if "-- Relevant Schema Context:" not in rag_prompt:
        raise AssertionError
    if "-- Table: users | Columns: id, name" not in rag_prompt:
        raise AssertionError
    if "-- Request:" not in rag_prompt:
        raise AssertionError
    if "-- Find all users" not in rag_prompt:
        raise AssertionError
    if "SELECT" not in rag_prompt:
        raise AssertionError

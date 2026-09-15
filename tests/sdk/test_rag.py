"""Tests for RAG-based schema contextualization module."""

import pytest

from gemma_4_sql.sdk import rag
from gemma_4_sql.sdk.rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema


def test_extract_schema_entities() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
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
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    ddl = "\n\n    -- This is a comment\n    CREATE TABLE test (\n        col1 INT\n    );\n    "
    schema = extract_schema_entities(ddl)
    if "test" not in schema:
        raise AssertionError
    if not schema["test"] == ["col1"]:
        raise AssertionError


def test_retrieve_relevant_schema() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
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
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    schema = {"users": ["id", "name"], "orders": ["order_id", "user_id", "amount"]}
    context = retrieve_relevant_schema("Show everything", schema, top_k_tables=1)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError


def test_retrieve_relevant_schema_empty() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    context = retrieve_relevant_schema("Show everything", {})
    if context != "":
        raise AssertionError


class MockSentenceTransformer:
    """Provide class docstring."""

    def __init__(self, name: str) -> None:
        """Execute function."""

    def encode(self, docs: list[str]) -> list[list[float]]:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [[1.0] for _ in docs]


class MockSimilarities:
    """Provide class docstring."""

    def argsort(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]

    def __getitem__(self, idx: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if isinstance(idx, int):
            return 1.0
        return self


def mock_cosine_similarity(_a: object, _b: object) -> list:
    """Execute function.

    Returns:
        object: Description of return.

    """
    return [MockSimilarities()]


def test_retrieve_relevant_schema_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(rag, "SentenceTransformer", MockSentenceTransformer)
    monkeypatch.setattr(rag, "cosine_similarity", mock_cosine_similarity)
    schema = {"users": ["id", "name"]}
    context = retrieve_relevant_schema("Show everything", schema, top_k_tables=1)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError


def test_retrieve_relevant_schema_semantic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(rag, "SentenceTransformer", MockSentenceTransformer)
    __import__("typing", fromlist=[""])

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(rag, "cosine_similarity", raise_err)
    schema = {"users": ["id", "name"]}
    context = retrieve_relevant_schema("users", schema, top_k_tables=1)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError


def test_build_rag_prompt_no_ddl() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    if not build_rag_prompt("Find users") == "Find users":
        raise AssertionError


def test_build_rag_prompt_with_ddl() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
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


def test_rag_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test import fallback."""
    import sys

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "sentence_transformers", None)
        if "gemma_4_sql.sdk.rag" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.rag"]
        rag = __import__("gemma_4_sql.sdk.rag", fromlist=[""])
        assert rag.SentenceTransformer is None
        assert rag.cosine_similarity is None


def test_rag_no_relevant_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test semantic search when similarity is lower than MIN_SIMILARITY.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import gemma_4_sql.sdk.rag as rg

    monkeypatch.setattr(rg, "MIN_SIMILARITY", 100.0)
    schema = {"users": ["id"], "orders": ["id"]}

    class MockModel:
        """Mock SentenceTransformer model."""

        def encode(self, _x: object) -> list[list[float]]:
            """Return mock embedding.

            Args:
                _x: Input object.

            Returns:
                Mock embedding list.
            """
            return [[1.0]]

    monkeypatch.setattr(rg, "SentenceTransformer", lambda *a, **k: MockModel())

    class MockSim:
        """Mock similarity array."""

        def argsort(self) -> list[int]:
            """Return sort order.

            Returns:
                Index list.
            """
            return [0, 1]

        def __getitem__(self, _i: int) -> float:
            """Return similarity score.

            Args:
                _i: Index.

            Returns:
                Similarity float.
            """
            return 0.5

    monkeypatch.setattr(rg, "cosine_similarity", lambda _a, _b: [MockSim()])
    res = rg._semantic_search("hi", schema, ["users", "orders"], 2, min_similarity=0.9)
    assert len(res) == 2
    assert res == ["users", "orders"]


from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_rag_semantic_no_relevant() -> None:
    """Test rag semantic when all similarity scores are below threshold."""
    mock_st = MagicMock()
    mock_st.return_value.encode.return_value = np.array([[1.0]])

    with patch("gemma_4_sql.sdk.rag.SentenceTransformer", mock_st), patch("gemma_4_sql.sdk.rag.cosine_similarity", lambda _a, _b: np.array([[0.0, 0.0]])):
        schema = {"t1": ["c1"], "t2": ["c2"]}
        res = retrieve_relevant_schema("prompt", schema)
        assert "Table: t1" in res
        assert "Table: t2" in res


def test_extract_schema_entities_complex() -> None:
    """Test extract_schema_entities with parameterized types, constraints, and multiple tables."""
    ddl = """
    CREATE TABLE orders (
        order_id INT PRIMARY KEY,
        amount DECIMAL(10, 2),
        cust_id INT,
        CONSTRAINT fk_customer FOREIGN KEY (cust_id) REFERENCES customers(id),
        status VARCHAR(50),
        UNIQUE (order_id)
    );
    CREATE TABLE IF NOT EXISTS items (
        item_id INT,
        price NUMERIC(8, 2) CHECK (price > 0),
        title TEXT
    );
    """
    schema = extract_schema_entities(ddl)
    assert schema == {
        "orders": ["order_id", "amount", "cust_id", "status"],
        "items": ["item_id", "price", "title"],
    }


def test_extract_schema_entities_edge_cases() -> None:
    """Test extract_schema_entities with empty columns, trailing commas, and invalid entries."""
    ddl = "CREATE TABLE t (col1 INT, , @invalid INT, col2 TEXT, );\nCREATE TABLE t2 (a INT,);"
    schema = extract_schema_entities(ddl)
    assert schema == {"t": ["col1", "col2"], "t2": ["a"]}


def test_extract_schema_entities_quoted_identifiers() -> None:
    """Test extract_schema_entities with double-quoted, backtick, and bracket-quoted identifiers."""
    ddl = """
    CREATE TABLE "public"."customers" (
        "id" INT PRIMARY KEY,
        "email" VARCHAR(255),
        "created_at" TIMESTAMP
    );
    CREATE TABLE `orders` (
        `order_id` INT,
        `total` DECIMAL(12, 2),
        CONSTRAINT chk_total CHECK (`total` >= 0)
    );
    CREATE TABLE [inventory] (
        [item_id] INT,
        [count] INT
    );
    """
    schema = extract_schema_entities(ddl)
    assert schema == {
        "customers": ["id", "email", "created_at"],
        "orders": ["order_id", "total"],
        "inventory": ["item_id", "count"],
    }


def test_semantic_search_empty_tables() -> None:
    """Test that _semantic_search handles empty table lists without error."""
    from gemma_4_sql.sdk.rag import _semantic_search

    res = _semantic_search("find user", {}, [], top_k_tables=2)
    assert res == []


def test_rag_complex_ddl_and_ranking_boundaries() -> None:
    """Test extract_schema_entities with complex DDL, mixed casings, and multiple foreign keys."""
    ddl = """
    CREATE TABLE UserAccounts (
        UserID INT NOT NULL,
        UserName VARCHAR(50),
        CONSTRAINT pk_user PRIMARY KEY (UserID)
    );
    CREATE TABLE Orders_Archive (
        OrderID INT,
        CustID INT,
        SellerID INT,
        Amount NUMERIC(10, 2),
        FOREIGN KEY (CustID) REFERENCES UserAccounts(UserID),
        FOREIGN KEY (SellerID) REFERENCES UserAccounts(UserID)
    );
    """
    schema = extract_schema_entities(ddl)
    assert "UserAccounts" in schema
    assert schema["UserAccounts"] == ["UserID", "UserName"]
    assert "Orders_Archive" in schema
    assert schema["Orders_Archive"] == ["OrderID", "CustID", "SellerID", "Amount"]

    res = retrieve_relevant_schema("Find UserAccounts with high Amount", schema, top_k_tables=1)
    assert "UserAccounts" in res


def test_semantic_search_missing_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _semantic_search falls back to keyword search when sentence_transformers is missing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    from gemma_4_sql.sdk import rag

    monkeypatch.setattr(rag, "SentenceTransformer", None)
    schema = {"users": ["id", "name"], "orders": ["id", "user_id"]}
    res = rag._semantic_search("users", schema, ["users", "orders"], 1)
    assert res == ["users"]

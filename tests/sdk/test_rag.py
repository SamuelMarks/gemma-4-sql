"""
Tests for RAG-based schema contextualization module.
"""

from gemma_4_sql.sdk.rag import (
    build_rag_prompt,
    extract_schema_entities,
    retrieve_relevant_schema,
)


def test_extract_schema_entities() -> None:
    ddl = """
    CREATE TABLE users (
        id INT,
        name VARCHAR,
        PRIMARY KEY (id)
    );
    CREATE TABLE orders (
        order_id INT,
        user_id INT,
        amount DECIMAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    schema = extract_schema_entities(ddl)
    assert "users" in schema
    assert schema["users"] == ["id", "name"]
    assert "orders" in schema
    assert schema["orders"] == ["order_id", "user_id", "amount"]


def test_extract_schema_entities_ignore_comments() -> None:
    ddl = """
    -- This is a comment
    CREATE TABLE test (
        col1 INT
    );
    """
    schema = extract_schema_entities(ddl)
    assert "test" in schema
    assert schema["test"] == ["col1"]


def test_retrieve_relevant_schema() -> None:
    schema = {
        "users": ["id", "name"],
        "orders": ["order_id", "user_id", "amount"],
        "products": ["prod_id", "name", "price"],
    }

    # Should match 'users' heavily
    context = retrieve_relevant_schema("Find all users names", schema)
    assert "Table: users | Columns: id, name" in context
    assert "Table: orders" not in context
    assert "Table: products" not in context

    # Should match 'orders' and 'amount'
    context2 = retrieve_relevant_schema("What is the total amount for orders?", schema)
    assert "Table: orders | Columns: order_id, user_id, amount" in context2


def test_retrieve_relevant_schema_fallback() -> None:
    schema = {"users": ["id", "name"], "orders": ["order_id", "user_id", "amount"]}
    # No direct matches, should fallback to returning the top K tables
    context = retrieve_relevant_schema("Show everything", schema, top_k_tables=1)
    assert "Table: users | Columns: id, name" in context


def test_build_rag_prompt_no_ddl() -> None:
    assert build_rag_prompt("Find users") == "Find users"


def test_build_rag_prompt_with_ddl() -> None:
    ddl = "CREATE TABLE users (id INT, name VARCHAR);"
    prompt = "Find all users"

    rag_prompt = build_rag_prompt(prompt, ddl)
    assert "-- Relevant Schema Context:" in rag_prompt
    assert "-- Table: users | Columns: id, name" in rag_prompt
    assert "-- Request:" in rag_prompt
    assert "-- Find all users" in rag_prompt
    assert "SELECT" in rag_prompt

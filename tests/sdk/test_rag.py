"""Tests for RAG-based schema contextualization module."""

from gemma_4_sql.sdk.rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema


def test_extract_schema_entities() -> None:
    """Initialize function test_extract_schema_entities."""
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
    """Initialize function test_extract_schema_entities_ignore_comments."""
    ddl = "\n\n    -- This is a comment\n    CREATE TABLE test (\n        col1 INT\n    );\n    "
    schema = extract_schema_entities(ddl)
    if "test" not in schema:
        raise AssertionError
    if not schema["test"] == ["col1"]:
        raise AssertionError


def test_retrieve_relevant_schema() -> None:
    """Initialize function test_retrieve_relevant_schema."""
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
    """Initialize function test_retrieve_relevant_schema_fallback."""
    schema = {"users": ["id", "name"], "orders": ["order_id", "user_id", "amount"]}
    context = retrieve_relevant_schema("Show everything", schema, top_k_tables=1)
    if "Table: users | Columns: id, name" not in context:
        raise AssertionError


def test_build_rag_prompt_no_ddl() -> None:
    """Initialize function test_build_rag_prompt_no_ddl."""
    if not build_rag_prompt("Find users") == "Find users":
        raise AssertionError


def test_build_rag_prompt_with_ddl() -> None:
    """Initialize function test_build_rag_prompt_with_ddl."""
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

"""Unit tests for the pure SQL lexical scanner in `gemma_4_sql.sdk.sql_lexer`."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import SQLSafetyViolationError
from gemma_4_sql.sdk.sql_lexer import (
    DANGEROUS_KEYWORDS,
    extract_word_tokens,
    has_top_level_order_by,
    split_preserving_literals,
    strip_comments_and_literals,
    strip_sql_comments,
    validate_readonly_sql,
    validate_sql_identifier,
)

_BS = chr(92)
_NL = chr(10)


def test_strip_sql_comments() -> None:
    """Verify `strip_sql_comments` removes line and block comments while preserving quoted literals."""
    sql = f"SELECT 'a -- not comment /* still literal */', \"b -- col\", `c /* col */`, [d -- col], $$e -- dollar$$ FROM t -- line comment{_NL}/* block{_NL}comment */ WHERE x = 1"
    stripped = strip_sql_comments(sql)
    assert "'a -- not comment /* still literal */'" in stripped
    assert '"b -- col"' in stripped
    assert "`c /* col */`" in stripped
    assert "[d -- col]" in stripped
    assert "$$e -- dollar$$" in stripped
    assert "line comment" not in stripped
    assert f"block{_NL}comment" not in stripped
    assert "WHERE x = 1" in stripped

    nested = "SELECT /* outer /* inner comment */ still outer */ 42 FROM dual"
    assert strip_sql_comments(nested).strip() == "SELECT   42 FROM dual"


def test_strip_comments_and_literals_with_escapes() -> None:
    """Verify `strip_comments_and_literals` handles backslash escapes, doubled quotes, and dollar quotes."""
    doubled_single = "'O''Reilly'"
    escaped_single = f"'O{_BS}'Reilly'"
    escaped_path = f"'C:{_BS}{_BS}path{_BS}{_BS}to{_BS}{_BS}file'"
    escaped_double = f'"col{_BS}"name"'
    sql = f'SELECT {doubled_single}, {escaped_single}, {escaped_path}, "col""name", {escaped_double}, `back``tick`, $tag$DROP TABLE users;$tag$ FROM users'
    cleaned = strip_comments_and_literals(sql)
    assert "Reilly" not in cleaned
    assert "path" not in cleaned
    assert "DROP" not in cleaned
    assert "SELECT" in cleaned
    assert "FROM users" in cleaned


def test_split_preserving_literals() -> None:
    """Verify `split_preserving_literals` alternates unquoted SQL at even indices and verbatim literals at odd indices."""
    escaped_single = f"'O{_BS}'Reilly'"
    escaped_path = f"'C:{_BS}{_BS}temp'"
    sql = f'SELECT {escaped_single} AS author, "Wait  For  It" FROM books WHERE path = {escaped_path}'
    parts = split_preserving_literals(sql)
    assert len(parts) == 7
    assert parts[0] == "SELECT "
    assert parts[1] == escaped_single
    assert parts[2] == " AS author, "
    assert parts[3] == '"Wait  For  It"'
    assert parts[4] == " FROM books WHERE path = "
    assert parts[5] == escaped_path
    assert parts[6] == ""


def test_has_top_level_order_by() -> None:
    """Verify `has_top_level_order_by` distinguishes top-level ORDER BY from CTEs, subqueries, and window functions."""
    assert has_top_level_order_by("SELECT * FROM users ORDER BY created_at DESC") is True
    assert has_top_level_order_by("WITH cte AS (SELECT * FROM users ORDER BY id) SELECT * FROM cte") is False
    assert has_top_level_order_by("SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) FROM emp") is False
    assert has_top_level_order_by("SELECT * FROM (SELECT * FROM t ORDER BY a) sub ORDER BY b") is True
    assert has_top_level_order_by("SELECT 'ORDER BY a' FROM t") is False


@pytest.mark.parametrize("keyword", sorted(DANGEROUS_KEYWORDS))
def test_validate_readonly_sql_all_dangerous_keywords(keyword: str) -> None:
    """Verify `validate_readonly_sql` blocks all 25 dangerous keywords in read-only mode."""
    with pytest.raises(SQLSafetyViolationError, match=keyword):
        validate_readonly_sql(f"{keyword} something", read_only=True)
    # Bypassed when read_only=False
    validate_readonly_sql(f"{keyword} something", read_only=False)


def test_validate_readonly_sql_unclosed_and_stacked() -> None:
    """Verify `validate_readonly_sql` rejects unclosed comments, unclosed literals, and stacked statements."""
    validate_readonly_sql("SELECT 1;", read_only=True)
    escaped_single = f"'O{_BS}'Reilly'"
    validate_readonly_sql(f"SELECT 'O''Reilly', {escaped_single} FROM t;", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="Unclosed block comment"):
        validate_readonly_sql("SELECT * FROM t /* unclosed", read_only=True)

    for unclosed_query in [
        "SELECT 'unclosed string FROM t",
        'SELECT "unclosed identifier FROM t',
        "SELECT `unclosed backtick FROM t",
        "SELECT $$unclosed dollar FROM t",
        "SELECT $tag$unclosed tagged dollar FROM t",
    ]:
        with pytest.raises(SQLSafetyViolationError, match="Unclosed string literal or quoted identifier"):
            validate_readonly_sql(unclosed_query, read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="Multiple stacked SQL statements"):
        validate_readonly_sql("SELECT 1; SELECT 2", read_only=True)


def test_extract_word_tokens_and_validate_identifier() -> None:
    """Verify `extract_word_tokens` and `validate_sql_identifier` behavior."""
    assert extract_word_tokens("SELECT User_Name, Age_2 FROM Users!") == ["select", "user_name", "age_2", "from", "users"]
    assert validate_sql_identifier("valid_table_123") == "valid_table_123"
    assert validate_sql_identifier("_private_table") == "_private_table"

    for invalid_ident in ["", "123table", "users; DROP TABLE users", "users--", "schema.table", "table name"]:
        with pytest.raises(ValueError, match="Invalid or unsafe table name"):
            validate_sql_identifier(invalid_ident)


def test_sql_lexer_edge_cases_and_nested_depth() -> None:
    """Verify `has_top_level_order_by`, `validate_readonly_sql`, and `strip_sql_comments` on deeply nested expressions and trailing semicolons."""
    assert has_top_level_order_by("SELECT ((1 + 2)) FROM users") is False
    assert has_top_level_order_by("SELECT * FROM (SELECT * FROM (SELECT 1 ORDER BY 1) a ORDER BY 1) b") is False
    assert has_top_level_order_by("SELECT * FROM (SELECT 1) a\nORDER   BY\n  1") is True

    # Multiple trailing semicolons without a second statement are still rejected in read-only mode
    with pytest.raises(SQLSafetyViolationError, match="Multiple stacked SQL statements"):
        validate_readonly_sql("SELECT 1;;", read_only=True)

    # Dangerous keyword inside a comment or literal is allowed in read-only mode
    validate_readonly_sql("SELECT 'DROP TABLE users' AS msg /* DELETE FROM t */ FROM dual -- TRUNCATE t", read_only=True)
    assert strip_sql_comments("SELECT 1") == "SELECT 1"

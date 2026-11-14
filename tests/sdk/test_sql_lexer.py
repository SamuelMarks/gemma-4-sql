"""Unit and adversarial security tests for the SQL lexical scanner in `gemma_4_sql.sdk.sql_lexer`."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import SQLSafetyViolationError
from gemma_4_sql.sdk.sql_lexer import (
    ALWAYS_FORBIDDEN_KEYWORDS,
    DANGEROUS_FUNCTIONS,
    DANGEROUS_KEYWORDS,
    STATEMENT_VERB_KEYWORDS,
    SQLSegment,
    extract_word_tokens,
    has_top_level_order_by,
    scan_sql_segments,
    split_preserving_literals,
    strip_comments_and_literals,
    strip_sql_comments,
    validate_readonly_sql,
    validate_sql_identifier,
)

_BS = r"\s"[0]
_NL = """
"""


def test_strip_sql_comments() -> None:
    """Verify `strip_sql_comments` removes line and nested block comments while preserving quoted literals."""
    dbl_col = '"b -- col"'
    sql = (
        f"SELECT 'a -- not comment /* still literal */', {dbl_col}, `c /* col */`, "
        f"[d -- col], $$e -- dollar$$ FROM t -- line comment{_NL}"
        f"/* block{_NL}comment */ WHERE x = 1 -- eof comment"
    )
    stripped = strip_sql_comments(sql)
    assert "'a -- not comment /* still literal */'" in stripped
    assert '"b -- col"' in stripped
    assert "`c /* col */`" in stripped
    assert "[d -- col]" in stripped
    assert "$$e -- dollar$$" in stripped
    assert "line comment" not in stripped
    assert "eof comment" not in stripped
    assert f"block{_NL}comment" not in stripped
    assert "WHERE x = 1" in stripped

    nested = "SELECT /* outer /* inner comment */ still outer */ 42 FROM dual"
    assert strip_sql_comments(nested).strip() == "SELECT   42 FROM dual"


def test_strip_comments_and_literals_ansi_and_postgres_escapes() -> None:
    """Verify `strip_comments_and_literals` handles ANSI doubled quotes, PG `E'...'` escapes, and dollar quotes."""
    doubled_single = "'O''Reilly'"
    pg_escaped_single = f"E'O{_BS}'Reilly''s'"
    pg_lower_escaped_path = f"e'C:{_BS}{_BS}path{_BS}{_BS}to{_BS}{_BS}file'"
    doubled_double = '"col""name"'
    doubled_bracket = "[col]]name]"
    sql = (
        f"SELECT {doubled_single}, {pg_escaped_single}, {pg_lower_escaped_path}, "
        f"{doubled_double}, {doubled_bracket}, `back``tick`, "
        "$tag$DROP TABLE users;$tag$ FROM users WHERE id = $1 AND name ='alice'"
    )
    cleaned = strip_comments_and_literals(sql)
    assert "Reilly" not in cleaned
    assert "path" not in cleaned
    assert "DROP" not in cleaned
    assert "alice" not in cleaned
    assert "SELECT" in cleaned
    assert "FROM users WHERE id = $1 AND name =" in cleaned

    # Ensure identifier ending in E before quote (`tableE'lit'`) is scanned as code `tableE` + ANSI literal `'lit'`
    segs = scan_sql_segments("SELECT tableE'lit'")
    assert segs == [SQLSegment("code", "SELECT tableE"), SQLSegment("literal", "'lit'")]


def test_backslash_quote_smuggling_is_blocked() -> None:
    """Verify ANSI single-quoted literals do NOT treat backslash-quote as escaped, blocking SQL injection smuggling."""
    smuggled_drop = f"SELECT '{_BS}'; DROP TABLE users; --'"
    with pytest.raises(SQLSafetyViolationError, match="Multiple stacked SQL statements"):
        validate_readonly_sql(smuggled_drop, read_only=True)

    smuggled_no_semi = f"SELECT '{_BS}' UNION ALL SELECT pg_read_file('/etc/passwd') --'"
    with pytest.raises(SQLSafetyViolationError, match="PG_READ_FILE"):
        validate_readonly_sql(smuggled_no_semi, read_only=True)

    # Genuine PostgreSQL E'...' escape string with escaped quote remains valid and safe
    pg_safe = f"SELECT E'O{_BS}'Reilly' AS author FROM books;"
    validate_readonly_sql(pg_safe, read_only=True)


def test_split_preserving_literals() -> None:
    """Verify `split_preserving_literals` alternates unquoted SQL and verbatim literals."""
    pg_escaped = f"E'O{_BS}'Reilly'"
    ansi_path = f"'C:{_BS}temp'"
    sql = f'SELECT {pg_escaped} AS author, "Wait  For  It" FROM books WHERE path = {ansi_path}'
    parts = split_preserving_literals(sql)
    assert len(parts) == 7
    assert parts[0] == "SELECT "
    assert parts[1] == pg_escaped
    assert parts[2] == " AS author, "
    assert parts[3] == '"Wait  For  It"'
    assert parts[4] == " FROM books WHERE path = "
    assert parts[5] == ansi_path
    assert parts[6] == ""


def test_has_top_level_order_by() -> None:
    """Verify `has_top_level_order_by` distinguishes top-level ORDER BY from CTEs, subqueries, and window functions."""
    assert has_top_level_order_by("SELECT * FROM users ORDER BY created_at DESC") is True
    assert has_top_level_order_by("WITH cte AS (SELECT * FROM users ORDER BY id) SELECT * FROM cte") is False
    assert (
        has_top_level_order_by("SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) FROM emp")
        is False
    )
    assert has_top_level_order_by("SELECT * FROM (SELECT * FROM t ORDER BY a) sub ORDER BY b") is True
    assert has_top_level_order_by("SELECT 'ORDER BY a' FROM t") is False


@pytest.mark.parametrize("keyword", sorted(DANGEROUS_KEYWORDS))
def test_validate_readonly_sql_all_dangerous_keywords(keyword: str) -> None:
    """Verify `validate_readonly_sql` blocks every dangerous keyword and function when leading a statement."""
    assert DANGEROUS_KEYWORDS == (ALWAYS_FORBIDDEN_KEYWORDS | STATEMENT_VERB_KEYWORDS | DANGEROUS_FUNCTIONS)
    with pytest.raises(SQLSafetyViolationError, match=keyword):
        validate_readonly_sql(f"{keyword} something", read_only=True)
    validate_readonly_sql(f"{keyword} something", read_only=False)


def test_validate_readonly_sql_unclosed_stacked_ctes_and_functions() -> None:
    """Verify `validate_readonly_sql` rejects unclosed tokens, stacked SQL, mutating CTEs, and unsafe functions."""
    validate_readonly_sql("SELECT 1;", read_only=True)
    validate_readonly_sql("   ", read_only=True)
    validate_readonly_sql(";", read_only=True)
    validate_readonly_sql("(SELECT 1)", read_only=True)
    validate_readonly_sql("EXPLAIN SELECT * FROM users", read_only=True)

    for unclosed_comment in [
        "SELECT * FROM t /* unclosed",
        "SELECT * FROM t /* outer /* inner */ still unclosed",
    ]:
        with pytest.raises(SQLSafetyViolationError, match="Unclosed block comment"):
            validate_readonly_sql(unclosed_comment, read_only=True)

    for unclosed_query in [
        "SELECT 'unclosed string FROM t",
        f"SELECT E'unclosed escape{_BS}",
        "SELECT E'unclosed escape body",
        'SELECT "unclosed identifier FROM t',
        "SELECT `unclosed backtick FROM t",
        "SELECT [unclosed bracket FROM t",
        "SELECT $$unclosed dollar FROM t",
        "SELECT $tag$unclosed tagged dollar FROM t",
    ]:
        with pytest.raises(SQLSafetyViolationError, match="Unclosed string literal or quoted identifier"):
            validate_readonly_sql(unclosed_query, read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="Multiple stacked SQL statements"):
        validate_readonly_sql("SELECT 1; SELECT 2", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="DELETE"):
        validate_readonly_sql("WITH deleted AS (DELETE FROM users RETURNING id) SELECT * FROM deleted", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="REPLACE"):
        validate_readonly_sql("REPLACE INTO users VALUES (1, 'admin')", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="DO"):
        validate_readonly_sql("DO $$ BEGIN PERFORM 1; END $$", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="COPY"):
        validate_readonly_sql("WITH exported AS (COPY users TO '/tmp/out.csv') SELECT 1", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="EXPLAIN ANALYZE"):
        validate_readonly_sql("EXPLAIN ANALYZE SELECT * FROM users", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="EXPLAIN ANALYZE"):
        validate_readonly_sql("EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM users", read_only=True)

    for fn_call, expected in [
        ("SELECT load_extension('evil.so')", "LOAD_EXTENSION"),
        ("SELECT pg_read_file('/etc/passwd')", "PG_READ_FILE"),
        ("SELECT write_csv(tbl, '/tmp/a.csv') FROM tbl", "WRITE_CSV"),
    ]:
        with pytest.raises(SQLSafetyViolationError, match=expected):
            validate_readonly_sql(fn_call, read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="SELECT INTO"):
        validate_readonly_sql("SELECT * INTO new_users FROM users", read_only=True)

    with pytest.raises(SQLSafetyViolationError, match="SELECT INTO"):
        validate_readonly_sql("SELECT id, name INTO TEMP TABLE tmp_users FROM users", read_only=True)


def test_validate_readonly_sql_allows_benign_identifier_names() -> None:
    """Verify columns, tables, and aliases named like statement verbs (`export`, `load`, `copy`) are allowed."""
    benign_queries = [
        (
            "SELECT export, export_date, analyze, load, copy, call, set, reset, comment, security "
            "FROM import WHERE load > 0"
        ),
        "WITH export AS (SELECT id, load, analyze FROM metrics) SELECT * FROM export ORDER BY load DESC",
        (
            "SELECT m.do, m.lock, m.listen, m.notify, m.cluster, m.prepare, m.execute, m.declare, m.install "
            "FROM metrics m"
        ),
    ]
    for query in benign_queries:
        validate_readonly_sql(query, read_only=True)


def test_extract_word_tokens_and_validate_identifier() -> None:
    """Verify `extract_word_tokens` and `validate_sql_identifier` behavior."""
    assert extract_word_tokens("SELECT User_Name, Age_2 FROM Users!") == [
        "select",
        "user_name",
        "age_2",
        "from",
        "users",
    ]
    assert validate_sql_identifier("valid_table_123") == "valid_table_123"
    assert validate_sql_identifier("_private_table") == "_private_table"

    for invalid_ident in ["", "123table", "users; DROP TABLE users", "users--", "schema.table", "table name"]:
        with pytest.raises(ValueError, match="Invalid or unsafe table name"):
            validate_sql_identifier(invalid_ident)


def test_sql_lexer_edge_cases_and_nested_depth() -> None:
    """Verify SQL lexer helpers on deeply nested expressions and trailing semicolons."""
    assert has_top_level_order_by("SELECT ((1 + 2)) FROM users") is False
    assert has_top_level_order_by("SELECT * FROM (SELECT * FROM (SELECT 1 ORDER BY 1) a ORDER BY 1) b") is False
    assert has_top_level_order_by(f"SELECT * FROM (SELECT 1) a{_NL}ORDER   BY{_NL}  1") is True

    with pytest.raises(SQLSafetyViolationError, match="Multiple stacked SQL statements"):
        validate_readonly_sql("SELECT 1;;", read_only=True)

    validate_readonly_sql(
        "SELECT 'DROP TABLE users' AS msg /* DELETE FROM t */ FROM dual -- TRUNCATE t", read_only=True
    )
    assert strip_sql_comments("SELECT 1") == "SELECT 1"

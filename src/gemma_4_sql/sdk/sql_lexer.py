"""Pure SQL lexical scanner for comment stripping, literal masking, and read-only safety validation."""

from __future__ import annotations

import re

from gemma_4_sql.exceptions import SQLSafetyViolationError

DANGEROUS_KEYWORDS = frozenset({
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "RENAME",
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "UPSERT",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "COPY",
    "INSTALL",
    "LOAD",
    "VACUUM",
    "EXECUTE",
    "CALL",
    "GRANT",
    "REVOKE",
})

_SQL_TOKEN_PATTERN = re.compile(
    r"""
    (\$(?P<dtag>[A-Za-z0-9_]*)\$[\s\S]*?\$(?P=dtag)\$) | # PostgreSQL dollar-quoted string literal
    ('(?:''|\.|[^'\])*')      | # Single-quoted string literal (SQL & backslash escapes)
    ("(?:""|\.|[^"\])*")      | # Double-quoted identifier/literal
    (`(?:``|[^`])*`)            | # Backtick-quoted identifier
    (\[[^\]]*\])                | # Bracket-quoted identifier
    (/\*[\s\S]*?\*/)            | # Block comment
    (--[^
]*)                    # Single-line comment
    """,
    re.VERBOSE,
)

_SQL_COMMENT_PATTERN = re.compile(
    r"""
    (\$(?P<dtag>[A-Za-z0-9_]*)\$[\s\S]*?\$(?P=dtag)\$) | # PostgreSQL dollar-quoted string literal
    ('(?:''|\.|[^'\])*')      | # Single-quoted string literal
    ("(?:""|\.|[^"\])*")      | # Double-quoted identifier/literal
    (`(?:``|[^`])*`)            | # Backtick-quoted identifier
    (\[[^\]]*\])                | # Bracket-quoted identifier
    (?P<comment>/\*[\s\S]*?\*/|--[^
]*) # Block or single-line comment
    """,
    re.VERBOSE,
)

_SQL_LITERAL_SPLIT_PATTERN = re.compile(r"""(\$(?:[A-Za-z0-9_]*)\$[\s\S]*?\$(?:[A-Za-z0-9_]*)\$|'(?:''|\.|[^'\])*'|"(?:""|\.|[^"\])*"|`(?:``|[^`])*`|\[[^\]]*\])""")


def strip_sql_comments(query: str) -> str:
    """Strip SQL block and single-line comments while preserving quoted literals and identifiers.

    Args:
        query: Raw SQL query or DDL string.

    Returns:
        SQL string with comments replaced by spaces and literals preserved intact.

    """
    return _SQL_COMMENT_PATTERN.sub(lambda m: " " if m.group("comment") is not None else m.group(0), query)


def strip_comments_and_literals(query: str) -> str:
    """Replace SQL comments and quoted literals with spaces in a single lexical pass.

    Args:
        query: Raw SQL query string.

    Returns:
        SQL string with all comments and quoted segments replaced by a single space.

    """
    return _SQL_TOKEN_PATTERN.sub(" ", query)


def split_preserving_literals(query: str) -> list[str]:
    """Split a SQL query into alternating unquoted segments and quoted literal/identifier segments.

    Even indices (`0, 2, 4, ...`) are unquoted SQL text; odd indices (`1, 3, 5, ...`) are
    quoted literals or identifiers that must be preserved verbatim.

    Args:
        query: SQL string (typically after `strip_sql_comments`).

    Returns:
        List of alternating unquoted and quoted SQL substrings.

    """
    return _SQL_LITERAL_SPLIT_PATTERN.split(query)


def has_top_level_order_by(query: str) -> bool:
    """Detect presence of a top-level `ORDER BY` clause outside parentheses, comments, and literals.

    Args:
        query: SQL query string.

    Returns:
        `True` if a top-level `ORDER BY` clause is present.

    """
    cleaned = strip_comments_and_literals(query)
    cleaned = re.sub(r"\bORDER\s+BY\b", "ORDER_BY", cleaned.upper())
    tokens = re.finditer(r"\(|\)|ORDER_BY", cleaned)
    depth = 0
    for match in tokens:
        t = match.group(0)
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
        elif t == "ORDER_BY" and depth <= 0:
            return True
    return False


def validate_readonly_sql(query: str, *, read_only: bool = True) -> None:
    """Ensure the SQL query is safe to execute when `read_only` is True.

    Args:
        query: SQL query string to inspect.
        read_only: Whether read-only enforcement is active.

    Raises:
        SQLSafetyViolationError: If mutating statements or stacked queries are attempted in read-only mode.

    """
    if not read_only:
        return

    cleaned = strip_comments_and_literals(query)
    if "/*" in cleaned:
        msg = "Safety Violation: Unclosed block comment is not allowed in read-only mode."
        raise SQLSafetyViolationError(msg)

    statements = [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]
    if len(statements) > 1 or cleaned.count(";") > 1:
        msg = "Safety Violation: Multiple stacked SQL statements are not allowed in read-only mode."
        raise SQLSafetyViolationError(msg)

    words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", cleaned.upper())
    for word in words:
        if word in DANGEROUS_KEYWORDS:
            msg = f"Safety Violation: Mutating statements ({word}) are not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)


def validate_sql_identifier(identifier: str) -> str:
    """Validate that a SQL table or column identifier matches safe `[a-zA-Z_][a-zA-Z0-9_]*` syntax.

    Args:
        identifier: Raw table or column identifier string.

    Returns:
        The validated identifier string.

    Raises:
        ValueError: If `identifier` contains unsafe characters or does not match identifier syntax.

    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        msg = f"Invalid or unsafe table name: {identifier!r}"
        raise ValueError(msg)
    return identifier


__all__ = [
    "DANGEROUS_KEYWORDS",
    "has_top_level_order_by",
    "split_preserving_literals",
    "strip_comments_and_literals",
    "strip_sql_comments",
    "validate_readonly_sql",
    "validate_sql_identifier",
]

"""Pure SQL lexical scanner for comment stripping, literal masking, and read-only safety validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from gemma_4_sql.exceptions import SQLSafetyViolationError

type SQLSegmentKind = Literal["code", "literal", "ident", "comment", "unclosed_comment", "unclosed_quote"]


@dataclass(frozen=True, slots=True)
class SQLSegment:
    """Lexical segment produced by scanning a raw SQL string.

    Attributes:
        kind: Classification of the scanned SQL token span.
        text: Exact source substring covered by this segment.

    """

    kind: SQLSegmentKind
    text: str


ALWAYS_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    "ALTER",
    "ATTACH",
    "CHECKPOINT",
    "CREATE",
    "DEALLOCATE",
    "DELETE",
    "DETACH",
    "DISCARD",
    "DROP",
    "GRANT",
    "INSERT",
    "MERGE",
    "PRAGMA",
    "REINDEX",
    "RENAME",
    "REPLACE",
    "REVOKE",
    "TRUNCATE",
    "UNLISTEN",
    "UPDATE",
    "UPSERT",
    "VACUUM",
})

STATEMENT_VERB_KEYWORDS: frozenset[str] = frozenset({
    "ANALYZE",
    "CALL",
    "CLUSTER",
    "COMMENT",
    "COPY",
    "DECLARE",
    "DO",
    "EXECUTE",
    "EXPORT",
    "IMPORT",
    "INSTALL",
    "LISTEN",
    "LOAD",
    "LOCK",
    "NOTIFY",
    "PREPARE",
    "RESET",
    "SECURITY",
    "SET",
})

DANGEROUS_FUNCTIONS: frozenset[str] = frozenset({
    "DBLINK",
    "LOAD_EXTENSION",
    "LO_EXPORT",
    "LO_IMPORT",
    "PG_CANCEL_BACKEND",
    "PG_LS_DIR",
    "PG_READ_BINARY_FILE",
    "PG_READ_FILE",
    "PG_SLEEP",
    "PG_STAT_FILE",
    "PG_TERMINATE_BACKEND",
    "WRITE_CSV",
    "WRITE_PARQUET",
})

DANGEROUS_KEYWORDS: frozenset[str] = ALWAYS_FORBIDDEN_KEYWORDS | STATEMENT_VERB_KEYWORDS | DANGEROUS_FUNCTIONS

_WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b")
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_DOLLAR_TAG_OPEN_PATTERN = re.compile(r"^\$(?P<tag>|[A-Za-z_][A-Za-z0-9_]*)\$")
_LEADING_WORD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_PAREN_VERB_PATTERN = re.compile(r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\b")
_FUNC_CALL_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_EXPLAIN_ANALYZE_PATTERN = re.compile(r"^\s*EXPLAIN\b[\s\S]*?\bANALYZE\b")
_SELECT_INTO_PATTERN = re.compile(
    r"\bSELECT\b[\s\S]*?\bINTO\s+(?:TEMP\s+|TEMPORARY\s+|UNLOGGED\s+)?(?:TABLE\s+)?[A-Za-z_]"
)
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b")
_PAREN_OR_ORDER_PATTERN = re.compile(r"\(|\)|ORDER_BY")
_BACKSLASH = "\\"
_NEWLINE = "\n"


def _scan_doubled_delim(
    query: str, start: int, quote_start: int, delim: str, ok_kind: SQLSegmentKind
) -> tuple[SQLSegment, int]:
    """Scan a delimited literal or identifier where doubled delimiters escape themselves.

    Args:
        query: Full SQL input string.
        start: Start offset of the token (including any prefix such as `E`).
        quote_start: Index immediately after the opening delimiter character.
        delim: Closing delimiter character (`'`, `"`, `` ` ``, or `]`).
        ok_kind: Segment kind emitted when a closing delimiter is found.

    Returns:
        Tuple of `(segment, next_index)`.

    """
    n = len(query)
    j = quote_start
    while j < n:
        if query[j] == delim:
            if j + 1 < n and query[j + 1] == delim:
                j += 2
                continue
            return (SQLSegment(ok_kind, query[start : j + 1]), j + 1)
        j += 1
    return (SQLSegment("unclosed_quote", query[start:]), n)


def _scan_escape_string(query: str, start: int) -> tuple[SQLSegment, int]:
    """Scan a PostgreSQL `E'...'` escape string supporting both backslash and `''` escapes.

    Args:
        query: Full SQL input string.
        start: Index of the leading `E` or `e` character before `'`.

    Returns:
        Tuple of `(segment, next_index)`.

    """
    n = len(query)
    j = start + 2
    while j < n:
        ch = query[j]
        if ch == _BACKSLASH:
            if j + 1 >= n:
                return (SQLSegment("unclosed_quote", query[start:]), n)
            j += 2
            continue
        if ch == "'":
            if j + 1 < n and query[j + 1] == "'":
                j += 2
                continue
            return (SQLSegment("literal", query[start : j + 1]), j + 1)
        j += 1
    return (SQLSegment("unclosed_quote", query[start:]), n)


def _scan_block_comment(query: str, start: int) -> tuple[SQLSegment, int]:
    """Scan a potentially nested `/* ... */` SQL block comment.

    Args:
        query: Full SQL input string.
        start: Index of the opening `/*` sequence.

    Returns:
        Tuple of `(segment, next_index)`.

    """
    n = len(query)
    depth = 1
    j = start + 2
    while j < n and depth > 0:
        two = query[j : j + 2]
        if two == "/*":
            depth += 1
            j += 2
        elif two == "*/":
            depth -= 1
            j += 2
        else:
            j += 1
    if depth == 0:
        return (SQLSegment("comment", query[start:j]), j)
    return (SQLSegment("unclosed_comment", query[start:]), n)


def scan_sql_segments(query: str) -> list[SQLSegment]:
    """Tokenize a SQL string into code, literal, identifier, and comment segments in O(N) time.

    Supports:
    - ANSI single-quoted string literals (`'...'` with `''` escaping only; backslashes are literal).
    - PostgreSQL escape string literals (`E'...'` / `e'...'` with backslash and `''` escaping).
    - Dollar-quoted string literals (`$$...$$` and `$tag$...$tag$`).
    - Double-quoted (`"..."`), backtick-quoted (`` `...` ``), and bracket-quoted (`[...]`) identifiers.
    - Single-line (`-- ...`) and arbitrarily nested block (`/* ... /* ... */ ... */`) comments.

    Args:
        query: Raw SQL query or DDL text.

    Returns:
        Ordered list of `SQLSegment` spans whose concatenated `.text` equals `query`.

    """
    segments: list[SQLSegment] = []
    code_start = 0
    i = 0
    n = len(query)

    def _flush_code(upto: int) -> None:
        """Append accumulated unquoted SQL text before index `upto`.

        Args:
            upto: Exclusive end offset of the unquoted code span.

        """
        if upto > code_start:
            segments.append(SQLSegment("code", query[code_start:upto]))

    while i < n:
        ch = query[i]
        two = query[i : i + 2]

        if two == "--":
            _flush_code(i)
            nl_idx = query.find(_NEWLINE, i + 2)
            end = n if nl_idx == -1 else nl_idx
            segments.append(SQLSegment("comment", query[i:end]))
            i = end
            code_start = i
            continue

        if two == "/*":
            _flush_code(i)
            seg, i = _scan_block_comment(query, i)
            segments.append(seg)
            code_start = i
            continue

        if (
            ch in {"E", "e"}
            and i + 1 < n
            and query[i + 1] == "'"
            and (i == 0 or not (query[i - 1].isalnum() or query[i - 1] == "_"))
        ):
            _flush_code(i)
            seg, i = _scan_escape_string(query, i)
            segments.append(seg)
            code_start = i
            continue

        if ch == "'":
            _flush_code(i)
            seg, i = _scan_doubled_delim(query, i, i + 1, "'", "literal")
            segments.append(seg)
            code_start = i
            continue

        if ch == '"':
            _flush_code(i)
            seg, i = _scan_doubled_delim(query, i, i + 1, '"', "ident")
            segments.append(seg)
            code_start = i
            continue

        if ch == "`":
            _flush_code(i)
            seg, i = _scan_doubled_delim(query, i, i + 1, "`", "ident")
            segments.append(seg)
            code_start = i
            continue

        if ch == "[":
            _flush_code(i)
            seg, i = _scan_doubled_delim(query, i, i + 1, "]", "ident")
            segments.append(seg)
            code_start = i
            continue

        if ch == "$":
            tag_match = _DOLLAR_TAG_OPEN_PATTERN.match(query[i:])
            if tag_match is not None:
                _flush_code(i)
                delim = tag_match.group(0)
                close_idx = query.find(delim, i + len(delim))
                if close_idx == -1:
                    segments.append(SQLSegment("unclosed_quote", query[i:]))
                    i = n
                else:
                    end = close_idx + len(delim)
                    segments.append(SQLSegment("literal", query[i:end]))
                    i = end
                code_start = i
                continue

        i += 1

    _flush_code(n)
    return segments


def strip_sql_comments(query: str) -> str:
    """Strip SQL block (including nested `/* /* ... */ */`) and single-line comments while preserving quoted literals.

    Args:
        query: Raw SQL query or DDL string.

    Returns:
        SQL string with closed comments replaced by single spaces and literals preserved verbatim.

    """
    return "".join(" " if seg.kind == "comment" else seg.text for seg in scan_sql_segments(query))


def strip_comments_and_literals(query: str) -> str:
    """Replace SQL comments, string literals, and quoted identifiers with single spaces.

    Args:
        query: Raw SQL query string.

    Returns:
        SQL string with all comments and closed quoted segments replaced by spaces.

    """
    return "".join(" " if seg.kind in {"comment", "literal", "ident"} else seg.text for seg in scan_sql_segments(query))


def split_preserving_literals(query: str) -> list[str]:
    """Split a SQL query into alternating unquoted segments and quoted literal/identifier segments.

    Even indices (`0, 2, 4, ...`) are unquoted SQL text; odd indices (`1, 3, 5, ...`) are
    quoted literals or identifiers that must be preserved verbatim.

    Args:
        query: SQL string (typically after `strip_sql_comments`).

    Returns:
        Odd-length list of alternating unquoted and quoted SQL substrings.

    """
    parts: list[str] = [""]
    for seg in scan_sql_segments(query):
        if seg.kind in {"literal", "ident"}:
            parts.extend((seg.text, ""))
        else:
            parts[-1] += seg.text
    return parts


def has_top_level_order_by(query: str) -> bool:
    """Detect presence of a top-level `ORDER BY` clause outside parentheses, comments, and literals.

    Args:
        query: SQL query string.

    Returns:
        `True` if a top-level `ORDER BY` clause is present.

    """
    cleaned = strip_comments_and_literals(query)
    normalized = _ORDER_BY_PATTERN.sub("ORDER_BY", cleaned.upper())
    depth = 0
    for match in _PAREN_OR_ORDER_PATTERN.finditer(normalized):
        tok = match.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth <= 0:
            return True
    return False


def validate_readonly_sql(query: str, *, read_only: bool = True) -> None:
    """Ensure the SQL query is strictly read-only when `read_only` is `True`.

    Args:
        query: SQL query string to inspect.
        read_only: Whether read-only enforcement is active.

    Raises:
        SQLSafetyViolationError: If unclosed comments/literals, stacked statements, mutating DDL/DML,
            dangerous administrative verbs, `EXPLAIN ANALYZE`, `SELECT INTO`, or side-effecting SQL
            functions are present in read-only mode.

    """
    if not read_only:
        return

    segments = scan_sql_segments(query)
    for seg in segments:
        if seg.kind == "unclosed_comment":
            msg = "Safety Violation: Unclosed block comment is not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)
        if seg.kind == "unclosed_quote":
            msg = "Safety Violation: Unclosed string literal or quoted identifier is not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)

    cleaned = "".join(" " if seg.kind in {"comment", "literal", "ident"} else seg.text for seg in segments)
    statements = [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]
    if len(statements) > 1 or cleaned.count(";") > 1:
        msg = "Safety Violation: Multiple stacked SQL statements are not allowed in read-only mode."
        raise SQLSafetyViolationError(msg)

    if not statements:
        return

    upper_stmt = statements[0].upper()

    if _EXPLAIN_ANALYZE_PATTERN.match(upper_stmt):
        msg = "Safety Violation: Mutating statements (EXPLAIN ANALYZE) are not allowed in read-only mode."
        raise SQLSafetyViolationError(msg)

    lead_match = _LEADING_WORD_PATTERN.match(upper_stmt)
    if lead_match is not None:
        lead_word = lead_match.group(0)
        if lead_word in DANGEROUS_KEYWORDS:
            msg = f"Safety Violation: Mutating statements ({lead_word}) are not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)

    for word in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", upper_stmt):
        if word in ALWAYS_FORBIDDEN_KEYWORDS:
            msg = f"Safety Violation: Mutating statements ({word}) are not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)

    for paren_match in _PAREN_VERB_PATTERN.finditer(upper_stmt):
        paren_verb = paren_match.group(1)
        if paren_verb in STATEMENT_VERB_KEYWORDS:
            msg = f"Safety Violation: Mutating statements ({paren_verb}) are not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)

    for fn_match in _FUNC_CALL_PATTERN.finditer(upper_stmt):
        fn_name = fn_match.group(1)
        if fn_name in DANGEROUS_FUNCTIONS:
            msg = f"Safety Violation: Dangerous function ({fn_name}) is not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)

    if _SELECT_INTO_PATTERN.search(upper_stmt):
        msg = "Safety Violation: Mutating statements (SELECT INTO) are not allowed in read-only mode."
        raise SQLSafetyViolationError(msg)


def extract_word_tokens(text: str) -> list[str]:
    """Extract lowercased alphanumeric word tokens from a prompt or SQL fragment.

    Args:
        text: Input string to tokenize into words.

    Returns:
        List of lowercased word tokens.

    """
    return [w.lower() for w in _WORD_TOKEN_PATTERN.findall(text)]


def validate_sql_identifier(identifier: str) -> str:
    """Validate that a SQL table or column identifier matches safe `[a-zA-Z_][a-zA-Z0-9_]*` syntax.

    Args:
        identifier: Raw table or column identifier string.

    Returns:
        The validated identifier string.

    Raises:
        ValueError: If `identifier` contains unsafe characters or does not match identifier syntax.

    """
    if not _IDENTIFIER_PATTERN.match(identifier):
        msg = f"Invalid or unsafe table name: {identifier!r}"
        raise ValueError(msg)
    return identifier


__all__ = [
    "ALWAYS_FORBIDDEN_KEYWORDS",
    "DANGEROUS_FUNCTIONS",
    "DANGEROUS_KEYWORDS",
    "STATEMENT_VERB_KEYWORDS",
    "SQLSegment",
    "SQLSegmentKind",
    "extract_word_tokens",
    "has_top_level_order_by",
    "scan_sql_segments",
    "split_preserving_literals",
    "strip_comments_and_literals",
    "strip_sql_comments",
    "validate_readonly_sql",
    "validate_sql_identifier",
]

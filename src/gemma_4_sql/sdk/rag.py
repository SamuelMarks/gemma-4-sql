"""RAG-based schema contextualization module."""

from __future__ import annotations

import logging
import operator
import re

MIN_SIMILARITY = 0.1
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except (ImportError, ValueError, AttributeError, OSError):
    SentenceTransformer = None
    cosine_similarity = None


def _clean_identifier(identifier: str) -> str:
    """Strip surrounding quotes or brackets from a SQL identifier.

    Args:
        identifier: The raw SQL identifier.

    Returns:
        The unquoted identifier string.
    """
    return identifier.strip("\"'`[]")


def extract_schema_entities(ddl: str) -> dict[str, list[str]]:
    """Extract table names and their corresponding column names from a DDL string.

    Supports standard unquoted identifiers as well as double-quoted,
    backtick-quoted, and bracket-quoted identifiers with schema prefixes.

    Args:
        ddl: The Data Definition Language (DDL) string.

    Returns:
        A dictionary mapping table names to lists of column names.
    """
    schema: dict[str, list[str]] = {}
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[a-zA-Z0-9_]+)\.)?(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[a-zA-Z0-9_]+)\s*\(",
        re.IGNORECASE,
    )
    pos = 0
    while True:
        match = pattern.search(ddl, pos)
        if not match:
            break
        raw_table_name = match.group(1)
        table_name = _clean_identifier(raw_table_name)
        schema[table_name] = []
        start_idx = match.end()
        depth = 1
        i = start_idx
        while i < len(ddl) and depth > 0:
            if ddl[i] == "(":
                depth += 1
            elif ddl[i] == ")":
                depth -= 1
            i += 1
        columns_block = ddl[start_idx : i - 1]
        pos = i

        raw_cols: list[str] = []
        cur: list[str] = []
        d = 0
        for ch in columns_block:
            if ch == "(":
                d += 1
                cur.append(ch)
            elif ch == ")":
                d -= 1
                cur.append(ch)
            elif ch == "," and d == 0:
                raw_cols.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            raw_cols.append("".join(cur).strip())

        ignored_keywords = ("PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT", "UNIQUE", "CHECK", "INDEX")
        col_pattern = re.compile(r"^(?:\"([^\"]+)\"|`([^`]+)`|\[([^\]]+)\]|([a-zA-Z0-9_]+))")

        for col_def in raw_cols:
            c = col_def.strip()
            if not c:
                continue
            c_upper = c.upper()
            if any(c_upper.startswith(kw) for kw in ignored_keywords):
                continue
            col_match = col_pattern.match(c)
            if col_match:
                col_name = next(g for g in col_match.groups() if g is not None)
                schema[table_name].append(col_name)

    return schema


def _score_table(table: str, columns: list[str], prompt_words: set[str]) -> int:
    """Calculate the relevance score for a single table.

    Args:
        table: Table name string.
        columns: List of column names in the table.
        prompt_words: Set of lowercased tokens from the query prompt.

    Returns:
        The relevance score integer.
    """
    score = 0
    if table.lower() in prompt_words:
        score += 5
    for col in columns:
        if col.lower() in prompt_words:
            score += 1
    return score


def _keyword_search(prompt: str, schema: dict[str, list[str]], top_k_tables: int) -> list[str]:
    """Perform keyword matching between prompt and schema entities.

    Args:
        prompt: Natural language question.
        schema: Database schema mapping tables to column lists.
        top_k_tables: Maximum number of tables to return.

    Returns:
        A list of table names sorted by relevance score.
    """
    prompt_words = set(re.findall(r"\b\w+\b", prompt.lower()))
    table_scores = {}
    for table, columns in schema.items():
        table_scores[table] = _score_table(table, columns, prompt_words)
    sorted_tables = sorted(table_scores.items(), key=operator.itemgetter(1), reverse=True)
    relevant_tables = [t[0] for t in sorted_tables[:top_k_tables] if t[1] > 0]
    if not relevant_tables:
        relevant_tables = list(schema.keys())[:top_k_tables]
    return relevant_tables


def _semantic_search(
    prompt: str,
    schema: dict[str, list[str]],
    table_names: list[str],
    top_k_tables: int,
    min_similarity: float = MIN_SIMILARITY,
) -> list[str]:
    """Execute semantic vector embedding retrieval using cosine similarity.

    Args:
        prompt: Natural language query.
        schema: Database schema mapping.
        table_names: Candidate table names.
        top_k_tables: Maximum number of top tables to return.
        min_similarity: Minimum cosine similarity threshold.

    Returns:
        A list of relevant table names.
    """
    if not table_names:
        return []
    if SentenceTransformer is None or cosine_similarity is None:
        return _keyword_search(prompt, schema, top_k_tables)
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        table_docs = [f"Table {t} with columns: {', '.join(schema[t])}" for t in table_names]
        prompt_embedding = model.encode([prompt])
        table_embeddings = model.encode(table_docs)
        similarities = cosine_similarity(prompt_embedding, table_embeddings)[0]
        top_indices = similarities.argsort()[-top_k_tables:][::-1]
        relevant_tables = [table_names[i] for i in top_indices if similarities[i] > min_similarity]
        if not relevant_tables:
            relevant_tables = table_names[:top_k_tables]
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.warning("Failed to use semantic search: %s. Falling back to keyword search.", e)
        return _keyword_search(prompt, schema, top_k_tables)
    else:
        return relevant_tables


def retrieve_relevant_schema(prompt: str, schema: dict[str, list[str]], top_k_tables: int = 2) -> str:
    """Retrieve the most relevant tables and columns based on a natural language prompt.

    This uses semantic vector embeddings (via sentence-transformers) if available,
    falling back to keyword matching otherwise.

    Args:
        prompt: The natural language prompt.
        schema: The parsed database schema.
        top_k_tables: The maximum number of tables to include in the context.

    Returns:
        A formatted string describing the relevant schema parts.
    """
    table_names = list(schema.keys())
    if not table_names:
        return ""
    relevant_tables = _semantic_search(prompt, schema, table_names, top_k_tables) if SentenceTransformer is not None and cosine_similarity is not None else _keyword_search(prompt, schema, top_k_tables)
    context_lines = ["-- Relevant Schema Context:"]
    for table in relevant_tables:
        cols = ", ".join(schema[table])
        context_lines.append(f"-- Table: {table} | Columns: {cols}")
    return "\n".join(context_lines)


def build_rag_prompt(prompt: str, ddl: str | None = None) -> str:
    """Build a prompt augmented with relevant schema information retrieved via RAG.

    Args:
        prompt: The original natural language prompt.
        ddl: Optional DDL string to extract schema context from.

    Returns:
        The augmented prompt string.
    """
    if not ddl:
        return prompt
    schema = extract_schema_entities(ddl)
    context = retrieve_relevant_schema(prompt, schema)
    return f"{context}\n\n-- Request:\n-- {prompt}\n\nSELECT"

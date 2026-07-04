# Copyright 2024
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


def extract_schema_entities(ddl: str) -> dict[str, list[str]]:
    """Extract table names and their corresponding column names from a DDL string.

    Args:
    ----
        ddl: The SQL Data Definition Language string.

    Returns:
    -------
        A dictionary mapping table names to lists of column names.

    """
    schema = {}
    table_pattern = re.compile("CREATE\\s+TABLE\\s+(?:IF\\s+NOT\\s+EXISTS\\s+)?([a-zA-Z0-9_]+)\\s*\\((.*?)\\);?", re.IGNORECASE | re.DOTALL)
    for match in table_pattern.finditer(ddl):
        table_name = match.group(1)
        columns_block = match.group(2)
        schema[table_name] = []
        for raw_col in columns_block.split(","):
            c_def = raw_col.strip()
            if not c_def or c_def.upper().startswith("PRIMARY KEY") or c_def.upper().startswith("FOREIGN KEY"):
                continue
            col_match = re.match("([a-zA-Z0-9_]+)\\b", c_def)
            if col_match:  # pragma: no cover
                schema[table_name].append(col_match.group(1))
    return schema


def _score_table(table: str, columns: list[str], prompt_words: set[str]) -> int:
    """Calculate the relevance score for a single table."""
    score = 0
    if table.lower() in prompt_words:
        score += 5
    for col in columns:
        if col.lower() in prompt_words:
            score += 1
    return score


def _keyword_search(prompt: str, schema: dict[str, list[str]], top_k_tables: int) -> list[str]:
    """Fallback keyword search.

    Returns:
        object: The resulting output from the operation.

    """
    prompt_words = set(re.findall("\\b\\w+\\b", prompt.lower()))
    table_scores = {}
    for table, columns in schema.items():
        table_scores[table] = _score_table(table, columns, prompt_words)
    sorted_tables = sorted(table_scores.items(), key=operator.itemgetter(1), reverse=True)
    relevant_tables = [t[0] for t in sorted_tables[:top_k_tables] if t[1] > 0]
    if not relevant_tables:
        relevant_tables = list(schema.keys())[:top_k_tables]
    return relevant_tables


def _semantic_search(prompt: str, schema: dict[str, list[str]], table_names: list[str], top_k_tables: int) -> list[str]:
    """Execute semantic vector embedding retrieval.

    Returns:
        object: The resulting output from the operation.

    """
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        table_docs = [f"Table {t} with columns: {', '.join(schema[t])}" for t in table_names]
        prompt_embedding = model.encode([prompt])
        table_embeddings = model.encode(table_docs)
        similarities = cosine_similarity(prompt_embedding, table_embeddings)[0]
        top_indices = similarities.argsort()[-top_k_tables:][::-1]
        relevant_tables = [table_names[i] for i in top_indices if similarities[i] > MIN_SIMILARITY]
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
    ----
        prompt: The natural language prompt.
        schema: The parsed database schema.
        top_k_tables: The maximum number of tables to include in the context.

    Returns:
    -------
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
    ----
        prompt: The original natural language prompt.
        ddl: Optional DDL string to extract schema context from.

    Returns:
    -------
        The augmented prompt.

    """
    if not ddl:
        return prompt
    schema = extract_schema_entities(ddl)
    context = retrieve_relevant_schema(prompt, schema)
    return f"{context}\n\n-- Request:\n-- {prompt}\n\nSELECT"

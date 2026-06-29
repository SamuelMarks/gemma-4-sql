"""RAG-based schema contextualization module."""

from __future__ import annotations

import re


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
            if col_match:
                schema[table_name].append(col_match.group(1))
    return schema


def retrieve_relevant_schema(prompt: str, schema: dict[str, list[str]], top_k_tables: int = 2) -> str:
    """Retrieve the most relevant tables and columns based on a natural language prompt.

    This simulates a RAG retrieval step using basic keyword matching.

    Args:
    ----
        prompt: The natural language prompt.
        schema: The parsed database schema.
        top_k_tables: The maximum number of tables to include in the context.

    Returns:
    -------
        A formatted string describing the relevant schema parts.

    """
    prompt_words = set(re.findall("\\b\\w+\\b", prompt.lower()))
    table_scores = {}
    for table, columns in schema.items():
        score = 0
        if table.lower() in prompt_words:
            score += 5
        for col in columns:
            if col.lower() in prompt_words:
                score += 1
        table_scores[table] = score
    sorted_tables = sorted(table_scores.items(), key=lambda item: item[1], reverse=True)
    relevant_tables = [t[0] for t in sorted_tables[:top_k_tables] if t[1] > 0]
    if not relevant_tables:
        relevant_tables = list(schema.keys())[:top_k_tables]
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

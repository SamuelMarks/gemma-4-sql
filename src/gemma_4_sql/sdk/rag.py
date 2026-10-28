"""RAG-based schema contextualization module."""

from __future__ import annotations

import hashlib
import logging
import math
import operator
import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_inference import BoundedLRUCache, make_model_cache_key
from gemma_4_sql.backends.lazy_loader import optional_attr
from gemma_4_sql.exceptions import SchemaExtractionError
from gemma_4_sql.sdk.sql_lexer import extract_word_tokens, strip_sql_comments
from gemma_4_sql.type_hints import EmbedderModelProtocol, NDArrayProtocol

MIN_SIMILARITY = 0.1
MAX_SCHEMA_CACHE_SIZE = 128
EMBEDDING_DIM = 64
logger = logging.getLogger(__name__)


class SimilarityRowProtocol(Protocol):
    """Protocol for a single row of pairwise cosine similarity scores."""

    def argsort(self) -> Sequence[int]:
        """Return indices that would sort the similarity row in ascending order."""
        ...

    def __getitem__(self, index: int) -> float:
        """Return similarity score at `index`."""
        ...


type EmbeddingMatrix = NDArrayProtocol | Sequence[Sequence[float]] | Sequence[float]


@runtime_checkable
class SentenceTransformerConstructorProtocol(Protocol):
    """Protocol for the `SentenceTransformer` class constructor."""

    def __call__(self, model_name_or_path: str) -> EmbedderModelProtocol:
        """Instantiate a `SentenceTransformer` embedder model."""
        ...


@runtime_checkable
class CosineSimilarityCallableProtocol(Protocol):
    """Protocol for `sklearn.metrics.pairwise.cosine_similarity`."""

    def __call__(
        self,
        x: EmbeddingMatrix,
        y: EmbeddingMatrix,
    ) -> Sequence[SimilarityRowProtocol]:
        """Compute pairwise cosine similarity between two embedding matrices."""
        ...


SentenceTransformer: Callable[[str], EmbedderModelProtocol] | None = optional_attr(
    "sentence_transformers",
    "SentenceTransformer",
    SentenceTransformerConstructorProtocol,
)
cosine_similarity: Callable[[EmbeddingMatrix, EmbeddingMatrix], Sequence[SimilarityRowProtocol]] | None = optional_attr(
    "sklearn.metrics.pairwise",
    "cosine_similarity",
    CosineSimilarityCallableProtocol,
)

_SCHEMA_EMBEDDING_CACHE: BoundedLRUCache[EmbeddingMatrix] = BoundedLRUCache(maxsize=MAX_SCHEMA_CACHE_SIZE)
_SENTENCE_TRANSFORMER_CACHE: BoundedLRUCache[EmbedderModelProtocol] = BoundedLRUCache(maxsize=8)


def _compute_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Compute a deterministic L2-normalized 64-D projection vector over word tokens and character 3-grams.

    Args:
        text: Input text or schema table description.
        dim: Output embedding vector dimensionality (defaults to 64).

    Returns:
        L2-normalized list of floats of length `dim`.

    """
    vec = [0.0] * dim
    words = extract_word_tokens(text)
    features: list[tuple[str, float]] = []
    for w in words:
        features.append((f"w:{w}", 2.0))
        for part in _expand_identifier_tokens(w):
            features.append((f"p:{part}", 1.5))
        padded = f"<{w}>"
        for i in range(max(0, len(padded) - 2)):
            features.append((f"g:{padded[i : i + 3]}", 0.5))

    if not features:
        return vec

    for feat, weight in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        for idx in range(min(dim, len(digest) * 2)):
            byte_val = digest[idx % len(digest)]
            bit = (byte_val >> (idx % 8)) & 1
            sign = 1.0 if bit else -1.0
            vec[idx] += sign * weight

    norm = max(1e-12, math.sqrt(sum(v * v for v in vec)))
    return [v / norm for v in vec]


def clear_rag_model_cache() -> None:
    """Clear the cached SentenceTransformer models and schema embeddings in a thread-safe manner."""
    _SCHEMA_EMBEDDING_CACHE.clear()
    _SENTENCE_TRANSFORMER_CACHE.clear()


def _get_sentence_transformer(model_id: str = "all-MiniLM-L6-v2") -> EmbedderModelProtocol:
    """Return a cached `SentenceTransformer` instance keyed by class and model identifier.

    Args:
        model_id: Sentence-Transformer model identifier.

    Returns:
        Instantiated or cached `SentenceTransformer` model.

    """
    cache_key = make_model_cache_key(model_id, backend_name="rag", factories=(SentenceTransformer,))
    cached = _SENTENCE_TRANSFORMER_CACHE.get_model(cache_key)
    if cached is None and SentenceTransformer is not None:
        cached = SentenceTransformer(model_id)
        _SENTENCE_TRANSFORMER_CACHE.put_model(cache_key, cached)
    if cached is None:
        msg = "SentenceTransformer is unavailable"
        raise RuntimeError(msg)
    return cached


def _clean_identifier(identifier: str) -> str:
    """Strip surrounding quotes or brackets from a SQL identifier.

    Args:
        identifier: The raw SQL identifier.

    Returns:
        The unquoted identifier string.

    """
    return identifier.strip("\"`'[]")


def extract_schema_entities(ddl: str, *, strict: bool = False) -> dict[str, list[str]]:
    """Extract table names and their corresponding column names from a DDL string.

    Uses a balanced-parenthesis scanner and supports standard unquoted identifiers
    as well as double-quoted, backtick-quoted, and bracket-quoted identifiers with
    optional catalog/schema prefixes and `TEMPORARY`/`TEMP` modifiers.

    Args:
        ddl: The Data Definition Language (DDL) string.
        strict: If True, raises `SchemaExtractionError` when a `CREATE TABLE` block has unbalanced parentheses.

    Returns:
        A dictionary mapping table names to lists of column names.

    Raises:
        SchemaExtractionError: If `strict=True` and unbalanced parentheses are encountered.

    """
    cleaned_ddl = strip_sql_comments(ddl)
    schema: dict[str, list[str]] = {}
    ident_pat = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[a-zA-Z0-9_]+)'
    pattern = re.compile(
        rf"CREATE\s+(?:(?:OR\s+REPLACE|TEMPORARY|TEMP|MATERIALIZED)\s+)*(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:{ident_pat}\.)*({ident_pat})\s*\(",
        re.IGNORECASE,
    )
    pos = 0
    while True:
        match = pattern.search(cleaned_ddl, pos)
        if not match:
            break
        raw_table_name = match.group(1)
        table_name = _clean_identifier(raw_table_name)
        schema[table_name] = []
        start_idx = match.end()
        depth = 1
        i = start_idx
        while i < len(cleaned_ddl) and depth > 0:
            if cleaned_ddl[i] == "(":
                depth += 1
            elif cleaned_ddl[i] == ")":
                depth -= 1
            i += 1
        if depth > 0 and strict:
            raise SchemaExtractionError(f"Unbalanced parentheses in CREATE TABLE '{table_name}'")
        columns_block = cleaned_ddl[start_idx : i - 1]
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
        col_pattern = re.compile(r'^(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([a-zA-Z0-9_]+))')

        for col_def in raw_cols:
            c = col_def.strip()
            if not c:
                continue
            c_upper = " ".join(c.upper().split())
            if any(c_upper.startswith(kw) for kw in ignored_keywords):
                continue
            col_match = col_pattern.match(c)
            if col_match:
                col_name = next(g for g in col_match.groups() if g is not None)
                schema[table_name].append(col_name)

    return schema


def _expand_identifier_tokens(ident: str) -> list[str]:
    """Expand an identifier into its lowercased form plus snake_case sub-tokens.

    Args:
        ident: Table or column identifier string.

    Returns:
        List of lowercased tokens including full identifier and underscore-split parts.

    """
    low = ident.lower()
    parts = [p for p in low.split("_") if p and p != low]
    return [low, *parts]


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
    if any(tok in prompt_words for tok in _expand_identifier_tokens(table)):
        score += 5
    for col in columns:
        if any(tok in prompt_words for tok in _expand_identifier_tokens(col)):
            score += 1
    return score


def _bm25_search(
    prompt: str,
    schema: dict[str, list[str]],
    top_k_tables: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[str]:
    """Rank tables using Okapi BM25 scoring over table names and column descriptions.

    Args:
        prompt: Natural language query string.
        schema: Parsed schema mapping table names to column lists.
        top_k_tables: Maximum number of tables to select.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        List of prioritized table names.

    """
    prompt_tokens = extract_word_tokens(prompt)
    if not prompt_tokens or not schema:
        return list(schema.keys())[:top_k_tables]

    doc_tokens: dict[str, list[str]] = {}
    total_len = 0
    for t, cols in schema.items():
        t_expanded = _expand_identifier_tokens(t)
        c_expanded = [tok for c in cols for tok in _expand_identifier_tokens(c)]
        tokens = t_expanded * 3 + c_expanded
        doc_tokens[t] = tokens
        total_len += len(tokens)

    avg_dl = max(1.0, float(total_len) / float(len(schema)))
    n_docs = len(schema)

    df: dict[str, int] = {}
    for token in set(prompt_tokens):
        df[token] = sum(1 for tokens in doc_tokens.values() if token in tokens)

    scores: dict[str, float] = {}
    for t, tokens in doc_tokens.items():
        dl = len(tokens)
        score = 0.0
        for token in prompt_tokens:
            if token not in df or df[token] == 0:
                continue
            tf = tokens.count(token)
            if tf == 0:
                continue
            idf = math.log((n_docs - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
            num = tf * (k1 + 1.0)
            denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
            score += idf * (num / denom)
        scores[t] = score

    sorted_tables = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    selected = [t[0] for t in sorted_tables[:top_k_tables] if t[1] > 0.0]
    if not selected:
        p_vec = _compute_embedding(prompt)
        proj_scores: list[tuple[str, float]] = []
        for t, cols in schema.items():
            t_vec = _compute_embedding(f"{t} {' '.join(cols)}")
            dot = sum(a * b for a, b in zip(p_vec, t_vec, strict=False))
            proj_scores.append((t, dot))
        proj_scores.sort(key=operator.itemgetter(1), reverse=True)
        selected = [t[0] for t in proj_scores[:top_k_tables] if t[1] > 0.35]
    if not selected:
        selected = list(schema.keys())[:top_k_tables]
    return selected


def _keyword_search(prompt: str, schema: dict[str, list[str]], top_k_tables: int) -> list[str]:
    """Perform keyword matching between prompt and schema entities with BM25 refinement.

    Args:
        prompt: Natural language question.
        schema: Database schema mapping tables to column lists.
        top_k_tables: Maximum number of tables to return.

    Returns:
        A list of table names sorted by relevance score.

    """
    prompt_words = set(extract_word_tokens(prompt))
    table_scores: dict[str, float] = {}
    for table, columns in schema.items():
        table_scores[table] = _score_table(table, columns, prompt_words)
    sorted_tables: list[tuple[str, float]] = sorted(table_scores.items(), key=operator.itemgetter(1), reverse=True)
    relevant_tables: list[str] = [t[0] for t in sorted_tables[:top_k_tables] if t[1] > 0]
    if not relevant_tables:
        relevant_tables = _bm25_search(prompt, schema, top_k_tables)
    return relevant_tables


def _semantic_search(
    prompt: str,
    schema: dict[str, list[str]],
    table_names: list[str],
    top_k_tables: int,
    min_similarity: float = MIN_SIMILARITY,
) -> list[str]:
    """Execute semantic vector embedding retrieval using cosine similarity with model and embedding caching.

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
        model: EmbedderModelProtocol = _get_sentence_transformer("all-MiniLM-L6-v2")
        table_docs = [f"Table {t} with columns: {', '.join(schema[t])}" for t in table_names]

        schema_key = hashlib.sha256(";;".join(table_docs).encode()).hexdigest()
        _SCHEMA_EMBEDDING_CACHE.maxsize = MAX_SCHEMA_CACHE_SIZE
        cached_embeddings = _SCHEMA_EMBEDDING_CACHE.get_model(schema_key)
        if cached_embeddings is not None:
            table_embeddings = cached_embeddings
        else:
            table_embeddings = model.encode(table_docs)
            _SCHEMA_EMBEDDING_CACHE.put_model(schema_key, table_embeddings)

        prompt_embedding = model.encode([prompt])
        similarities = cosine_similarity(prompt_embedding, table_embeddings)[0]
        top_indices = list(similarities.argsort())[-top_k_tables:][::-1]
        relevant_tables: list[str] = [table_names[int(i)] for i in top_indices if float(similarities[int(i)]) > min_similarity]
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
    falling back to keyword matching and BM25 ranking otherwise.

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
    return chr(10).join(context_lines)


def build_rag_prompt(
    prompt: str,
    ddl: str | None = None,
    *,
    sql_prefix: str = "SELECT",
    top_k_tables: int = 2,
    image_path: str | None = None,
    audio_path: str | None = None,
) -> str:
    """Build a prompt augmented with relevant schema information retrieved via RAG.

    Args:
        prompt: The original natural language prompt.
        ddl: Optional DDL string to extract schema context from.
        sql_prefix: Optional trailing SQL continuation prefix (defaults to `'SELECT'`; pass `''` to omit).
        top_k_tables: Maximum number of relevant tables to include in the context.
        image_path: Optional path to a schema diagram or ERD image for multimodal formatting.
        audio_path: Optional path to a recorded query audio file for multimodal formatting.

    Returns:
        The augmented prompt string.

    """
    if not ddl:
        base_prompt = prompt
    else:
        schema = extract_schema_entities(ddl)
        context = retrieve_relevant_schema(prompt, schema, top_k_tables=top_k_tables)
        if sql_prefix:
            base_prompt = f"""{context}

-- Request:
-- {prompt}

{sql_prefix}"""
        else:
            base_prompt = f"""{context}

-- Request:
-- {prompt}"""

    if image_path is not None or audio_path is not None:
        from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

        formatted = format_multimodal_prompt(
            base_prompt,
            has_image=image_path is not None,
            has_audio=audio_path is not None,
        )
        return str(formatted.get("prompt", base_prompt))
    return base_prompt


__all__ = [
    "EMBEDDING_DIM",
    "MIN_SIMILARITY",
    "_compute_embedding",
    "build_rag_prompt",
    "clear_rag_model_cache",
    "extract_schema_entities",
    "retrieve_relevant_schema",
]

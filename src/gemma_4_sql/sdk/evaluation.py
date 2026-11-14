"""SDK Evaluation module for Text-to-SQL exact match and execution accuracy."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import starmap
from pathlib import Path

from gemma_4_sql._types_tensors import NestedComplexScalarList
from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.exceptions import EvaluationError, GemmaSQLError, is_failed_status
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.sdk.sql_lexer import split_preserving_literals, strip_sql_comments
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    ETLConfig,
    EvaluationMetricsDict,
    EvaluationResultDict,
    JSONValue,
    NestedScalarList,
    TensorProtocol,
    TensorType,
    ToListArrayProtocol,
    extract_float_kwarg,
    extract_int_kwarg,
    run_coroutine_sync,
)

MAX_BATCHES = 10
_COMPARISON_OP_PATTERN = re.compile(r"\s*(!=|<>|<=|>=|=|<|>)\s*")

type TokenSequenceItem = (
    Sequence[Sequence[int]]
    | Sequence[int]
    | Sequence[str]
    | Iterable[int]
    | Sequence[ToListArrayProtocol]
    | NestedScalarList
    | NestedComplexScalarList
    | ToListArrayProtocol
    | TensorProtocol
    | TensorType
    | int
    | float
    | str
    | bytes
    | None
)
type TokenSequenceBatch = (
    Mapping[str, TokenSequenceItem]
    | tuple[TokenSequenceItem, TokenSequenceItem]
    | Sequence[TokenSequenceItem]
    | int
    | str
    | None
)


def normalize_sql(sql: str) -> str:
    """Normalize SQL comments, punctuation/operator spacing, and unquoted casing.

    Args:
        sql: The string representing the SQL query.

    Returns:
        The normalized SQL query string.

    """
    cleaned = strip_sql_comments(sql)

    parts = split_preserving_literals(cleaned)
    for i in range(0, len(parts), 2):
        segment = parts[i].lower()
        segment = re.sub(r"\s*,\s*", ", ", segment)
        segment = re.sub(r"\(\s+", "(", segment)
        segment = re.sub(r"\s+\)", ")", segment)
        segment = _COMPARISON_OP_PATTERN.sub(r" \1 ", segment)
        parts[i] = segment

    res = "".join(parts)
    return " ".join(res.strip().rstrip(";").split())


async def compute_metrics_async(
    engine: LiveDatabaseEngine,
    preds: list[str],
    truths: list[str],
) -> EvaluationMetricsDict:
    """Compute exact match, valid SQL, and execution accuracy asynchronously.

    Args:
        engine: The LiveDatabaseEngine instance for SQL execution.
        preds: List of predicted SQL query strings.
        truths: List of ground truth SQL query strings.

    Returns:
        A dictionary containing `exact_match`, `valid_sql`, and `execution_accuracy` metrics.

    Raises:
        ValueError: If predictions and ground truths lengths differ.

    """
    if len(preds) != len(truths):
        msg = f"Length mismatch: {len(preds)} predictions vs {len(truths)} truths"
        raise ValueError(msg)

    exact_matches = 0
    valid_sqls = 0
    exec_matches = 0

    async def process_pair(p: str, t: str) -> tuple[int, int, int]:
        """Process a single prediction and ground truth pair.

        Args:
            p: Predicted SQL query string.
            t: Ground truth SQL query string.

        Returns:
            Tuple of `(exact_match, valid_sql, execution_match)` binary flags.

        """
        em = 1 if normalize_sql(p) == normalize_sql(t) else 0
        try:
            (p_success, p_results, _) = await engine.execute_with_feedback_async(p)
            vs = 1 if p_success else 0
            (t_success, t_results, _) = await engine.execute_with_feedback_async(t)

            if p_success and t_success:
                strict_order = engine.has_top_level_order_by(t)
                ex = 1 if engine.compare_results(p_results, t_results, strict_order) else 0
            else:
                ex = 0
        except (GemmaSQLError, RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as exc:
            import logging

            logging.getLogger(__name__).warning("Query execution failed: %s", exc)
            vs = 0
            ex = 0
        return (em, vs, ex)

    results = await asyncio.gather(*list(starmap(process_pair, zip(preds, truths, strict=False))))
    for em, vs, ex in results:
        exact_matches += em
        valid_sqls += vs
        exec_matches += ex
    total = len(preds) if preds else 1
    return {
        "exact_match": exact_matches / total,
        "valid_sql": valid_sqls / total,
        "execution_accuracy": exec_matches / total,
    }


def compute_metrics(
    engine: LiveDatabaseEngine,
    preds: list[str],
    truths: list[str],
) -> EvaluationMetricsDict:
    """Compute exact match, valid SQL, and execution accuracy synchronously.

    Args:
        engine: The LiveDatabaseEngine instance.
        preds: A sequence of predicted SQL queries.
        truths: A sequence of ground truth SQL queries.

    Returns:
        Dictionary containing `exact_match`, `valid_sql`, and `execution_accuracy` metrics.

    """
    res: EvaluationMetricsDict = run_coroutine_sync(lambda: compute_metrics_async(engine, preds, truths))
    return res


def _to_int_seq(seq: TokenSequenceItem) -> list[int]:
    """Convert a 1D tensor, array, or sequence of token IDs into a Python `list[int]`.

    Args:
        seq: 1D tensor, array, or sequence of token IDs.

    Returns:
        List of integer token IDs.

    """
    unwrapped: TokenSequenceItem = seq.tolist() if isinstance(seq, ToListArrayProtocol) else seq
    if isinstance(unwrapped, (int, float, bool)):
        return [int(unwrapped)]
    if isinstance(unwrapped, (str, bytes)) or unwrapped is None or not isinstance(unwrapped, Iterable):
        return []
    out: list[int] = [
        int(elem)
        for elem in unwrapped
        if isinstance(elem, (int, float, bool)) or (isinstance(elem, str) and elem.lstrip("-").isdigit())
    ]
    return out


def _normalize_id_matrix(raw_item: TokenSequenceItem) -> list[list[int]]:
    """Normalize a 1D or 2D tensor/sequence of token IDs into a 2D `list[list[int]]`.

    Args:
        raw_item: Raw inputs or targets object from a batch.

    Returns:
        2D list of integer token ID sequences.

    """
    if raw_item is None or isinstance(raw_item, (str, bytes)):
        return [[]]
    unwrapped: TokenSequenceItem | list[TokenSequenceItem]
    if isinstance(raw_item, ToListArrayProtocol):
        unwrapped = raw_item.tolist()
    elif isinstance(raw_item, Iterable):
        try:
            unwrapped = list(raw_item)
        except TypeError:
            return [[]]
    else:
        return [[]]

    if isinstance(unwrapped, (int, float, bool, complex)):
        scalar_i = int(unwrapped.real) if isinstance(unwrapped, complex) else int(unwrapped)
        return [[scalar_i]]
    if isinstance(unwrapped, (str, bytes)) or len(unwrapped) == 0:
        return [[]]
    first = unwrapped[0]
    if isinstance(first, (int, float, bool)):
        return [[int(x) for x in unwrapped if isinstance(x, (int, float, bool))]]
    return [_to_int_seq(row) for row in unwrapped]


def _coerce_multimodal_json(
    raw_val: TensorType
    | Sequence[int]
    | Sequence[Sequence[int]]
    | TensorProtocol
    | ToListArrayProtocol
    | NestedComplexScalarList
    | JSONValue
    | bytes,
) -> JSONValue:
    """Convert a batch tensor or nested sequence into a genuine `JSONValue` for `generate_sql`.

    Args:
        raw_val: Multimodal `pixel_values` or `audio_values` from an evaluation batch.

    Returns:
        Genuine JSON-compatible nested list or scalar (`None` if unsupported).

    """
    candidate: _MultimodalCandidateType = raw_val.tolist() if isinstance(raw_val, ToListArrayProtocol) else raw_val
    if candidate is None or isinstance(candidate, (bool, int, float, str)):
        return candidate
    if isinstance(candidate, Mapping):
        return {str(k): _coerce_multimodal_json(v) for k, v in candidate.items()}
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        return [_coerce_multimodal_json(elem) for elem in candidate]
    return None


type _MultimodalCandidateType = (
    TensorType
    | Sequence[int]
    | Sequence[Sequence[int]]
    | TensorProtocol
    | ToListArrayProtocol
    | NestedComplexScalarList
    | JSONValue
    | bytes
)


def _extract_batch_ids(batch: TokenSequenceBatch) -> tuple[list[list[int]], list[list[int]]]:
    """Extract all input and target token ID sequences from a batch structure.

    Handles PyTorch tensors, JAX arrays, MLX arrays, Keras tensors, lists, and dicts uniformly.

    Args:
        batch: Batch tuple, list, or dictionary containing `'inputs'` and `'targets'`.

    Returns:
        Tuple of `(batch_input_ids, batch_target_ids)` where each is a `list[list[int]]`.

    """
    min_batch_tuple_length = 2
    raw_inputs: TokenSequenceItem = None
    raw_targets: TokenSequenceItem = None

    if isinstance(batch, (tuple, list)) and len(batch) >= min_batch_tuple_length:
        raw_inputs = batch[0]
        raw_targets = batch[1]
    elif isinstance(batch, Mapping) and "inputs" in batch and "targets" in batch:
        raw_inputs = batch["inputs"]
        raw_targets = batch["targets"]

    if raw_inputs is None or raw_targets is None:
        return ([[]], [[]])

    return (_normalize_id_matrix(raw_inputs), _normalize_id_matrix(raw_targets))


def _process_batch_inputs(batch: TokenSequenceBatch) -> tuple[list[int], list[int]]:
    """Extract first-element input and target IDs from a batch.

    Args:
        batch: The batch data structure (tuple, list, or mapping).

    Returns:
        Tuple of `(input_ids, target_ids)` lists for the first batch item.

    """
    in_seqs, tgt_seqs = _extract_batch_ids(batch)
    return (in_seqs[0] if in_seqs else [], tgt_seqs[0] if tgt_seqs else [])


def _run_evaluation_inference(
    model_name: str,
    dataset_name: str,
    backend_impl: BackendProtocol,
    image_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    modality: str = "text",
    max_batches: int | None = MAX_BATCHES,
    **kwargs: JSONValue,
) -> tuple[list[str], list[str], list[float]]:
    """Run inference for evaluation across text and multimodal benchmarks.

    Args:
        model_name: Target model name.
        dataset_name: Target dataset name.
        backend_impl: Backend protocol implementation.
        image_path: Optional path to schema diagram or ERD image.
        audio_path: Optional path to recorded query audio.
        modality: Explicit modality selector (`'text'`, `'vision'`, `'audio'`, `'multimodal'`).
        max_batches: Maximum number of batches to evaluate (`None` for default `MAX_BATCHES`).
        **kwargs: Additional inference arguments.

    Returns:
        Tuple of `(predictions, truths, confidence_scores)`.

    Raises:
        EvaluationError: If valid iterable dataloader is missing.

    """
    preds: list[str] = []
    truths: list[str] = []
    confidence_scores: list[float] = []
    raw_tok_name = kwargs.get("tokenizer_name")
    tokenizer_name = str(raw_tok_name) if isinstance(raw_tok_name, str) and raw_tok_name else None
    data_dict = backend_impl.build_dataloader(
        ETLConfig(
            dataset_name=dataset_name,
            split="test",
            batch_size=1,
            modality=modality,
            tokenizer_name=tokenizer_name,
        ),
    )
    dataloader = data_dict.get("loader", None)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    if dataloader is None:
        msg = "Valid iterable dataloader is required for evaluation; dataset could not be constructed."
        raise EvaluationError(msg, model_name=model_name, operation="evaluate")

    limit = MAX_BATCHES if max_batches is None else int(max_batches)
    for i, batch in enumerate(dataloader):
        if limit > 0 and i >= limit:
            break

        batch_inputs, batch_targets = _extract_batch_ids(batch)
        for inp_seq, tgt_seq in zip(batch_inputs, batch_targets, strict=False):
            prompt_text = tokenizer.decode(inp_seq)
            truth_text = tokenizer.decode(tgt_seq)
            if prompt_text and truth_text.startswith(prompt_text) and len(truth_text) > len(prompt_text):
                truth_text = truth_text[len(prompt_text) :].strip()

            img_arg: JSONValue = str(image_path) if image_path is not None else None
            pv_arg: JSONValue = (
                _coerce_multimodal_json(batch["pixel_values"])
                if (image_path is None and "pixel_values" in batch)
                else None
            )

            aud_arg: JSONValue = str(audio_path) if audio_path is not None else None
            av_arg: JSONValue = (
                _coerce_multimodal_json(batch["audio_values"])
                if (audio_path is None and "audio_values" in batch)
                else None
            )

            gen_res = backend_impl.generate_sql(
                model_name,
                prompt_text,
                modality=modality,
                image_path=img_arg,
                pixel_values=pv_arg,
                audio_path=aud_arg,
                audio_values=av_arg,
            )
            sql_pred = "" if is_failed_status(gen_res) else str(gen_res.get("sql", ""))
            conf_score = 0.0 if is_failed_status(gen_res) else extract_float_kwarg(gen_res, "confidence_score", 0.0)
            preds.append(sql_pred)
            confidence_scores.append(conf_score)
            truths.append(truth_text)
    return (preds, truths, confidence_scores)


def evaluate(
    model_name: str,
    dataset_name: str,
    backend: str = DEFAULT_BACKEND,
    db_path: str = ":memory:",
    ddl: str | None = None,
    max_batches: int = MAX_BATCHES,
    **kwargs: JSONValue,
) -> EvaluationResultDict:
    """Evaluate a Text-to-SQL model on exact match and execution accuracy.

    Args:
        model_name: The name or path of the model.
        dataset_name: The dataset to evaluate against.
        backend: The backend framework ('jax', 'keras', 'pytorch', etc.).
        db_path: Path to the database for execution accuracy.
        ddl: Optional DDL to set up the schema.
        max_batches: Maximum number of dataset batches to evaluate.
        **kwargs: Additional options such as `db_type`, `db_kwargs`, `image_path`, and `audio_path`.

    Returns:
        Evaluation results dictionary containing status and metrics.

    """
    db_type = kwargs.get("db_type", "sqlite")
    db_kwargs = kwargs.get("db_kwargs")
    db_kwargs_dict = db_kwargs if isinstance(db_kwargs, dict) else {}
    image_path = kwargs.get("image_path")
    audio_path = kwargs.get("audio_path")
    modality = str(kwargs.get("modality", "text"))

    with LiveDatabaseEngine(db_path=db_path, ddl=ddl, db_type=str(db_type), db_kwargs=db_kwargs_dict) as engine:
        backend_impl = get_backend(backend)

        eval_kwargs = dict(kwargs)
        eval_kwargs.pop("image_path", None)
        eval_kwargs.pop("audio_path", None)
        eval_kwargs.pop("modality", None)
        resolved_max_batches = extract_int_kwarg(eval_kwargs, "max_batches", max_batches)
        eval_kwargs.pop("max_batches", None)

        (preds, truths, confidence_scores) = _run_evaluation_inference(
            model_name,
            dataset_name,
            backend_impl,
            image_path=str(image_path) if image_path else None,
            audio_path=str(audio_path) if audio_path else None,
            modality=modality,
            max_batches=resolved_max_batches,
            **eval_kwargs,
        )

        metrics = compute_metrics(engine, preds, truths)
        if confidence_scores:
            metrics["mean_confidence"] = sum(confidence_scores) / len(confidence_scores)
    res: EvaluationResultDict = {
        "backend": backend,
        "model": model_name,
        "dataset": dataset_name,
        "modality": modality,
        "status": "completed",
        "metrics": metrics,
    }
    if image_path:
        res["image_path"] = str(image_path)
    if audio_path:
        res["audio_path"] = str(audio_path)
    return res


__all__ = [
    "MAX_BATCHES",
    "TokenSequenceBatch",
    "TokenSequenceItem",
    "_process_batch_inputs",
    "compute_metrics",
    "compute_metrics_async",
    "evaluate",
    "normalize_sql",
]

"""SDK Evaluation module for Text-to-SQL exact match and execution accuracy."""

from __future__ import annotations

import asyncio
import re
import typing
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig

MAX_BATCHES = 10
if TYPE_CHECKING:
    from pathlib import Path

    from gemma_4_sql.sdk.protocols import BackendProtocol
    from gemma_4_sql.type_hints import EvaluationMetricsDict, EvaluationResultDict, JSONValue


def normalize_sql(sql: str) -> str:
    """Normalize SQL by stripping comments, standardizing punctuation spacing, and lowercasing unquoted text.

    Args:
        sql: The string representing the SQL query.

    Returns:
        The normalized SQL query string.

    """
    cleaned = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    cleaned = re.sub(r"--.*$", " ", cleaned, flags=re.MULTILINE)

    parts = re.split(r"""('[^']*'|"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\])""", cleaned)
    for i in range(0, len(parts), 2):
        segment = parts[i].lower()
        segment = re.sub(r"\s*,\s*", ", ", segment)
        segment = re.sub(r"\(\s+", "(", segment)
        segment = re.sub(r"\s+\)", ")", segment)
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
        """Process a single prediction and ground truth pair."""
        em = 1 if normalize_sql(p) == normalize_sql(t) else 0
        try:
            (p_success, p_results, _) = await engine.execute_with_feedback_async(p)
            vs = 1 if p_success else 0
            (t_success, t_results, _) = await engine.execute_with_feedback_async(t)

            if p_success and t_success:
                strict_order = engine._has_top_level_order_by(t)
                ex = 1 if engine._compare_results(p_results, t_results, strict_order) else 0
            else:
                ex = 0
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("Query execution failed: %s", exc)
            vs = 0
            ex = 0
        return (em, vs, ex)

    results = await asyncio.gather(*[process_pair(p, t) for p, t in zip(preds, truths)])
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
    return asyncio.run(compute_metrics_async(engine, preds, truths))


def _process_batch_inputs(batch: object) -> tuple[list[int], list[int]]:
    """Extract input and target IDs from a batch.

    Args:
        batch: The batch data structure (tuple, list, or mapping).

    Returns:
        Tuple of `(input_ids, target_ids)` lists.

    """
    min_batch_tuple_length = 2
    input_ids: Any = []
    target_ids: Any = []

    if isinstance(batch, (tuple, list)) and len(cast(list[Any], batch)) >= min_batch_tuple_length:
        batch_list = cast(list[Any], batch)
        input_ids = batch_list[0][0].tolist() if hasattr(batch_list[0][0], "tolist") else batch_list[0][0]
        target_ids = batch_list[1][0].tolist() if hasattr(batch_list[1][0], "tolist") else batch_list[1][0]
    elif isinstance(batch, dict):
        batch_dict = cast(dict[str, Any], batch)
        if "inputs" in batch_dict and "targets" in batch_dict:
            inputs = batch_dict["inputs"]
            targets = batch_dict["targets"]
            if hasattr(inputs, "__getitem__") and hasattr(targets, "__getitem__"):
                in0 = inputs[0]
                tgt0 = targets[0]
                input_ids = in0.tolist() if hasattr(in0, "tolist") else in0
                target_ids = tgt0.tolist() if hasattr(tgt0, "tolist") else tgt0

    return list(cast(list[int], input_ids)), list(cast(list[int], target_ids))


def _run_evaluation_inference(
    model_name: str,
    dataset_name: str,
    backend_impl: BackendProtocol,
    image_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    modality: str = "text",
    max_batches: int | None = MAX_BATCHES,
    **kwargs: object,
) -> tuple[list[str], list[str], list[float]]:
    """Run inference for evaluation across text and multimodal benchmarks.

    Args:
        model_name: Target model name.
        dataset_name: Target dataset name.
        backend_impl: Backend protocol implementation.
        image_path: Optional path to schema diagram or ERD image.
        audio_path: Optional path to recorded query audio.
        modality: Explicit modality selector ('text', 'vision', 'audio', 'multimodal').
        max_batches: Maximum number of batches to evaluate (`None` for default `MAX_BATCHES`).
        **kwargs: Additional inference arguments.

    Returns:
        Tuple of `(predictions, truths, confidence_scores)`.

    Raises:
        ValueError: If valid iterable dataloader is missing.

    """
    preds: list[str] = []
    truths: list[str] = []
    confidence_scores: list[float] = []
    data_dict = backend_impl.build_dataloader(ETLConfig(dataset_name=dataset_name, split="test", batch_size=1, modality=modality))
    dataloader = data_dict.get("loader", None)
    tokenizer = SQLTokenizer(model_name=None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
        msg = "Valid iterable dataloader is required for evaluation; dataset could not be constructed."
        raise ValueError(msg)

    limit = MAX_BATCHES if max_batches is None else int(max_batches)
    for i, batch in enumerate(cast(typing.Iterable[Any], dataloader)):
        if i >= limit:
            break

        (input_ids, target_ids) = _process_batch_inputs(batch)

        prompt_text = tokenizer.decode(input_ids)
        truth_text = tokenizer.decode(target_ids)

        gen_kwargs: dict[str, Any] = {"modality": modality}
        if image_path is not None:
            gen_kwargs["image_path"] = str(image_path)
        elif isinstance(batch, dict) and "pixel_values" in batch:
            gen_kwargs["pixel_values"] = batch["pixel_values"]

        if audio_path is not None:
            gen_kwargs["audio_path"] = str(audio_path)
        elif isinstance(batch, dict) and "audio_values" in batch:
            gen_kwargs["audio_values"] = batch["audio_values"]

        gen_res = backend_impl.generate_sql(model_name, prompt_text, **gen_kwargs)
        preds.append(str(gen_res.get("sql", "")))
        confidence_scores.append(float(str(gen_res.get("confidence_score", 0.0))))
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

    engine = LiveDatabaseEngine(db_path=db_path, ddl=ddl, db_type=str(db_type), db_kwargs=db_kwargs_dict)

    backend_impl = get_backend(backend)

    eval_kwargs = dict(kwargs)
    eval_kwargs.pop("image_path", None)
    eval_kwargs.pop("audio_path", None)
    eval_kwargs.pop("modality", None)
    raw_max_batches = eval_kwargs.pop("max_batches", max_batches)
    resolved_max_batches = int(raw_max_batches) if isinstance(raw_max_batches, (int, float, str)) else max_batches

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
    engine.close()
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


__all__ = ["MAX_BATCHES", "compute_metrics", "compute_metrics_async", "evaluate", "normalize_sql"]

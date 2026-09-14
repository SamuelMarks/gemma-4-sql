"""SDK Evaluation module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.tokenization import SQLTokenizer

MAX_BATCHES = 10
if TYPE_CHECKING:
    from gemma_4_sql.sdk.protocols import BackendProtocol
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def normalize_sql(sql: str) -> str:
    """Normalize SQL by stripping whitespace and lowercasing.

    Args:
        sql: The string representing the SQL query.

    Returns:
        The normalized SQL query string.
    """
    return " ".join(sql.strip().lower().split())


async def compute_metrics_async(
    engine: LiveDatabaseEngine,
    preds: list[str],
    truths: list[str],
) -> dict[str, float]:
    """Compute exact match, valid SQL, and execution accuracy asynchronously.

    Args:
        engine: The LiveDatabaseEngine instance for SQL execution.
        preds: List of predicted SQL query strings.
        truths: List of ground truth SQL query strings.

    Returns:
        A dictionary containing exact_match, valid_sql, and execution_accuracy metrics.
    """
    exact_matches = 0
    valid_sqls = 0
    exec_matches = 0

    async def process_pair(p: str, t: str) -> tuple[int, int, int]:
        """Process a single prediction and ground truth pair.

        Args:
            p: Predicted SQL query string.
            t: Ground truth SQL query string.

        Returns:
            Tuple of (exact_match, valid_sql, execution_match) indicator flags (0 or 1).
        """
        em = 1 if normalize_sql(p) == normalize_sql(t) else 0
        (p_success, p_results, _) = await engine.execute_with_feedback_async(p)
        vs = 1 if p_success else 0
        (t_success, t_results, _) = await engine.execute_with_feedback_async(t)

        # Execution accuracy requires valid execution on both and matching results
        ex = 1 if (p_success and t_success and p_results == t_results) else 0
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
) -> dict[str, float]:
    """Compute exact match, valid SQL, and execution accuracy.

    Args:
        engine: The LiveDatabaseEngine instance.
        preds: A sequence of predicted SQL queries.
        truths: A sequence of ground truth SQL queries.

    Returns:
        Dictionary containing exact_match, valid_sql, and execution_accuracy metrics.
    """
    return asyncio.run(compute_metrics_async(engine, preds, truths))


def _process_batch_inputs(batch: object) -> tuple[list[int], list[int]]:
    """Extract input and target IDs from a batch.

    Args:
        batch: The batch data structure (tuple, list, or mapping).

    Returns:
        Tuple of (input_ids, target_ids) lists.
    """
    min_batch_tuple_length = 2
    if isinstance(batch, (tuple, list)) and len(batch) >= min_batch_tuple_length:
        input_ids = batch[0][0].tolist() if hasattr(batch[0][0], "tolist") else batch[0][0]
        target_ids = batch[1][0].tolist() if hasattr(batch[1][0], "tolist") else batch[1][0]
    elif isinstance(batch, dict):
        input_ids = batch["inputs"][0].tolist() if hasattr(batch["inputs"][0], "tolist") else batch["inputs"][0]
        target_ids = batch["targets"][0].tolist() if hasattr(batch["targets"][0], "tolist") else batch["targets"][0]
    else:
        input_ids, target_ids = [], []
    return list(input_ids), list(target_ids)


def _run_evaluation_inference(
    model_name: str,
    dataset_name: str,
    backend_impl: BackendProtocol,
) -> tuple[list[str], list[str], list[float]]:
    """Run inference for evaluation.

    Args:
        model_name: Target model name.
        dataset_name: Target dataset name.
        backend_impl: Backend protocol implementation.

    Returns:
        Tuple of (predictions, truths, confidence_scores).
    """
    preds: list[str] = []
    truths: list[str] = []
    confidence_scores: list[float] = []
    ETLConfig = __import__("gemma_4_sql.type_hints", fromlist=["ETLConfig"]).ETLConfig
    data_dict = backend_impl.build_dataloader(ETLConfig(dataset_name=dataset_name, split="test", batch_size=1))
    dataloader = data_dict.get("loader", None)
    tokenizer = SQLTokenizer(model_name=None)
    if dataloader is not None and hasattr(dataloader, "__iter__"):
        for i, batch in enumerate(dataloader):
            if i >= MAX_BATCHES:
                break

            (input_ids, target_ids) = _process_batch_inputs(batch)

            prompt_text = tokenizer.decode(input_ids)
            truth_text = tokenizer.decode(target_ids)
            gen_res = backend_impl.generate_sql(model_name, prompt_text)
            preds.append(str(gen_res.get("sql", "")))
            confidence_scores.append(float(str(gen_res.get("confidence_score", 0.0))))
            truths.append(truth_text)
    else:
        simulated_prompts = ["Get all users", "Find user with id 1"]
        truths = ["SELECT * FROM users", "SELECT * FROM users WHERE id = 1"]
        for prompt in simulated_prompts:
            gen_res = backend_impl.generate_sql(model_name, prompt)
            preds.append(str(gen_res.get("sql", "SELECT 1")))
            confidence_scores.append(float(str(gen_res.get("confidence_score", 0.0))))
    return (preds, truths, confidence_scores)


def evaluate(
    model_name: str,
    dataset_name: str,
    backend: str = "jax",
    db_path: str = ":memory:",
    ddl: str | None = None,
    **kwargs: JSONValue,
) -> JSONDict:
    """Evaluate a Text-to-SQL model.

    Args:
        model_name: The name or path of the model.
        dataset_name: The dataset to evaluate against.
        backend: The backend framework ('jax', 'keras', 'pytorch', etc.).
        db_path: Path to the database for execution accuracy.
        ddl: Optional DDL to set up the schema.
        **kwargs: Evaluation overrides, mock_predictions, mock_truths, db_type, etc.

    Returns:
        Evaluation results dictionary containing status and metrics.
    """
    db_type = kwargs.get("db_type", "sqlite")
    db_kwargs = kwargs.get("db_kwargs")
    db_kwargs_dict = db_kwargs if isinstance(db_kwargs, dict) else {}
    engine = LiveDatabaseEngine(db_path=db_path, ddl=ddl, db_type=str(db_type), db_kwargs=db_kwargs_dict)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend(backend)

    mock_preds = kwargs.get("mock_predictions")
    mock_truths = kwargs.get("mock_truths")
    if isinstance(mock_preds, list) and isinstance(mock_truths, list):
        preds = [str(x) for x in mock_preds]
        truths = [str(x) for x in mock_truths]
        confidence_scores = [1.0] * len(preds)
    else:
        (preds, truths, confidence_scores) = _run_evaluation_inference(model_name, dataset_name, backend_impl)

    metrics = asyncio.run(compute_metrics_async(engine, preds, truths))
    if confidence_scores:
        metrics["mean_confidence"] = sum(confidence_scores) / len(confidence_scores)
    engine.close()
    return {
        "backend": backend,
        "model": model_name,
        "dataset": dataset_name,
        "status": "completed",
        "metrics": metrics,
    }

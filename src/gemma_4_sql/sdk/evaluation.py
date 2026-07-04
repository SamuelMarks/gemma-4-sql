# Copyright 2024
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

    Returns:
        object: The resulting output from the operation.

    """
    return " ".join(sql.strip().lower().split())


async def compute_metrics_async(engine: LiveDatabaseEngine, preds: list[str], truths: list[str]) -> dict[str, float]:
    """Compute exact match, valid SQL, and execution accuracy asynchronously.

    Returns:
        object: The resulting output from the operation.

    """
    exact_matches = 0
    valid_sqls = 0
    exec_matches = 0

    async def process_pair(p: str, t: str) -> tuple[int, int, int]:
        """Process a pair of prediction and truth.

        Returns:
            object: The resulting output from the operation.

        """
        em = 1 if normalize_sql(p) == normalize_sql(t) else 0
        (success, _, _) = await engine.execute_with_feedback_async(p)
        vs = 1 if success else 0
        ex = 1 if await engine.compare_queries_async(p, t) else 0
        return (em, vs, ex)

    results = await asyncio.gather(*list(map(process_pair, preds, truths)))
    for em, vs, ex in results:
        exact_matches += em
        valid_sqls += vs
        exec_matches += ex
    total = len(preds) if preds else 1
    return {"exact_match": exact_matches / total, "valid_sql": valid_sqls / total, "execution_accuracy": exec_matches / total}


def compute_metrics(engine: LiveDatabaseEngine, preds: list[str], truths: list[str]) -> dict[str, float]:
    """Compute exact match, valid SQL, and execution accuracy.

    Returns:
        object: The resulting output from the operation.

    """
    return asyncio.run(compute_metrics_async(engine, preds, truths))


def _process_batch_inputs(batch: object) -> tuple[list[int], list[int]]:
    """Extract input and target IDs from a batch.

    Returns:
        object: Description of return.

    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        input_ids = batch[0][0].tolist() if hasattr(batch[0][0], "tolist") else batch[0][0]
        target_ids = batch[1][0].tolist() if hasattr(batch[1][0], "tolist") else batch[1][0]
    else:
        input_ids = batch["inputs"][0].tolist() if hasattr(batch["inputs"][0], "tolist") else batch["inputs"][0]
        target_ids = batch["targets"][0].tolist() if hasattr(batch["targets"][0], "tolist") else batch["targets"][0]
    return input_ids, target_ids


def _run_evaluation_inference(model_name: str, dataset_name: str, backend_impl: BackendProtocol) -> tuple[list[str], list[str], list[float]]:
    """Run inference for evaluation.

    Returns:
        object: The resulting output from the operation.

    """
    preds = []
    truths = []
    confidence_scores = []
    ETLConfig = __import__("gemma_4_sql.type_hints", fromlist=["ETLConfig"]).ETLConfig
    data_dict = backend_impl.build_dataloader(ETLConfig(dataset_name=dataset_name, split="test", batch_size=1))
    dataloader = data_dict.get("loader", None)
    tokenizer = SQLTokenizer(model_name=None)
    if dataloader is not None and hasattr(dataloader, "__iter__"):
        for i, batch in enumerate(dataloader):
            if i >= MAX_BATCHES:
                break  # pragma: no cover

            (input_ids, target_ids) = _process_batch_inputs(batch)

            prompt_text = tokenizer.decode(input_ids)
            truth_text = tokenizer.decode(target_ids)
            gen_res = backend_impl.generate_sql(model_name, prompt_text)
            preds.append(str(gen_res.get("sql", "")))
            confidence_scores.append(float(gen_res.get("confidence_score", 0.0)))
            truths.append(truth_text)
    else:
        simulated_prompts = ["Get all users", "Find user with id 1"]
        truths = ["SELECT * FROM users", "SELECT * FROM users WHERE id = 1"]
        for prompt in simulated_prompts:
            gen_res = backend_impl.generate_sql(model_name, prompt)
            preds.append(str(gen_res.get("sql", "SELECT 1")))
            confidence_scores.append(float(gen_res.get("confidence_score", 0.0)))
    return (preds, truths, confidence_scores)


def evaluate(model_name: str, dataset_name: str, backend: str = "jax", db_path: str = ":memory:", ddl: str | None = None, **kwargs: JSONValue) -> JSONDict:
    """Evaluate a Text-to-SQL model.

    Args:
    ----
        model_name: The name or path of the model.
        dataset_name: The dataset to evaluate against.
        backend: The backend framework ('jax', 'keras', or 'maxtext').
        db_path: Path to the SQLite database for execution accuracy.
        ddl: Optional DDL to set up the schema.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        Evaluation results dictionary.

    """
    db_type = kwargs.get("db_type", "sqlite")
    db_kwargs = kwargs.get("db_kwargs")
    mock_predictions = kwargs.get("mock_predictions")
    mock_truths = kwargs.get("mock_truths")
    engine = LiveDatabaseEngine(db_path=db_path, ddl=ddl, db_type=str(db_type), db_kwargs=db_kwargs)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend(backend)
    confidence_scores = []
    if mock_predictions is not None and mock_truths is not None:
        preds = list(mock_predictions)
        truths = list(mock_truths)
    else:
        (preds, truths, confidence_scores) = _run_evaluation_inference(model_name, dataset_name, backend_impl)
    metrics = asyncio.run(compute_metrics_async(engine, preds, truths))
    if confidence_scores:
        metrics["mean_confidence"] = sum(confidence_scores) / len(confidence_scores)
    engine.close()
    return {"backend": backend, "model": model_name, "dataset": dataset_name, "status": "completed", "metrics": metrics}

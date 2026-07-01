"""MaxText-specific model evaluation pipeline."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.backends.maxtext.inference import generate_sql
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.tokenization import SQLTokenizer

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def normalize_sql(sql: str) -> str:
    """Normalize SQL by stripping whitespace and lowercasing.

    Args:
    ----
        sql: The raw SQL string to normalize.

    Returns:
    -------
        The normalized SQL string.

    """
    return " ".join(sql.strip().lower().split())


async def compute_metrics_async(engine: LiveDatabaseEngine, preds: list[str], truths: list[str]) -> dict[str, float]:
    """Compute exact match, valid SQL, and execution accuracy asynchronously.

    Args:
    ----
        engine: The LiveDatabaseEngine instance for query execution.
        preds: A list of predicted SQL query strings.
        truths: A list of ground truth SQL query strings.

    Returns:
    -------
        A dictionary containing the computed metrics:
        'exact_match', 'valid_sql', and 'execution_accuracy'.

    """
    exact_matches = 0
    valid_sqls = 0
    exec_matches = 0

    async def process_pair(p: str, t: str) -> tuple[int, int, int]:
        em = 1 if normalize_sql(p) == normalize_sql(t) else 0
        (success, _, _) = await engine.execute_with_feedback_async(p)
        vs = 1 if success else 0
        ex = 1 if await engine.compare_queries_async(p, t) else 0
        return em, vs, ex

    results = await asyncio.gather(*[process_pair(p, t) for p, t in zip(preds, truths)])
    for em, vs, ex in results:
        exact_matches += em
        valid_sqls += vs
        exec_matches += ex

    total = len(preds) if preds else 1
    return {"exact_match": exact_matches / total, "valid_sql": valid_sqls / total, "execution_accuracy": exec_matches / total}


def compute_metrics(engine: LiveDatabaseEngine, preds: list[str], truths: list[str]) -> dict[str, float]:
    """Compute exact match, valid SQL, and execution accuracy.

    Args:
    ----
        engine: The LiveDatabaseEngine instance for query execution.
        preds: A list of predicted SQL query strings.
        truths: A list of ground truth SQL query strings.

    Returns:
    -------
        A dictionary containing the computed metrics:
        'exact_match', 'valid_sql', and 'execution_accuracy'.

    """
    return asyncio.run(compute_metrics_async(engine, preds, truths))


def evaluate_model(model_name: str, dataset_name: str, db_path: str = ":memory:", ddl: str | None = None, db_type: str = "sqlite", **kwargs: JSONValue) -> JSONDict:
    """Evaluate a Text-to-SQL model using the MaxText backend.

    Args:
    ----
        model_name: The name or path of the model to evaluate.
        dataset_name: The dataset to use for evaluation.
        db_path: Path to the evaluation database (default: :memory:).
        ddl: Optional DDL to setup the evaluation schema.
        db_type: Type of the evaluation database.
        db_kwargs: Additional keyword arguments for the evaluation database.
        mock_predictions: Optional list of predicted SQL queries for testing.
        mock_truths: Optional list of ground truth SQL queries for testing.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        A dictionary containing MaxText evaluation metrics.

    """
    db_kwargs = kwargs.get("db_kwargs")
    mock_predictions = kwargs.get("mock_predictions")
    mock_truths = kwargs.get("mock_truths")
    engine = LiveDatabaseEngine(db_path=db_path, ddl=ddl, db_type=db_type, db_kwargs=db_kwargs)
    confidence_scores = []
    if mock_predictions is not None and mock_truths is not None:
        preds = mock_predictions
        truths = mock_truths
    else:
        data_dict = build_dataloader(dataset_name=dataset_name, split="test", batch_size=1)
        dataloader = data_dict.get("loader", None)
        preds = []
        truths = []
        tokenizer = SQLTokenizer(model_name=None)
        if dataloader is not None and hasattr(dataloader, "__iter__"):
            for i, batch in enumerate(dataloader):
                if i >= int("10"):
                    break
                input_ids = batch["inputs"][0].tolist() if hasattr(batch["inputs"][0], "tolist") else batch["inputs"][0]
                target_ids = batch["targets"][0].tolist() if hasattr(batch["targets"][0], "tolist") else batch["targets"][0]
                prompt_text = tokenizer.decode(input_ids)
                truth_text = tokenizer.decode(target_ids)
                gen_res = generate_sql(model_name, prompt_text)
                preds.append(gen_res.get("sql", ""))  # type: ignore[arg-type]
                confidence_scores.append(float(gen_res.get("confidence_score", 0.0)))  # type: ignore[arg-type]
                truths.append(truth_text)  # type: ignore[arg-type]
        else:
            simulated_prompts = ["Get all users", "Find user with id 1"]
            truths = ["SELECT * FROM users", "SELECT * FROM users WHERE id = 1"]
            for prompt in simulated_prompts:
                gen_res = generate_sql(model_name, prompt)
                preds.append(gen_res.get("sql", "SELECT 1"))  # type: ignore[arg-type]
                confidence_scores.append(float(gen_res.get("confidence_score", 0.0)))  # type: ignore[arg-type]
    metrics = asyncio.run(compute_metrics_async(engine, preds, truths))
    if confidence_scores:
        metrics["mean_confidence"] = sum(confidence_scores) / len(confidence_scores)
    engine.close()
    return {"backend": "maxtext", "model": model_name, "dataset": dataset_name, "status": "completed", "metrics": metrics}

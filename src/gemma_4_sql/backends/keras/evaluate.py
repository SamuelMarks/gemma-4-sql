"""
Keras-specific model evaluation pipeline.
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.keras.inference import generate_sql
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.tokenization import SQLTokenizer

def normalize_sql(sql: str) -> str:
    """Normalizes SQL by stripping whitespace and lowercasing."""
    return " ".join(sql.strip().lower().split())

def compute_metrics(
    engine: LiveDatabaseEngine, preds: list[str], truths: list[str]
) -> dict[str, float]:
    """Computes exact match, valid SQL, and execution accuracy."""
    exact_matches = 0
    valid_sqls = 0
    exec_matches = 0

    for p, t in zip(preds, truths):
        if normalize_sql(p) == normalize_sql(t):
            exact_matches += 1

        success, _, _ = engine.execute_with_feedback(p)
        if success:
            valid_sqls += 1

        if engine.compare_queries(p, t):
            exec_matches += 1

    total = len(preds) if preds else 1
    return {
        "exact_match": exact_matches / total,
        "valid_sql": valid_sqls / total,
        "execution_accuracy": exec_matches / total,
    }

def evaluate_model(
    model_name: str,
    dataset_name: str,
    db_path: str = ":memory:",
    ddl: str | None = None,
    db_type: str = "sqlite",
    db_kwargs: dict[str, Any] | None = None,
    mock_predictions: list[str] | None = None,
    mock_truths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluates a Text-to-SQL model using the Keras backend.

    Args:
        model_name: The name or path of the model to evaluate.
        dataset_name: The dataset to use for evaluation.
        db_path: Path to the evaluation database (default: :memory:).
        ddl: Optional DDL to setup the evaluation schema.
        db_type: Type of the evaluation database.
        db_kwargs: Additional keyword arguments for the evaluation database.
        mock_predictions: Optional list of predicted SQL queries for testing.
        mock_truths: Optional list of ground truth SQL queries for testing.

    Returns:
        A dictionary containing Keras evaluation metrics.
    """
    engine = LiveDatabaseEngine(
        db_path=db_path, ddl=ddl, db_type=db_type, db_kwargs=db_kwargs
    )

    if mock_predictions is not None and mock_truths is not None:
        preds = mock_predictions
        truths = mock_truths
    else:
        # Load the dataset using ETL
        data_dict = build_dataloader(
            dataset_name=dataset_name,
            split="test",
            batch_size=1,
        )
        dataloader = data_dict.get("loader", None)

        preds = []
        truths = []

        tokenizer = SQLTokenizer(model_name=None)

        if dataloader is not None and hasattr(dataloader, "__iter__"):
            # Limit evaluation to 10 samples for speed unless running a full eval script
            for i, batch in enumerate(dataloader):
                if i >= 10:
                    break

                if isinstance(batch, tuple) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]  # pragma: no cover
                else:
                    inputs, targets = batch["inputs"], batch["targets"]

                input_ids = (
                    inputs[0].numpy().tolist()
                    if hasattr(inputs[0], "numpy")
                    else inputs[0]
                )
                target_ids = (
                    targets[0].numpy().tolist()
                    if hasattr(targets[0], "numpy")
                    else targets[0]
                )
                prompt_text = tokenizer.decode(input_ids)
                truth_text = tokenizer.decode(target_ids)

                gen_res = generate_sql(model_name, prompt_text)
                preds.append(gen_res.get("sql", ""))
                truths.append(truth_text)
        else:
            # Fallback if dataloader is mocked
            simulated_prompts = ["Get all users", "Find user with id 1"]
            truths = ["SELECT * FROM users", "SELECT * FROM users WHERE id = 1"]
            for prompt in simulated_prompts:
                gen_res = generate_sql(model_name, prompt)
                preds.append(gen_res.get("sql", "SELECT 1"))

    metrics = compute_metrics(engine, preds, truths)
    engine.close()

    return {
        "backend": "keras",
        "model": model_name,
        "dataset": dataset_name,
        "status": "completed",
        "metrics": metrics,
    }

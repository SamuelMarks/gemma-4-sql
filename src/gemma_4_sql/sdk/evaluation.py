"""
SDK Evaluation module.
"""

from __future__ import annotations

from typing import Any

def evaluate(
    model_name: str,
    dataset_name: str,
    backend: str = "jax",
    db_path: str = ":memory:",
    ddl: str | None = None,
    db_type: str = "sqlite",
    db_kwargs: dict[str, Any] | None = None,
    mock_predictions: list[str] | None = None,
    mock_truths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluates a Text-to-SQL model.

    Args:
        model_name: The name or path of the model.
        dataset_name: The dataset to evaluate against.
        backend: The backend framework ('jax', 'keras', or 'maxtext').
        db_path: Path to the SQLite database for execution accuracy.
        ddl: Optional DDL to set up the schema.
        db_type: Type of database engine ('sqlite', 'postgresql', 'snowflake').
        db_kwargs: Additional keyword arguments for DB engine connection.
        mock_predictions: Optional predictions to mock execution.
        mock_truths: Optional ground truths to mock execution.

    Returns:
        Evaluation results dictionary.
    """
    kwargs = {
        "db_path": db_path,
        "ddl": ddl,
        "db_type": db_type,
        "db_kwargs": db_kwargs,
        "mock_predictions": mock_predictions,
        "mock_truths": mock_truths,
    }

    if backend == "jax":
        from gemma_4_sql.backends.jax.evaluate import evaluate_model

        return evaluate_model(model_name, dataset_name, **kwargs)
    elif backend == "keras":
        from gemma_4_sql.backends.keras.evaluate import evaluate_model

        return evaluate_model(model_name, dataset_name, **kwargs)
    elif backend == "maxtext":
        from gemma_4_sql.backends.maxtext.evaluate import evaluate_model

        return evaluate_model(model_name, dataset_name, **kwargs)
    elif backend == "pytorch":
        from gemma_4_sql.backends.pytorch.evaluate import evaluate_model

        return evaluate_model(model_name, dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

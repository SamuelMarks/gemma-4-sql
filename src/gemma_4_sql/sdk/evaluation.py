"""SDK Evaluation module."""

from __future__ import annotations


def evaluate(model_name: str, dataset_name: str, backend: str = "jax", db_path: str = ":memory:", ddl: str | None = None, **kwargs: object) -> dict[str, object]:
    """Evaluate a Text-to-SQL model.

    Args:
    ----
        model_name: The name or path of the model.
        dataset_name: The dataset to evaluate against.
        backend: The backend framework ('jax', 'keras', or 'maxtext').
        db_path: Path to the SQLite database for execution accuracy.
        ddl: Optional DDL to set up the schema.
        db_type: Type of database engine ('sqlite', 'postgresql', 'snowflake').
        db_kwargs: Additional keyword arguments for DB engine connection.
        mock_predictions: Optional predictions to mock execution.
        mock_truths: Optional ground truths to mock execution.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        Evaluation results dictionary.

    """
    db_type = kwargs.get("db_type", "sqlite")
    db_kwargs = kwargs.get("db_kwargs")
    mock_predictions = kwargs.get("mock_predictions")
    mock_truths = kwargs.get("mock_truths")
    kwargs = {"model_name": model_name, "dataset_name": dataset_name, "db_path": db_path, "ddl": ddl, "db_type": db_type, "db_kwargs": db_kwargs, "mock_predictions": mock_predictions, "mock_truths": mock_truths}
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).evaluate_model(**kwargs)

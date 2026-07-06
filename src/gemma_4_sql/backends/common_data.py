"""Shared dataset and loading utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import LazyLoader

if TYPE_CHECKING:
    from gemma_4_sql.tokenization import SQLTokenizer
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)


def _create_hf_data_source(base_ds: type) -> type:
    """Execute function.

    Args:
        base_ds: The base ds.

    Returns:
        The execution result.
    """

    class HFDataSource(base_ds):
        """Data source wrapping a Hugging Face dataset."""

        def __init__(self, hf_ds: object) -> None:
            """Execute function.

            Args:
                hf_ds: The hf ds.
            """
            self._ds = hf_ds

        def __len__(self) -> int:
            """Execute function.

            Returns:
                The execution result.

            """
            return len(self._ds)

        def __getitem__(self, idx: int) -> object:
            """Execute function.

            Returns:
                The execution result.

            """
            return self._ds[idx]

    return HFDataSource


def _create_base_format_transform(base_map: type) -> type:
    """Execute function.

    Returns:
        The execution result.

    """

    class BaseFormatTransform(base_map):
        """Transforms data into numpy/JAX/TF compatible formats."""

        def __init__(self, tokenizer: SQLTokenizer) -> None:
            """Execute function.

            Args:
                element: The element.

            Returns:
                A dictionary containing the results.
            """
            self.tokenizer = tokenizer

        def map(self, element: JSONDict) -> JSONDict:
            """Execute function.

            Returns:
                The execution result.

            """
            prompt = element.get("sql_prompt", element.get("question", ""))
            target = element.get("sql", element.get("query", ""))
            return {"inputs": self.tokenizer.encode(str(prompt)), "targets": self.tokenizer.encode(str(target))}

    return BaseFormatTransform


def _get_grain_classes(grain_module: object) -> tuple[type, type]:
    """Dynamically construct Grain classes.

    Args:
    ----
        grain_module: The loaded grain module.

    Returns:
    -------
        A tuple of (HFDataSource, BaseFormatTransform) classes.

    """
    base_ds = getattr(grain_module, "RandomAccessDataSource", object)
    base_map = getattr(grain_module, "MapTransform", object)

    return (_create_hf_data_source(base_ds), _create_base_format_transform(base_map))


def _load_duckdb_dataset(db_path: str, table: str) -> list[JSONDict]:
    """Load a dataset from a DuckDB database.

    Args:
    ----
        db_path: Path to DuckDB database file.
        table: Table name to read from.

    Returns:
    -------
        A list of dictionaries representing the dataset.

    Raises:
    RuntimeError: If DuckDB is not available or query fails.

    """
    duckdb_module = LazyLoader("duckdb").get_module()
    if duckdb_module is None:
        msg = "duckdb is required. Install with `pip install duckdb`."  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    try:
        conn = duckdb_module.connect(db_path, read_only=True)
        results = conn.execute(f"SELECT * FROM {table}").fetchall()
        columns = [desc[0] for desc in getattr(conn, "description", [("col" + str(i),) for i in range(len(results[0]))] if results else [])]
        conn.close()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to load dataset from DuckDB")  # pragma: no cover
        msg = f"DuckDB error: {e}"  # pragma: no cover
        raise RuntimeError(msg) from e  # pragma: no cover

"""Shared dataset and loading utilities."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

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
        """Adapter to turn Hugging Face Dataset into Grain RandomAccessDataSource."""

        _ds: Any

        def __init__(self, hf_ds: Any) -> None:
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
            """Initialize transform with tokenizer.

            Args:
                tokenizer: The SQL tokenizer instance.
            """
            self.tokenizer = tokenizer

        def map(self, element: JSONDict) -> JSONDict:
            """Execute data mapping transformation with multimodal support.

            Args:
                element: Input dictionary element from dataset.

            Returns:
                Dictionary with tokenized inputs, targets, and optional multimodal features.
            """
            from gemma_4_sql.backends.common_multimodal import (
                format_multimodal_prompt,
                process_audio,
                process_image,
            )

            prompt = str(element.get("sql_prompt", element.get("question", "")))
            target = str(element.get("sql", element.get("query", "")))

            image_input = element.get("image_bytes") or element.get("image_url") or element.get("image")
            audio_input = element.get("audio_clip") or element.get("audio")

            formatted = format_multimodal_prompt(
                prompt,
                has_image=image_input is not None,
                has_audio=audio_input is not None,
            )
            res: dict[str, Any] = {
                "inputs": self.tokenizer.encode(str(formatted["prompt"])),
                "targets": self.tokenizer.encode(str(target)),
            }
            if image_input is not None:
                img_data = process_image(cast(Any, image_input))
                res["pixel_values"] = img_data["pixel_values"]
            if audio_input is not None:
                aud_data = process_audio(cast(Any, audio_input))
                res["audio_values"] = aud_data["audio_values"]
            return res

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
        db_path: Path to DuckDB database file.
        table: Table name to read from.

    Returns:
        A list of dictionaries representing the dataset.

    Raises:
        RuntimeError: If DuckDB is not available or query fails.
        ValueError: If the table name is invalid or unsafe.
    """
    duckdb_module = LazyLoader("duckdb").get_module()
    if duckdb_module is None:
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise RuntimeError(msg)

    try:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
            msg = f"Invalid or unsafe table name: {table!r}"
            raise ValueError(msg)
        conn = duckdb_module.connect(db_path, read_only=True)
        results = conn.execute(f'SELECT * FROM "{table}"').fetchall()
        columns = [desc[0] for desc in getattr(conn, "description", [("col" + str(i),) for i in range(len(results[0]))] if results else [])]
        conn.close()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        logger.exception("Failed to load dataset from DuckDB")
        msg = f"DuckDB error: {e}"
        raise RuntimeError(msg) from e

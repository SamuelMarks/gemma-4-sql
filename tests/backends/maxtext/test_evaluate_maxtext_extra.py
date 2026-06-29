"""Module docstring."""

import typing
from unittest import mock

from gemma_4_sql.backends.maxtext.evaluate import evaluate_model


def test_maxtext_evaluate_no_tolist_and_break() -> object:  # type: ignore[return]
    """Initialize function test_maxtext_evaluate_no_tolist_and_break."""

    class DummyLoader:
        """Initialize class DummyLoader."""

        def __iter__(self: typing.Any) -> object:
            """Initialize function __iter__."""
            for _ in range(15):
                yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

    with mock.patch("gemma_4_sql.backends.maxtext.evaluate.build_dataloader") as mock_bdl:
        mock_bdl.return_value = {"loader": DummyLoader()}
        with mock.patch("gemma_4_sql.backends.maxtext.evaluate.generate_sql") as mock_gen:
            mock_gen.return_value = {"sql": "SELECT 1"}
            res = evaluate_model("dummy", "dummy")
            if not res["status"] == "completed":
                raise AssertionError

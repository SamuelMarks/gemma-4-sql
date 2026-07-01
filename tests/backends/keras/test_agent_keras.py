"""Tests for Keras Agentic Loop."""

import pytest

from gemma_4_sql.backends.keras.agent import run_agentic_loop


class MockLiveDatabaseEngine:
    """Mock LiveDatabaseEngine that fails once then succeeds."""

    def __init__(self, **_kwargs: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        kwargs: Description of kwargs.

        """
        self.call_count = 0

    async def execute_with_feedback_async(self, _query: str) -> tuple[bool, list[object], str]:
        self.call_count += 1
        if self.call_count == 1:
            return (False, [], "Syntax error")
        return (True, [(1,)], "")

    def execute_with_feedback(self, _query: str) -> tuple[bool, list[object], str]:
        """Initialize function execute_with_feedback.

        Args:
        ----
        query: Description of query.

        """
        self.call_count += 1
        if self.call_count == 1:
            return (False, [], "Syntax error")
        return (True, [(1,)], "")

    def close(self) -> None:
        """Initialize function close."""


@pytest.fixture
def _mock_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function mock_engine.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    keras_agent = __import__("gemma_4_sql.backends.keras.agent", fromlist=[""])
    monkeypatch.setattr(keras_agent, "LiveDatabaseEngine", MockLiveDatabaseEngine)

    def mock_generate_sql(_model_name: str, _prompt: str) -> dict[str, object]:
        """Initialize function mock_generate_sql.

        Args:
        ----
        model_name: Description of model_name.
        prompt: Description of prompt.

        """
        return {"sql": "SELECT * FROM t"}

    monkeypatch.setattr(keras_agent, "generate_sql", mock_generate_sql)


@pytest.mark.usefixtures("_mock_engine")
def test_run_agentic_loop() -> None:
    """Test Keras agentic loop."""
    res = run_agentic_loop(model_name="m", prompt="p", max_retries=3)
    if not res["attempts"] == int("2"):
        raise AssertionError
    if res["success"] is not True:
        raise AssertionError
    if not len(res["history"]) == int("2"):  # type: ignore[arg-type]
        raise AssertionError
    if res["history"][0]["success"] is not False:  # type: ignore[index]
        raise AssertionError
    if not res["history"][0]["error"] == "Syntax error":  # type: ignore[index]
        raise AssertionError
    if res["history"][1]["success"] is not True:  # type: ignore[index]
        raise AssertionError
    if "Syntax error" not in res["history"][1]["prompt"]:  # type: ignore[index]
        raise AssertionError


def test_run_agentic_loop_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    keras_agent = __import__("gemma_4_sql.backends.keras.agent", fromlist=[""])

    class MockLiveDatabaseEngineFail:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def execute_with_feedback_async(self, _query: str) -> tuple[bool, list[object], str]:
            return (False, [], "Syntax error that never fixes")

        def execute_with_feedback(self, _query: str) -> tuple[bool, list[object], str]:
            return (False, [], "Syntax error that never fixes")

        def close(self) -> None:
            pass

    monkeypatch.setattr(keras_agent, "LiveDatabaseEngine", MockLiveDatabaseEngineFail)

    def mock_generate_sql(_model_name: str, _prompt: str) -> dict[str, object]:
        return {"sql": "SELECT * FROM t"}

    monkeypatch.setattr(keras_agent, "generate_sql", mock_generate_sql)

    res = run_agentic_loop(model_name="m", prompt="p", max_retries=2)
    assert res["attempts"] == 2
    assert res["success"] is False
    assert len(res["history"]) == 2

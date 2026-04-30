"""
Tests for MaxText Agentic Loop.
"""

from typing import Any

import pytest

from gemma_4_sql.backends.maxtext.agent import run_agentic_loop


class MockLiveDatabaseEngine:
    """Mock LiveDatabaseEngine that fails once then succeeds."""

    def __init__(self, **kwargs: Any) -> None:
        self.call_count = 0

    def execute_with_feedback(self, query: str) -> tuple[bool, list[Any], str]:
        self.call_count += 1
        if self.call_count == 1:
            return False, [], "Syntax error"
        return True, [(1,)], ""

    def close(self) -> None:
        pass


@pytest.fixture
def mock_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.maxtext.agent as maxtext_agent

    monkeypatch.setattr(maxtext_agent, "LiveDatabaseEngine", MockLiveDatabaseEngine)

    def mock_generate_sql(model_name: str, prompt: str) -> dict[str, Any]:
        return {"sql": "SELECT * FROM t"}

    monkeypatch.setattr(maxtext_agent, "generate_sql", mock_generate_sql)


def test_run_agentic_loop(mock_engine: None) -> None:
    """Test MaxText agentic loop."""
    res = run_agentic_loop(model_name="m", prompt="p", max_retries=3)
    assert res["status"] == "completed"
    assert res["attempts"] == 2
    assert res["success"] is True
    assert len(res["history"]) == 2
    assert res["history"][0]["success"] is False
    assert res["history"][0]["error"] == "Syntax error"
    assert res["history"][1]["success"] is True
    assert "Syntax error" in res["history"][1]["prompt"]

import argparse

from gemma_4_sql.cli_misc import rag_cmd
from gemma_4_sql.cli_serve import agent_cmd, chat_cmd, evaluate_cmd, generate_cmd


def test_evaluate_cmd_false_branches(monkeypatch):
    args = argparse.Namespace(model="d", dataset="d", backend="d", db_path="d", ddl="d", db_type="d", db_kwargs="")
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda **k: {})
    evaluate_cmd(args)


def test_generate_cmd_false_branches(monkeypatch):
    args = argparse.Namespace(model="d", prompt="d", backend="d", beam_width=3, max_length=50, show_confidence=False)
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda **k: {"sql": "a", "confidence_score": 0.5})
    generate_cmd(args)


def test_agent_cmd_false_branches(monkeypatch):
    args = argparse.Namespace(model="d", prompt="d", backend="d", db_path="d", ddl="d", db_type="d", max_retries=1, db_kwargs="")
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda **k: {})
    agent_cmd(args)


def test_chat_cmd_false_branches(monkeypatch):
    args = argparse.Namespace(model="d", prompt="d", backend="d", history="")
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda **k: {})
    chat_cmd(args)


def test_rag_cmd_false_branches(monkeypatch, capsys):
    args = argparse.Namespace(action="build", prompt="d", ddl="d", image_path=None, audio_path=None)
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda **k: "rag")
    rag_cmd(args)

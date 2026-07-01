"""Tests for MLX backend."""

import pytest

import gemma_4_sql.backends.mlx.inference as inf
import gemma_4_sql.backends.mlx.train as tr
from gemma_4_sql.backends.mlx import etl, peft


def test_train_mocked():
    res = tr.train_model("sft", "mod", "dat", 1, 0.1)
    assert res["status"] == "mocked_missing_mlx"


def test_etl_mocked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(etl, "datasets", None)
    res = etl.build_dataloader("ds", "train")
    assert res["status"] == "mocked"


def test_inference_mocked():
    res = inf.generate_sql("mod", "prompt")
    assert res["status"] == "mocked_missing_mlx"


def test_peft_mocked():
    res = peft.apply_lora("mod", ["q_proj"], 8, 16, 0.1)
    assert res["status"] == "mocked_missing_mlx"


def test_all_other_methods_mocked():
    from gemma_4_sql.backends.mlx import agent, benchmark, chat, dpo, evaluate, export, few_shot, logging, quantize

    assert agent.run_agentic_loop("m", "p")["backend"] == "mlx"
    assert benchmark.benchmark_model("m", "cpu", 1)["backend"] == "mlx"
    assert chat.chat_turn("m", [], "p")["backend"] == "mlx"
    assert dpo.run_dpo("m", "d")["backend"] == "mlx"
    assert evaluate.evaluate_model("m", "d")["backend"] == "mlx"
    assert export.export_model("m", "p")["backend"] == "mlx"
    assert few_shot.build_few_shot_prompt("m", "p", [])["backend"] == "mlx"
    assert logging.log_metrics({"l": 1.0}, 1, "d")["backend"] == "mlx"
    assert quantize.quantize_model("m", "awq")["backend"] == "mlx"

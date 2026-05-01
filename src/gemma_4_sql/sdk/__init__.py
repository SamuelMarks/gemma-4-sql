"""
SDK module for gemma-4-sql.
"""

from __future__ import annotations

from ..tokenization import SQLTokenizer
from .agent import run_agentic_loop
from .benchmark import benchmark
from .chat import chat_turn
from .db_engine import LiveDatabaseEngine
from .dpo import run_dpo
from .duckdb_extension import embed_in_duckdb
from .etl import etl_posttrain, etl_pretrain, etl_sft
from .evaluation import evaluate
from .export import export_model
from .few_shot import build_few_shot_prompt
from .inference import generate
from .logging import log_metrics
from .models import posttrain_model, pretrain_model, sft_model, train_from_scratch
from .peft import apply_peft
from .quantize import quantize_model
from .rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema
from .serve import serve_model

__all__ = [
    "SQLTokenizer",
    "LiveDatabaseEngine",
    "benchmark",
    "run_agentic_loop",
    "run_dpo",
    "embed_in_duckdb",
    "etl_pretrain",
    "etl_sft",
    "etl_posttrain",
    "evaluate",
    "export_model",
    "generate",
    "log_metrics",
    "train_from_scratch",
    "pretrain_model",
    "sft_model",
    "posttrain_model",
    "apply_peft",
    "quantize_model",
    "build_rag_prompt",
    "extract_schema_entities",
    "retrieve_relevant_schema",
    "serve_model",
    "chat_turn",
    "build_few_shot_prompt",
]

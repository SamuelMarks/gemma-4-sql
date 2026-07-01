"""Command Line Interface for gemma-4-sql."""

from __future__ import annotations

import argparse
import json

from gemma_4_sql.sdk import (
    LiveDatabaseEngine,
    SQLTokenizer,
    apply_peft,
    benchmark,
    build_few_shot_prompt,
    build_rag_prompt,
    chat_turn,
    embed_in_duckdb,
    etl_posttrain,
    etl_pretrain,
    etl_sft,
    evaluate,
    export_model,
    extract_schema_entities,
    generate,
    log_metrics,
    posttrain_model,
    pretrain_model,
    retrieve_relevant_schema,
    run_agentic_loop,
    run_dpo,
    serve_model,
    sft_model,
    train_from_scratch,
)
from gemma_4_sql.sdk.quantize import quantize_model


def tokenize_cmd(args: argparse.Namespace) -> None:
    """Tokenize or detokenize text/tokens."""
    tokenizer = SQLTokenizer(vocab_size=args.vocab_size, model_name=args.hf_model)
    if args.decode:
        try:
            tokens = [int(t.strip()) for t in args.decode.split(",")]
            tokenizer.decode(tokens)
        except ValueError:
            pass
    elif args.encode:
        tokenizer.encode(args.encode)
    else:
        pass


def db_execute_cmd(args: argparse.Namespace) -> None:
    """Execute a SQL query against the LiveDatabaseEngine."""
    db_kwargs = {}
    if getattr(args, "db_kwargs", ""):
        db_kwargs = json.loads(args.db_kwargs)
    engine = LiveDatabaseEngine(db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, db_kwargs=db_kwargs)
    (success, _results, _error) = engine.execute_with_feedback(args.query)
    if success:
        pass
    else:
        pass
    engine.close()


def embed_duckdb_cmd(args: argparse.Namespace) -> None:
    """Embed Gemma as a UDF in DuckDB and execute a prompt."""
    try:
        duckdb = __import__("duckdb")
    except ImportError:
        return
    conn = duckdb.connect(args.db_path)
    if args.ddl:
        conn.execute(args.ddl)
    embed_in_duckdb(conn=conn, model_name=args.model, backend=args.backend, db_path=args.db_path, max_retries=args.max_retries)
    if args.prompt:
        conn.execute("SELECT ask_gemma(?)", [args.prompt]).fetchall()
    else:
        pass
    conn.close()


def etl_pretrain_cmd(args: argparse.Namespace) -> None:
    """Run ETL for pretraining SQL datasets."""
    etl_pretrain(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, backend=args.backend, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)


def etl_sft_cmd(args: argparse.Namespace) -> None:
    """Run SFT ETL for SQL datasets."""
    etl_sft(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, backend=args.backend, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)


def etl_posttrain_cmd(args: argparse.Namespace) -> None:
    """Run posttrain ETL for SQL datasets."""
    etl_posttrain(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, backend=args.backend, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)


def train_cmd(args: argparse.Namespace) -> None:
    """Train a new model from scratch."""
    train_from_scratch(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none"))


def pretrain_cmd(args: argparse.Namespace) -> None:
    """Pretrain an existing model."""
    pretrain_model(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none"))


def sft_cmd(args: argparse.Namespace) -> None:
    """Supervised fine-tune an existing model."""
    sft_model(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none"))


def posttrain_cmd(args: argparse.Namespace) -> None:
    """Post-train an existing model."""
    posttrain_model(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none"))


def dpo_cmd(args: argparse.Namespace) -> None:
    """Run Direct Preference Optimization (DPO)."""
    run_dpo(model_name=args.model, dataset=args.dataset, backend=args.backend, beta=args.beta)


def peft_cmd(args: argparse.Namespace) -> None:
    """Apply PEFT / LoRA to an existing model."""
    target_modules = args.target_modules.split(",") if args.target_modules else None
    apply_peft(model_name=args.model, target_modules=target_modules, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, backend=args.backend)


def quantize_cmd(args: argparse.Namespace) -> None:
    """Quantize a model."""
    quantize_model(model_name=args.model, method=args.method, backend=args.backend)


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Evaluate an existing model."""
    preds = args.predictions.split(";") if args.predictions else None
    truths = args.truths.split(";") if args.truths else None
    db_kwargs = {}
    if args.db_kwargs:
        json = __import__("json")
        db_kwargs = json.loads(args.db_kwargs)
    evaluate(model_name=args.model, dataset_name=args.dataset, backend=args.backend, db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, db_kwargs=db_kwargs, mock_predictions=preds, mock_truths=truths)


def export_cmd(args: argparse.Namespace) -> None:
    """Export a trained model."""
    export_model(model_name=args.model, export_path=args.path, backend=args.backend)


def generate_cmd(args: argparse.Namespace) -> None:
    """Generate SQL from text."""
    generate(model_name=args.model, prompt=args.prompt, backend=args.backend, beam_width=args.beam_width, max_length=args.max_length, show_confidence=getattr(args, "show_confidence", False))


def agent_cmd(args: argparse.Namespace) -> None:
    """Run agentic self-correction loop."""
    db_kwargs = {}
    if args.db_kwargs:
        json = __import__("json")
        db_kwargs = json.loads(args.db_kwargs)
    run_agentic_loop(model_name=args.model, prompt=args.prompt, backend=args.backend, db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, db_kwargs=db_kwargs, max_retries=args.max_retries, min_confidence=getattr(args, "min_confidence", 0.0))


def rag_cmd(args: argparse.Namespace) -> None:
    """Build a RAG-augmented prompt or extract schema context."""
    if getattr(args, "action", "build") == "extract":
        extract_schema_entities(args.ddl)
    elif getattr(args, "action", "build") == "retrieve":
        schema = extract_schema_entities(args.ddl)
        retrieve_relevant_schema(args.prompt, schema)
    else:
        build_rag_prompt(prompt=args.prompt, ddl=args.ddl)


def log_metrics_cmd(args: argparse.Namespace) -> None:
    """Log training metrics."""
    metrics_dict = {}
    if args.metrics:
        for m in args.metrics.split(","):
            (k, v) = m.split("=")
            metrics_dict[k.strip()] = float(v.strip())
    log_metrics(metrics=metrics_dict, step=args.step, log_dir=args.log_dir, backend=args.backend)


def serve_cmd(args: argparse.Namespace) -> None:
    """Serve a model using continuous batching."""
    serve_model(model_name=args.model, port=args.port, max_batch_size=args.max_batch_size, backend=args.backend)


def chat_cmd(args: argparse.Namespace) -> None:
    """Run a multi-turn conversational SQL chat turn."""
    history = []
    if getattr(args, "history", ""):
        try:
            history = json.loads(args.history)
        except json.JSONDecodeError:
            return
    chat_turn(model_name=args.model, history=history, new_prompt=args.prompt, backend=args.backend)


def few_shot_cmd(args: argparse.Namespace) -> None:
    """Run dynamic few-shot prompting."""
    examples = []
    if getattr(args, "examples", ""):
        try:
            examples = json.loads(args.examples)
        except json.JSONDecodeError:
            return
    build_few_shot_prompt(model_name=args.model, prompt=args.prompt, examples=examples, backend=args.backend)


def benchmark_cmd(args: argparse.Namespace) -> None:
    """Benchmark a model on target hardware."""
    benchmark(model_name=args.model, hardware=args.hardware, batch_size=args.batch_size, backend=args.backend)


def _add_etl_parsers(subparsers: object) -> None:
    parser_etl = subparsers.add_parser("etl", help="Run ETL to prepare SQL training datasets.")  # type: ignore[attr-defined]
    etl_subparsers = parser_etl.add_subparsers(dest="etl_command", required=True)
    parser_etl_pretrain = etl_subparsers.add_parser("pretrain", help="Run ETL for pretraining SQL datasets.")
    parser_etl_pretrain.add_argument("--dataset", default="seeklhy/SynSQL-2.5M", help="Hugging Face dataset name.")
    parser_etl_pretrain.add_argument("--split", default="train", help="Dataset split.")
    parser_etl_pretrain.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser_etl_pretrain.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_etl_pretrain.add_argument("--distributed", action="store_true", help="Enable distributed sharding.")
    parser_etl_pretrain.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer model name.")
    parser_etl_pretrain.add_argument("--duckdb-path", default=None, help="Optional path to DuckDB database.")
    parser_etl_pretrain.add_argument("--duckdb-table", default=None, help="Optional DuckDB table name.")
    parser_etl_pretrain.set_defaults(func=etl_pretrain_cmd)
    parser_etl_sft = etl_subparsers.add_parser("sft", help="Run ETL for SFT SQL datasets.")
    parser_etl_sft.add_argument("--dataset", default="gretelai/synthetic_text_to_sql", help="Hugging Face dataset name.")
    parser_etl_sft.add_argument("--split", default="train", help="Dataset split.")
    parser_etl_sft.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser_etl_sft.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_etl_sft.add_argument("--distributed", action="store_true", help="Enable distributed sharding.")
    parser_etl_sft.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer model name.")
    parser_etl_sft.add_argument("--duckdb-path", default=None, help="Optional path to DuckDB database.")
    parser_etl_sft.add_argument("--duckdb-table", default=None, help="Optional DuckDB table name.")
    parser_etl_sft.set_defaults(func=etl_sft_cmd)
    parser_etl_posttrain = etl_subparsers.add_parser("posttrain", help="Run ETL for post-training/RLHF SQL datasets.")
    parser_etl_posttrain.add_argument("--dataset", default="xlangai/spider2-lite", help="Hugging Face dataset name.")
    parser_etl_posttrain.add_argument("--split", default="train", help="Dataset split.")
    parser_etl_posttrain.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser_etl_posttrain.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_etl_posttrain.add_argument("--distributed", action="store_true", help="Enable distributed sharding.")
    parser_etl_posttrain.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer model name.")
    parser_etl_posttrain.add_argument("--duckdb-path", default=None, help="Optional path to DuckDB database.")
    parser_etl_posttrain.add_argument("--duckdb-table", default=None, help="Optional DuckDB table name.")
    parser_etl_posttrain.set_defaults(func=etl_posttrain_cmd)


def _add_training_parsers(subparsers: object) -> None:
    parser_train = subparsers.add_parser("train", help="Train a new model from scratch.")  # type: ignore[attr-defined]
    parser_train.add_argument("--model", default="gemma-4", help="Model name.")
    parser_train.add_argument("--dataset", default="dummy_dataset", help="Training dataset.")
    parser_train.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser_train.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate.")
    parser_train.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_train.add_argument("--distributed-strategy", default="none", choices=["none", "ddp", "fsdp"], help="Distributed strategy for PyTorch.")
    parser_train.set_defaults(func=train_cmd)
    parser_pretrain = subparsers.add_parser("pretrain", help="Pretrain an existing model.")  # type: ignore[attr-defined]
    parser_pretrain.add_argument("--model", default="gemma-4", help="Model name.")
    parser_pretrain.add_argument("--dataset", default="dummy_dataset", help="Training dataset.")
    parser_pretrain.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser_pretrain.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate.")
    parser_pretrain.add_argument("--backend", default="maxtext", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_pretrain.add_argument("--distributed-strategy", default="none", choices=["none", "ddp", "fsdp"], help="Distributed strategy for PyTorch.")
    parser_pretrain.set_defaults(func=pretrain_cmd)
    parser_sft = subparsers.add_parser("sft", help="Supervised fine-tune an existing model.")  # type: ignore[attr-defined]
    parser_sft.add_argument("--model", default="gemma-4", help="Model name.")
    parser_sft.add_argument("--dataset", default="dummy_dataset", help="Training dataset.")
    parser_sft.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser_sft.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate.")
    parser_sft.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_sft.add_argument("--distributed-strategy", default="none", choices=["none", "ddp", "fsdp"], help="Distributed strategy for PyTorch.")
    parser_sft.set_defaults(func=sft_cmd)
    parser_posttrain = subparsers.add_parser("posttrain", help="Post-train an existing model (e.g. RLHF/DPO).")  # type: ignore[attr-defined]
    parser_posttrain.add_argument("--model", default="gemma-4", help="Model name.")
    parser_posttrain.add_argument("--dataset", default="dummy_dataset", help="Training dataset.")
    parser_posttrain.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser_posttrain.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate.")
    parser_posttrain.add_argument("--backend", default="keras", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_posttrain.add_argument("--distributed-strategy", default="none", choices=["none", "ddp", "fsdp"], help="Distributed strategy for PyTorch.")
    parser_posttrain.set_defaults(func=posttrain_cmd)
    parser_dpo = subparsers.add_parser("dpo", help="Run Direct Preference Optimization (DPO).")  # type: ignore[attr-defined]
    parser_dpo.add_argument("--model", default="gemma-4", help="Model name.")
    parser_dpo.add_argument("--dataset", default="dummy_dataset", help="Training dataset.")
    parser_dpo.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter.")
    parser_dpo.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_dpo.set_defaults(func=dpo_cmd)
    parser_peft = subparsers.add_parser("peft", help="Apply PEFT / LoRA configuration to a model.")  # type: ignore[attr-defined]
    parser_peft.add_argument("--model", default="gemma-4", help="Model name.")
    parser_peft.add_argument("--target-modules", default="q_proj,v_proj", help="Comma-separated target modules (e.g. 'q_proj,v_proj').")
    parser_peft.add_argument("--lora-r", type=int, default=8, help="LoRA attention dimension (rank).")
    parser_peft.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser_peft.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser_peft.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_peft.set_defaults(func=peft_cmd)
    parser_quantize = subparsers.add_parser("quantize", help="Quantize a model (e.g., AWQ, GPTQ, GGUF, int8).")  # type: ignore[attr-defined]
    parser_quantize.add_argument("--model", default="gemma-4", help="Model name.")
    parser_quantize.add_argument("--method", default="int8", choices=["int8", "awq", "gptq", "gguf"], help="Quantization method.")
    parser_quantize.add_argument("--backend", default="pytorch", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_quantize.set_defaults(func=quantize_cmd)


def _add_evaluate_parsers(subparsers: object) -> None:
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained model.")  # type: ignore[attr-defined]
    parser_evaluate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_evaluate.add_argument("--dataset", default="test-data", help="Dataset to evaluate on.")
    parser_evaluate.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_evaluate.add_argument("--db-path", default=":memory:", help="Path to SQLite db for evaluation.")
    parser_evaluate.add_argument("--db-type", default="sqlite", help="Type of database backend (sqlite, postgresql, snowflake).")
    parser_evaluate.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs (e.g. user, password).")
    parser_evaluate.add_argument("--ddl", default="", help="DDL string to setup the evaluation schema.")
    parser_evaluate.add_argument("--predictions", default=None, help="Semicolon separated mock predictions.")
    parser_evaluate.add_argument("--truths", default=None, help="Semicolon separated mock truths.")
    parser_evaluate.set_defaults(func=evaluate_cmd)
    parser_few_shot = subparsers.add_parser("few-shot", help="Build a dynamic few-shot prompt.")  # type: ignore[attr-defined]
    parser_few_shot.add_argument("--model", default="gemma-4", help="Model name.")
    parser_few_shot.add_argument("--prompt", required=True, help="New user prompt.")
    parser_few_shot.add_argument("--examples", default="[]", help="JSON string representing few-shot examples.")
    parser_few_shot.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_few_shot.set_defaults(func=few_shot_cmd)
    parser_chat = subparsers.add_parser("chat", help="Execute a turn in a multi-turn conversational SQL chat.")  # type: ignore[attr-defined]
    parser_chat.add_argument("--model", default="gemma-4", help="Model name.")
    parser_chat.add_argument("--prompt", required=True, help="New user prompt.")
    parser_chat.add_argument("--history", default="[]", help="JSON string representing the previous conversation history.")
    parser_chat.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_chat.set_defaults(func=chat_cmd)


def _add_inference_parsers(subparsers: object) -> None:
    parser_serve = subparsers.add_parser("serve", help="Serve a model using continuous batching (vLLM).")  # type: ignore[attr-defined]
    parser_serve.add_argument("--model", default="gemma-4", help="Model name.")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    parser_serve.add_argument("--max-batch-size", type=int, default=256, help="Maximum batch size.")
    parser_serve.add_argument("--backend", default="pytorch", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_serve.set_defaults(func=serve_cmd)
    parser_export = subparsers.add_parser("export", help="Export and save a trained model.")  # type: ignore[attr-defined]
    parser_export.add_argument("--model", default="gemma-4", help="Model name.")
    parser_export.add_argument("--path", default="./checkpoints", help="Export destination path.")
    parser_export.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_export.set_defaults(func=export_cmd)
    parser_generate = subparsers.add_parser("generate", help="Generate SQL from text using a trained model.")  # type: ignore[attr-defined]
    parser_generate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_generate.add_argument("--prompt", required=True, help="Natural language prompt.")
    parser_generate.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_generate.add_argument("--beam-width", type=int, default=3, help="Number of beams for generation.")
    parser_generate.add_argument("--max-length", type=int, default=50, help="Maximum generation length.")
    parser_generate.add_argument("--show-confidence", action="store_true", help="Display the model's confidence score alongside the query.")
    parser_generate.set_defaults(func=generate_cmd)
    parser_agent = subparsers.add_parser("agent", help="Run agentic self-correction loop.")  # type: ignore[attr-defined]
    parser_agent.add_argument("--model", default="gemma-4", help="Model name.")
    parser_agent.add_argument("--prompt", required=True, help="Natural language prompt.")
    parser_agent.add_argument("--db-path", default=":memory:", help="Path to database for execution.")
    parser_agent.add_argument("--db-type", default="sqlite", help="Type of database backend (sqlite, postgresql, snowflake).")
    parser_agent.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs (e.g. user, password).")
    parser_agent.add_argument("--ddl", default="", help="DDL string to setup the evaluation schema.")
    parser_agent.add_argument("--max-retries", type=int, default=3, help="Maximum number of self-correction attempts.")
    parser_agent.add_argument("--min-confidence", type=float, default=0.0, help="Minimum generation confidence score required.")
    parser_agent.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_agent.set_defaults(func=agent_cmd)
    parser_rag = subparsers.add_parser("rag", help="Build a RAG prompt or extract schema context.")  # type: ignore[attr-defined]
    parser_rag.add_argument("--action", default="build", choices=["build", "extract", "retrieve"], help="Action to perform.")
    parser_rag.add_argument("--prompt", default="", help="Natural language prompt (required for build and retrieve).")
    parser_rag.add_argument("--ddl", required=True, help="DDL string to extract schema context from.")
    parser_rag.set_defaults(func=rag_cmd)
    parser_log = subparsers.add_parser("log", help="Log metrics to the backend.")  # type: ignore[attr-defined]
    parser_log.add_argument("--step", type=int, default=0, help="Training step.")
    parser_log.add_argument("--metrics", default="", help="Comma separated key=value metrics (e.g. loss=0.5,acc=0.9).")
    parser_log.add_argument("--log-dir", default="logs", help="Directory to save TensorBoard logs.")
    parser_log.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_log.set_defaults(func=log_metrics_cmd)


def _add_misc_parsers(subparsers: object) -> None:
    parser_tokenize = subparsers.add_parser("tokenize", help="Encode or decode text using SQLTokenizer.")  # type: ignore[attr-defined]
    parser_tokenize.add_argument("--encode", type=str, help="Text to encode.")
    parser_tokenize.add_argument("--decode", type=str, help="Comma-separated tokens to decode.")
    parser_tokenize.add_argument("--hf-model", type=str, default=None, help="Hugging Face model name.")
    parser_tokenize.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size for fallback char-level encoding.")
    parser_tokenize.set_defaults(func=tokenize_cmd)
    parser_execute = subparsers.add_parser("execute", help="Execute SQL against a live database.")  # type: ignore[attr-defined]
    parser_execute.add_argument("--query", required=True, help="SQL query to execute.")
    parser_execute.add_argument("--db-path", default=":memory:", help="Path to database.")
    parser_execute.add_argument("--db-type", default="sqlite", help="Type of database (sqlite, postgresql, snowflake, duckdb).")
    parser_execute.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs.")
    parser_execute.add_argument("--ddl", default="", help="DDL string to initialize the schema.")
    parser_execute.set_defaults(func=db_execute_cmd)
    parser_embed = subparsers.add_parser("embed-duckdb", help="Embed Gemma as a UDF in DuckDB.")  # type: ignore[attr-defined]
    parser_embed.add_argument("--model", default="gemma-4", help="Model name.")
    parser_embed.add_argument("--db-path", default=":memory:", help="DuckDB database path.")
    parser_embed.add_argument("--prompt", default="", help="Prompt to execute via the UDF.")
    parser_embed.add_argument("--ddl", default="", help="Optional DDL to setup the schema.")
    parser_embed.add_argument("--backend", default="jax", help="Backend to use.")
    parser_embed.add_argument("--max-retries", type=int, default=3, help="Max self-correction attempts.")
    parser_embed.set_defaults(func=embed_duckdb_cmd)
    parser_benchmark = subparsers.add_parser("benchmark", help="Benchmark a model on target hardware.")  # type: ignore[attr-defined]
    parser_benchmark.add_argument("--model", default="gemma-4", help="Model name.")
    parser_benchmark.add_argument("--hardware", default="gpu", choices=["gpu", "tpu", "cpu"], help="Target hardware.")
    parser_benchmark.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark.")
    parser_benchmark.add_argument("--backend", default="jax", help="Backend to use (jax, keras, maxtext, pytorch).")
    parser_benchmark.set_defaults(func=benchmark_cmd)


def cli(args: list[str] | None = None) -> None:
    """Run main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="CLI for gemma-4-sql dataset generation and model training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_etl_parsers(subparsers)
    _add_training_parsers(subparsers)
    _add_evaluate_parsers(subparsers)
    _add_inference_parsers(subparsers)
    _add_misc_parsers(subparsers)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    cli()

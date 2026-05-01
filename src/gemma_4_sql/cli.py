"""
Command Line Interface for gemma-4-sql.
"""

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
            result = tokenizer.decode(tokens)
            print(f"Decoded: {result}")
        except ValueError:
            print("Error: --decode requires a comma-separated list of integers.")
    elif args.encode:
        result = tokenizer.encode(args.encode)
        print(f"Encoded: {result}")
    else:
        print("Must provide either --encode or --decode")


def db_execute_cmd(args: argparse.Namespace) -> None:
    """Execute a SQL query against the LiveDatabaseEngine."""
    db_kwargs = {}
    if getattr(args, "db_kwargs", ""):
        db_kwargs = json.loads(args.db_kwargs)

    engine = LiveDatabaseEngine(
        db_path=args.db_path,
        ddl=args.ddl,
        db_type=args.db_type,
        db_kwargs=db_kwargs,
    )

    success, results, error = engine.execute_with_feedback(args.query)

    if success:
        print("Execution Successful!")
        print(f"Results: {results}")
    else:
        print("Execution Failed!")
        print(f"Error: {error}")

    engine.close()


def embed_duckdb_cmd(args: argparse.Namespace) -> None:
    """Embed Gemma as a UDF in DuckDB and execute a prompt."""
    try:
        import duckdb
    except ImportError:
        print("duckdb is required. Install with `pip install duckdb`.")
        return

    print(f"Embedding Gemma in DuckDB: model={args.model}, db_path={args.db_path}")
    conn = duckdb.connect(args.db_path)

    if args.ddl:
        conn.execute(args.ddl)

    embed_in_duckdb(
        conn=conn,
        model_name=args.model,
        backend=args.backend,
        db_path=args.db_path,
        max_retries=args.max_retries,
    )

    if args.prompt:
        print(f"Executing prompt: '{args.prompt}'")
        res = conn.execute("SELECT ask_gemma(?)", [args.prompt]).fetchall()
        print(f"Result: {res[0][0]}")
    else:
        print("UDF 'ask_gemma' registered. Provide a --prompt to execute it.")

    conn.close()


def etl_pretrain_cmd(args: argparse.Namespace) -> None:
    """Run ETL for pretraining SQL datasets."""
    print(f"Running pretrain ETL for dataset: {args.dataset} (split: {args.split})")
    result = etl_pretrain(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        backend=args.backend,
        distributed=args.distributed,
        tokenizer_name=args.tokenizer,
    )
    print(f"Result: {result}")


def etl_sft_cmd(args: argparse.Namespace) -> None:
    """Run SFT ETL for SQL datasets."""
    print(f"Running SFT ETL for dataset: {args.dataset} (split: {args.split})")
    result = etl_sft(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        backend=args.backend,
        distributed=args.distributed,
        tokenizer_name=args.tokenizer,
    )
    print(f"Result: {result}")


def etl_posttrain_cmd(args: argparse.Namespace) -> None:
    """Run posttrain ETL for SQL datasets."""
    print(f"Running posttrain ETL for dataset: {args.dataset} (split: {args.split})")
    result = etl_posttrain(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        backend=args.backend,
        distributed=args.distributed,
        tokenizer_name=args.tokenizer,
    )
    print(f"Result: {result}")


def train_cmd(args: argparse.Namespace) -> None:
    """Train a new model from scratch."""
    print(f"Training from scratch: model={args.model}, backend={args.backend}")
    result = train_from_scratch(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        backend=args.backend,
    )
    print(f"Result: {result}")


def pretrain_cmd(args: argparse.Namespace) -> None:
    """Pretrain an existing model."""
    print(f"Pretraining: model={args.model}, backend={args.backend}")
    result = pretrain_model(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        backend=args.backend,
    )
    print(f"Result: {result}")


def sft_cmd(args: argparse.Namespace) -> None:
    """Supervised fine-tune an existing model."""
    print(f"SFT: model={args.model}, backend={args.backend}")
    result = sft_model(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        backend=args.backend,
    )
    print(f"Result: {result}")


def posttrain_cmd(args: argparse.Namespace) -> None:
    """Post-train an existing model."""
    print(f"Post-training: model={args.model}, backend={args.backend}")
    result = posttrain_model(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        backend=args.backend,
    )
    print(f"Result: {result}")


def dpo_cmd(args: argparse.Namespace) -> None:
    """Run Direct Preference Optimization (DPO)."""
    print(f"DPO: model={args.model}, backend={args.backend}, beta={args.beta}")
    result = run_dpo(
        model_name=args.model,
        dataset=args.dataset,
        backend=args.backend,
        beta=args.beta,
    )
    print(f"Result: {result}")


def peft_cmd(args: argparse.Namespace) -> None:
    """Apply PEFT / LoRA to an existing model."""
    print(f"Applying PEFT: model={args.model}, backend={args.backend}")
    target_modules = args.target_modules.split(",") if args.target_modules else None
    result = apply_peft(
        model_name=args.model,
        target_modules=target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backend=args.backend,
    )
    print(f"Result: {result}")


def quantize_cmd(args: argparse.Namespace) -> None:
    """Quantize a model."""
    print(
        f"Quantizing: model={args.model}, backend={args.backend}, method={args.method}"
    )
    result = quantize_model(
        model_name=args.model,
        method=args.method,
        backend=args.backend,
    )
    print(f"Result: {result}")


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Evaluate an existing model."""
    print(
        f"Evaluating: model={args.model}, dataset={args.dataset}, backend={args.backend}"
    )
    preds = args.predictions.split(";") if args.predictions else None
    truths = args.truths.split(";") if args.truths else None

    db_kwargs = {}
    if args.db_kwargs:
        import json

        db_kwargs = json.loads(args.db_kwargs)

    result = evaluate(
        model_name=args.model,
        dataset_name=args.dataset,
        backend=args.backend,
        db_path=args.db_path,
        ddl=args.ddl,
        db_type=args.db_type,
        db_kwargs=db_kwargs,
        mock_predictions=preds,
        mock_truths=truths,
    )
    print(f"Result: {result}")


def export_cmd(args: argparse.Namespace) -> None:
    """Export a trained model."""
    print(f"Exporting: model={args.model}, path={args.path}, backend={args.backend}")
    result = export_model(
        model_name=args.model, export_path=args.path, backend=args.backend
    )
    print(f"Result: {result}")


def generate_cmd(args: argparse.Namespace) -> None:
    """Generate SQL from text."""
    print(
        f"Generating: model={args.model}, prompt='{args.prompt}', backend={args.backend}"
    )
    result = generate(model_name=args.model, prompt=args.prompt, backend=args.backend)
    print(f"Result: {result}")


def agent_cmd(args: argparse.Namespace) -> None:
    """Run agentic self-correction loop."""
    print(
        f"Running Agentic Loop: model={args.model}, prompt='{args.prompt}', backend={args.backend}"
    )
    db_kwargs = {}
    if args.db_kwargs:
        import json

        db_kwargs = json.loads(args.db_kwargs)

    result = run_agentic_loop(
        model_name=args.model,
        prompt=args.prompt,
        backend=args.backend,
        db_path=args.db_path,
        ddl=args.ddl,
        db_type=args.db_type,
        db_kwargs=db_kwargs,
        max_retries=args.max_retries,
    )
    print(f"Result: {result}")


def rag_cmd(args: argparse.Namespace) -> None:
    """Build a RAG-augmented prompt or extract schema context."""
    if getattr(args, "action", "build") == "extract":
        print("Extracting schema entities...")
        result = extract_schema_entities(args.ddl)
        print(f"Result:\n{json.dumps(result, indent=2)}")
    elif getattr(args, "action", "build") == "retrieve":
        print("Retrieving relevant schema...")
        schema = extract_schema_entities(args.ddl)
        result = retrieve_relevant_schema(args.prompt, schema)
        print(f"Result:\n{result}")
    else:
        print("Building RAG prompt...")
        result = build_rag_prompt(prompt=args.prompt, ddl=args.ddl)
        print(f"Result:\n{result}")


def log_metrics_cmd(args: argparse.Namespace) -> None:
    """Log training metrics."""
    print(
        f"Logging: step={args.step}, metrics={args.metrics}, "
        f"log_dir={args.log_dir}, backend={args.backend}"
    )
    metrics_dict = {}
    if args.metrics:
        for m in args.metrics.split(","):
            k, v = m.split("=")
            metrics_dict[k.strip()] = float(v.strip())

    result = log_metrics(
        metrics=metrics_dict, step=args.step, log_dir=args.log_dir, backend=args.backend
    )
    print(f"Result: {result}")


def serve_cmd(args: argparse.Namespace) -> None:
    """Serve a model using continuous batching."""
    print(
        f"Serving: model={args.model}, port={args.port}, "
        f"max_batch_size={args.max_batch_size}, backend={args.backend}"
    )
    result = serve_model(
        model_name=args.model,
        port=args.port,
        max_batch_size=args.max_batch_size,
        backend=args.backend,
    )
    print(f"Result: {result}")


def chat_cmd(args: argparse.Namespace) -> None:
    """Run a multi-turn conversational SQL chat turn."""
    history = []
    if getattr(args, "history", ""):
        try:
            history = json.loads(args.history)
        except json.JSONDecodeError:
            print("Error: --history must be a valid JSON list of dictionaries.")
            return

    print(
        f"Chat: model={args.model}, prompt='{args.prompt}', "
        f"history_length={len(history)}, backend={args.backend}"
    )
    result = chat_turn(
        model_name=args.model,
        history=history,
        new_prompt=args.prompt,
        backend=args.backend,
    )
    print(f"Result: {result}")


def few_shot_cmd(args: argparse.Namespace) -> None:
    """Run dynamic few-shot prompting."""
    examples = []
    if getattr(args, "examples", ""):
        try:
            examples = json.loads(args.examples)
        except json.JSONDecodeError:
            print("Error: --examples must be a valid JSON list of dictionaries.")
            return

    print(
        f"Few-Shot: model={args.model}, prompt='{args.prompt}', "
        f"num_examples={len(examples)}, backend={args.backend}"
    )
    result = build_few_shot_prompt(
        model_name=args.model,
        prompt=args.prompt,
        examples=examples,
        backend=args.backend,
    )
    print(f"Result: {result}")


def benchmark_cmd(args: argparse.Namespace) -> None:
    """Benchmark a model on target hardware."""
    print(
        f"Benchmarking: model={args.model}, hardware={args.hardware}, "
        f"batch_size={args.batch_size}, backend={args.backend}"
    )
    result = benchmark(
        model_name=args.model,
        hardware=args.hardware,
        batch_size=args.batch_size,
        backend=args.backend,
    )
    print(f"Result: {result}")


def cli(args: list[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="CLI for gemma-4-sql dataset generation and model training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ETL Group
    parser_etl = subparsers.add_parser(
        "etl", help="Run ETL to prepare SQL training datasets."
    )
    etl_subparsers = parser_etl.add_subparsers(dest="etl_command", required=True)

    # ETL Pretrain
    parser_etl_pretrain = etl_subparsers.add_parser(
        "pretrain", help="Run ETL for pretraining SQL datasets."
    )
    parser_etl_pretrain.add_argument(
        "--dataset", default="seeklhy/SynSQL-2.5M", help="Hugging Face dataset name."
    )
    parser_etl_pretrain.add_argument("--split", default="train", help="Dataset split.")
    parser_etl_pretrain.add_argument(
        "--batch-size", type=int, default=32, help="Batch size."
    )
    parser_etl_pretrain.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_etl_pretrain.add_argument(
        "--distributed", action="store_true", help="Enable distributed sharding."
    )
    parser_etl_pretrain.add_argument(
        "--tokenizer", default=None, help="Hugging Face tokenizer model name."
    )
    parser_etl_pretrain.set_defaults(func=etl_pretrain_cmd)

    # ETL SFT
    parser_etl_sft = etl_subparsers.add_parser(
        "sft", help="Run ETL for SFT SQL datasets."
    )
    parser_etl_sft.add_argument(
        "--dataset",
        default="gretelai/synthetic_text_to_sql",
        help="Hugging Face dataset name.",
    )
    parser_etl_sft.add_argument("--split", default="train", help="Dataset split.")
    parser_etl_sft.add_argument(
        "--batch-size", type=int, default=32, help="Batch size."
    )
    parser_etl_sft.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_etl_sft.add_argument(
        "--distributed", action="store_true", help="Enable distributed sharding."
    )
    parser_etl_sft.add_argument(
        "--tokenizer", default=None, help="Hugging Face tokenizer model name."
    )
    parser_etl_sft.set_defaults(func=etl_sft_cmd)

    # ETL Posttrain
    parser_etl_posttrain = etl_subparsers.add_parser(
        "posttrain", help="Run ETL for post-training/RLHF SQL datasets."
    )
    parser_etl_posttrain.add_argument(
        "--dataset", default="xlangai/spider2-lite", help="Hugging Face dataset name."
    )
    parser_etl_posttrain.add_argument("--split", default="train", help="Dataset split.")
    parser_etl_posttrain.add_argument(
        "--batch-size", type=int, default=32, help="Batch size."
    )
    parser_etl_posttrain.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_etl_posttrain.add_argument(
        "--distributed", action="store_true", help="Enable distributed sharding."
    )
    parser_etl_posttrain.add_argument(
        "--tokenizer", default=None, help="Hugging Face tokenizer model name."
    )
    parser_etl_posttrain.set_defaults(func=etl_posttrain_cmd)

    # Train
    parser_train = subparsers.add_parser(
        "train", help="Train a new model from scratch."
    )
    parser_train.add_argument("--model", default="gemma-4", help="Model name.")
    parser_train.add_argument(
        "--dataset", default="dummy_dataset", help="Training dataset."
    )
    parser_train.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser_train.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )
    parser_train.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_train.set_defaults(func=train_cmd)

    # Pretrain
    parser_pretrain = subparsers.add_parser(
        "pretrain", help="Pretrain an existing model."
    )
    parser_pretrain.add_argument("--model", default="gemma-4", help="Model name.")
    parser_pretrain.add_argument(
        "--dataset", default="dummy_dataset", help="Training dataset."
    )
    parser_pretrain.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser_pretrain.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )
    parser_pretrain.add_argument(
        "--backend",
        default="maxtext",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_pretrain.set_defaults(func=pretrain_cmd)

    # SFT
    parser_sft = subparsers.add_parser(
        "sft", help="Supervised fine-tune an existing model."
    )
    parser_sft.add_argument("--model", default="gemma-4", help="Model name.")
    parser_sft.add_argument(
        "--dataset", default="dummy_dataset", help="Training dataset."
    )
    parser_sft.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser_sft.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )
    parser_sft.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_sft.set_defaults(func=sft_cmd)

    # Posttrain
    parser_posttrain = subparsers.add_parser(
        "posttrain", help="Post-train an existing model (e.g. RLHF/DPO)."
    )
    parser_posttrain.add_argument("--model", default="gemma-4", help="Model name.")
    parser_posttrain.add_argument(
        "--dataset", default="dummy_dataset", help="Training dataset."
    )
    parser_posttrain.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser_posttrain.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )
    parser_posttrain.add_argument(
        "--backend",
        default="keras",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_posttrain.set_defaults(func=posttrain_cmd)

    # DPO
    parser_dpo = subparsers.add_parser(
        "dpo", help="Run Direct Preference Optimization (DPO)."
    )
    parser_dpo.add_argument("--model", default="gemma-4", help="Model name.")
    parser_dpo.add_argument(
        "--dataset", default="dummy_dataset", help="Training dataset."
    )
    parser_dpo.add_argument(
        "--beta", type=float, default=0.1, help="DPO temperature parameter."
    )
    parser_dpo.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_dpo.set_defaults(func=dpo_cmd)

    # PEFT / LoRA
    parser_peft = subparsers.add_parser(
        "peft", help="Apply PEFT / LoRA configuration to a model."
    )
    parser_peft.add_argument("--model", default="gemma-4", help="Model name.")
    parser_peft.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="Comma-separated target modules (e.g. 'q_proj,v_proj').",
    )
    parser_peft.add_argument(
        "--lora-r", type=int, default=8, help="LoRA attention dimension (rank)."
    )
    parser_peft.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA alpha parameter."
    )
    parser_peft.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout probability."
    )
    parser_peft.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_peft.set_defaults(func=peft_cmd)

    # Quantize
    parser_quantize = subparsers.add_parser(
        "quantize", help="Quantize a model (e.g., AWQ, GPTQ, GGUF, int8)."
    )
    parser_quantize.add_argument("--model", default="gemma-4", help="Model name.")
    parser_quantize.add_argument(
        "--method",
        default="int8",
        choices=["int8", "awq", "gptq", "gguf"],
        help="Quantization method.",
    )
    parser_quantize.add_argument(
        "--backend",
        default="pytorch",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_quantize.set_defaults(func=quantize_cmd)

    # Evaluate
    parser_evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a trained model."
    )
    parser_evaluate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_evaluate.add_argument(
        "--dataset", default="test-data", help="Dataset to evaluate on."
    )
    parser_evaluate.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_evaluate.add_argument(
        "--db-path", default=":memory:", help="Path to SQLite db for evaluation."
    )
    parser_evaluate.add_argument(
        "--db-type",
        default="sqlite",
        help="Type of database backend (sqlite, postgresql, snowflake).",
    )
    parser_evaluate.add_argument(
        "--db-kwargs",
        default="",
        help="JSON string of DB kwargs (e.g. user, password).",
    )
    parser_evaluate.add_argument(
        "--ddl", default="", help="DDL string to setup the evaluation schema."
    )
    parser_evaluate.add_argument(
        "--predictions", default=None, help="Semicolon separated mock predictions."
    )
    parser_evaluate.add_argument(
        "--truths", default=None, help="Semicolon separated mock truths."
    )
    parser_evaluate.set_defaults(func=evaluate_cmd)

    # Few-Shot
    parser_few_shot = subparsers.add_parser(
        "few-shot", help="Build a dynamic few-shot prompt."
    )
    parser_few_shot.add_argument("--model", default="gemma-4", help="Model name.")
    parser_few_shot.add_argument("--prompt", required=True, help="New user prompt.")
    parser_few_shot.add_argument(
        "--examples", default="[]", help="JSON string representing few-shot examples."
    )
    parser_few_shot.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_few_shot.set_defaults(func=few_shot_cmd)

    # Chat
    parser_chat = subparsers.add_parser(
        "chat", help="Execute a turn in a multi-turn conversational SQL chat."
    )
    parser_chat.add_argument("--model", default="gemma-4", help="Model name.")
    parser_chat.add_argument("--prompt", required=True, help="New user prompt.")
    parser_chat.add_argument(
        "--history",
        default="[]",
        help="JSON string representing the previous conversation history.",
    )
    parser_chat.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_chat.set_defaults(func=chat_cmd)

    # Serve
    parser_serve = subparsers.add_parser(
        "serve", help="Serve a model using continuous batching (vLLM)."
    )
    parser_serve.add_argument("--model", default="gemma-4", help="Model name.")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    parser_serve.add_argument(
        "--max-batch-size", type=int, default=256, help="Maximum batch size."
    )
    parser_serve.add_argument(
        "--backend",
        default="pytorch",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_serve.set_defaults(func=serve_cmd)

    # Export
    parser_export = subparsers.add_parser(
        "export", help="Export and save a trained model."
    )
    parser_export.add_argument("--model", default="gemma-4", help="Model name.")
    parser_export.add_argument(
        "--path", default="./checkpoints", help="Export destination path."
    )
    parser_export.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_export.set_defaults(func=export_cmd)

    # Generate
    parser_generate = subparsers.add_parser(
        "generate", help="Generate SQL from text using a trained model."
    )
    parser_generate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_generate.add_argument(
        "--prompt", required=True, help="Natural language prompt."
    )
    parser_generate.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_generate.set_defaults(func=generate_cmd)

    # Agent
    parser_agent = subparsers.add_parser(
        "agent", help="Run agentic self-correction loop."
    )
    parser_agent.add_argument("--model", default="gemma-4", help="Model name.")
    parser_agent.add_argument(
        "--prompt", required=True, help="Natural language prompt."
    )
    parser_agent.add_argument(
        "--db-path", default=":memory:", help="Path to database for execution."
    )
    parser_agent.add_argument(
        "--db-type",
        default="sqlite",
        help="Type of database backend (sqlite, postgresql, snowflake).",
    )
    parser_agent.add_argument(
        "--db-kwargs",
        default="",
        help="JSON string of DB kwargs (e.g. user, password).",
    )
    parser_agent.add_argument(
        "--ddl", default="", help="DDL string to setup the evaluation schema."
    )
    parser_agent.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of self-correction attempts.",
    )
    parser_agent.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_agent.set_defaults(func=agent_cmd)

    # RAG Contextualization
    parser_rag = subparsers.add_parser(
        "rag", help="Build a RAG prompt or extract schema context."
    )
    parser_rag.add_argument(
        "--action",
        default="build",
        choices=["build", "extract", "retrieve"],
        help="Action to perform.",
    )
    parser_rag.add_argument(
        "--prompt",
        default="",
        help="Natural language prompt (required for build and retrieve).",
    )
    parser_rag.add_argument(
        "--ddl", required=True, help="DDL string to extract schema context from."
    )
    parser_rag.set_defaults(func=rag_cmd)

    # Log Metrics
    parser_log = subparsers.add_parser("log", help="Log metrics to the backend.")
    parser_log.add_argument("--step", type=int, default=0, help="Training step.")
    parser_log.add_argument(
        "--metrics",
        default="",
        help="Comma separated key=value metrics (e.g. loss=0.5,acc=0.9).",
    )
    parser_log.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to save TensorBoard logs.",
    )
    parser_log.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_log.set_defaults(func=log_metrics_cmd)

    # Tokenize
    parser_tokenize = subparsers.add_parser(
        "tokenize", help="Encode or decode text using SQLTokenizer."
    )
    parser_tokenize.add_argument("--encode", type=str, help="Text to encode.")
    parser_tokenize.add_argument(
        "--decode", type=str, help="Comma-separated tokens to decode."
    )
    parser_tokenize.add_argument(
        "--hf-model", type=str, default=None, help="Hugging Face model name."
    )
    parser_tokenize.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="Vocabulary size for fallback char-level encoding.",
    )
    parser_tokenize.set_defaults(func=tokenize_cmd)

    # Database Execution
    parser_execute = subparsers.add_parser(
        "execute", help="Execute SQL against a live database."
    )
    parser_execute.add_argument("--query", required=True, help="SQL query to execute.")
    parser_execute.add_argument(
        "--db-path", default=":memory:", help="Path to database."
    )
    parser_execute.add_argument(
        "--db-type",
        default="sqlite",
        help="Type of database (sqlite, postgresql, snowflake, duckdb).",
    )
    parser_execute.add_argument(
        "--db-kwargs", default="", help="JSON string of DB kwargs."
    )
    parser_execute.add_argument(
        "--ddl", default="", help="DDL string to initialize the schema."
    )
    parser_execute.set_defaults(func=db_execute_cmd)

    # DuckDB Embed
    parser_embed = subparsers.add_parser(
        "embed-duckdb", help="Embed Gemma as a UDF in DuckDB."
    )
    parser_embed.add_argument("--model", default="gemma-4", help="Model name.")
    parser_embed.add_argument(
        "--db-path", default=":memory:", help="DuckDB database path."
    )
    parser_embed.add_argument(
        "--prompt", default="", help="Prompt to execute via the UDF."
    )
    parser_embed.add_argument(
        "--ddl", default="", help="Optional DDL to setup the schema."
    )
    parser_embed.add_argument("--backend", default="jax", help="Backend to use.")
    parser_embed.add_argument(
        "--max-retries", type=int, default=3, help="Max self-correction attempts."
    )
    parser_embed.set_defaults(func=embed_duckdb_cmd)

    # Benchmark
    parser_benchmark = subparsers.add_parser(
        "benchmark", help="Benchmark a model on target hardware."
    )
    parser_benchmark.add_argument("--model", default="gemma-4", help="Model name.")
    parser_benchmark.add_argument(
        "--hardware",
        default="gpu",
        choices=["gpu", "tpu", "cpu"],
        help="Target hardware.",
    )
    parser_benchmark.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for benchmark."
    )
    parser_benchmark.add_argument(
        "--backend",
        default="jax",
        help="Backend to use (jax, keras, maxtext, pytorch).",
    )
    parser_benchmark.set_defaults(func=benchmark_cmd)

    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    cli()

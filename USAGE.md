# `gemma-4-sql` Usage Guide

This guide covers the primary workflows for using `gemma-4-sql` via the Command Line Interface (CLI) and the Python SDK. The framework is designed for end-to-end Text-to-SQL tasks: from dataset preparation and model training to evaluation, quantization, and deployment.

## Installation

The core package now installs everything by default, including all backends (PyTorch, JAX, Keras, MaxText) and DuckDB support:

```bash
# Install everything (all backends + DuckDB)
pip install .

# You can also use the default/all extras explicitly:
pip install ".[all]"
```

## Development

To contribute to `gemma-4-sql` or modify its logic, install the development dependencies. This project uses `hatch` for package management, `ruff` for linting and formatting, `mypy` for strict type-checking, and `pytest` for testing.

1. **Install Dev Dependencies:**
   ```bash
   pip install -e ".[dev]"
   # Or directly: pip install -r requirements-dev.txt
   ```

2. **Run Linting & Formatting (`ruff`):**
   ```bash
   ruff check .
   ruff format .
   ```

3. **Run Type Checking (`mypy`):**
   ```bash
   mypy src/gemma_4_sql
   ```

4. **Run Tests with Coverage (`pytest`):**
   ```bash
   pytest --cov=src/gemma_4_sql tests/
   ```

---

## 1. Data Preparation (ETL)

The ETL process standardizes Hugging Face datasets into optimized formats using Google's `grain` library. This is crucial for distributing training data to different backend engines (JAX, Keras, MaxText, PyTorch).

### CLI

```bash
# Pretrain: Extract from seeklhy/SynSQL-2.5M
gemma-4-sql etl pretrain --dataset seeklhy/SynSQL-2.5M --batch-size 32 --backend jax

# SFT: Supervised Fine-Tuning ETL
gemma-4-sql etl sft --dataset gretelai/synthetic_text_to_sql --batch-size 16 --backend keras

# Posttrain / RLHF ETL
gemma-4-sql etl posttrain --dataset xlangai/spider2-lite --batch-size 8 --backend maxtext

# ETL using DuckDB (bypassing Hugging Face datasets)
gemma-4-sql etl pretrain --duckdb-path my_dataset.duckdb --duckdb-table pretrain_data --backend jax
```

### SDK

```python
from gemma_4_sql.sdk.etl import etl_pretrain, etl_sft

# Prepare pretraining dataloader for JAX
pretrain_data = etl_pretrain(
    dataset_name="seeklhy/SynSQL-2.5M", 
    batch_size=64, 
    backend="jax"
)
jax_loader = pretrain_data["loader"]

# Prepare SFT dataloader for Keras
sft_data = etl_sft(
    dataset_name="gretelai/synthetic_text_to_sql", 
    batch_size=16, 
    backend="keras"
)
keras_loader = sft_data["loader"]
```

---

## 2. Model Training & Fine-Tuning

The framework supports multiple training stages out-of-the-box. The backend flag implicitly routes the execution to the chosen engine (`jax`, `keras`, `maxtext`, `pytorch`).

### CLI

```bash
# 1. Train from Scratch
gemma-4-sql train --model gemma-4 --dataset my_dataset --backend pytorch

# 2. Pretrain (Domain Adaptation)
gemma-4-sql pretrain --model gemma-4 --dataset seeklhy/SynSQL-2.5M --backend maxtext

# 3. Supervised Fine-Tuning (SFT)
gemma-4-sql sft --model gemma-4 --dataset gretelai/synthetic_text_to_sql --backend jax

# 4. Post-Training
gemma-4-sql posttrain --model gemma-4 --dataset xlangai/spider2-lite --backend keras

# 5. Direct Preference Optimization (DPO)
gemma-4-sql dpo --model gemma-4 --dataset my_dpo_dataset --beta 0.1 --backend jax
```

### SDK

```python
from gemma_4_sql.sdk.models import pretrain_model, sft_model
from gemma_4_sql.sdk.dpo import run_dpo

# Pretrain using MaxText
pretrain_model(model_name="gemma-4", dataset="seeklhy/SynSQL-2.5M", backend="maxtext")

# Run DPO using JAX
run_dpo(model_name="gemma-4", dataset="my_dpo_dataset", beta=0.1, backend="jax")
```

---

## 3. Model Exporting

You can export trained models to various checkpoint formats like `safetensors`, `.keras`, or `orbax` depending on the backend used.

### CLI

```bash
# Export using PyTorch to safetensors
gemma-4-sql export --model gemma-4 --export-path ./exported/gemma-4-pt --backend pytorch

# Export using JAX to orbax
gemma-4-sql export --model gemma-4 --export-path ./exported/gemma-4-jax --backend jax
```

### SDK

```python
from gemma_4_sql.sdk.export import export_model

# Export using Keras
result = export_model(
    model_name="gemma-4",
    export_path="./exported/gemma-4-keras",
    backend="keras"
)
print(f"Exported to: {result['path']}")
```

---

## 4. PEFT & Quantization

To save compute and memory footprints, you can inject LoRA adapters (PEFT) and quantize models.

### CLI

```bash
# Apply LoRA adapters
gemma-4-sql peft --model gemma-4 --target-modules q_proj,v_proj --lora-r 16 --backend jax

# Quantize the model
gemma-4-sql quantize --model gemma-4 --method int8 --backend pytorch
```

### SDK

```python
from gemma_4_sql.sdk.peft import apply_peft
from gemma_4_sql.sdk.quantize import quantize_model

# Apply PEFT
peft_config = apply_peft(
    model_name="gemma-4",
    target_modules=["q_proj", "v_proj"],
    lora_r=16,
    backend="jax"
)

# Quantize
quantized = quantize_model(model_name="gemma-4", method="int8", backend="pytorch")
```

---

## 5. Raw Inference (Generation)

You can use the `generate` command to generate SQL directly from a prompt without full evaluation or the agentic loop.

### CLI

```bash
gemma-4-sql generate \
    --model gemma-4 \
    --prompt "List all users who signed up today." \
    --backend maxtext \
    --beam-width 5 \
    --max-length 100
```

### SDK

```python
from gemma_4_sql.sdk.inference import generate

# Generate SQL from a prompt
result = generate(
    model_name="gemma-4",
    prompt="List all users who signed up today.",
    backend="pytorch"
)
print("Generated SQL:", result["sql"])
```

---

## 6. Evaluation & Execution Engine

Gemma-4-SQL provides a live database execution engine to measure Execution Accuracy (EX). It runs the generated SQL against a real database backend (SQLite, PostgreSQL, Snowflake, DuckDB) and compares the output with the ground truth result.

### CLI

```bash
# Evaluate against an in-memory SQLite database
gemma-4-sql evaluate --model gemma-4 \
    --dataset test-data \
    --db-type sqlite \
    --ddl "CREATE TABLE users (id INT, name TEXT);" \
    --backend jax

# Directly execute SQL against DuckDB
gemma-4-sql execute \
    --db-type duckdb \
    --db-path "analytics.duckdb" \
    --query "SELECT COUNT(*) FROM logs;"
```

### SDK

```python
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.evaluation import evaluate

# Instantiate the live engine
engine = LiveDatabaseEngine(
    db_type="sqlite", 
    db_path=":memory:", 
    ddl="CREATE TABLE t (id INT);"
)
engine.execute_query("INSERT INTO t VALUES (1);")

# Run evaluation on an entire dataset
metrics = evaluate(
    model_name="gemma-4",
    dataset_name="test-data",
    db_type="duckdb",
    backend="jax"
)
```

---

## 5. Agentic Loop (Self-Correction)

For real-world inference, models often generate malformed SQL. The **Agentic Loop** runs the generated SQL, captures runtime errors from the database, and injects them back into the prompt for the model to self-correct in real-time.

### CLI

```bash
gemma-4-sql agent --model gemma-4 \
    --prompt "Show the total sales for 2024" \
    --db-path "sqlite:///my_database.db" \
    --max-retries 3 \
    --backend jax
```

### SDK

```python
from gemma_4_sql.sdk.agent import run_agentic_loop

result = run_agentic_loop(
    model_name="gemma-4",
    prompt="Show the total sales for 2024",
    backend="jax",
    db_type="sqlite",
    db_path=":memory:",
    max_retries=3
)
```

---

## 8. DuckDB UDF Integration

You can natively embed Gemma inside a DuckDB runtime, turning the model into a callable SQL User-Defined Function (UDF).

```bash
# Embeds the model and tests a prompt via the registered `ask_gemma` UDF
gemma-4-sql embed-duckdb \
    --model gemma-4 \
    --db-path data.duckdb \
    --prompt "How many users joined yesterday?"
```

---

## 7. Serving & Chat Interfaces

### Multi-Turn Chat
```bash
gemma-4-sql chat --model gemma-4 \
    --prompt "And what about 2025?" \
    --history '[{"role": "user", "content": "Show sales for 2024"}, {"role": "assistant", "content": "SELECT * FROM sales WHERE year = 2024"}]' \
    --backend jax
```

### High-Throughput Serving (Continuous Batching)
```bash
gemma-4-sql serve --model gemma-4 --port 8000 --max-batch-size 256 --backend pytorch
```

---

## 8. RAG Contextualization & Few-Shot Prompting

Injecting database schemas (DDL) and contextual examples dramatically improves SQL generation.

```bash
# Build RAG prompt with schema
gemma-4-sql rag --prompt "Total sales?" --ddl "CREATE TABLE sales (amount DECIMAL);"

# Few-shot examples
gemma-4-sql few-shot --model gemma-4 \
    --prompt "Total sales 2024?" \
    --examples '[{"input": "Total sales 2023?", "output": "SELECT SUM(amount) FROM sales WHERE year=2023;"}]'
```

---

## 9. Hardware Benchmarking

Before deploying your models, you can benchmark their throughput, latency, and memory characteristics on your target hardware (GPU, TPU, CPU) using the newly integrated benchmark tools.

### CLI

```bash
# Benchmark PyTorch backend on GPU
gemma-4-sql benchmark --model gemma-4 --hardware gpu --batch-size 32 --backend pytorch

# Benchmark MaxText backend on TPU
gemma-4-sql benchmark --model gemma-4 --hardware tpu --batch-size 128 --backend maxtext
```

### SDK

```python
from gemma_4_sql.sdk.benchmark import benchmark

metrics = benchmark(
    model_name="gemma-4",
    hardware="tpu",
    batch_size=128,
    backend="maxtext"
)
print(f"Tokens/sec: {metrics['tokens_per_sec']}")
```

---

## 10. Tokenization

You can manually encode strings or decode tokens using the same tokenization logic the ETL process uses. It defaults to character-level encoding, or it can use a Hugging Face tokenizer if provided.

### CLI

```bash
# Encode text
gemma-4-sql tokenize --encode "SELECT * FROM users"

# Decode tokens
gemma-4-sql tokenize --decode "83, 69, 76, 69, 67, 84"

# Use a Hugging Face model
gemma-4-sql tokenize --encode "SELECT * FROM users" --hf-model "google/gemma-2b"
```

---

## 11. Logging (TensorBoard Integration)

Gemma-4-SQL provides native integration with TensorBoard across all backend topologies (`jax`, `maxtext`, `keras`, `pytorch`). You can log arbitrary metrics (loss, accuracy, execution accuracy) during training, pretraining, or evaluation.

### CLI

```bash
# Log loss and accuracy to the default "logs" directory using PyTorch TB
gemma-4-sql log --step 100 --metrics "loss=0.5,acc=0.9" --backend pytorch

# Specify a custom log directory
gemma-4-sql log --step 200 --metrics "loss=0.3,acc=0.95" --log-dir "./runs/experiment_1" --backend jax
```

### SDK

```python
from gemma_4_sql.sdk.logging import log_metrics

# Log metrics directly to TensorBoard via the Keras backend
result = log_metrics(
    metrics={"loss": 0.5, "execution_accuracy": 0.88},
    step=100,
    log_dir="./runs/experiment_1",
    backend="keras"
)
print(f"Status: {result['status']}")
```


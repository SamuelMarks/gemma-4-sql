gemma-4-sql
===========

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- badges -->
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen) ![Docs](https://img.shields.io/badge/Docs-100%25-brightgreen)
<!-- /badges -->

Natural text to SQL with Gemma 4; with DuckDB support and swappable-backends: PyTorch; Keras ; JAX / Bonsai; JAX / MaxText.

`gemma-4-sql` is a specialized SDK and CLI tool designed for orchestrating Text-to-SQL training pipelines. It provides an end-to-end framework capable of ingesting diverse Text-to-SQL datasets, transforming them using Google's `grain` library into consistent multidimensional formats, and preparing them for modern AI-Hypercomputer workloads.

We explicitly integrate with and support the following Gemma 4 model architectures across different ecosystems:
*   **PyTorch**: Directly imports and uses `Gemma4ForCausalLM` from **[Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)**;
*   **MaxText**: Directly imports and uses `Gemma4Model` from **[AI-Hypercomputer MaxText](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/models/gemma4.py)**;
*   **JAX**: Directly imports and uses `BonsaiModel` from **[bonsai](https://github.com/jax-ml/bonsai)** (specifically at my [fork with a brand new Gemma 4 implementation](https://github.com/SamuelMarks/bonsai/tree/gemma4));
*   **Keras**: Supports generic [Keras 3](https://keras.io) workflows.

### Feature Support Matrix

| Feature | PyTorch Backend | Keras 3 Backend | JAX (Bonsai) | MaxText (XLA) |
| :--- | :--- | :--- | :--- | :--- |
| **ETL (Data Loading)** | ✅ Native `DataLoader` | ✅ Grain + `MapTransform` | ✅ Grain + `JAXDistributed` | ✅ Grain + `JAXDistributed` |
| **Training (Fit/JIT)** | ✅ `Gemma4ForCausalLM` | ✅ `keras.Model.fit()` | ✅ `@jax.jit` loop | ✅ `@jax.jit` loop |
| **PEFT / LoRA** | ✅ `peft` | ✅ Native Keras | ✅ `optax` | ✅ Native JAX |
| **Inference (Beam)** | ✅ Tensor-based Search | ✅ TF Native Search | ✅ Compiled `argsort` | ✅ Compiled `argsort` |
| **Evaluation (DB)** | ✅ Live `sqlite3` Loop | ✅ Live `sqlite3` Loop | ✅ Live `sqlite3` Loop | ✅ Live `sqlite3` Loop |
| **Export (Ckpt)** | ✅ `safetensors` | ✅ `.keras` v3 format | ✅ `orbax` Checkpointer | ✅ `orbax` Checkpointer |
| **Agentic Loop** | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction |

## How to Install

Install the core project via `pip`. Depending on your target execution environment, you can specify one of the optional extras to install backend-specific dependencies.

**Core Installation (No specific backend):**
```bash
pip install .
```

**Ecosystem-specific Installation:**
Choose the backend that matches your training infrastructure:

```bash
# Install with Native JAX support
pip install ".[jax]"

# Install with Keras support
pip install ".[keras]"

# Install with MaxText support (typically used on TPUs)
pip install ".[maxtext]"
```

## How to Develop

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

## How to do the SQL ETL

The Grain ETL pipeline normalizes and prepares Hugging Face datasets into highly-performant iterables mapped for your specific ecosystem (e.g., standardizing `(inputs, targets)` for JAX, `(x, y)` tuples for Keras, or `inputs, targets, segment_ids, positions` for MaxText).

You can use the pipeline either via the Python SDK or the Command Line Interface (CLI).

### Using the CLI

```bash
# Pretraining Dataset (Defaults to seeklhy/SynSQL-2.5M)
gemma-4-sql etl pretrain --dataset seeklhy/SynSQL-2.5M --batch-size 32

# SFT Dataset (Defaults to gretelai/synthetic_text_to_sql)
gemma-4-sql etl sft --dataset gretelai/synthetic_text_to_sql --batch-size 16

# Post-Training Dataset (Defaults to xlangai/spider2-lite)
gemma-4-sql etl posttrain --dataset xlangai/spider2-lite --batch-size 8
```

### Using the Python SDK

```python
from gemma_4_sql.sdk.etl import etl_pretrain, etl_sft
from gemma_4_sql.sdk.peft import apply_peft
from gemma_4_sql.sdk.quantize import quantize_model

# Load pretraining data for a JAX pipeline
pretrain_data = etl_pretrain(
    dataset_name="seeklhy/SynSQL-2.5M", 
    batch_size=64, 
    backend="jax"
)
jax_dataloader = pretrain_data["loader"]

# Load fine-tuning data for a Keras pipeline
sft_data = etl_sft(
    dataset_name="gretelai/synthetic_text_to_sql", 
    batch_size=16, 
    backend="keras"
)
keras_dataloader = sft_data["loader"]

# Apply LoRA/PEFT before training
peft_config = apply_peft(
    model_name="gemma-4",
    target_modules=["q_proj", "v_proj"],
    lora_r=8,
    backend="jax"
)

# Quantize the model
quantize_config = quantize_model(
    model_name="gemma-4",
    method="int8",
    backend="pytorch"
)
```

## How to Train, Pretrain, and Fine-Tune

Training a state-of-the-art Text-to-SQL model involves different stages depending on your goals. The `gemma-4-sql` CLI exposes dedicated commands for each.

1. **Train from Scratch (`train`)**:
   Initializes a completely fresh model and trains it from the ground up on your dataset.
   ```bash
   gemma-4-sql train --model gemma-4 --backend jax
   ```

2. **Pretraining (`pretrain`)**:
   Continues pretraining an existing base model on millions of synthesized or aggregated SQL statements. This phase focuses on learning the syntax of SQL and the relational algebra fundamentals.
   ```bash
   gemma-4-sql pretrain --model gemma-4 --backend maxtext
   ```

3. **Supervised Fine-Tuning (`sft`)**:
   Adapts a pretrained model to strictly formatted instruction-response pairs, ensuring the generated SQL matches the exact dialect and schema requested by the user's natural language query.
   ```bash
   gemma-4-sql sft --model gemma-4 --backend jax
   ```

4. **Posttraining (`posttrain`)**:
   General post-training adaptation. In this stage, models are aligned to avoid dangerous schema modifications (e.g., dropping tables) and to prefer highly optimized queries (e.g., proper joins).
   ```bash
   gemma-4-sql posttrain --model gemma-4 --backend keras
   ```

4. **Direct Preference Optimization (`dpo`)**:
   Specific RLHF optimization aligning model generation to preferred SQL over rejected SQL.
   ```bash
   gemma-4-sql dpo --model gemma-4 --dataset dummy_dataset --beta 0.1 --backend jax
   ```

5. **Parameter-Efficient Fine-Tuning (`peft`)**:
   Apply LoRA adapters to an existing model to save memory and compute during fine-tuning.
   ```bash
   gemma-4-sql peft --model gemma-4 --target-modules q_proj,v_proj --lora-r 16 --backend jax
   ```

6. **Model Quantization (`quantize`)**:
   Quantize an existing model (e.g., to int8, awq, gptq, or gguf) to reduce its memory footprint and improve inference speed.
   ```bash
   gemma-4-sql quantize --model gemma-4 --method int8 --backend pytorch
   ```

### 7. Execution and Delivery Phase

Once your model has been trained, aligned, and optionally quantized, you can evaluate its performance, export it for production serving, or test it manually via generation.

**Evaluation (`evaluate`)**:
Runs the model against a test dataset and calculates Execution Accuracy (EX) and Exact Match (EM) by testing the generated queries against an actual live database.
```bash
gemma-4-sql evaluate --model gemma-4 \
    --dataset test-data \
    --db-path sqlite:///eval_db.sqlite \
    --backend jax
```

**Export (`export`)**:
Saves the fully trained model to disk in a format native to your chosen backend (e.g., `.keras`, `safetensors`, or `orbax` checkpoints).
```bash
gemma-4-sql export --model gemma-4 --path ./release/gemma-4-sql-v1 --backend maxtext
```

**Inference (`generate`)**:
Run a one-off text-to-SQL generation.
```bash
gemma-4-sql generate --model gemma-4 --prompt "How many active users are there?" --backend pytorch
```

### Tokenization (`tokenize`)
You can manually encode strings or decode tokens using the same tokenization logic the ETL process uses. It defaults to character-level encoding, or it can use a Hugging Face tokenizer if provided.

```bash
# Encode text
gemma-4-sql tokenize --encode "SELECT * FROM users"

# Decode tokens
gemma-4-sql tokenize --decode "83, 69, 76, 69, 67, 84"

# Use a Hugging Face model
gemma-4-sql tokenize --encode "SELECT * FROM users" --hf-model "google/gemma-2b"
```

### Live Database Execution (`execute`)
The `LiveDatabaseEngine` powers the model evaluation and self-correction agent. You can also use it directly to execute SQL queries and test connection strings across various backend systems (SQLite, PostgreSQL, Snowflake, DuckDB).

```bash
# Execute a basic query against an in-memory SQLite DB
gemma-4-sql execute --query "SELECT 1" --db-type sqlite

# Initialize a schema and run a query against DuckDB
gemma-4-sql execute \
  --db-type duckdb \
  --ddl "CREATE TABLE test (id INT);" \
  --query "INSERT INTO test VALUES (1); SELECT * FROM test;"
```

### Training Metrics Logging (`log`)

You can log custom training or evaluation metrics to the specified backend (like TensorBoard for Keras/JAX). This uses the underlying logging mechanisms of your chosen backend framework.

```bash
gemma-4-sql log --step 100 --metrics "loss=0.5,acc=0.9" --backend jax
```

### RAG Contextualization (`rag`)

Many models perform better when contextualized with the database schema (DDL) directly in the prompt. You can extract schema entities, retrieve relevant tables based on a prompt, or build fully augmented RAG prompts.

#### Using the CLI
```bash
# Build a full RAG augmented prompt
gemma-4-sql rag \
    --prompt "Show the total sales for 2024" \
    --ddl "CREATE TABLE sales (id INT, amount DECIMAL, year INT);"

# Extract schema entities as JSON
gemma-4-sql rag --action extract \
    --ddl "CREATE TABLE sales (id INT, amount DECIMAL, year INT);"

# Retrieve relevant schema based on prompt
gemma-4-sql rag --action retrieve \
    --prompt "Show the total sales for 2024" \
    --ddl "CREATE TABLE sales (id INT, amount DECIMAL, year INT);"
```

#### Using the Python SDK
```python
from gemma_4_sql.sdk.rag import (
    build_rag_prompt,
    extract_schema_entities,
    retrieve_relevant_schema,
)

prompt = "Show the total sales for 2024"
ddl = "CREATE TABLE sales (id INT, amount DECIMAL, year INT);"

# 1. Extract schema entities
schema = extract_schema_entities(ddl)

# 2. Retrieve relevant schema context
context = retrieve_relevant_schema(prompt=prompt, schema=schema)

# 3. Build a fully augmented prompt
augmented_prompt = build_rag_prompt(prompt=prompt, ddl=ddl)
print(augmented_prompt)
```

### Agentic Loop (Self-Correction)

For enterprise text-to-SQL applications, it is critical that the model can recover from execution failures. You can invoke the agentic self-correction loop, which will attempt to generate SQL, run it against a live database, and feedback the error message into the model until a successful query is produced (or max retries are hit).

#### Using the CLI
```bash
gemma-4-sql agent --model gemma-4 \
    --prompt "Show the total sales for 2024" \
    --db-path "sqlite:///my_database.db" \
    --max-retries 3 \
    --backend jax
```

#### Using the Python SDK
```python
from gemma_4_sql.sdk.agent import run_agentic_loop

result = run_agentic_loop(
    model_name="gemma-4",
    prompt="Show the total sales for 2024",
    backend="jax",
    db_path="sqlite:///my_database.db",
    max_retries=3
)
print(f"Generated SQL: {result.get('sql')}")
print(f"Execution Status: {result.get('status')}")
```

### Deployment and Full Tutorial

For a comprehensive guide on running these training scripts across distributed infrastructure (like Google Cloud TPU VMs) using MaxText or JAX, please refer to the **[DEPLOY.md](./DEPLOY.md)** file. It contains step-by-step instructions for provisioning hardware, installing TPU-optimized wheels, and executing the parallelized ETL and Training phases.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

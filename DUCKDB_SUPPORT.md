# DuckDB Support in Gemma-4-SQL

## Overview

`gemma-4-sql` fully supports using DuckDB to:
1. Load data for Pretraining, Retraining (SFT), and Posttraining (RLHF) directly from DuckDB databases.
2. Evaluate and self-correct SQL generation via the Agent loop running against DuckDB.
3. Embed a Gemma-4 Text-to-SQL UDF (User Defined Function) natively into a DuckDB connection for natural language querying.

## Requirements

DuckDB is now installed by default with the main package:
```bash
pip install .
```

## 1. Pretraining, Retraining, and Posttraining

You can pass a DuckDB database path and table name directly to the ETL functions. The ETL pipeline will extract the dataset directly from DuckDB, bypassing Hugging Face datasets.

```python
from gemma_4_sql.type_hints import ETLConfig
from gemma_4_sql.sdk.etl import etl_pretrain, etl_sft, etl_posttrain

# Pretrain from DuckDB
config = ETLConfig(dataset_name="", split="train", duckdb_path="my_dataset.duckdb", duckdb_table="pretrain_data")
dataloader_info = etl_pretrain(config=config, backend="jax")

# Retrain (SFT) from DuckDB
config = ETLConfig(dataset_name="", split="train", duckdb_path="my_dataset.duckdb", duckdb_table="sft_data")
dataloader_info = etl_sft(config=config, backend="jax")

# Posttrain (RLHF) from DuckDB
config = ETLConfig(dataset_name="", split="train", duckdb_path="my_dataset.duckdb", duckdb_table="rlhf_data")
dataloader_info = etl_posttrain(config=config, backend="jax")
```

**Note:** The DuckDB table is automatically mapped using `tokenize_dataset_element` in `backends/common_data.py`, supporting `sql_prompt` / `question` / `prompt` text columns, `sql` / `query` target columns, DPO preference columns (`chosen` / `chosen_sql` and `rejected` / `rejected_sql`), and optional multimodal columns (`image` / `image_url` / `image_bytes` and `audio` / `audio_clip`).

## 2. Using DuckDB as the Live Evaluation Engine

Gemma-4-SQL's `LiveDatabaseEngine` supports executing models directly against DuckDB for feedback loops and evaluation metrics (Execution Accuracy - EX). By default (`read_only=True`), `DuckDBAdapter` enforces both dialect-aware SQL lexer checks and DuckDB driver-level external access lockdown (`SET enable_external_access = false` and `READ_ONLY` file mode).

```python
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

engine = LiveDatabaseEngine(db_type="duckdb", db_path="my_database.duckdb")

success, results, error = engine.execute_with_feedback("SELECT * FROM users")
print("Results:", results)
```

## 3. Embedding Gemma 4 in DuckDB

You can register Gemma 4 as a UDF in any DuckDB connection to query the database using natural language natively via SQL.

```python
import duckdb
from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb

conn = duckdb.connect("my_database.duckdb")

# This registers the `ask_gemma` scalar function
embed_in_duckdb(conn=conn, model_name="google/gemma-4-E2B-it", backend="jax", db_path="my_database.duckdb")

# Query the database in natural language using standard SQL
result = conn.execute("SELECT ask_gemma('What is the total revenue for 2024?')").fetchall()

# Result is a JSON string containing prompt, generated_sql, results, success (and error on failure)
print(result[0][0])
```

*Note on Backends:* The `backend` parameter defaults to `"jax"` and supports any registered execution engine or TPU comparison alias: `"jax"`, `"jax_native"`, `"pytorch"`, `"pytorch_hf"`, `"hf_transformers"`, `"pytorch_native"`, `"keras"`, `"keras_jax"`, `"keras_tensorflow"`, `"keras_tf"`, `"keras_pytorch"`, `"keras_torch"`, `"maxtext"`, or `"mlx"`.

### Command Line Usage

You can also test the DuckDB UDF directly from the command line:

```bash
gemma-4-sql embed-duckdb \
  --db-path test.duckdb \
  --prompt "Show all users" \
  --ddl "CREATE TABLE users (id INTEGER, name VARCHAR);" \
  --backend jax
```

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
from gemma_4_sql.sdk.etl import etl_pretrain, etl_sft, etl_posttrain

# Pretrain from DuckDB
dataloader_info = etl_pretrain(
    backend="jax",
    duckdb_path="my_dataset.duckdb",
    duckdb_table="pretrain_data"
)

# Retrain (SFT) from DuckDB
dataloader_info = etl_sft(
    backend="jax",
    duckdb_path="my_dataset.duckdb",
    duckdb_table="sft_data"
)

# Posttrain (RLHF) from DuckDB
dataloader_info = etl_posttrain(
    backend="jax",
    duckdb_path="my_dataset.duckdb",
    duckdb_table="rlhf_data"
)
```

**Note:** The DuckDB table should contain the standard format expected by `gemma_4_sql`, primarily `sql_prompt` (or `question`) and `sql` (or `query`) text columns.

## 2. Using DuckDB as the Live Evaluation Engine

Gemma-4-SQL's `LiveDatabaseEngine` supports executing models directly against DuckDB for feedback loops and evaluation metrics (Execution Accuracy - EX).

```python
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

engine = LiveDatabaseEngine(
    db_type="duckdb",
    db_path="my_database.duckdb"
)

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
embed_in_duckdb(
    conn=conn,
    model_name="google/gemma-4-sql-2b",
    backend="jax",
    db_path="my_database.duckdb" 
)

# Query the database in natural language using standard SQL
result = conn.execute("SELECT ask_gemma('What is the total revenue for 2024?')").fetchall()

# Result is a JSON string containing the generated SQL and execution success/data
print(result[0][0])
```

### Command Line Usage

You can also test the DuckDB UDF directly from the command line:

```bash
gemma-4-sql embed-duckdb \
  --db-path test.duckdb \
  --prompt "Show all users" \
  --ddl "CREATE TABLE users (id INTEGER, name VARCHAR);"
```

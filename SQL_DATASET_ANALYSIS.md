# Text-to-SQL Training Datasets Analysis (2024-2025)

Based on the latest trends in the open-source community, the text-to-SQL landscape has shifted from small academic benchmarks to large-scale synthetic corpora and complex enterprise-level datasets. 

If you are looking to pretrain, instruction-tune, or post-train a model like `gemma-4` for SQL tasks, here are the best options currently available on Hugging Face:

## 1. Large-Scale Pretraining & Base Fine-Tuning
If you need massive volume and variety to teach a base LLM the fundamental grammar, relationships, and reasoning required for SQL:

*   **SynSQL-2.5M (`seeklhy/SynSQL-2.5M`)**
    *   **Scale:** 2.5 million samples across 16,583 synthetic databases.
    *   **Why use it:** This is widely considered the largest and most diverse synthetic text-to-SQL dataset. It includes **Chain-of-Thought (CoT)** reasoning traces for queries and covers 9 different linguistic styles (formal, colloquial, vague). Excellent for teaching a model *how* to construct SQL step-by-step.
*   **SQaLe (`trl-lab/SQaLe-text-to-SQL-dataset`)**
    *   **Scale:** ~517,000 validated triples.
    *   **Why use it:** Unlike purely synthetic datasets, SQaLe is grounded in 23,000 real-world schemas sourced from SchemaPile. It bridges the gap between synthetic scale and the messiness of real-world database architectures.

## 2. High-Quality Instruction Tuning
If your model already knows basic SQL and you want to align it to follow instructions accurately, format code cleanly, and handle standard queries:

*   **Gretel Synthetic Text-to-SQL (`gretelai/synthetic_text_to_sql`)**
    *   **Scale:** ~106,000 samples.
    *   **Why use it:** Highly regarded for its data cleanliness and quality. It provides excellent cross-domain logic and is perfect for standard instruction tuning without overwhelming the model with noise.
*   **SQL Create Context (`b-mc2/sql-create-context`)**
    *   **Scale:** ~78,000 samples.
    *   **Why use it:** A very popular, lightweight dataset built from a combination of WikiSQL and Spider. It specifically focuses on the triplet of `[Question, Table Context (DDL), Target SQL]`, which is the exact format most RAG-based text-to-SQL applications use.

## 3. Enterprise, Agentic & Complex Reasoning (Post-Training)
If you want to train your model for "industrial-strength" SQL, data analysis workflows, or self-correction (RLHF/DPO):

*   **Spider 2.0 (`xlangai/spider2-lite` or `spider2`)**
    *   **Why use it:** The evolution of the classic Spider benchmark. It moves beyond simple `SELECT` statements into complex data transformation, analytics workflows, and handling massive context windows (e.g., metadata for 1,000+ columns across cloud dialects like BigQuery and Snowflake).
*   **BIRD-Critic (`birdsql/bird-critic-1.0-open`) / BIRD-SQL**
    *   **Why use it:** BIRD-SQL focuses on massive databases with real-world, noisy data values. BIRD-Critic specifically trains models on **SQL debugging and self-correction** (fixing buggy SQL based on user feedback/errors). This is critical for training agentic SQL assistants that can recover from database execution errors.

## Recommendation Summary
*   **For Pretraining/Continual Pretraining:** Use **SynSQL-2.5M** mixed with your standard code/text data.
*   **For Supervised Fine-Tuning (SFT):** Use a blend of **Gretel's synthetic data** and **SQaLe** to get high-quality instruction following over realistic schemas.
*   **For Post-Training / RLHF:** Use **Spider 2.0** and **BIRD-Critic** to teach the model how to handle enterprise dialect complexity and fix its own errors.
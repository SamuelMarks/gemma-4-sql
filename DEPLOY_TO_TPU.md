# Deploying and Training `gemma-4-sql` on Google Cloud TPUs

This document provides a comprehensive guide to deploying, training, and serving the `gemma-4-sql` Large Language Model on Google Cloud TPUs utilizing the `libscript` infrastructure automation ecosystem. 

Because `gemma-4-sql` natively supports multiple backends (`jax`, `maxtext`, `pytorch`, `keras`) and data sources (HuggingFace datasets, DuckDB), this guide details how to leverage `libscript` to orchestrate both single-node and multi-node (distributed) training architectures.

---

## 0. Parameters & Environment Variables

Before running the commands in this guide, export the following environment variables. These parameters configure the hardware, model, datasets, and storage buckets used across the workflows.

```bash
# HuggingFace Token (Required for gated models like Gemma)
export HF_TOKEN="your_huggingface_token_here"

# Model & Dataset Configuration
export MODEL_NAME="google/gemma-4-sql-it"
export DATASET_NAME="seeklhy/SynSQL-2.5M"
export DUCKDB_URL="https://example.com/my_dataset.duckdb"
export DUCKDB_PATH="analytics.duckdb"
export DUCKDB_TABLE="pretrain_data"

# Google Cloud Project Configuration
export GCP_PROJECT_ID="your_google_cloud_project_id"
export GCP_ZONE="us-central2-b"

# TPU Hardware Configuration
export TPU_ZONE="us-central2-b"
export TPU_ACCELERATOR_TYPE="v4-8"

# Single-Node TPU VM Config
export TPU_NAME="gemma-train-node"
export SERVE_TPU_NAME="gemma-serve-node"
export TPU_DATA_DISK_SIZE="500" # Size in GB for persistent disk caching

# Multi-Node Distributed GKE/XPK Config
export XPK_CLUSTER_NAME="gemma-training-cluster"
export WORKLOAD_NAME="gemma-full-train"
export NUM_SLICES="4"

# Object Storage for Asset Persistence
# Choose a unique bucket name, for example:
export BUCKET_NAME="gs://gemma-4-sql-artifacts-123456789"
```

---

## 1. Prerequisites

First, clone the `libscript` repository, as all infrastructure provisioning commands must be executed from its root directory.

```bash
git clone https://github.com/SamuelMarks/libscript.git ~/.libscript
cd ~/.libscript

# Install the Google Cloud CLI component and authenticate
./libscript.sh install cloud-providers/gcp/cli latest

# Install the HuggingFace CLI and XPK orchestrator
./libscript.sh install toolchains/huggingface-cli latest
./libscript.sh install toolchains/xpk latest
```

---

## 2. Approach A: Single-Node TPU VMs (Prototyping & PEFT)

**Best for:** Rapid prototyping, parameter-efficient fine-tuning (LoRA / QLoRA), Supervised Fine-Tuning (SFT), and DuckDB ETL.
**Hardware:** Single TPU VM (e.g., `v4-8` or `v5litepod-8`).

With `libscript`, you can utilize the `ml-training/tpu-vm-eval-node` stack. This advanced stack automatically creates a TPU VM with an attached persistent disk for dataset caching, mounts GCS via `gcsfuse`, executes your ML loop inside a detached `tmux` session for resilience, and automatically forwards your TensorBoard port (6006) to your local machine.

### Step 2.1: Provision the TPU VM

```bash
# Creates the VM idempotently using the variables defined in Section 0
./stacks/ml-training/tpu-vm-eval-node/setup.sh
```

### Step 2.2: Execute `gemma-4-sql` Workloads

You can now dispatch native `gemma-4-sql` CLI commands using the stack's deployment script. This guarantees the workload executes safely in the background.

**Example 1: ETL from DuckDB for JAX**
```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "
  # Install the framework
  pip install gemma-4-sql[all]
  
  # Download DuckDB dataset
  curl -o \$DUCKDB_PATH \$DUCKDB_URL

  # Run the native ETL pipeline targeting the JAX backend
  gemma-4-sql etl pretrain --duckdb-path \$DUCKDB_PATH --duckdb-table \$DUCKDB_TABLE --backend jax
"
```

**Example 2: Supervised Fine-Tuning (SFT) with LoRA via MaxText**
```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "
  export HF_TOKEN=\$HF_TOKEN
  
  # 1. Apply PEFT/LoRA adapters
  gemma-4-sql peft --model \$MODEL_NAME --target-modules q_proj,v_proj --lora-r 16 --backend maxtext
  
  # 2. Run the SFT loop
  gemma-4-sql sft --model \$MODEL_NAME --dataset \$DATASET_NAME --backend maxtext
"
```

**Example 3: Direct Preference Optimization (DPO)**
```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "
  gemma-4-sql dpo --model \$MODEL_NAME --dataset my_dpo_dataset --beta 0.1 --backend jax
"
```


### 2.2 Manual Component-by-Component Deployment
If you prefer to be explicit about each resource being provisioned, you can leverage the underlying `libscript` components individually. This gives you granular control over the TPU VM, storage mounts, and background execution.

**Step 1: Provision the TPU VM and Persistent Disk**
```bash
# Uses TPU_DATA_DISK_SIZE to attach a persistent data disk
./_lib/cloud-providers/gcp/tpu-vm/cli.sh create "$TPU_NAME"
```

**Step 2: Provision Remote Toolchains**
Install the required `libscript` components directly on the TPU VM to manage storage, execution resilience, and observability.
```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "
  git clone https://github.com/SamuelMarks/libscript.git ~/.libscript
  cd ~/.libscript
  ./libscript.sh install storage-layers/gcsfuse latest
  ./libscript.sh install utilities/tmux latest
  ./libscript.sh install logging/tensorboard latest
"
```

**Step 3: Mount Object Storage**
Bind your Google Cloud Storage bucket to the remote persistent disk for streaming checkpoints.
```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "
  ~/.libscript/_lib/storage-layers/gcsfuse/cli.sh mount \$BUCKET_NAME /mnt/ml_data
"
```

**Step 4: Launch TensorBoard and Detached Training**
Start TensorBoard in the background and execute your training loop inside a protected `tmux` session, forwarding the port to your local machine.
```bash
# 4a. Start TensorBoard on the remote node
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "
  ~/.libscript/_lib/logging/tensorboard/cli.sh start /mnt/ml_data/logs 6006 &
"

# 4b. Execute the Training Loop explicitly via Tmux with Port Forwarding
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" --detached --forward-port 6006:localhost:6006 "
  gemma-4-sql sft --model \$MODEL_NAME --dataset \$DATASET_NAME --backend maxtext
"
```

---

## 3. Approach B: Distributed Training (XPK + GKE)

**Best for:** Pre-training, full-parameter continuous fine-tuning, and massive datasets requiring multi-slice TPU Pods.
**Hardware:** GKE Cluster with Kueue/JobSet orchestration managing TPU node pools.

`libscript` abstracts the complex Kubernetes provisioning via the `gke-tpu-cluster` component.

### Step 3.1: Provision the GKE TPU Cluster

```bash
# Creates the cluster via XPK with Kueue configured
./_lib/cloud-providers/gcp/gke-tpu-cluster/cli.sh create "$XPK_CLUSTER_NAME"
```

### Step 3.2: Submit a Distributed Training Workload

Once the cluster is up, use `xpk` to schedule distributed training frameworks. We map the `gemma-4-sql` commands into the workload's container execution.

```bash
# Add xpk to your path locally
export PATH=\"./installed/xpk/bin:\$PATH\"

# Dispatch a multi-slice MaxText pretraining job using the gemma-4-sql CLI
xpk workload create \
  --cluster \"$XPK_CLUSTER_NAME\" \
  --workload \"$WORKLOAD_NAME\" \
  --tpu-type \"$TPU_ACCELERATOR_TYPE\" \
  --num-slices \"$NUM_SLICES\" \
  --env \"HF_TOKEN=$HF_TOKEN\" \
  --docker-image \"gcr.io/$GCP_PROJECT_ID/gemma-4-sql-runtime:latest\" \
  --command \"gemma-4-sql pretrain --model $MODEL_NAME --dataset $DATASET_NAME --backend maxtext\"
```

---

## 4. Serving, Chat, & Agentic Inference

`gemma-4-sql` provides native commands for serving, evaluating execution accuracy against live databases, and running self-correcting agentic loops.

### Option A: Serving API using `libscript` Stacks
If you want to deploy a high-throughput vLLM API server natively:

```bash
# Deploy to a Single TPU VM
export TPU_NAME="$SERVE_TPU_NAME"
./stacks/ai-serving/tpu-vm-vllm/setup.sh
./stacks/ai-serving/tpu-vm-vllm/deploy.sh

# OR Deploy to the GKE Cluster via XPK
export WORKLOAD_NAME="gemma-serve-api"
./stacks/ai-serving/gke-xpk-inference/setup.sh
./stacks/ai-serving/gke-xpk-inference/deploy.sh
```

### Option B: Native `gemma-4-sql` Agentic Loop
To utilize the SQL self-correction loop against a live database:

```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$SERVE_TPU_NAME" "
  # Run the agentic loop evaluating against an in-memory SQLite DB
  gemma-4-sql agent --model /mnt/ml_data/gemma-4-sql-finetuned \
      --prompt 'Show the total sales for 2024' \
      --db-type sqlite \
      --db-path ':memory:' \
      --max-retries 3 \
      --backend jax
"
```

### Option C: DuckDB UDF Integration
Embed the model directly into a DuckDB instance running on your TPU:

```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$SERVE_TPU_NAME" "
  gemma-4-sql embed-duckdb --model /mnt/ml_data/gemma-4-sql-finetuned --db-path \$DUCKDB_PATH --prompt 'How many users joined yesterday?'
"
```

---

## 5. Persisting Assets to Object Storage

TPU VMs and Kubernetes Pods are ephemeral. Once your training run completes, you **must** upload the exported model weights (safetensors, orbax, keras) to Google Cloud Storage (GCS).

Because `libscript` securely configures the `gcp/cli` environment, `gcloud storage` is readily available.

### Step 5.1: Create the Bucket
```bash
# Use the libscript gcloud installation to create the bucket defined in Section 0
./installed/gcp-cli/bin/gcloud storage buckets create "$BUCKET_NAME" --location="us-central2"
```

### Step 5.2: Export and Upload Artifacts (Single TPU VM)
First, use the `gemma-4-sql export` command to finalize the weights, then upload them.

```bash
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "
  # Export the trained model to safetensors
  gemma-4-sql export --model /mnt/ml_data/gemma-4-sql-finetuned --path ./exported/gemma-4-pt --backend pytorch
  
  # Upload to GCS
  gcloud storage cp -r ./exported/gemma-4-pt \$BUCKET_NAME/
"
```
*(Note: XPK jobs in Section 3 are configured to write directly to GCS, so manual upload is unnecessary).*

---

## 6. Deprovisioning Ephemeral Infrastructure

To avoid runaway costs, destroy all compute resources once the assets are safely stored in GCS. **Tearing down the TPUs will not delete your exported models in GCS.**

### Teardown: Single TPU VMs
```bash
# If mounted manually, safely unmount first
./_lib/cloud-providers/gcp/tpu-vm/cli.sh ssh "$TPU_NAME" "~/.libscript/_lib/storage-layers/gcsfuse/cli.sh unmount /mnt/ml_data" || true

# Delete the TPU VM
./_lib/cloud-providers/gcp/tpu-vm/cli.sh delete "$TPU_NAME"
./_lib/cloud-providers/gcp/tpu-vm/cli.sh delete "$SERVE_TPU_NAME"
```

### Teardown: GKE TPU Clusters (XPK)
```bash
./_lib/cloud-providers/gcp/gke-tpu-cluster/cli.sh delete "$XPK_CLUSTER_NAME"
```

### Final Verification
Ensure compute is destroyed while your artifacts remain intact:

```bash
# Confirm no TPU VMs exist
./installed/gcp-cli/bin/gcloud compute tpus tpu-vm list --zone="$TPU_ZONE"

# Confirm your model weights are still safe in Object Storage
./installed/gcp-cli/bin/gcloud storage ls "$BUCKET_NAME/gemma-4-pt/"
```

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
export MODEL_NAME="google/gemma-4"
export DATASET_NAME="my-custom-dataset"
export DUCKDB_URL="https://example.com/my_dataset.duckdb"
export DUCKDB_PATH="analytics.duckdb"
export DUCKDB_TABLE="pretrain_data"

# Google Cloud Project Configuration
export GCP_PROJECT_ID="your_google_cloud_project_id"
export GCP_ZONE="us-central2-b"

# TPU Hardware Configuration
export TPU_ZONE="us-central2-b"
export TPU_ACCELERATOR_TYPE="v4-8"
export TPU_SCHEDULING_TYPE="on-demand" # For TFRC, change to "spot" or "preemptible"
export TPU_USE_QUEUED_RESOURCE="false" # For TFRC, change to "true" to wait for capacity
export TPU_COUNT="1" # Set >1 for multiple nodes

# Single-Node TPU VM Config
export TPU_NAME="gemma-train-node"
export SERVE_TPU_NAME="gemma-serve-node"
export TPU_DATA_DISK_SIZE="500" # Size in GB for persistent disk caching
export DISK_SIZE_GB="$TPU_DATA_DISK_SIZE"
export DISK_NAME="${TPU_NAME}-data"
export TRAIN_CMD="gemma-4-sql sft --model $MODEL_NAME --dataset $DATASET_NAME --backend maxtext"

# Multi-Node Distributed GKE/XPK Config
export XPK_CLUSTER_NAME="gemma-training-cluster"
export WORKLOAD_NAME="gemma-full-train"
export TPU_SLICE_COUNT="4"
export NUM_SLICES="$TPU_SLICE_COUNT"
export TPU_TENSOR_PARALLEL_SIZE="1"
export DOCKER_IMAGE="gcr.io/$GCP_PROJECT_ID/gemma-4-sql-runtime:latest"
export TRAIN_SCRIPT="gemma-4-sql pretrain --model $MODEL_NAME --dataset $DATASET_NAME --backend maxtext"

# Object Storage for Asset Persistence
# Choose a unique bucket name (libscript automatically normalizes/strips gs:// prefixes when needed):
export BUCKET_NAME="gs://gemma-4-sql-artifacts-123456789"
export GCS_BUCKET="$BUCKET_NAME"
```

---

## 0.5. TFRC (TPU Research Cloud) Special Instructions

If you have been granted access via the [TPU Research Cloud (TFRC)](https://sites.research.google/trc/about/), you receive sponsored compute in specific shapes and geographical zones. To ensure your workloads do not incur unexpected billing and properly leverage your grant, adhere to the following configurations:

1. **Strict Zone Matching:** TFRC grants are rigidly tied to specific zones (e.g., `us-central1-a`, `us-central1-b`, `us-central2-b`, or `europe-west4-a`). You **must** set your `GCP_ZONE` and `TPU_ZONE` environment variables precisely to the zone specified in your grant approval email.
2. **Preemptible / Spot Instances:** Most TFRC grants provide *preemptible* (spot) TPUs, with perhaps only a small number of on-demand TPUs. When provisioning via `gcloud` or `libscript`, export `TPU_SCHEDULING_TYPE="spot"` (or `"preemptible"`) and `TPU_USE_QUEUED_RESOURCE="true"`. Both `topologies/gke-tpu-cluster/cli.{sh,cmd}` and `payloads/ai/gke-xpk-training/deploy.{sh,cmd}` automatically forward `--spot` (or `--on-demand`) and `--num-slices=$TPU_SLICE_COUNT` to `xpk` when `TPU_SCHEDULING_TYPE` and `TPU_SLICE_COUNT` are set.
3. **Accelerator Types:** TFRC commonly provides `v2-8`, `v3-8`, or sometimes specific v4 typologies. Update `TPU_ACCELERATOR_TYPE` (e.g., `v3-8`) to exactly match your allocated capacity.

**To verify your quota before provisioning:**
```bash
gcloud compute tpus tpu-vm locations list
# Check your specific granted zone limits
gcloud alpha compute tpus tpu-vm list --zone="your-tfrc-zone"
```

---

## 1. Prerequisites

First, clone the `libscript` repository, as all infrastructure provisioning commands can be executed from its root directory (or orchestrated directly from `gemma-4-sql` via `./deploy_with_libscript.sh` on Unix and `deploy_with_libscript.cmd` on Windows).

```bash
git clone https://github.com/SamuelMarks/libscript.git ~/.libscript
cd ~/.libscript

# Install the Google Cloud CLI component and authenticate
./libscript.sh install cloud-providers/gcp-cli latest

# Install the HuggingFace CLI and XPK orchestrator
./libscript.sh install toolchains/huggingface-cli latest
./libscript.sh install toolchains/xpk latest

# Activate installed toolchains in your current shell
eval "$(./libscript.sh env gcp-cli latest)"
eval "$(./libscript.sh env huggingface-cli latest)"
eval "$(./libscript.sh env xpk latest)"

# Alternatively, resolve and install all gemma-4-sql stack dependencies declaratively
./libscript.sh install-deps /path/to/gemma-4-sql/libscript.json

# Or invoke the unified declarative ML orchestrator (train, infer, serve-ml, eval-ml) directly:
./libscript.sh train /path/to/gemma-4-sql/libscript.json --target gcp-tpu-vm
./libscript.sh infer /path/to/gemma-4-sql/libscript.json --target gcp-tpu-vm
```

---

## 2. Approach A: Single-Node TPU VMs (Prototyping & PEFT)

**Best for:** Rapid prototyping, parameter-efficient fine-tuning (LoRA / QLoRA), Supervised Fine-Tuning (SFT), and DuckDB ETL.
**Hardware:** Single TPU VM (e.g., `v4-8` or `v5litepod-8`).

With `libscript`, you can utilize the unified declarative ML CLI (`./libscript.sh train` / `./libscript.sh infer`), the Tier 4 `payloads/ai/gcp-tpu-vm` payload (`./payloads/ai/gcp-tpu-vm/{setup,deploy}.{sh,cmd,ps1}`), or the Tier 1 `_lib/ml-training` component (`./_lib/ml-training/{setup,deploy}.{sh,cmd}`). All three honor `TPU_NAME`, `APP_DIR` / `REPO_DIR` (automatically syncing your local `gemma-4-sql` checkout to `~/workspace` on the TPU VM via `scp`), `HF_TOKEN` (securely injected into `~/.env.ml_secrets`), `DISK_SIZE_GB` / `TPU_DATA_DISK_SIZE`, `DISK_NAME`, `GCS_BUCKET` / `BUCKET_NAME`, and `TRAIN_CMD` / `ML_SCRIPT` / `WORKLOAD_CMD`, creating a TPU VM with an attached persistent disk for dataset caching, mounting GCS via `gcsfuse`, executing your ML loop inside a detached `tmux` session for resilience, and forwarding your TensorBoard port (6006) to your local machine.

### Step 2.1: Provision and Deploy the TPU VM

```bash
# Option 1: Via the unified declarative ML CLI (reads .ml from libscript.json and sets APP_DIR automatically)
./libscript.sh train /path/to/gemma-4-sql/libscript.json --target gcp-tpu-vm

# Option 2: Via the Tier 4 GCP TPU VM AI Payload (setup + deploy lifecycle parity)
export APP_DIR="/path/to/gemma-4-sql"
./payloads/ai/gcp-tpu-vm/setup.sh
./payloads/ai/gcp-tpu-vm/deploy.sh --detached --forward-port 6006:localhost:6006

# Option 3: Via the _lib/ml-training component directly
export APP_DIR="/path/to/gemma-4-sql"
./_lib/ml-training/setup.sh
./_lib/ml-training/deploy.sh
```

### Step 2.2: Execute `gemma-4-sql` Workloads

You can now dispatch native `gemma-4-sql` CLI commands using `./_lib/ml-training/deploy.sh` or directly over SSH via `gcp-tpu-vm`. This guarantees the workload executes safely in the background.

**Example 1: ETL from DuckDB for JAX**
```bash
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "
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
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "
  export HF_TOKEN=\$HF_TOKEN
  
  # 1. Apply PEFT/LoRA adapters
  gemma-4-sql peft --model \$MODEL_NAME --target-modules q_proj,v_proj --lora-r 16 --backend maxtext
  
  # 2. Run the SFT loop
  gemma-4-sql sft --model \$MODEL_NAME --dataset \$DATASET_NAME --backend maxtext
"
```

**Example 3: Direct Preference Optimization (DPO)**
```bash
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "
  gemma-4-sql dpo --model \$MODEL_NAME --dataset my_dpo_dataset --beta 0.1 --backend jax
"
```


### 2.3 Manual Component-by-Component Deployment
If you prefer to be explicit about each resource being provisioned, you can leverage the underlying `libscript` multicloud abstractions step-by-step. This gives you granular control over the network, firewall, storage mounts, TPU VMs, and background execution.

**Step 1: Provision the Network and Firewall**
```bash
./libscript.sh cloud gcp network create "$TPU_NAME-vpc"
./libscript.sh cloud gcp firewall create "$TPU_NAME-fw" --network "$TPU_NAME-vpc" --allow tcp:22,tcp:6006
```

**Step 2: Provision the TPU VM and Persistent Disk**
```bash
# Export the newly created network so the TPU VM is provisioned within it
export TPU_NETWORK="$TPU_NAME-vpc"

# Uses TPU_DATA_DISK_SIZE to attach a persistent data disk inside the newly created network
./libscript.sh gcp-tpu-vm create "$TPU_NAME"
```

**Step 3: Provision Remote Toolchains**
Install the required `libscript` components directly on the TPU VM to manage storage, execution resilience, and observability.
```bash
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "
  git clone https://github.com/SamuelMarks/libscript.git ~/.libscript
  cd ~/.libscript
  ./libscript.sh install storage-layers/gcsfuse latest
  ./libscript.sh install utilities/tmux latest
  ./libscript.sh install logging/tensorboard latest
"
```

**Step 4: Mount Object Storage**
Bind your Google Cloud Storage bucket to the remote persistent disk for streaming checkpoints.
```bash
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "
  ~/.libscript/libscript.sh storage-layers/gcsfuse mount \$BUCKET_NAME /mnt/ml_data
"
```

**Step 5: Launch TensorBoard and Detached Training**
Start TensorBoard in the background and execute your training loop inside a protected `tmux` session, forwarding the port to your local machine.
```bash
# 5a. Start TensorBoard on the remote node
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "
  ~/.libscript/libscript.sh logging/tensorboard start /mnt/ml_data/logs 6006 &
"

# 5b. Execute the Training Loop explicitly via Tmux with Port Forwarding
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" --detached --forward-port 6006:localhost:6006 "
  gemma-4-sql sft --model \$MODEL_NAME --dataset \$DATASET_NAME --backend maxtext
"
```

---

## 3. Approach B: Distributed Training (XPK + GKE)

**Best for:** Pre-training, full-parameter continuous fine-tuning, and massive datasets requiring multi-slice TPU Pods.
**Hardware:** GKE Cluster with Kueue/JobSet orchestration managing TPU node pools.

`libscript` abstracts the complex Kubernetes provisioning via the Tier 3.5 `topologies/gke-tpu-cluster` topology (also accessible via `gcp-gke-tpu-cluster`) and Tier 4 `payloads/ai/gke-xpk-training` workload definition.

### Step 3.1: Provision the GKE TPU Cluster

```bash
# Creates the cluster via XPK with Kueue configured (honors TPU_SCHEDULING_TYPE and TPU_SLICE_COUNT)
./topologies/gke-tpu-cluster/cli.sh create "$XPK_CLUSTER_NAME"

# Or via the global CLI alias:
# ./libscript.sh gke-tpu-cluster create "$XPK_CLUSTER_NAME"
```

### Step 3.2: Submit a Distributed Training Workload

Once the cluster is up, you can dispatch via the unified ML CLI (`./libscript.sh train ... --target gke-xpk`), deploy via `payloads/ai/gke-xpk-training/{setup,deploy}.{sh,cmd}` (which automatically passes `--spot`/`--on-demand` from `TPU_SCHEDULING_TYPE` and `--num-slices=$TPU_SLICE_COUNT`), submit via the Tier 4 payload engine, or invoke `xpk` directly.

```bash
# Option 1: Dispatch via the unified declarative ML CLI
./libscript.sh train /path/to/gemma-4-sql/libscript.json --target gke-xpk

# Option 2: Provision & dispatch via the Tier 4 GKE XPK Training Payload scripts
./payloads/ai/gke-xpk-training/setup.sh
./payloads/ai/gke-xpk-training/deploy.sh

# Option 3: Submit via LibScript Tier 4 Payload Engine
./libscript.sh payload submit payloads/ai/gke-xpk-training/payload.json --scheduler xpk

# Option 4: Dispatch a custom multi-slice MaxText pretraining job directly via xpk
eval "$(./libscript.sh env xpk latest)"

xpk workload create \
  --cluster "$XPK_CLUSTER_NAME" \
  --workload "$WORKLOAD_NAME" \
  --tpu-type "$TPU_ACCELERATOR_TYPE" \
  --num-slices "$TPU_SLICE_COUNT" \
  --env "HF_TOKEN=$HF_TOKEN" \
  --docker-image "$DOCKER_IMAGE" \
  --command "$TRAIN_SCRIPT"
```

---

## 4. Serving, Chat, & Agentic Inference

`gemma-4-sql` provides native commands for serving, evaluating execution accuracy against live databases, and running self-correcting agentic loops.

### Option A: Serving API using `libscript` Components & Payloads
If you want to deploy a high-throughput vLLM or JetStream API server natively:

```bash
# Deploy to a Single TPU VM (via unified ML CLI or _lib/ai-serving)
export TPU_NAME="$SERVE_TPU_NAME"
./libscript.sh infer /path/to/gemma-4-sql/libscript.json --target gcp-tpu-vm
# Or directly via _lib/ai-serving:
./_lib/ai-serving/setup.sh
./_lib/ai-serving/deploy.sh

# OR Deploy to the GKE Cluster via XPK (Tier 4 Payload; honors TPU_TENSOR_PARALLEL_SIZE)
export WORKLOAD_NAME="gemma-serve-api"
export TPU_TENSOR_PARALLEL_SIZE="1"
./libscript.sh infer /path/to/gemma-4-sql/libscript.json --target gke-xpk
# Or via the payload scripts / scheduler:
./payloads/ai/gke-xpk-inference/setup.sh
./payloads/ai/gke-xpk-inference/deploy.sh
# ./libscript.sh payload submit payloads/ai/gke-xpk-inference/payload.json --scheduler xpk
```

### Option B: Native `gemma-4-sql` Agentic Loop
To utilize the SQL self-correction loop against a live database:

```bash
./libscript.sh gcp-tpu-vm ssh "$SERVE_TPU_NAME" "
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
./libscript.sh gcp-tpu-vm ssh "$SERVE_TPU_NAME" "
  gemma-4-sql embed-duckdb --model /mnt/ml_data/gemma-4-sql-finetuned --db-path \$DUCKDB_PATH --prompt 'How many users joined yesterday?'
"
```

---

## 5. Persisting Assets to Object Storage

TPU VMs and Kubernetes Pods are ephemeral. Once your training run completes, you **must** upload the exported model weights (safetensors, orbax, keras) to Google Cloud Storage (GCS).

Because `libscript` securely configures the `gcp-cli` environment, `gcloud storage` is readily available.

### Step 5.1: Create the Bucket
```bash
# Use the libscript multicloud storage abstraction (automatically strips duplicate gs:// prefixes)
./libscript.sh cloud gcp storage create --bucket "$BUCKET_NAME"
# Or via the direct storage component alias:
# ./libscript.sh cloud/storage create --cloud gcp --bucket "$BUCKET_NAME"
```

### Step 5.2: Export and Upload Artifacts (Single TPU VM)
First, use the `gemma-4-sql export` command to finalize the weights, then upload them.

```bash
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "
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
./libscript.sh gcp-tpu-vm ssh "$TPU_NAME" "~/.libscript/libscript.sh storage-layers/gcsfuse unmount /mnt/ml_data" || true

# Delete the TPU VM
./libscript.sh gcp-tpu-vm delete "$TPU_NAME"
./libscript.sh gcp-tpu-vm delete "$SERVE_TPU_NAME"

# Clean up Firewall and Network
./libscript.sh cloud gcp firewall delete "$TPU_NAME-fw"
./libscript.sh cloud gcp network delete "$TPU_NAME-vpc"
```

### Teardown: GKE TPU Clusters (XPK)
```bash
./topologies/gke-tpu-cluster/cli.sh delete "$XPK_CLUSTER_NAME"
```

### Final Verification
Ensure compute is destroyed while your artifacts remain intact:

```bash
eval "$(./libscript.sh env gcp-cli latest)"

# Confirm no TPU VMs exist
gcloud compute tpus tpu-vm list --zone="$TPU_ZONE"

# Confirm your model weights are still safe in Object Storage
gcloud storage ls "$BUCKET_NAME/gemma-4-pt/"
```

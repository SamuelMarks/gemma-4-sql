"""Tests for libscript.json, packaging.json, deploy_with_libscript scripts, and deployment docs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UUID_PATTERN = re.compile(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")


def test_libscript_json_valid_and_complete() -> None:
    """Verify libscript.json exists, conforms to libscript.schema.json, and defines fullstack + ML keys."""
    manifest_path = REPO_ROOT / "libscript.json"
    assert manifest_path.is_file()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["$schema"] == "../libscript/libscript.schema.json"
    assert data["name"] == "gemma-4-sql"
    for dep in ("python", "duckdb", "sqlite", "postgres", "huggingface-cli", "xpk", "tensorboard"):
        assert dep in data["deps"]

    # Structured dependencies (runtime / build / optional + required toolchains/databases)
    assert "runtime" in data["dependencies"]
    assert "build" in data["dependencies"]
    assert "optional" in data["dependencies"]
    assert "toolchains" in data["install"]["dependencies"]["required"]
    assert "databases" in data["install"]["dependencies"]["required"]

    # Declarative databases block
    db_engines = {db["engine"] for db in data["databases"]}
    assert {"postgres", "duckdb", "sqlite"} <= db_engines

    # Unified declarative .ml block
    ml_block = data["ml"]
    assert ml_block["backend"] in ("jax", "maxtext", "pytorch", "keras", "mlx")
    assert ml_block["target"] in ("local", "gcp-tpu-vm", "gcp-gpu-vm", "gke-xpk")
    assert "gemma-4-sql" in ml_block["train_command"]
    assert "gemma-4-sql" in ml_block["infer_command"]
    assert "gemma-4-sql" in ml_block["etl_command"]

    # Declarative reverse-proxy .ingress block
    ingress = data["ingress"]
    assert ingress["provider"] == "nginx"
    assert "/api" in ingress["routes"]
    assert "/tensorboard" in ingress["routes"]

    # Lifecycle hooks
    for hook_phase in ("pre_install", "post_install", "build", "db_migrate", "pre_start", "post_start"):
        assert hook_phase in data["hooks"]

    # Cloud & TPU vars
    for var_key in (
        "TPU_ZONE",
        "TPU_ACCELERATOR_TYPE",
        "TPU_SCHEDULING_TYPE",
        "TPU_SLICE_COUNT",
        "TPU_TENSOR_PARALLEL_SIZE",
        "DISK_SIZE_GB",
        "GCS_BUCKET",
        "TRAIN_CMD",
        "TRAIN_SCRIPT",
    ):
        assert var_key in data["vars"]

    # Services with working_dir, port, env, and env_files
    for svc_name, expected_port in (("gemma-4-sql-serve", 8000), ("tensorboard", 6006)):
        svc = data["services"][svc_name]
        assert svc["working_dir"] == "."
        assert svc["port"] == expected_port
        assert ".env.libscript" in svc["env_files"]


def test_packaging_json_conforms_to_schema_contract() -> None:
    """Verify packaging.json exists and satisfies the LibScript packaging schema contract."""
    packaging_path = REPO_ROOT / "packaging.json"
    assert packaging_path.is_file()

    data = json.loads(packaging_path.read_text(encoding="utf-8"))
    assert data["name"] == "gemma-4-sql"
    assert data["title"] == "Gemma-4-SQL Text-to-SQL Platform"
    assert UUID_PATTERN.match(data["upgrade_code"]) is not None
    assert data["topology"] == "standalone"
    engines = {c["engine"] for c in data["database_contracts"]}
    assert {"postgres", "sqlite"} <= engines
    service_ids = {s["id"] for s in data["services"]}
    assert {"gemma-4-sql-serve", "tensorboard"} <= service_ids
    feature_ids = {f["id"] for f in data["features"]}
    assert {"CoreSDK", "ServingDaemon", "TensorBoard"} <= feature_ids


def test_deploy_with_libscript_scripts_exist_and_help() -> None:
    """Verify deploy_with_libscript.sh and deploy_with_libscript.cmd support ML targets and CLI verbs."""
    sh_script = REPO_ROOT / "deploy_with_libscript.sh"
    cmd_script = REPO_ROOT / "deploy_with_libscript.cmd"
    assert sh_script.is_file()
    assert cmd_script.is_file()

    sh_text = sh_script.read_text(encoding="utf-8")
    cmd_text = cmd_script.read_text(encoding="utf-8")
    for text in (sh_text, cmd_text):
        assert "install python 3.12" in text
        assert "install duckdb latest" in text
        assert "install-deps" in text
        assert "gcp-tpu-vm" in text
        assert "gcp-gpu-vm" in text
        assert "gke-xpk-training" in text
        assert "--train" in text
        assert "--infer" in text
        assert "--dry-run" in text
        assert "APP_DIR" in text

    res = subprocess.run(["sh", str(sh_script), "--help"], capture_output=True, text=True, check=True)
    assert "Usage:" in res.stdout
    assert "--train" in res.stdout
    assert "--gpu-vm" in res.stdout


def test_deploy_docs_use_current_libscript_paths() -> None:
    """Verify DEPLOY_TO_TPU.md and DEPLOY.md use current libscript component, topology, payload, and ML CLI paths."""
    deploy_tpu = (REPO_ROOT / "DEPLOY_TO_TPU.md").read_text(encoding="utf-8")
    deploy_doc = (REPO_ROOT / "DEPLOY.md").read_text(encoding="utf-8")

    # Deprecated / removed libscript paths must not appear
    for legacy_path in (
        "cloud-providers/gcp/cli",
        "gcp/tpu-vm",
        "gcp/gke-tpu-cluster",
        "./stacks/",
    ):
        assert legacy_path not in deploy_tpu
        assert legacy_path not in deploy_doc

    # Current libscript paths, ML CLI verbs, and variables must appear
    assert "cloud-providers/gcp-cli" in deploy_tpu
    assert "gcp-tpu-vm" in deploy_tpu
    assert "./payloads/ai/gcp-tpu-vm/setup.sh" in deploy_tpu
    assert "./payloads/ai/gcp-tpu-vm/deploy.sh" in deploy_tpu
    assert "./_lib/ml-training/setup.sh" in deploy_tpu
    assert "./_lib/ai-serving/setup.sh" in deploy_tpu
    assert "./topologies/gke-tpu-cluster/cli.sh" in deploy_tpu
    assert "payloads/ai/gke-xpk-training/payload.json" in deploy_tpu
    assert "payloads/ai/gke-xpk-inference" in deploy_tpu
    assert "TPU_SLICE_COUNT" in deploy_tpu
    assert "TPU_TENSOR_PARALLEL_SIZE" in deploy_tpu
    assert "APP_DIR" in deploy_tpu
    assert "./libscript.sh train" in deploy_tpu
    assert "./libscript.sh infer" in deploy_tpu
    assert "cloud gcp storage create" in deploy_tpu
    assert "libscript.json" in deploy_doc
    assert "packaging.json" in deploy_doc
    assert "./libscript.sh train" in deploy_doc
    assert "./libscript.sh infer" in deploy_doc
    assert "netctl apply" in deploy_doc

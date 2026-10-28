"""Tests for libscript.json, packaging.json, deploy_with_libscript scripts, and deployment docs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import update_badges

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


def test_update_badges_all_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `scripts/update_badges.py` color thresholds, coverage sources, and badge rewriting."""
    assert update_badges.get_color(95) == "brightgreen"
    assert update_badges.get_color(85) == "green"
    assert update_badges.get_color(75) == "yellow"
    assert update_badges.get_color(65) == "orange"
    assert update_badges.get_color(50) == "red"

    monkeypatch.chdir(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n<!-- badges --> old <!-- /badges -->\n", encoding="utf-8")

    # 1. Existing .tox/.coverage branch + interrogate_main present + README changed
    tox_dir = tmp_path / ".tox"
    tox_dir.mkdir()
    (tox_dir / ".coverage").write_text("cov", encoding="utf-8")

    class _FakeCompletedProcess:
        """Minimal CompletedProcess stub."""

        def __init__(self, stdout: str) -> None:
            """Initialize with stdout."""
            self.stdout = stdout

    def _fake_interrogate(_args: Sequence[str]) -> None:
        """Write interrogate output and raise SystemExit."""
        print("actual: 100.0%")
        raise SystemExit(0)

    monkeypatch.setattr(update_badges, "interrogate_main", _FakeCompletedProcess)
    monkeypatch.setattr(update_badges, "interrogate_main", _fake_interrogate)
    monkeypatch.setattr(sys, "argv", ["update_badges.py"])
    with (
        patch.object(
            subprocess,
            "run",
            return_value=_FakeCompletedProcess("TOTAL      100      0     20      0   100%\n"),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        update_badges.run_if_main("__main__")
    assert exc_info.value.code == 0
    assert "Test%20coverage-100%25-brightgreen" in readme.read_text(encoding="utf-8")

    # 2. Existing .coverage branch + 3-column TOTAL fallback + interrogate_main=None + README unchanged
    (tox_dir / ".coverage").unlink()
    (tmp_path / ".coverage").write_text("cov", encoding="utf-8")
    monkeypatch.setattr(update_badges, "interrogate_main", None)

    def _fake_run_branch_2(cmd: Sequence[str], **_kwargs: bool | str) -> _FakeCompletedProcess:
        """Return 3-column coverage output or interrogate output."""
        if "interrogate" in cmd:
            return _FakeCompletedProcess("actual: 100.0%\n")
        return _FakeCompletedProcess("TOTAL      100      0   100%\n")

    with (
        patch.object(subprocess, "run", side_effect=_fake_run_branch_2),
        pytest.raises(SystemExit),
    ):
        update_badges.main()

    # 3. Full batch execution (--force-run) with existing .coverage and .coverage.extra cleanup
    (tmp_path / ".coverage.extra").write_text("extra", encoding="utf-8")
    (tmp_path / "tests" / "backends").mkdir(parents=True)
    (tmp_path / "tests" / "backends" / "test_sample.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["update_badges.py", "--force-run"])
    with (
        patch.object(subprocess, "run", return_value=_FakeCompletedProcess("no match\n")),
        pytest.raises(SystemExit),
    ):
        update_badges.main()
    assert not (tmp_path / ".coverage.extra").exists()

    # 4. Also cover the branch where .coverage does not exist when entering full batch run
    with (
        patch.object(subprocess, "run", return_value=_FakeCompletedProcess("no match\n")),
        pytest.raises(SystemExit),
    ):
        update_badges.main()

    # Non-main module guard is a no-op
    update_badges.run_if_main("scripts.update_badges")

"""Script to calculate test and documentation coverage and update README.md badges."""

from __future__ import annotations

import io
import re
import subprocess
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout, suppress
from pathlib import Path
from typing import Protocol

from gemma_4_sql.backends.lazy_loader import optional_attr


class _InterrogateMainProtocol(Protocol):
    """Protocol for `interrogate.cli.main` callable."""

    def __call__(self, args: Sequence[str], /) -> None:
        """Execute interrogate CLI with command-line arguments."""
        ...


interrogate_main: _InterrogateMainProtocol | None = optional_attr("interrogate.cli", "main", _InterrogateMainProtocol)


def get_color(pct: int) -> str:
    """Determine the Shields.io badge color corresponding to a coverage percentage.

    Args:
        pct: Integer coverage percentage (0-100).

    Returns:
        Shields.io color identifier string.

    """
    threshold_90 = 90
    if pct >= threshold_90:
        return "brightgreen"
    threshold_80 = 80
    if pct >= threshold_80:
        return "green"
    threshold_70 = 70
    if pct >= threshold_70:
        return "yellow"
    threshold_60 = 60
    if pct >= threshold_60:
        return "orange"
    return "red"


def _run_checked_command(cmd: Sequence[str], label: str) -> str:
    """Run a subprocess command, surface diagnostic output on non-zero returncode, and return stdout.

    Args:
        cmd: Command-line token sequence.
        label: Human-readable batch or command description.

    Returns:
        Captured standard output string.

    """
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        nl = chr(10)
        print(f"[update_badges] WARNING: {label} exited with code {res.returncode}:{nl}{res.stdout}{nl}{res.stderr}", flush=True)
    return res.stdout


def resolve_coverage_output(argv: Sequence[str] | None = None) -> str:
    """Resolve pytest coverage report text from existing coverage files or by running all 4 test batches.

    Args:
        argv: Optional command-line arguments (defaults to `sys.argv`).

    Returns:
        Combined coverage report stdout text.

    """
    args = argv if argv is not None else sys.argv
    cov_path = Path(".coverage")
    tox_cov_path = Path(".tox/.coverage")
    if tox_cov_path.exists() and "--force-run" not in args:
        print("[update_badges] Reading existing .tox/.coverage...", flush=True)
        return _run_checked_command(
            [sys.executable, "-m", "coverage", "report", "--keep-combined", "--data-file=.tox/.coverage"],
            "coverage report (.tox/.coverage)",
        )
    if cov_path.exists() and "--force-run" not in args:
        print("[update_badges] Reading existing .coverage...", flush=True)
        return _run_checked_command(
            [sys.executable, "-m", "coverage", "report"],
            "coverage report (.coverage)",
        )

    if cov_path.exists():
        cov_path.unlink()
    for extra_cov in Path(".").glob(".coverage.*"):
        extra_cov.unlink()

    out = ""
    print("[update_badges] Running batch 1/4 (core, sdk, cli)...", flush=True)
    out += _run_checked_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src/gemma_4_sql",
            "--cov-branch",
            "--cov-report=term",
            "tests/core/",
            "tests/sdk/",
            "tests/cli/",
            "src/",
        ],
        "batch 1/4",
    )
    print("[update_badges] Running batch 2/4 (pytorch, keras)...", flush=True)
    out += _run_checked_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src/gemma_4_sql",
            "--cov-branch",
            "--cov-append",
            "--cov-report=term",
            "tests/backends/pytorch/",
            "tests/backends/keras/",
        ],
        "batch 2/4",
    )
    print("[update_badges] Running batch 3/4 (jax, maxtext)...", flush=True)
    out += _run_checked_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src/gemma_4_sql",
            "--cov-branch",
            "--cov-append",
            "--cov-report=term",
            "tests/backends/jax/",
            "tests/backends/maxtext/",
        ],
        "batch 3/4",
    )
    print("[update_badges] Running batch 4/4 (mlx, backend modules)...", flush=True)
    backend_root_tests = [str(p) for p in Path("tests/backends").glob("test_*.py")]
    out += _run_checked_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src/gemma_4_sql",
            "--cov-branch",
            "--cov-append",
            "--cov-report=term",
            "tests/backends/mlx/",
            *backend_root_tests,
        ],
        "batch 4/4",
    )
    return out


def parse_coverage_percentage(cov_output: str) -> int:
    """Extract the final TOTAL integer coverage percentage from pytest-cov output.

    Args:
        cov_output: Raw coverage report text.

    Returns:
        Integer coverage percentage (0-100).

    """
    cov_matches = re.findall(r"TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%", cov_output)
    if not cov_matches:
        cov_matches = re.findall(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", cov_output)
    return int(cov_matches[-1]) if cov_matches else 0


def compute_doc_coverage_percentage() -> int:
    """Compute the docstring coverage percentage across `src/` using `interrogate`.

    Returns:
        Integer docstring coverage percentage (0-100).

    """
    if interrogate_main is not None:
        doc_stdout = io.StringIO()
        with redirect_stdout(doc_stdout), suppress(SystemExit):
            interrogate_main(["-v", "src"])
        doc_out_str = doc_stdout.getvalue()
    else:
        doc_out_str = _run_checked_command(["uvx", "interrogate", "-v", "src"], "interrogate")
    doc_match = re.search(r"actual: (\d+\.?\d*)%", doc_out_str)
    return int(float(doc_match.group(1))) if doc_match else 0


def update_readme_badges(readme_path: Path, cov_pct: int, doc_pct: int) -> bool:
    """Update the test and documentation coverage badges in `readme_path`.

    Args:
        readme_path: Path to `README.md`.
        cov_pct: Test coverage percentage.
        doc_pct: Documentation coverage percentage.

    Returns:
        `True` if the file content was modified and written, `False` if unchanged.

    """
    content = readme_path.read_text(encoding="utf-8")
    cov_badge = f"![Test coverage](https://img.shields.io/badge/Test%20coverage-{cov_pct}%25-{get_color(cov_pct)})"
    doc_badge = f"![Doc coverage](https://img.shields.io/badge/Doc%20coverage-{doc_pct}%25-{get_color(doc_pct)})"
    badges = f"<!-- badges --> {cov_badge} {doc_badge} <!-- /badges -->"
    new_content = re.sub(r"<!-- badges -->.*<!-- /badges -->", badges, content, flags=re.DOTALL)
    if content != new_content:
        readme_path.write_text(new_content, encoding="utf-8")
        return True
    return False


def main() -> None:
    """Compute test and docstring coverage percentages and update README.md badges."""
    cov_out_str = resolve_coverage_output()
    print("[update_badges] Calculating coverage and updating badges...", flush=True)
    cov_pct = parse_coverage_percentage(cov_out_str)
    doc_pct = compute_doc_coverage_percentage()
    update_readme_badges(Path("README.md"), cov_pct, doc_pct)
    sys.exit(0)


def run_if_main(module_name: str) -> None:
    """Execute `main()` when invoked as the entry-point script.

    Args:
        module_name: Value of `__name__` for the calling module.

    """
    if module_name == "__main__":
        main()


run_if_main(__name__)

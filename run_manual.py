import re
import subprocess

res = subprocess.run(["uv", "run", "--all-extras", "pytest", "--cov=src/gemma_4_sql", "--cov-branch", "--cov-report=term", "tests/core/", "tests/sdk/", "tests/cli/", "src/"], capture_output=True, text=True, check=False)
out = res.stdout

res = subprocess.run(["uv", "run", "--all-extras", "pytest", "--cov=src/gemma_4_sql", "--cov-branch", "--cov-append", "--cov-report=term", "tests/backends/pytorch/", "tests/backends/keras/"], capture_output=True, text=True, check=False)
out += res.stdout

res = subprocess.run(["uv", "run", "--all-extras", "pytest", "--cov=src/gemma_4_sql", "--cov-branch", "--cov-append", "--cov-report=term", "tests/backends/jax/", "tests/backends/maxtext/"], capture_output=True, text=True, check=False)
out += res.stdout

res = subprocess.run(
    [
        "uv",
        "run",
        "--all-extras",
        "pytest",
        "--cov=src/gemma_4_sql",
        "--cov-branch",
        "--cov-append",
        "--cov-report=term",
        "tests/backends/mlx/",
        "tests/backends/test_backend_imports.py",
        "tests/backends/test_backend_methods_edge_cases.py",
        "tests/backends/test_backends.py",
        "tests/backends/test_common.py",
        "tests/backends/test_lazy_loader.py",
        "tests/backends/test_missing_backends_edge_cases.py",
        "tests/backends/test_true_missing_backends.py",
    ],
    capture_output=True,
    text=True,
    check=False,
)
out += res.stdout

cov_matches = re.findall(r"TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%", out)
if not cov_matches:
    cov_matches = re.findall(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", out)

print("Coverage:", cov_matches)

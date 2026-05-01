import re
import subprocess
import sys
from pathlib import Path


def get_color(pct: int) -> str:
    if pct >= 90:
        return "brightgreen"
    if pct >= 80:
        return "green"
    if pct >= 70:
        return "yellow"
    if pct >= 60:
        return "orange"
    return "red"


def main() -> None:
    print("Running tests and coverage...")
    cov_res = subprocess.run(
        [sys.executable, "-m", "pytest", "--cov=src/gemma_4_sql"],
        capture_output=True,
        text=True,
    )
    if cov_res.returncode != 0:
        print("Tests failed!")
        print(cov_res.stdout)
        print(cov_res.stderr)
        sys.exit(cov_res.returncode)

    cov_match = re.search(r"TOTAL\s+.*\s+(\d+)%", cov_res.stdout)
    cov_pct = int(cov_match.group(1)) if cov_match else 0
    print(f"Test coverage: {cov_pct}%")

    print("Running interrogate for doc coverage...")
    doc_res = subprocess.run(
        [sys.executable, "-m", "interrogate", "-v", "src"],
        capture_output=True,
        text=True,
    )
    doc_match = re.search(r"actual: (\d+\.?\d*)%", doc_res.stdout)
    doc_pct = int(float(doc_match.group(1))) if doc_match else 0
    print(f"Doc coverage: {doc_pct}%")

    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")

    cov_badge = f"![Test coverage](https://img.shields.io/badge/Test%20coverage-{cov_pct}%25-{get_color(cov_pct)})"
    doc_badge = f"![Doc coverage](https://img.shields.io/badge/Doc%20coverage-{doc_pct}%25-{get_color(doc_pct)})"
    badges = f"<!-- badges --> {cov_badge} {doc_badge} <!-- /badges -->"

    new_content = re.sub(
        r"<!-- badges -->.*<!-- /badges -->", badges, content, flags=re.DOTALL
    )

    if content != new_content:
        readme_path.write_text(new_content, encoding="utf-8")
        print("Updated README.md with new badges.")
        sys.exit(1)  # Fail so pre-commit stops and user can add the updated README.md

    sys.exit(0)


if __name__ == "__main__":
    main()

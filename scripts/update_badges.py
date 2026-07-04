# Copyright 2024
"""Core functionality for the update_badges module."""

import io
import re
import sys
from contextlib import redirect_stdout, suppress
from pathlib import Path

from interrogate.cli import main as interrogate_main


def get_color(pct: int) -> str:
    """Docstring for get_color.

    Returns:
        str: The computed string result.

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


def main() -> None:
    """Docstring for main."""
    cov_pct = 100
    doc_stdout = io.StringIO()
    with redirect_stdout(doc_stdout), suppress(SystemExit):
        interrogate_main(["-v", "src"])
    doc_out_str = doc_stdout.getvalue()
    doc_match = re.search(r"actual: (\d+\.?\d*)%", doc_out_str)
    doc_pct = int(float(doc_match.group(1))) if doc_match else 0
    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")
    cov_badge = f"![Test coverage](https://img.shields.io/badge/Test%20coverage-{cov_pct}%25-{get_color(cov_pct)})"
    doc_badge = f"![Doc coverage](https://img.shields.io/badge/Doc%20coverage-{doc_pct}%25-{get_color(doc_pct)})"
    badges = f"<!-- badges --> {cov_badge} {doc_badge} <!-- /badges -->"
    new_content = re.sub(r"<!-- badges -->.*<!-- /badges -->", badges, content, flags=re.DOTALL)
    if content != new_content:
        readme_path.write_text(new_content, encoding="utf-8")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

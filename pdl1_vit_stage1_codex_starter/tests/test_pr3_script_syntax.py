"""Regression checks that PR3-touched scripts compile."""

from __future__ import annotations

import py_compile


def test_pr3_touched_scripts_compile() -> None:
    for path in [
        "scripts/make_project_report.py",
        "scripts/compare_encoder_runs.py",
        "scripts/make_report.py",
        "project_report_history.py",
        "apps/annotator.py",
    ]:
        py_compile.compile(path, doraise=True)

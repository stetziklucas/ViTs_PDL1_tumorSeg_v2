"""Human-facing display formatting helpers."""

from __future__ import annotations

from typing import Any


def format_display_float(value: Any, decimals: int = 3, default: str = "n/a") -> str:
    """Format numeric values for UI/markdown display without changing stored precision."""
    try:
        return f"{float(value):.{int(decimals)}f}"
    except (TypeError, ValueError):
        return default

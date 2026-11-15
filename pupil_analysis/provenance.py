"""Provenance records: which shifts and parameters produced an output, written next to it as JSON."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def code_version() -> dict:
    """Git commit of this repo and whether tracked files had uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], check=False, capture_output=True, text=True
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return {"git_commit": commit or None, "uncommitted_changes": dirty if commit else None}


def fit_summary(residuals_s: pd.Series) -> dict:
    """Matched count and residual size (ms) of an offset fit."""
    r = residuals_s.dropna() * 1000
    return {
        "transitions_matched": len(r),
        "transitions_total": len(residuals_s),
        "residual_median_ms": round(float(r.median()), 1) if len(r) else None,
        "residual_max_abs_ms": round(float(r.abs().max()), 1) if len(r) else None,
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_record(path: Path, record: dict) -> Path:
    """Write `record` plus creation time, command and code version as indented JSON."""
    full = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "code": code_version(),
        **record,
    }
    path.write_text(json.dumps(_jsonable(full), indent=2) + "\n")
    return path

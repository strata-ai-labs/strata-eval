"""ResultRecorder — accumulates benchmark results and writes unified JSON reports."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .schema import BenchmarkReport, BenchmarkResult, RunMetadata
from .system_info import (
    capture_hardware,
    get_sdk_version,
    git_branch,
    git_is_dirty,
    git_short_commit,
)


def _json_default(obj: object) -> object:
    """Handle numpy and other non-standard types in JSON serialization."""
    # numpy scalar types
    type_name = type(obj).__module__
    if type_name == "numpy":
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ResultRecorder:
    """Collects BenchmarkResult entries and writes a BenchmarkReport JSON file.

    Captures hardware and git metadata at construction time so all results
    in a single report share the same snapshot.
    """

    def __init__(self, category: str):
        self.category = category
        now = datetime.now(timezone.utc)
        commit = git_short_commit()

        self._report = BenchmarkReport(
            metadata=RunMetadata(
                timestamp=now.isoformat(),
                git_commit=commit,
                git_branch=git_branch(),
                git_dirty=git_is_dirty(),
                sdk="python",
                sdk_version=get_sdk_version(),
                hardware=capture_hardware(),
            ),
        )
        self._timestamp_slug = now.strftime("%Y-%m-%dT%H-%M-%SZ")
        self._commit_slug = commit or "unknown"

    def record(self, result: BenchmarkResult) -> None:
        self._report.results.append(result)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.category}-{self._timestamp_slug}-{self._commit_slug}.json"
        path = output_dir / filename

        # Atomic write: serialize to temp file, then rename.
        fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._report.to_dict(), f, indent=2, default=_json_default)
            os.rename(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        print(f"\nResults saved to {path}")
        return path

"""Unified benchmark result schema, compatible with strata-benchmarks (Rust) schema.rs."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class HardwareInfo:
    cpu: str = ""
    cores: int = 0
    ram_gb: float = 0.0
    os: str = ""
    arch: str = ""


@dataclass
class RunMetadata:
    timestamp: str = ""
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    sdk: str = "python"
    sdk_version: str = ""
    hardware: HardwareInfo = field(default_factory=HardwareInfo)


@dataclass
class BenchmarkResult:
    benchmark: str          # e.g. "ycsb/workload-a/100k-zipfian"
    category: str           # e.g. "ycsb"
    parameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    baselines: dict | None = None


@dataclass
class BenchmarkReport:
    schema_version: int = 1
    metadata: RunMetadata = field(default_factory=RunMetadata)
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Strip None baselines to keep JSON clean
        for r in d["results"]:
            if r.get("baselines") is None:
                del r["baselines"]
        # Strip None metadata fields
        meta = d["metadata"]
        for key in list(meta):
            if meta[key] is None:
                del meta[key]
        return d

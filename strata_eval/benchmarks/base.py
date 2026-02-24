"""Base class for all benchmark suites."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

from ..schema import BenchmarkResult


class BaseBenchmark(ABC):
    """Abstract base for all benchmark suites.

    Each benchmark registers its own CLI arguments, implements a run method
    that returns results, and provides a download method for datasets.
    """

    name: str = ""

    @abstractmethod
    def register_args(self, parser: argparse.ArgumentParser) -> None:
        """Add benchmark-specific CLI arguments to *parser*."""

    @abstractmethod
    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        """Execute the benchmark. Returns a list of result entries."""

    @abstractmethod
    def download(self, args: argparse.Namespace) -> None:
        """Download required datasets."""

    def validate(self, args: argparse.Namespace) -> bool:
        """Check prerequisites (datasets, deps). Override to add checks."""
        return True

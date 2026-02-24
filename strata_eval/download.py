"""Dataset download dispatcher."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from .benchmarks import get_benchmarks


def register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bench", required=True,
        help="Benchmark suite to download datasets for (e.g. ann, graphalytics, beir)",
    )
    parser.add_argument(
        "--dataset", nargs="*", default=None,
        help="Specific dataset(s) to download (default: all for the suite)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override dataset directory",
    )


def run_download(args: argparse.Namespace) -> None:
    benchmarks = get_benchmarks()
    name = args.bench
    if name not in benchmarks:
        available = ", ".join(sorted(benchmarks.keys()))
        print(f"Unknown benchmark: {name}")
        print(f"Available: {available}")
        return
    bench = benchmarks[name]()
    bench.download(args)


def download_file(url: str, dest: Path, desc: str | None = None) -> Path:
    """Download a file with progress bar. Skips if dest already exists.

    Downloads to a temporary file first, then atomically renames to *dest*
    so interrupted downloads do not leave corrupt files behind.
    """
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    import urllib.request

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    print(f"  Downloading {desc or dest.name}...")

    # Download to a temp file in the same directory so os.rename is atomic.
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".download")
    os.close(fd)

    try:
        if tqdm:
            class _Progress:
                def __init__(self):
                    self.bar = None
                def __call__(self, block_num, block_size, total_size):
                    if self.bar is None:
                        self.bar = tqdm(
                            total=max(total_size, 0),
                            unit="B",
                            unit_scale=True,
                        )
                    self.bar.update(block_size)
                def close(self):
                    if self.bar is not None:
                        self.bar.close()

            progress = _Progress()
            try:
                urllib.request.urlretrieve(url, tmp_path, reporthook=progress)
            finally:
                progress.close()
        else:
            urllib.request.urlretrieve(url, tmp_path)
            print(f"  Done: {dest}")

        os.rename(tmp_path, dest)
    except Exception:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return dest

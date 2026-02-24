"""HDF5 dataset loading utilities for ANN benchmarks."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Open an ANN benchmark HDF5 file and return its arrays.

    The standard ann-benchmarks HDF5 format contains four datasets:
      - "train"     : (N, D) float32 — index vectors
      - "test"      : (Q, D) float32 — query vectors
      - "neighbors" : (Q, K) int32   — ground-truth neighbor indices
      - "distances" : (Q, K) float32 — ground-truth distances

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict with keys "train", "test", "neighbors", "distances",
    each mapping to a numpy array.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        return {
            "train": np.array(f["train"]),
            "test": np.array(f["test"]),
            "neighbors": np.array(f["neighbors"]),
            "distances": np.array(f["distances"]),
        }

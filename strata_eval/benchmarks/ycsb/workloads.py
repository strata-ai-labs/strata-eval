"""YCSB workload definitions A-F with operation proportions and key distribution generators."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# WorkloadSpec — declarative workload definition
# ---------------------------------------------------------------------------

@dataclass
class WorkloadSpec:
    """Describes a YCSB workload by its operation mix and key distribution."""

    name: str
    description: str
    read_proportion: float = 0.0
    update_proportion: float = 0.0
    insert_proportion: float = 0.0
    scan_proportion: float = 0.0
    rmw_proportion: float = 0.0
    distribution: str = "zipfian"  # "zipfian", "uniform", or "latest"


# ---------------------------------------------------------------------------
# Standard YCSB workloads A-F
# ---------------------------------------------------------------------------

WORKLOADS: dict[str, WorkloadSpec] = {
    "a": WorkloadSpec(
        name="workload-a",
        description="Update heavy -- 50% read, 50% update, Zipfian",
        read_proportion=0.50,
        update_proportion=0.50,
        distribution="zipfian",
    ),
    "b": WorkloadSpec(
        name="workload-b",
        description="Read mostly -- 95% read, 5% update, Zipfian",
        read_proportion=0.95,
        update_proportion=0.05,
        distribution="zipfian",
    ),
    "c": WorkloadSpec(
        name="workload-c",
        description="Read only -- 100% read, Zipfian",
        read_proportion=1.0,
        distribution="zipfian",
    ),
    "d": WorkloadSpec(
        name="workload-d",
        description="Read latest -- 95% read, 5% insert, Latest distribution",
        read_proportion=0.95,
        insert_proportion=0.05,
        distribution="latest",
    ),
    "e": WorkloadSpec(
        name="workload-e",
        description="Short scans -- 95% scan, 5% insert, Zipfian",
        scan_proportion=0.95,
        insert_proportion=0.05,
        distribution="zipfian",
    ),
    "f": WorkloadSpec(
        name="workload-f",
        description="Read-modify-write -- 50% read, 50% RMW, Zipfian",
        read_proportion=0.50,
        rmw_proportion=0.50,
        distribution="zipfian",
    ),
}


# ---------------------------------------------------------------------------
# Key formatting (YCSB standard: "user" + 12-digit zero-padded integer)
# ---------------------------------------------------------------------------

def format_key(i: int) -> str:
    """Return the YCSB-standard key string for integer *i*."""
    return f"user{i:012d}"


# ---------------------------------------------------------------------------
# FNV-1a 64-bit hash (used by scrambled Zipfian, matching YCSB reference)
# ---------------------------------------------------------------------------

_FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325
_FNV_PRIME_64 = 0x100000001B3
_MASK_64 = (1 << 64) - 1


def _fnv1a_64(value: int) -> int:
    """Compute FNV-1a 64-bit hash of an integer, matching the YCSB Java
    reference implementation's scramble via ``Utils.FNVhash64``."""
    h = _FNV_OFFSET_BASIS_64
    # Hash the 8 bytes of the 64-bit integer, little-endian.
    for _ in range(8):
        octet = value & 0xFF
        h ^= octet
        h = (h * _FNV_PRIME_64) & _MASK_64
        value >>= 8
    return h


# ---------------------------------------------------------------------------
# Distribution generators
# ---------------------------------------------------------------------------

class ZipfianGenerator:
    """Scrambled Zipfian distribution over [0, n), matching the YCSB
    reference implementation.

    Parameters
    ----------
    n : int
        Number of items (must be >= 1).
    theta : float
        Zipfian constant (default 0.99, same as YCSB).
    """

    def __init__(self, n: int, theta: float = 0.99) -> None:
        if n < 1:
            raise ValueError(f"ZipfianGenerator requires n >= 1, got {n}")
        self._n = n
        self._theta = theta
        self._alpha = 1.0 / (1.0 - theta)
        self._zeta_n = self._compute_zeta(n, theta)
        self._zeta_2 = self._compute_zeta(min(2, n), theta)
        self._eta = (1.0 - (2.0 / n) ** (1.0 - theta)) / (
            1.0 - self._zeta_2 / self._zeta_n
        )

    # -- Zeta computation (harmonic number with exponent) ------------------

    @staticmethod
    def _compute_zeta(n: int, theta: float) -> float:
        """Compute the generalized harmonic number H_{n,theta}."""
        total = 0.0
        for i in range(1, n + 1):
            total += 1.0 / (i ** theta)
        return total

    @staticmethod
    def _compute_zeta_incremental(prev_zeta: float, prev_n: int, new_n: int, theta: float) -> float:
        """Incrementally update zeta from prev_n to new_n in O(new_n - prev_n)."""
        total = prev_zeta
        for i in range(prev_n + 1, new_n + 1):
            total += 1.0 / (i ** theta)
        return total

    # -- Core Zipfian sampling (unscrambled) --------------------------------

    def _next_zipfian(self) -> int:
        """Return a Zipfian-distributed integer in [0, n)."""
        u = random.random()
        uz = u * self._zeta_n
        if uz < 1.0:
            return 0
        if uz < 1.0 + 0.5 ** self._theta:
            return 1
        # Clamp to [0, n-1] to prevent floating-point edge case where
        # u very close to 1.0 could produce exactly n.
        return min(
            int(self._n * ((self._eta * u - self._eta + 1.0) ** self._alpha)),
            self._n - 1,
        )

    # -- Public interface ---------------------------------------------------

    def next(self) -> int:
        """Return a scrambled Zipfian-distributed integer in [0, n).

        Scrambling via FNV hash ensures the hot keys are spread across
        the key-space rather than clustered at the low end, matching the
        YCSB reference implementation.
        """
        raw = self._next_zipfian()
        return _fnv1a_64(raw) % self._n

    def update_n(self, new_n: int) -> None:
        """Incrementally update the item count (O(delta) not O(n))."""
        if new_n <= self._n:
            return
        old_n = self._n
        self._n = new_n
        self._zeta_n = self._compute_zeta_incremental(
            self._zeta_n, old_n, new_n, self._theta,
        )
        self._eta = (1.0 - (2.0 / new_n) ** (1.0 - self._theta)) / (
            1.0 - self._zeta_2 / self._zeta_n
        )


class UniformGenerator:
    """Uniform random distribution over [0, n)."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"UniformGenerator requires n >= 1, got {n}")
        self._n = n

    def next(self) -> int:
        return random.randint(0, self._n - 1)


class LatestGenerator:
    """Exponentially weighted distribution favoring the most recent keys.

    Key *n-1* (the latest) is most likely; probability decays
    exponentially toward key 0.  Implemented via a Zipfian over reverse
    offsets from the latest key.
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._zipfian = ZipfianGenerator(n)

    def next(self) -> int:
        # Zipfian gives an offset from the latest key.
        offset = self._zipfian._next_zipfian()
        return max(0, self._n - 1 - offset)

    def set_n(self, n: int) -> None:
        """Update the item count — uses incremental zeta update (O(delta))."""
        if n <= self._n:
            return
        self._n = n
        self._zipfian.update_n(n)


# ---------------------------------------------------------------------------
# Value generation
# ---------------------------------------------------------------------------

_ALPHANUM = string.ascii_letters + string.digits


def generate_value(field_count: int, field_length: int) -> dict[str, str]:
    """Generate a dict of *field_count* random alphanumeric fields, each
    *field_length* bytes long.  Field names follow YCSB convention:
    ``field0``, ``field1``, ...
    """
    return {
        f"field{i}": "".join(random.choices(_ALPHANUM, k=field_length))
        for i in range(field_count)
    }

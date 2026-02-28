"""Hardware, git, and platform metadata capture."""

from __future__ import annotations

import os
import platform
import subprocess

from .schema import HardwareInfo


def capture_hardware() -> HardwareInfo:
    """Detect CPU model, core count, RAM, OS, and architecture."""
    cpu = _read_cpu_model()
    cores = os.cpu_count() or 0
    ram_gb = _read_ram_gb()
    return HardwareInfo(
        cpu=cpu,
        cores=cores,
        ram_gb=round(ram_gb, 1),
        os=platform.system().lower(),
        arch=platform.machine(),
    )


def git_short_commit() -> str | None:
    return _git("rev-parse", "--short", "HEAD")


def git_branch() -> str | None:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def git_is_dirty() -> bool | None:
    out = _git("status", "--porcelain")
    if out is None:
        return None
    return len(out.strip()) > 0


def get_sdk_version() -> str:
    """Get the Strata version from the CLI binary via ping."""
    import tempfile
    try:
        from lib.strata_client import StrataClient
        with tempfile.TemporaryDirectory() as d:
            with StrataClient(db_path=d, cache=True) as client:
                return client.ping()
    except Exception:
        return "not installed"


# -------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _read_cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return platform.processor() or "unknown"


def _read_ram_gb() -> float:
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return 0.0

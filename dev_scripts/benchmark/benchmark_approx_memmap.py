# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Benchmark profile script.
#
# Author: Maximilian Salomon
# Created with assistance from GPT-5.3-Codex.
#
# Copyright (C) 2025 Maximilian Salomon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Run the Approx Memmap benchmark profile and persist benchmark metrics.

File Description
----------------
This script executes the full MDXplain benchmark workflow for the
"Approx Memmap" profile and writes per-step metrics, run summaries,
plot artifacts, and a profile-level summary JSON.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/benchmark_approx_memmap.py``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mdxplain import PipelineManager
import psutil


results_dir = Path("benchmark_results_approx_memmap")
cache_root = Path("cache/benchmark_approx_memmap")
dataset_factors = [1, 2, 3, 5, 10, 30, 50, 500]
supported_dataset_factors = [1, 2, 3, 5, 10, 30, 50, 500, 1000]

data_root = Path("data/benchmarks")
base_dataset = Path("data/2RJY")

CONTACT_SELECTOR = "contacts_only"
DECOMPOSITION_NAME = "ContactKernelPCA"
CLUSTER_NAME = "DPA_ContactKernelPCA"
COMPARISON_NAME = "cluster_comparison"
FEATURE_IMPORTANCE_NAME = "feature_importance"

DECISION_TREE_KWARGS = {
    "feature_importance_name": FEATURE_IMPORTANCE_NAME,
    "save_fig": True,
    "render": False,
    "short_layout": True,
    "hide_path": False,
    "separate_trees": True,
    "width_scale_factor": 0.4,
    "height_scale_factor": 0.5,
    "edge_symbol_fontsize": 18,
    "file_format": "svg",
}
VIOLIN_KWARGS = {
    "feature_importance_name": FEATURE_IMPORTANCE_NAME,
    "n_top": 2,
    "save_fig": True,
    "max_cols": 7,
    "file_format": "svg",
    "tick_fontsize": 26,
    "ylabel_fontsize": 26,
    "subplot_title_fontsize": 26,
}
DENSITY_KWARGS = {
    "feature_importance_name": FEATURE_IMPORTANCE_NAME,
    "n_top": 2,
    "save_fig": True,
    "max_cols": 7,
    "file_format": "svg",
    "tick_fontsize": 18,
    "ylabel_fontsize": 18,
    "subplot_title_fontsize": 18,
}
TIME_SERIES_KWARGS = {
    "feature_importance_name": FEATURE_IMPORTANCE_NAME,
    "n_top": 2,
    "save_fig": True,
    "max_cols": 3,
    "membership_per_feature": True,
    "clustering_name": CLUSTER_NAME,
    "file_format": "svg",
    "tick_fontsize": 26,
    "ylabel_fontsize": 26,
    "xlabel_fontsize": 26,
    "subplot_title_fontsize": 26,
}
MEMBERSHIP_KWARGS = {
    "clustering_name": CLUSTER_NAME,
    "save_fig": True,
    "file_format": "svg",
    "tick_fontsize": 18,
    "xlabel_fontsize": 18,
    "ylabel_fontsize": 18,
}
LANDSCAPE_KWARGS = {
    "decomposition_name": DECOMPOSITION_NAME,
    "dimensions": [0, 1, 2, 3],
    "clustering_name": CLUSTER_NAME,
    "save_fig": True,
    "file_format": "svg",
    "tick_fontsize": 26,
    "xlabel_fontsize": 26,
    "ylabel_fontsize": 26,
    "contour_label_fontsize": 26,
}

StepCallable = Callable[[], None]


@dataclass(frozen=True)
class _BenchmarkProfile:
    """Immutable benchmark profile configuration.

    Parameters
    ----------
    name : str
        Human-readable benchmark profile name.
    results_dir : Path
        Output directory for per-dataset benchmark results.
    cache_root : Path
        Cache root directory used while executing pipeline steps.
    dataset_factors : list[int]
        Dataset scaling factors to execute in this profile.
    use_memmap : bool
        Pipeline memmap mode flag.
    chunk_size : int or None
        Optional pipeline chunk size.
    use_nystrom : bool
        Enables Nyström approximation for contact-kernel PCA.
    n_landmarks : int or None
        Landmark count used when Nyström is enabled.
    dpa_method : str
        DPA method value forwarded to clustering configuration.

    Returns
    -------
    None
        Dataclass instances are consumed by the profile runner.

    Notes
    -----
    Paths are interpreted relative to project root when this script is executed
    from repository root.
    """

    name: str
    results_dir: Path
    cache_root: Path
    dataset_factors: list[int]
    use_memmap: bool
    chunk_size: Optional[int]
    use_nystrom: bool
    n_landmarks: Optional[int]
    dpa_method: str


@dataclass
class _StepResult:
    """Per-step benchmark metrics.

    Parameters
    ----------
    name : str
        Executed step name.
    seconds : float
        Step execution duration in seconds.
    rss_start_mb : float
        RSS memory before step execution in MB.
    rss_end_mb : float
        RSS memory after step execution in MB.
    rss_peak_mb : float
        Peak RSS observed during step execution in MB.
    non_cache_start_mb : float
        Non-cache memory before step execution in MB.
    non_cache_end_mb : float
        Non-cache memory after step execution in MB.
    non_cache_peak_mb : float
        Peak non-cache memory observed during step in MB.
    private_start_mb : float or None
        Private memory before step execution in MB (Windows).
    private_end_mb : float or None
        Private memory after step execution in MB (Windows).
    private_peak_mb : float or None
        Peak private memory observed during step in MB (Windows).
    min_necessary_ram_start_mb : float
        Canonical minimum necessary RAM before step execution in MB.
    min_necessary_ram_end_mb : float
        Canonical minimum necessary RAM after step execution in MB.
    min_necessary_ram_peak_mb : float
        Canonical minimum necessary RAM peak observed during step in MB.
    mem_available_start_mb : float or None
        MemAvailable before step execution in MB.
    mem_available_end_mb : float or None
        MemAvailable after step execution in MB.
    mem_available_min_mb : float or None
        Minimum MemAvailable observed while step ran in MB.
    cgroup_current_start_mb : float or None
        cgroup memory.current value before step execution in MB.
    cgroup_current_end_mb : float or None
        cgroup memory.current value after step execution in MB.
    cgroup_current_peak_mb : float or None
        Peak cgroup current value observed while step ran in MB.
    cgroup_limit_mb : float or None
        cgroup memory limit in MB when available.
    cache_size_mb : float
        Cache directory size after step execution in MB.

    Returns
    -------
    None
        Dataclass instances are serialized into JSON output files.

    Notes
    -----
    Unit conversion to MB is done before storing values here.
    """

    name: str
    seconds: float
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    non_cache_start_mb: float
    non_cache_end_mb: float
    non_cache_peak_mb: float
    private_start_mb: Optional[float]
    private_end_mb: Optional[float]
    private_peak_mb: Optional[float]
    min_necessary_ram_start_mb: float
    min_necessary_ram_end_mb: float
    min_necessary_ram_peak_mb: float
    mem_available_start_mb: Optional[float]
    mem_available_end_mb: Optional[float]
    mem_available_min_mb: Optional[float]
    cgroup_current_start_mb: Optional[float]
    cgroup_current_end_mb: Optional[float]
    cgroup_current_peak_mb: Optional[float]
    cgroup_limit_mb: Optional[float]
    cache_size_mb: float


_CGROUP_FILES: tuple[Optional[Path], Optional[Path]] | None = None


def _bytes_to_mb(value: int) -> float:
    """Convert raw bytes to MB.

    Parameters
    ----------
    value : int
        Byte count.

    Returns
    -------
    float
        Megabyte value using 1024-based conversion.

    Notes
    -----
    This helper keeps MB conversion consistent across all metrics.
    """
    return float(value) / (1024.0 * 1024.0)


def _bytes_to_mb_optional(value: Optional[int]) -> Optional[float]:
    """Convert optional byte value to MB.

    Parameters
    ----------
    value : int or None
        Optional byte count.

    Returns
    -------
    float or None
        Converted MB value, or ``None`` when input is missing.

    Notes
    -----
    ``None`` values are preserved to avoid fake zero-values.
    """
    return None if value is None else _bytes_to_mb(value)


def _fmt_optional_mb(value: Optional[float]) -> str:
    """Format optional MB values for log output.

    Parameters
    ----------
    value : float or None
        Optional MB value.

    Returns
    -------
    str
        Formatted text representation for console output.

    Notes
    -----
    Missing values are rendered as ``n/a``.
    """
    return "n/a" if value is None else f"{value:.2f}"


def _dir_size_bytes(path: Path) -> int:
    """Calculate recursive directory size in bytes.

    Parameters
    ----------
    path : Path
        Directory to inspect.

    Returns
    -------
    int
        Total file size in bytes.

    Notes
    -----
    I/O errors are ignored to keep benchmark runs resilient.
    """
    # Skip missing directories quickly to avoid unnecessary traversals.
    if not path.exists():
        return 0

    # Aggregate file sizes recursively while tolerating transient file errors.
    total = 0
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            total += file_path.stat().st_size
        except OSError:
            continue
    return total


def _read_smaps_rollup_kb(pid: int) -> dict[str, int]:
    """Read Linux ``smaps_rollup`` values in kB for one process.

    Parameters
    ----------
    pid : int
        Process identifier.

    Returns
    -------
    dict[str, int]
        Parsed rollup fields in kilobytes.

    Notes
    -----
    Returns an empty mapping when ``/proc`` data is unavailable.
    """
    # Resolve process rollup path and fail fast when file is missing.
    path = Path(f"/proc/{pid}/smaps_rollup")
    if not path.exists():
        return {}

    # Parse ``Key: value kB`` lines into integer dictionary values.
    stats: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", maxsplit=1)
            stats[key] = int(rest.strip().split()[0])
    except (OSError, ValueError):
        return {}
    return stats


def _benchmark_os_key() -> str:
    """Return canonical benchmark OS key.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Canonical OS key: ``windows``, ``linux``, ``macos``, or ``other``.
    """
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("darwin"):
        return "macos"
    return "other"


def _get_process_tree() -> list[psutil.Process]:
    """Resolve current process and recursive children.

    Parameters
    ----------
    None

    Returns
    -------
    list[psutil.Process]
        Process tree including the current process as first item.

    Notes
    -----
    Returns an empty list on transient psutil failures.
    """
    try:
        process = psutil.Process(os.getpid())
        return [process] + process.children(recursive=True)
    except (psutil.Error, OSError):
        return []


def _get_rss_bytes() -> int:
    """Measure current RSS across process tree.

    Parameters
    ----------
    None

    Returns
    -------
    int
        RSS total in bytes.

    Notes
    -----
    Child processes are included recursively to reflect full benchmark usage.
    """
    # Resolve process tree once and gracefully handle psutil failures.
    processes = _get_process_tree()
    if not processes:
        return 0

    # Sum RSS for all alive processes while ignoring transient process exits.
    total_rss = 0
    for proc in processes:
        try:
            total_rss += int(proc.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total_rss


def _get_private_bytes() -> Optional[int]:
    """Measure total private bytes across process tree on Windows.

    Parameters
    ----------
    None

    Returns
    -------
    int or None
        Private bytes on Windows, otherwise ``None``.

    Notes
    -----
    ``None`` indicates the metric is not available on the active platform.
    """
    if not sys.platform.startswith("win"):
        return None

    processes = _get_process_tree()
    if not processes:
        return 0

    total_private = 0
    for proc in processes:
        try:
            total_private += int(getattr(proc.memory_info(), "private", 0))
        except (psutil.Error, OSError):
            continue
    return total_private


def _get_non_cache_bytes() -> int:
    """Estimate non-cache memory pressure in bytes.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Approximate non-cache memory in bytes.

    Notes
    -----
    On Linux this subtracts clean file-backed pages from RSS via
    ``smaps_rollup``. On Windows this reads ``private`` memory bytes
    which exclude file cache. Other platforms fall back to RSS.
    """
    if sys.platform.startswith("win"):
        private_bytes = _get_private_bytes()
        return int(private_bytes) if private_bytes is not None else _get_rss_bytes()

    # Use direct RSS fallback for non-Linux/Windows environments.
    if not sys.platform.startswith("linux"):
        return _get_rss_bytes()

    # Collect process list and fall back to RSS on psutil errors.
    processes = _get_process_tree()
    if not processes:
        return _get_rss_bytes()

    # Aggregate rollup-based non-cache usage across the process tree.
    total_non_cache = 0
    found_any = False
    for proc in processes:
        stats = _read_smaps_rollup_kb(proc.pid)
        if not stats:
            continue
        found_any = True
        rss_kb = int(stats.get("Rss", 0))
        clean_file_kb = int(stats.get("Shared_Clean", 0)) + int(stats.get("Private_Clean", 0))
        total_non_cache += max(0, rss_kb - clean_file_kb) * 1024

    # Use RSS fallback when no rollup information was available.
    return total_non_cache if found_any else _get_rss_bytes()


def _compute_min_necessary_ram_bytes(
    rss_bytes: int,
    non_cache_bytes: int,
    private_bytes: Optional[int],
) -> int:
    """Compute canonical minimum necessary RAM pressure in bytes.

    Parameters
    ----------
    rss_bytes : int
        RSS memory bytes.
    non_cache_bytes : int
        Non-cache memory bytes.
    private_bytes : int or None
        Windows private memory bytes when available.

    Returns
    -------
    int
        Canonical minimum necessary RAM pressure bytes.

    Notes
    -----
    Linux uses non-cache RAM. Windows uses ``max(RSS, private)``.
    Other platforms fall back to non-cache, then RSS.
    """
    os_key = _benchmark_os_key()
    if os_key == "windows":
        if private_bytes is None:
            return max(int(rss_bytes), int(non_cache_bytes))
        return max(int(rss_bytes), int(private_bytes))
    if os_key == "linux":
        return int(non_cache_bytes)
    return int(non_cache_bytes) if int(non_cache_bytes) > 0 else int(rss_bytes)


def _read_int_from_file(path: Optional[Path]) -> Optional[int]:
    """Read integer value from filesystem path.

    Parameters
    ----------
    path : Path or None
        Input file path.

    Returns
    -------
    int or None
        Parsed integer value, or ``None`` when unavailable.

    Notes
    -----
    The cgroup literal ``max`` is interpreted as unlimited and mapped to
    ``None``.
    """
    # Validate path before attempting to read.
    if path is None or not path.exists():
        return None

    # Parse integer text and filter unsupported/unlimited sentinels.
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_cgroup_relative_path() -> str:
    """Resolve current process cgroup relative path.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Relative cgroup path without leading slash.

    Notes
    -----
    Returns an empty string for root-level cgroup placement.
    """
    # Default to root-level path when proc cgroup file is missing.
    cgroup_path = Path("/proc/self/cgroup")
    if not cgroup_path.exists():
        return ""

    # Parse proc cgroup lines and prefer unified hierarchy entry.
    for line in cgroup_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue
        if parts[0] == "0":
            return parts[2].lstrip("/")

    # Fall back to memory-controller entry for cgroup v1 systems.
    for line in cgroup_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue
        controllers = parts[1].split(",")
        if "memory" in controllers:
            return parts[2].lstrip("/")
    return ""


def _resolve_cgroup_v2_files() -> tuple[Optional[Path], Optional[Path]]:
    """Resolve cgroup v2 memory paths.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[Path or None, Path or None]
        Tuple of ``(memory.current, memory.max)`` paths.

    Notes
    -----
    Missing files are returned as ``None`` entries.
    """
    # Detect cgroup v2 root and abort quickly when not available.
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").exists():
        return None, None

    # Build cgroup-specific paths for current process membership.
    rel_path = _read_cgroup_relative_path()
    base = root / rel_path if rel_path else root
    current = base / "memory.current"
    limit = base / "memory.max"
    return (current if current.exists() else None, limit if limit.exists() else None)


def _resolve_cgroup_v1_files() -> tuple[Optional[Path], Optional[Path]]:
    """Resolve cgroup v1 memory paths.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[Path or None, Path or None]
        Tuple of ``(memory.usage_in_bytes, memory.limit_in_bytes)`` paths.

    Notes
    -----
    Several common cgroup v1 mount layouts are tested.
    """
    # Build candidate mount roots and cgroup-specific relative directory.
    rel_path = _read_cgroup_relative_path()
    candidates = [Path("/sys/fs/cgroup/memory"), Path("/sys/fs/cgroup")]

    # Return first valid pair that exposes memory usage metrics.
    for root in candidates:
        base = root / rel_path if rel_path else root
        current = base / "memory.usage_in_bytes"
        limit = base / "memory.limit_in_bytes"
        if current.exists():
            return current, (limit if limit.exists() else None)
    return None, None


def _resolve_cgroup_memory_files() -> tuple[Optional[Path], Optional[Path]]:
    """Resolve and cache cgroup memory files for current process.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[Path or None, Path or None]
        Cached tuple of current and limit file paths.

    Notes
    -----
    v2 paths are preferred over v1 when both layouts are discoverable.
    """
    # Return cached value immediately when already resolved.
    global _CGROUP_FILES
    if _CGROUP_FILES is not None:
        return _CGROUP_FILES

    # Resolve best available cgroup layout and cache outcome.
    current, limit = _resolve_cgroup_v2_files()
    if current is None and limit is None:
        current, limit = _resolve_cgroup_v1_files()
    _CGROUP_FILES = current, limit
    return _CGROUP_FILES


def _get_cgroup_memory_bytes() -> tuple[Optional[int], Optional[int]]:
    """Read current and limit cgroup memory values.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[int or None, int or None]
        Tuple of ``(current_bytes, limit_bytes)``.

    Notes
    -----
    Missing or non-numeric values are returned as ``None``.
    """
    # Resolve memory files once and parse integer values from disk.
    current_path, limit_path = _resolve_cgroup_memory_files()
    return _read_int_from_file(current_path), _read_int_from_file(limit_path)


def _read_mem_available_from_proc() -> Optional[int]:
    """Read ``MemAvailable`` from ``/proc/meminfo``.

    Parameters
    ----------
    None

    Returns
    -------
    int or None
        Available host memory in bytes.

    Notes
    -----
    Returns ``None`` on non-Linux systems or parsing errors.
    """
    # Restrict proc parsing to Linux environments.
    if not sys.platform.startswith("linux"):
        return None

    # Parse MemAvailable in kB and convert to bytes.
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1]) * 1024
    return None


def _get_mem_available_bytes() -> Optional[int]:
    """Estimate currently available memory in bytes.

    Parameters
    ----------
    None

    Returns
    -------
    int or None
        Conservative available memory estimate.

    Notes
    -----
    On Linux the minimum of host MemAvailable and cgroup availability is used
    when both values exist.
    """
    # Resolve host-level and cgroup-level availability candidates.
    host_available = _read_mem_available_from_proc()
    cgroup_current, cgroup_limit = _get_cgroup_memory_bytes()

    # Compute cgroup-available bytes only for finite limits.
    cgroup_available = None
    if cgroup_current is not None and cgroup_limit is not None and cgroup_limit > 0:
        if cgroup_limit < (1 << 60):
            cgroup_available = max(0, cgroup_limit - cgroup_current)

    # Combine both availability signals conservatively.
    if host_available is None:
        return cgroup_available
    if cgroup_available is None:
        return host_available
    return min(host_available, cgroup_available)


class MemorySampler:
    """Sample memory metrics in a background thread during one pipeline step.

    Parameters
    ----------
    interval_sec : float, default=0.2
        Sampling interval in seconds.

    Returns
    -------
    None
        Metrics are exposed via instance attributes.

    Notes
    -----
    The sampler captures RSS, non-cache memory, private bytes, minimum necessary
    RAM, MemAvailable, and cgroup usage.
    """

    def __init__(self, interval_sec: float = 0.2):
        """Initialize sampler state.

        Parameters
        ----------
        interval_sec : float, default=0.2
            Sampling interval in seconds.

        Returns
        -------
        None
            Initializes counters and thread primitives.

        Notes
        -----
        Metrics are initialized from current process state.
        """
        # Store static runtime settings and initialize metrics.
        self.interval_sec = interval_sec
        self.max_rss = _get_rss_bytes()
        self.max_non_cache = _get_non_cache_bytes()
        self.max_private = _get_private_bytes()
        self.max_min_necessary_ram = _compute_min_necessary_ram_bytes(
            self.max_rss,
            self.max_non_cache,
            self.max_private,
        )
        self.min_mem_available = _get_mem_available_bytes()

        # Initialize cgroup tracking values once at sampler creation time.
        current, limit = _get_cgroup_memory_bytes()
        self.max_cgroup_current = current or 0
        self.cgroup_limit = limit

        # Prepare background thread state for cooperative shutdown.
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample_once(self) -> None:
        """Sample one memory snapshot and update extrema.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Updates sampler attributes in place.

        Notes
        -----
        This method is called repeatedly by the background thread.
        """
        # Update memory peaks from current process snapshot.
        current_rss = _get_rss_bytes()
        current_non_cache = _get_non_cache_bytes()
        current_private = _get_private_bytes()
        current_min_necessary_ram = _compute_min_necessary_ram_bytes(
            current_rss,
            current_non_cache,
            current_private,
        )

        self.max_rss = max(self.max_rss, current_rss)
        self.max_non_cache = max(self.max_non_cache, current_non_cache)
        if current_private is not None:
            self.max_private = (
                current_private
                if self.max_private is None
                else max(self.max_private, current_private)
            )
        self.max_min_necessary_ram = max(self.max_min_necessary_ram, current_min_necessary_ram)

        # Update minimum available memory when this metric is available.
        available = _get_mem_available_bytes()
        if available is not None:
            if self.min_mem_available is None:
                self.min_mem_available = available
            else:
                self.min_mem_available = min(self.min_mem_available, available)

        # Update cgroup peaks and persist discovered finite memory limit.
        cgroup_current, cgroup_limit = _get_cgroup_memory_bytes()
        if cgroup_current is not None:
            self.max_cgroup_current = max(self.max_cgroup_current, cgroup_current)
        if cgroup_limit is not None:
            self.cgroup_limit = cgroup_limit

    def _run(self) -> None:
        """Run periodic sampling loop until stopped.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Loops until stop event is set.

        Notes
        -----
        Sampling exceptions are suppressed to avoid interrupting benchmark runs.
        """
        # Poll with timeout-based waiting so stop requests are responsive.
        while not self._stop_event.wait(self.interval_sec):
            try:
                self._sample_once()
            except Exception:  # pragma: no cover - defensive sampling guard
                continue

    def __enter__(self) -> "MemorySampler":
        """Start background sampling and return sampler instance.

        Parameters
        ----------
        None

        Returns
        -------
        MemorySampler
            Running sampler instance.

        Notes
        -----
        Context-manager usage guarantees thread cleanup in ``__exit__``.
        """
        # Prime metrics with one immediate sample before thread starts.
        self._sample_once()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Stop background sampling thread.

        Parameters
        ----------
        exc_type : type or None
            Exception class raised in context, if any.
        exc : BaseException or None
            Exception instance raised in context, if any.
        tb : traceback or None
            Traceback object raised in context, if any.

        Returns
        -------
        None
            Thread is signaled to stop and joined.

        Notes
        -----
        Exceptions from context body are never suppressed.
        """
        # Signal shutdown and wait briefly for background thread completion.
        self._stop_event.set()
        self._thread.join(timeout=2.0)


def _close_fig(obj) -> None:
    """Close matplotlib figure objects returned by plotting calls.

    Parameters
    ----------
    obj : object
        Object potentially containing a matplotlib figure.

    Returns
    -------
    None
        Figure is closed when possible.

    Notes
    -----
    Some plotting APIs return figure handles while others return ``None``.
    """
    # Close only direct Figure instances and ignore all other objects.
    if isinstance(obj, Figure):
        plt.close(obj)


def _build_pipeline(profile: _BenchmarkProfile, cache_dir: Path) -> PipelineManager:
    """Build one pipeline instance for benchmark execution.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.
    cache_dir : Path
        Cache directory used by pipeline internals.

    Returns
    -------
    PipelineManager
        Configured pipeline instance.

    Notes
    -----
    ``chunk_size`` is only passed when configured by profile.
    """
    # Build base pipeline kwargs from profile-independent defaults.
    kwargs: dict[str, object] = {
        "use_memmap": profile.use_memmap,
        "show_progress": False,
        "cache_dir": str(cache_dir),
    }

    # Add optional chunk size only when profile requests it explicitly.
    if profile.chunk_size is not None:
        kwargs["chunk_size"] = profile.chunk_size
    return PipelineManager(**kwargs)


def _contact_kernel_kwargs(profile: _BenchmarkProfile) -> dict[str, object]:
    """Build contact-kernel PCA kwargs for selected benchmark profile.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.

    Returns
    -------
    dict[str, object]
        Keyword arguments for ``contact_kernel_pca``.

    Notes
    -----
    Nyström options are injected only for profiles that enable them.
    """
    # Build baseline decomposition kwargs shared by all profiles.
    kwargs: dict[str, object] = {
        "n_components": 4,
        "selection_name": CONTACT_SELECTOR,
        "decomposition_name": DECOMPOSITION_NAME,
    }

    # Add Nyström approximation settings when configured.
    if profile.use_nystrom and profile.n_landmarks is not None:
        kwargs["use_nystrom"] = True
        kwargs["n_landmarks"] = profile.n_landmarks
    return kwargs


def _build_ingest_steps(pipeline: PipelineManager, dataset_dir: Path) -> list[tuple[str, StepCallable]]:
    """Build trajectory ingestion and feature preparation step list.

    Parameters
    ----------
    pipeline : PipelineManager
        Pipeline object executing benchmark operations.
    dataset_dir : Path
        Dataset directory for the current benchmark run.

    Returns
    -------
    list[tuple[str, StepCallable]]
        Ordered ingestion steps.

    Notes
    -----
    These steps are common across all benchmark profiles.
    """
    # Define initial data loading and feature-selection preparation steps.
    return [
        ("load_trajectories", lambda: pipeline.trajectory.load_trajectories(str(dataset_dir))),
        ("add_labels", lambda: pipeline.trajectory.add_labels(traj_selection="all")),
        ("add_distances", lambda: pipeline.feature.add.distances()),
        ("add_contacts", lambda: pipeline.feature.add.contacts(cutoff=4.5)),
        ("create_selector", lambda: pipeline.feature_selector.create(CONTACT_SELECTOR)),
        ("select_contacts", lambda: pipeline.feature_selector.add.contacts(CONTACT_SELECTOR, "all")),
        ("apply_selector", lambda: pipeline.feature_selector.select(CONTACT_SELECTOR)),
    ]


def _build_model_steps(pipeline: PipelineManager, profile: _BenchmarkProfile) -> list[tuple[str, StepCallable]]:
    """Build decomposition, clustering, and importance-model steps.

    Parameters
    ----------
    pipeline : PipelineManager
        Pipeline object executing benchmark operations.
    profile : _BenchmarkProfile
        Benchmark profile configuration.

    Returns
    -------
    list[tuple[str, StepCallable]]
        Ordered model-building steps.

    Notes
    -----
    Profile-specific options are injected for decomposition and clustering.
    """
    # Resolve profile-dependent kwargs for decomposition and clustering steps.
    ckpca_kwargs = _contact_kernel_kwargs(profile)
    dpa_kwargs = {
        "Z": 2.5,
        "metric": "euclidean",
        "affinity": "nearest_neighbors",
        "cluster_name": CLUSTER_NAME,
        "method": profile.dpa_method,
    }

    # Define modeling steps from decomposition through feature importance.
    return [
        ("contact_kernel_pca", lambda: pipeline.decomposition.add.contact_kernel_pca(**ckpca_kwargs)),
        ("dpa", lambda: pipeline.clustering.add.dpa(DECOMPOSITION_NAME, **dpa_kwargs)),
        (
            "data_selector",
            lambda: pipeline.data_selector.create_from_clusters(group_name="cluster", clustering_name=CLUSTER_NAME),
        ),
        (
            "comparison",
            lambda: pipeline.comparison.create_comparison(
                name=COMPARISON_NAME,
                mode="one_vs_rest",
                feature_selector=CONTACT_SELECTOR,
                data_selector_groups="cluster",
            ),
        ),
        (
            "feature_importance",
            lambda: pipeline.feature_importance.add.decision_tree(
                comparison_name=COMPARISON_NAME,
                analysis_name=FEATURE_IMPORTANCE_NAME,
                max_samples=100000,
            ),
        ),
    ]


def _build_plot_steps(pipeline: PipelineManager) -> list[tuple[str, StepCallable]]:
    """Build visualization step list for benchmark workflow.

    Parameters
    ----------
    pipeline : PipelineManager
        Pipeline object executing benchmark operations.

    Returns
    -------
    list[tuple[str, StepCallable]]
        Ordered visualization steps.

    Notes
    -----
    Figure-returning APIs are wrapped by ``_close_fig`` to release memory.
    """
    # Define plotting and textual reporting steps using shared kwargs maps.
    return [
        ("plot_decision_trees", lambda: pipeline.plots.feature_importance.decision_trees(**DECISION_TREE_KWARGS)),
        (
            "print_top_features",
            lambda: pipeline.feature_importance.print_top_n_features(analysis_name=FEATURE_IMPORTANCE_NAME),
        ),
        ("plot_violins", lambda: _close_fig(pipeline.plots.feature_importance.violins(**VIOLIN_KWARGS))),
        ("plot_densities", lambda: _close_fig(pipeline.plots.feature_importance.densities(**DENSITY_KWARGS))),
        (
            "plot_time_series",
            lambda: _close_fig(pipeline.plots.feature_importance.time_series(**TIME_SERIES_KWARGS)),
        ),
        ("plot_membership", lambda: _close_fig(pipeline.plots.clustering.membership(**MEMBERSHIP_KWARGS))),
        ("plot_landscape", lambda: _close_fig(pipeline.plots.landscape(**LANDSCAPE_KWARGS))),
    ]


def _build_output_steps(pipeline: PipelineManager, output_root: Path) -> list[tuple[str, StepCallable]]:
    """Build artifact-export step list for benchmark workflow.

    Parameters
    ----------
    pipeline : PipelineManager
        Pipeline object executing benchmark operations.
    output_root : Path
        Dataset-specific benchmark output directory.

    Returns
    -------
    list[tuple[str, StepCallable]]
        Ordered artifact generation steps.

    Notes
    -----
    Archive output path is deterministic per dataset run.
    """
    # Define structure export and archive creation steps.
    return [
        (
            "beta_factor_pdb",
            lambda: pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
                structure_viz_name="structure_viz",
                feature_importance_name=FEATURE_IMPORTANCE_NAME,
            ),
        ),
        ("create_archive", lambda: pipeline.create_sharable_archive(str(output_root / "pipeline.tar.zst"))),
    ]


def _build_steps(
    pipeline: PipelineManager,
    dataset_dir: Path,
    output_root: Path,
    profile: _BenchmarkProfile,
) -> list[tuple[str, StepCallable]]:
    """Build full ordered step list for one dataset run.

    Parameters
    ----------
    pipeline : PipelineManager
        Pipeline object executing benchmark operations.
    dataset_dir : Path
        Dataset directory for current run.
    output_root : Path
        Dataset-specific benchmark output directory.
    profile : _BenchmarkProfile
        Benchmark profile configuration.

    Returns
    -------
    list[tuple[str, StepCallable]]
        Ordered step list spanning ingest to archive output.

    Notes
    -----
    Step ordering is intentionally stable across benchmark runs.
    """
    # Concatenate all step groups in deterministic execution order.
    return [
        *_build_ingest_steps(pipeline, dataset_dir),
        *_build_model_steps(pipeline, profile),
        *_build_plot_steps(pipeline),
        *_build_output_steps(pipeline, output_root),
    ]


def _run_step(name: str, func: StepCallable, cache_dir: Path) -> _StepResult:
    """Execute one step and capture detailed resource metrics.

    Parameters
    ----------
    name : str
        Step name.
    func : StepCallable
        Callable executing step logic.
    cache_dir : Path
        Cache directory used for post-step cache-size measurement.

    Returns
    -------
    _StepResult
        Captured execution and memory metrics.

    Notes
    -----
    Memory peaks are sampled asynchronously while the step runs.
    """
    # Capture baseline metrics before running benchmark step logic.
    rss_start = _get_rss_bytes()
    non_cache_start = _get_non_cache_bytes()
    private_start = _get_private_bytes()
    min_necessary_start = _compute_min_necessary_ram_bytes(
        rss_start,
        non_cache_start,
        private_start,
    )
    mem_available_start = _get_mem_available_bytes()
    cgroup_current_start, cgroup_limit_start = _get_cgroup_memory_bytes()

    # Execute step under active background memory sampling.
    start_time = time.perf_counter()
    with MemorySampler() as sampler:
        func()
    elapsed = time.perf_counter() - start_time

    # Capture post-step metrics and map everything into dataclass output.
    rss_end = _get_rss_bytes()
    non_cache_end = _get_non_cache_bytes()
    private_end = _get_private_bytes()
    min_necessary_end = _compute_min_necessary_ram_bytes(
        rss_end,
        non_cache_end,
        private_end,
    )
    mem_available_end = _get_mem_available_bytes()
    cgroup_current_end, cgroup_limit_end = _get_cgroup_memory_bytes()
    cgroup_limit = cgroup_limit_start or cgroup_limit_end or sampler.cgroup_limit
    cgroup_peak = sampler.max_cgroup_current if sampler.max_cgroup_current > 0 else None
    return _StepResult(
        name=name,
        seconds=elapsed,
        rss_start_mb=_bytes_to_mb(rss_start),
        rss_end_mb=_bytes_to_mb(rss_end),
        rss_peak_mb=_bytes_to_mb(sampler.max_rss),
        non_cache_start_mb=_bytes_to_mb(non_cache_start),
        non_cache_end_mb=_bytes_to_mb(non_cache_end),
        non_cache_peak_mb=_bytes_to_mb(sampler.max_non_cache),
        private_start_mb=_bytes_to_mb_optional(private_start),
        private_end_mb=_bytes_to_mb_optional(private_end),
        private_peak_mb=_bytes_to_mb_optional(sampler.max_private),
        min_necessary_ram_start_mb=_bytes_to_mb(min_necessary_start),
        min_necessary_ram_end_mb=_bytes_to_mb(min_necessary_end),
        min_necessary_ram_peak_mb=_bytes_to_mb(sampler.max_min_necessary_ram),
        mem_available_start_mb=_bytes_to_mb_optional(mem_available_start),
        mem_available_end_mb=_bytes_to_mb_optional(mem_available_end),
        mem_available_min_mb=_bytes_to_mb_optional(sampler.min_mem_available),
        cgroup_current_start_mb=_bytes_to_mb_optional(cgroup_current_start),
        cgroup_current_end_mb=_bytes_to_mb_optional(cgroup_current_end),
        cgroup_current_peak_mb=_bytes_to_mb_optional(cgroup_peak),
        cgroup_limit_mb=_bytes_to_mb_optional(cgroup_limit),
        cache_size_mb=_bytes_to_mb(_dir_size_bytes(cache_dir)),
    )


def _write_step_results(results: list[_StepResult], results_path: Path, dataset_name: str) -> None:
    """Write incremental step metrics and print step status line.

    Parameters
    ----------
    results : list[_StepResult]
        Current step results for active dataset.
    results_path : Path
        Path to ``steps.json`` output file.
    dataset_name : str
        Dataset label for console logging.

    Returns
    -------
    None
        Result JSON is written to disk.

    Notes
    -----
    This function rewrites the full steps file after each step to support
    progress recovery.
    """
    # Persist full incremental step history for current dataset run.
    results_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")

    # Print compact step-level status line for live CLI feedback.
    step = results[-1]
    print(
        f"[{dataset_name}] step={step.name} "
        f"seconds={step.seconds:.2f} "
        f"min_necessary_ram_peak_mb={step.min_necessary_ram_peak_mb:.2f} "
        f"rss_peak_mb={step.rss_peak_mb:.2f} "
        f"private_peak_mb={_fmt_optional_mb(step.private_peak_mb)} "
        f"non_cache_peak_mb={step.non_cache_peak_mb:.2f} "
        f"mem_available_min_mb={_fmt_optional_mb(step.mem_available_min_mb)} "
        f"cgroup_peak_mb={_fmt_optional_mb(step.cgroup_current_peak_mb)}",
        flush=True,
    )


def _build_summary(results: list[_StepResult], cache_dir: Path) -> dict[str, object]:
    """Aggregate run-level summary metrics from step results.

    Parameters
    ----------
    results : list[_StepResult]
        Completed step results for one dataset run.
    cache_dir : Path
        Cache directory used for final cache-size calculation.

    Returns
    -------
    dict[str, object]
        Summary values persisted into ``summary.json``.

    Notes
    -----
    Summary keys are intentionally stable for downstream analysis scripts.
    """
    # Compute scalar aggregate metrics across all completed step entries.
    total_seconds = sum(item.seconds for item in results)
    benchmark_os = _benchmark_os_key()
    peak_rss_mb = max((item.rss_peak_mb for item in results), default=0.0)
    peak_non_cache_mb = max((item.non_cache_peak_mb for item in results), default=0.0)
    peak_private_values = [item.private_peak_mb for item in results if item.private_peak_mb is not None]
    peak_private_mb = max(peak_private_values) if peak_private_values else None
    peak_min_necessary_ram_mb = max((item.min_necessary_ram_peak_mb for item in results), default=0.0)
    mem_available_values = [item.mem_available_min_mb for item in results if item.mem_available_min_mb is not None]
    cgroup_current_values = [item.cgroup_current_peak_mb for item in results if item.cgroup_current_peak_mb is not None]

    # Derive optional values that might be missing on some platforms.
    min_mem_available_mb = min(mem_available_values) if mem_available_values else None
    peak_cgroup_current_mb = max(cgroup_current_values) if cgroup_current_values else None
    cgroup_limit_mb = next((item.cgroup_limit_mb for item in results if item.cgroup_limit_mb is not None), None)
    cgroup_peak_of_limit_pct = None
    if cgroup_limit_mb is not None and peak_cgroup_current_mb is not None:
        cgroup_peak_of_limit_pct = (peak_cgroup_current_mb / cgroup_limit_mb) * 100.0

    # Return summary payload using stable schema for all benchmark profiles.
    return {
        "benchmark_os": benchmark_os,
        "total_seconds": total_seconds,
        "peak_rss_mb": peak_rss_mb,
        "peak_private_mb": peak_private_mb,
        "peak_non_cache_mb": peak_non_cache_mb,
        "peak_min_necessary_ram_mb": peak_min_necessary_ram_mb,
        "min_mem_available_mb": min_mem_available_mb,
        "peak_cgroup_current_mb": peak_cgroup_current_mb,
        "cgroup_limit_mb": cgroup_limit_mb,
        "cgroup_peak_of_limit_pct": cgroup_peak_of_limit_pct,
        "cache_size_mb": _bytes_to_mb(_dir_size_bytes(cache_dir)),
    }


def _write_summary(summary: dict[str, object], output_root: Path) -> None:
    """Write dataset-level summary JSON and print one summary line.

    Parameters
    ----------
    summary : dict[str, object]
        Summary payload from ``_build_summary``.
    output_root : Path
        Dataset-specific output directory.

    Returns
    -------
    None
        Summary file is written to disk.

    Notes
    -----
    Console output intentionally mirrors JSON keys for easy grepping.
    """
    # Persist summary payload for downstream packaging and analysis.
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print one compact run-level status line.
    print(
        f"[{output_root.name}] total_seconds={summary['total_seconds']:.2f}, "
        f"benchmark_os={summary['benchmark_os']}, "
        f"peak_min_necessary_ram_mb={summary['peak_min_necessary_ram_mb']:.2f}, "
        f"peak_rss_mb={summary['peak_rss_mb']:.2f}, "
        f"peak_private_mb={_fmt_optional_mb(summary['peak_private_mb'])}, "
        f"peak_non_cache_mb={summary['peak_non_cache_mb']:.2f}, "
        f"min_mem_available_mb={_fmt_optional_mb(summary['min_mem_available_mb'])}, "
        f"peak_cgroup_current_mb={_fmt_optional_mb(summary['peak_cgroup_current_mb'])}, "
        f"cgroup_limit_mb={_fmt_optional_mb(summary['cgroup_limit_mb'])}, "
        f"cgroup_peak_of_limit_pct={_fmt_optional_mb(summary['cgroup_peak_of_limit_pct'])}, "
        f"cache_size_mb={summary['cache_size_mb']:.2f}"
    )


def _copy_plot_artifacts(cache_dir: Path, output_root: Path) -> None:
    """Copy generated plot artifacts from cache to result directory.

    Parameters
    ----------
    cache_dir : Path
        Cache directory containing generated plot files.
    output_root : Path
        Dataset-specific benchmark output directory.

    Returns
    -------
    None
        Plot files are copied when present.

    Notes
    -----
    Existing files are overwritten by ``shutil.copy2``.
    """
    # Ensure destination directory exists before copying artifact files.
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Copy common vector/raster plot formats produced by benchmark pipeline.
    for extension in (".svg", ".png", ".pdf"):
        for source in cache_dir.rglob(f"*{extension}"):
            shutil.copy2(source, plots_dir / source.name)


def _copy_structure_artifacts(cache_dir: Path, output_root: Path) -> None:
    """Copy structure visualization artifacts from cache to result directory.

    Parameters
    ----------
    cache_dir : Path
        Cache directory containing ``structure_viz`` output.
    output_root : Path
        Dataset-specific benchmark output directory.

    Returns
    -------
    None
        Directory is copied when source exists.

    Notes
    -----
    Destination directory is replaced to avoid stale file accumulation.
    """
    # Resolve source/destination paths and skip when structure output is absent.
    source_dir = cache_dir / "structure_viz"
    if not source_dir.exists():
        return

    # Replace previous destination tree and copy current structure artifacts.
    target_dir = output_root / "structure_viz"
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir)


def _copy_artifacts(cache_dir: Path, output_root: Path) -> None:
    """Copy all non-JSON artifacts to dataset output directory.

    Parameters
    ----------
    cache_dir : Path
        Cache directory containing generated artifacts.
    output_root : Path
        Dataset-specific benchmark output directory.

    Returns
    -------
    None
        Artifact copy actions are executed in sequence.

    Notes
    -----
    Plot artifacts and structure visualization artifacts are copied separately.
    """
    # Delegate artifact handling to dedicated copy helpers.
    _copy_plot_artifacts(cache_dir, output_root)
    _copy_structure_artifacts(cache_dir, output_root)


def _prepare_run_dirs(profile: _BenchmarkProfile, dataset_name: str, remove: bool) -> tuple[Path, Path, Path]:
    """Create and normalize run-specific paths for one dataset.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.
    dataset_name : str
        Dataset label used to derive output paths.
    remove : bool
        Enables cleanup/overwrite behavior for existing run directories.

    Returns
    -------
    tuple[Path, Path, Path]
        Tuple of ``(results_path, output_root, run_cache)``.

    Notes
    -----
    Output and cache directories are created as needed.
    """
    # Build deterministic per-dataset output and cache locations.
    output_root = profile.results_dir / dataset_name
    results_path = output_root / "steps.json"
    run_cache = profile.cache_root / dataset_name

    # When remove is disabled, never overwrite existing run folders.
    if output_root.exists():
        if not remove:
            raise FileExistsError(f"Run output already exists and remove=False: {output_root}")
        shutil.rmtree(output_root, ignore_errors=True)

    # Ensure output exists and optionally reset cache to avoid contamination.
    output_root.mkdir(parents=True, exist_ok=True)
    if run_cache.exists():
        if not remove:
            raise FileExistsError(f"Run cache already exists and remove=False: {run_cache}")
        shutil.rmtree(run_cache, ignore_errors=True)
    run_cache.mkdir(parents=True, exist_ok=True)
    return results_path, output_root, run_cache


def _run_dataset(profile: _BenchmarkProfile, dataset_name: str, dataset_dir: Path, remove: bool) -> None:
    """Execute full benchmark workflow for one dataset directory.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.
    dataset_name : str
        Dataset label.
    dataset_dir : Path
        Dataset directory.
    remove : bool
        Enables cleanup/overwrite behavior for existing run directories.

    Returns
    -------
    None
        Step and summary files are written under profile results directory.

    Notes
    -----
    This function is the smallest atomic benchmark execution unit.
    """
    # Prepare run directories and initialize pipeline and step list.
    results_path, output_root, run_cache = _prepare_run_dirs(profile, dataset_name, remove=remove)
    pipeline = _build_pipeline(profile, run_cache)
    steps = _build_steps(pipeline, dataset_dir, output_root, profile)

    # Execute all steps with incremental writes for resilience.
    results: list[_StepResult] = []
    for name, func in steps:
        results.append(_run_step(name, func, run_cache))
        _write_step_results(results, results_path, dataset_name)

    # Persist run summary and copy generated artifacts.
    summary = _build_summary(results, run_cache)
    _write_summary(summary, output_root)
    _copy_artifacts(run_cache, output_root)
    pipeline.close()


def _dataset_name(factor: int) -> str:
    """Build benchmark dataset name for one scale factor.

    Parameters
    ----------
    factor : int
        Dataset scaling factor.

    Returns
    -------
    str
        Dataset name used by benchmark scripts.

    Notes
    -----
    Factor ``1`` maps to the base ``2RJY`` dataset name.
    """
    return "2RJY" if factor == 1 else f"2RJY_stack{factor}x"


def _dataset_factor_key(dataset_name: str) -> int:
    """Extract numeric factor key from dataset name for stable sorting.

    Parameters
    ----------
    dataset_name : str
        Dataset name such as ``2RJY`` or ``2RJY_stack50x``.

    Returns
    -------
    int
        Parsed factor key, defaults to ``1`` for base dataset.

    Notes
    -----
    Unknown names fall back to ``1``.
    """
    if dataset_name == "2RJY":
        return 1
    match = re.search(r"_stack(\d+)x$", dataset_name)
    return int(match.group(1)) if match else 1


def _dataset_path(factor: int) -> Path:
    """Resolve benchmark dataset path for one scale factor.

    Parameters
    ----------
    factor : int
        Dataset scaling factor.

    Returns
    -------
    Path
        Dataset path used by benchmark scripts.

    Notes
    -----
    Factor ``1`` points to original source dataset path.
    """
    return base_dataset if factor == 1 else data_root / _dataset_name(factor)


def _resolve_datasets(factors: list[int]) -> list[tuple[str, Path]]:
    """Resolve and validate dataset directories for selected factors.

    Parameters
    ----------
    factors : list[int]
        Dataset scale factors to execute.

    Returns
    -------
    list[tuple[str, Path]]
        Ordered ``(dataset_name, dataset_path)`` pairs.

    Notes
    -----
    Missing datasets raise ``FileNotFoundError`` with recovery hint.
    """
    # Build deterministic dataset list while validating on-disk availability.
    datasets: list[tuple[str, Path]] = []
    for factor in factors:
        name = _dataset_name(factor)
        path = _dataset_path(factor)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}. "
                "Run dev_scripts/benchmark/benchmark_generate_data.py first."
            )
        datasets.append((name, path))
    return datasets


def _load_dataset_outputs(profile: _BenchmarkProfile, dataset_name: str) -> dict[str, object]:
    """Load per-dataset output JSON files for profile summary.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.
    dataset_name : str
        Dataset label.

    Returns
    -------
    dict[str, object]
        Combined ``steps`` and ``summary`` payload.

    Notes
    -----
    This helper centralizes summary file loading and schema creation.
    """
    # Resolve expected per-dataset output files.
    output_root = profile.results_dir / dataset_name
    steps_path = output_root / "steps.json"
    summary_path = output_root / "summary.json"

    # Load JSON payloads and return stable schema used by packer/analysis.
    return {
        "steps": json.loads(steps_path.read_text(encoding="utf-8")),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def _write_profile_summary(profile: _BenchmarkProfile, datasets: list[tuple[str, Path]]) -> None:
    """Write profile-level summary JSON combining all dataset runs.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.
    datasets : list[tuple[str, Path]]
        Executed dataset list.

    Returns
    -------
    None
        Profile summary is written to ``results_dir/summary.json``.

    Notes
    -----
    Summary file contains one object keyed by dataset name.
    """
    # Build full summary payload in deterministic dataset iteration order.
    payload: dict[str, object] = {}
    for dataset_name, _ in datasets:
        payload[dataset_name] = _load_dataset_outputs(profile, dataset_name)

    # Ensure result directory exists before writing profile summary file.
    profile.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = profile.results_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_existing_result_datasets(profile: _BenchmarkProfile) -> list[tuple[str, Path]]:
    """Collect datasets that already have complete result JSON files.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.

    Returns
    -------
    list[tuple[str, Path]]
        Dataset names with corresponding output directories.

    Notes
    -----
    Only directories containing both ``steps.json`` and ``summary.json`` are
    included.
    """
    if not profile.results_dir.exists():
        return []

    datasets: list[tuple[str, Path]] = []
    for run_dir in sorted(path for path in profile.results_dir.iterdir() if path.is_dir()):
        if (run_dir / "steps.json").exists() and (run_dir / "summary.json").exists():
            datasets.append((run_dir.name, run_dir))
    return sorted(datasets, key=lambda item: (_dataset_factor_key(item[0]), item[0]))


def _run_profile(profile: _BenchmarkProfile, remove: bool = True) -> int:
    """Run benchmark workflow for every dataset in selected profile.

    Parameters
    ----------
    profile : _BenchmarkProfile
        Benchmark profile configuration.
    remove : bool, default=True
        Enables cleanup/overwrite behavior for existing run directories.

    Returns
    -------
    int
        Process-style exit code (``0`` on success).

    Notes
    -----
    This is the shared execution entry point used by all benchmark profiles.
    """
    # Resolve datasets once, then execute one run per dataset.
    datasets = _resolve_datasets(profile.dataset_factors)
    for dataset_name, dataset_dir in datasets:
        _run_dataset(profile, dataset_name, dataset_dir, remove=remove)

    # Rewrite summary from all complete result folders currently on disk.
    _write_profile_summary(profile, _collect_existing_result_datasets(profile))
    return 0


def _approx_memmap_profile() -> _BenchmarkProfile:
    """Build Approx Memmap benchmark profile configuration.

    Parameters
    ----------
    None

    Returns
    -------
    _BenchmarkProfile
        Approx Memmap profile instance.

    Notes
    -----
    These values preserve the existing benchmark behavior.
    """
    # Configure memmap + Nyström Approx Memmap profile parameters.
    return _BenchmarkProfile(
        name="approx_memmap",
        results_dir=results_dir,
        cache_root=cache_root,
        dataset_factors=list(dataset_factors),
        use_memmap=True,
        chunk_size=2000,
        use_nystrom=True,
        n_landmarks=2000,
        dpa_method="knn_sampling",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Approx Memmap benchmark execution.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.

    Notes
    -----
    Defaults preserve the existing full-profile behavior.
    """
    parser = argparse.ArgumentParser(description="Run Approx Memmap benchmark profile.")
    parser.add_argument(
        "--stacks",
        nargs="+",
        type=int,
        choices=supported_dataset_factors,
        default=list(dataset_factors),
        help="Stack factors to run. Supported: 1,2,3,5,10,30,50,500,1000. Default: all configured factors.",
    )
    parser.add_argument("--remove", type=_parse_bool, default=True, help="Allow cleanup/overwrite behavior (true/false). Default: true.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional custom cache directory root.")
    return parser.parse_args()


def _parse_bool(raw_value: str) -> bool:
    """Parse boolean CLI string values.

    Parameters
    ----------
    raw_value : str
        Raw CLI value.

    Returns
    -------
    bool
        Parsed boolean value.

    Notes
    -----
    Accepted true values: ``true,1,yes,on``.
    Accepted false values: ``false,0,no,off``.
    """
    text = str(raw_value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value for --remove: {raw_value!r}")


def main() -> int:
    """Run the Approx Memmap benchmark profile.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Process-style exit code.

    Notes
    -----
    This CLI entry point runs the predefined Approx Memmap configuration.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/benchmark_approx_memmap.py
    """
    # Build profile from CLI selection and execute via shared benchmark engine.
    args = parse_args()
    profile = _approx_memmap_profile()
    if args.cache_dir is not None:
        # Resolve to absolute path to avoid issues if working directory changes
        cache_dir = args.cache_dir.resolve() if getattr(args.cache_dir, 'is_absolute', lambda: False)() else Path(os.getcwd()) / args.cache_dir
        profile = replace(profile, cache_root=cache_dir)
    profile = replace(profile, dataset_factors=list(args.stacks))
    return _run_profile(profile, remove=bool(args.remove))


if __name__ == "__main__":
    raise SystemExit(main())

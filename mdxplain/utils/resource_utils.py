# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0).
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Utilities for applying process-level resource limits.

These limits are best-effort and platform-dependent. Requires psutil and
threadpoolctl for portable process priority, I/O priority, CPU affinity,
and BLAS/OpenMP thread limiting.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import mmap
import os
import sys

import psutil
from threadpoolctl import threadpool_limits


class ResourceUtils:
    """
    Utility class for best-effort process-level resource limits.
    """
    _blas_limit_context = None

    @staticmethod
    def recommend_cpu_affinity(reserve_cores: int = 2) -> Optional[List[int]]:
        """
        Recommend a CPU affinity mask leaving a number of cores free.

        Parameters
        ----------
        reserve_cores : int, default=2
            Number of cores to leave unused for system responsiveness. The
            resulting affinity always keeps at least one core, even when
            reserve_cores is large relative to the allocation.

        Returns
        -------
        list of int or None
            Recommended CPU affinity list, or None if unavailable. The list is
            derived from the currently allowed CPUs (e.g., cgroups on HPC),
            so it will not exceed the scheduler allocation.

        Notes
        -----
        This does not apply the affinity; it only calculates the recommended
        mask. Use apply_process_limits to enforce the recommendation.
        """
        if reserve_cores < 0:
            raise ValueError("reserve_cores must be >= 0")

        available = ResourceUtils._get_allowed_cpus()
        if not available:
            return None
        if reserve_cores == 0 or len(available) <= 1:
            return available
        if reserve_cores >= len(available):
            return available
        return available[: max(1, len(available) - reserve_cores)]

    @staticmethod
    def apply_auto_limits(
        reserve_cores: int = 2,
        nice: Optional[int] = 15,
        io_priority: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply recommended CPU affinity plus optional nice/I/O priority.

        Parameters
        ----------
        reserve_cores : int, default=2
            Number of cores to leave free.
        nice : int, optional
            POSIX nice value (or Windows priority mapping). Use 0 for normal
            priority. Positive values yield to the desktop or scheduler. The
            default of 15 is intentionally conservative to keep the system
            responsive during heavy I/O workloads.
        io_priority : str, optional
            I/O priority hint ("idle", "low", "normal", "high"). Use "low"
            when working with large memmaps to reduce I/O starvation.

        Returns
        -------
        dict
            Dictionary containing applied values and any errors encountered.

        Notes
        -----
        This is a convenience helper that combines recommend_cpu_affinity and
        apply_process_limits. It is best called early in a process before
        heavy compute starts.
        """
        cpu_affinity = ResourceUtils.recommend_cpu_affinity(reserve_cores)
        return ResourceUtils.apply_process_limits(
            nice=nice,
            io_priority=io_priority,
            cpu_affinity=cpu_affinity,
        )

    @staticmethod
    def apply_blas_thread_limits(max_threads: Optional[int]) -> Dict[str, Any]:
        """
        Apply global BLAS/OpenMP thread limits for the current process.

        Parameters
        ----------
        max_threads : int or None
            Maximum number of BLAS/OpenMP threads. None resets to defaults.
            Set to the effective CPU affinity size to avoid oversubscription.

        Returns
        -------
        dict
            Dictionary containing applied values and any errors encountered.

        Notes
        -----
        This uses threadpoolctl to update active BLAS/OpenMP thread pools.
        The limit is process-wide and persistent until reset or changed.
        It does not alter environment variables and does not interrupt
        already-running compute kernels.

        When using joblib or other parallel backends (n_jobs > 1), prefer
        keeping this limit at the effective CPU count and rely on per-algorithm
        auto_limit_blas logic to drop BLAS threads to 1 for those workloads.
        """
        result: Dict[str, Any] = {"max_threads": None, "errors": []}

        if ResourceUtils._blas_limit_context is not None:
            try:
                ResourceUtils._blas_limit_context.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                result["errors"].append(f"blas_reset: {exc}")
            ResourceUtils._blas_limit_context = None

        if max_threads is None:
            return result
        if max_threads < 1:
            raise ValueError("max_threads must be >= 1")

        try:
            ctx = threadpool_limits(limits=max_threads)
            ctx.__enter__()
            ResourceUtils._blas_limit_context = ctx
            result["max_threads"] = max_threads
        except Exception as exc:  # pragma: no cover - platform-specific failures
            result["errors"].append(f"blas_limit: {exc}")
        return result

    @staticmethod
    def apply_process_limits(
        nice: Optional[int] = None,
        io_priority: Optional[str] = None,
        cpu_affinity: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Apply process-level limits for CPU priority, I/O priority, and affinity.

        Parameters
        ----------
        nice : int, optional
            POSIX nice value. On Windows, maps to a best-effort priority class.
        io_priority : str, optional
            I/O priority hint ("idle", "low", "normal", "high"). Best-effort.
            Some platforms ignore this setting or require elevated privileges.
        cpu_affinity : sequence of int, optional
            CPU affinity mask (list of CPU indices). Best-effort.

        Returns
        -------
        dict
            Dictionary containing applied values and any errors encountered.

        Notes
        -----
        CPU affinity constrains where all threads in this process may run.
        This is the most reliable way to keep a few cores free for system
        responsiveness. I/O priority is useful for large memmap workloads,
        but exact behavior varies by OS and filesystem.
        """
        result: Dict[str, Any] = {
            "nice": None,
            "io_priority": None,
            "cpu_affinity": None,
            "errors": [],
        }

        if nice is None and io_priority is None and cpu_affinity is None:
            return result

        proc = psutil.Process()

        if nice is not None:
            try:
                if sys.platform.startswith("win"):
                    proc.nice(ResourceUtils._map_windows_priority(nice))
                else:
                    proc.nice(nice)
                result["nice"] = nice
            except Exception as exc:  # pragma: no cover - OS-specific errors
                result["errors"].append(f"nice: {exc}")

        if cpu_affinity is not None:
            if hasattr(proc, "cpu_affinity"):
                try:
                    proc.cpu_affinity(list(cpu_affinity))
                    result["cpu_affinity"] = list(cpu_affinity)
                except Exception as exc:  # pragma: no cover - OS-specific errors
                    result["errors"].append(f"cpu_affinity: {exc}")
            else:
                result["errors"].append("cpu_affinity: not supported on this platform")

        if io_priority is not None:
            try:
                ResourceUtils._apply_io_priority(proc, io_priority)
                result["io_priority"] = io_priority
            except Exception as exc:  # pragma: no cover - OS-specific errors
                result["errors"].append(f"io_priority: {exc}")

        return result

    @staticmethod
    def _get_allowed_cpus() -> List[int]:
        """
        Return the list of CPUs currently allowed for this process.

        Uses psutil when available (respects cgroups/cpuset) and falls back to
        os.sched_getaffinity or os.cpu_count. This is a helper for auto
        affinity selection and has no side effects.
        """
        if hasattr(psutil.Process(), "cpu_affinity"):
            return sorted(psutil.Process().cpu_affinity())
        if hasattr(os, "sched_getaffinity"):
            return sorted(os.sched_getaffinity(0))
        count = os.cpu_count() or 1
        return list(range(count))

    @staticmethod
    def tune_memmap(
        array: Any, 
        strategy: str, 
        start_offset: int = 0, 
        length: int = 0
    ) -> Dict[str, Any]:
        """
        Apply a best-effort `madvise` hint to a memmap (or memmap-backed view).

        Parameters
        ----------
        array : Any
            Memmap (or ndarray view backed by a memmap).
        strategy : str
            One of: ``"sequential"``, ``"random"``, ``"dontneed"``.
        start_offset : int
            Byte offset for range-based hints. Default ``0`` (full mapping).
        length : int
            Byte length for range-based hints. For known mapping sizes, values
            ``<=0`` are interpreted as \"until end of mapping\".

        Returns
        -------
        dict
            ``{\"strategy\": ..., \"applied\": bool, \"errors\": [..]}``.
        """
        result: Dict[str, Any] = {"strategy": strategy, "applied": False, "errors": []}
        if array is None:
            return result

        mm = ResourceUtils._resolve_madvise_handle(array)
        if mm is None:
            result["errors"].append("madvise not supported for this array")
            return result

        option = ResourceUtils._resolve_madvise_option(strategy)
        if option is None:
            result["errors"].append(
                f"madvise option not available for {strategy.strip().lower()}"
            )
            return result

        use_range, start, range_length, range_error = ResourceUtils._normalize_madvise_range(
            mm, start_offset, length
        )
        if range_error is not None:
            result["errors"].append(range_error)
            return result

        try:
            if use_range:
                mm.madvise(option, start, range_length)
            else:
                mm.madvise(option)
            result["applied"] = True
        except (AttributeError, OSError, ValueError, TypeError) as exc:  # pragma: no cover - OS-specific
            result["errors"].append(f"madvise failed: {exc}")

        return result

    @staticmethod
    def _resolve_madvise_handle(array: Any) -> Optional[Any]:
        """
        Resolve the first mmap-like object in an array/base chain that supports ``madvise``.

        Parameters
        ----------
        array : Any
            Candidate memmap or ndarray view. The function traverses ``array.base``
            until it finds an object exposing a usable ``madvise`` method.

        Returns
        -------
        object or None
            Resolved mmap-like object that supports ``madvise``, or ``None`` when
            no suitable handle is available.

        Notes
        -----
        A ``seen`` set is used to avoid infinite loops in unusual base chains.
        """
        mm = getattr(array, "_mmap", None)
        if mm is not None and hasattr(mm, "madvise"):
            return mm

        base = getattr(array, "base", None)
        seen = set()
        while base is not None and id(base) not in seen:
            seen.add(id(base))
            mm = getattr(base, "_mmap", None)
            if mm is not None and hasattr(mm, "madvise"):
                return mm
            if hasattr(base, "madvise"):
                return base
            base = getattr(base, "base", None)

        return None

    @staticmethod
    def _resolve_madvise_option(strategy: str) -> Optional[int]:
        """
        Map a strategy string to the corresponding platform ``madvise`` constant.

        Parameters
        ----------
        strategy : str
            Access hint strategy. Supported values are ``\"sequential\"``,
            ``\"random\"``, and ``\"dontneed\"``.

        Returns
        -------
        int or None
            Platform-specific ``MADV_*`` constant, or ``None`` if the strategy
            exists but is not exposed on the current platform.

        Raises
        ------
        ValueError
            If ``strategy`` is not one of the supported values.
        """
        strategy_key = strategy.strip().lower()
        if strategy_key == "sequential":
            return getattr(mmap, "MADV_SEQUENTIAL", None)
        if strategy_key == "random":
            return getattr(mmap, "MADV_RANDOM", None)
        if strategy_key == "dontneed":
            return getattr(mmap, "MADV_DONTNEED", None)
        raise ValueError("strategy must be one of: sequential, random, dontneed")

    @staticmethod
    def _mmap_size_bytes(mmap_obj: Any) -> Optional[int]:
        """
        Best-effort extraction of mapping size in bytes.

        Parameters
        ----------
        mmap_obj : Any
            mmap-like object. The method first tries ``size()``, then falls back
            to ``len()`` when available.

        Returns
        -------
        int or None
            Non-negative mapping size in bytes, or ``None`` if size cannot be
            determined reliably.
        """
        if hasattr(mmap_obj, "size"):
            try:
                size_value = int(mmap_obj.size())
                if size_value >= 0:
                    return size_value
            except (TypeError, ValueError, OSError, OverflowError):
                pass
        if hasattr(mmap_obj, "__len__"):
            try:
                size_value = int(len(mmap_obj))
                if size_value >= 0:
                    return size_value
            except (TypeError, ValueError, OverflowError):
                pass
        return None

    @staticmethod
    def _normalize_madvise_range(
        mmap_obj: Any,
        start_offset: int,
        length: int,
    ) -> Tuple[bool, int, int, Optional[str]]:
        """
        Normalize and validate range arguments for ranged ``madvise`` calls.

        Parameters
        ----------
        mmap_obj : Any
            mmap-like object used to determine mapping size when available.
        start_offset : int
            Requested start offset in bytes.
        length : int
            Requested length in bytes. For known mapping sizes, values ``<= 0``
            are interpreted as \"until end of mapping\".

        Returns
        -------
        tuple
            Tuple ``(use_range, start, normalized_length, error)`` where:
            
            - ``use_range`` indicates whether ranged ``madvise`` should be used.
            - ``start`` is the validated byte offset.
            - ``normalized_length`` is the validated/clamped byte length.
            - ``error`` is ``None`` on success, otherwise a human-readable message.

        Notes
        -----
        If mapping size is unknown, a strictly positive length is required for
        ranged calls.
        """
        try:
            start = int(start_offset)
            normalized_length = int(length)
        except (TypeError, ValueError, OverflowError):
            return False, 0, 0, "start_offset and length must be integers"

        if start < 0 or normalized_length < 0:
            return False, 0, 0, "start_offset and length must be non-negative"

        use_range = start > 0 or normalized_length > 0
        if not use_range:
            return False, 0, 0, None

        map_size = ResourceUtils._mmap_size_bytes(mmap_obj)
        if map_size is None:
            if normalized_length <= 0:
                return False, 0, 0, "length must be > 0 when mmap size is unavailable"
            return True, start, normalized_length, None

        if start >= map_size:
            return (
                False,
                0,
                0,
                f"start_offset {start} out of bounds for mapping size {map_size}",
            )

        if normalized_length <= 0:
            normalized_length = map_size - start
        else:
            normalized_length = min(normalized_length, map_size - start)

        if normalized_length <= 0:
            return (
                False,
                0,
                0,
                f"length {length} resolves to empty range at offset {start}",
            )

        return True, start, normalized_length, None

    @staticmethod
    def _map_windows_priority(nice: int) -> int:
        """
        Map a POSIX-style nice value to a Windows priority class.

        This keeps the public API consistent across platforms while still
        applying a reasonable Windows priority class under the hood.
        """
        if nice >= 15:
            return psutil.IDLE_PRIORITY_CLASS
        if nice >= 10:
            return psutil.BELOW_NORMAL_PRIORITY_CLASS
        if nice <= -10:
            return psutil.HIGH_PRIORITY_CLASS
        if nice <= -5:
            return psutil.ABOVE_NORMAL_PRIORITY_CLASS
        return psutil.NORMAL_PRIORITY_CLASS

    @staticmethod
    def _apply_io_priority(proc: "psutil.Process", io_priority: str) -> None:
        """
        Apply best-effort I/O priority to a psutil.Process.

        Parameters
        ----------
        proc : psutil.Process
            Process handle to update.
        io_priority : str
            I/O priority hint ("idle", "low", "normal", "high").

        Raises
        ------
        ValueError
            If io_priority is not one of the supported strings.
        RuntimeError
            If the platform does not support setting I/O priority.

        Notes
        -----
        On Windows, I/O priority may map to process priority classes and can
        be limited. Using nice >= 15 (IDLE_PRIORITY_CLASS) is typically the
        most reliable way to reduce I/O pressure on the system.
        """
        level = io_priority.strip().lower()
        if level not in {"idle", "low", "normal", "high"}:
            raise ValueError("io_priority must be one of: idle, low, normal, high")

        if hasattr(psutil, "IOPRIO_CLASS_IDLE"):
            mapping = {
                "idle": (psutil.IOPRIO_CLASS_IDLE, 0),
                "low": (psutil.IOPRIO_CLASS_BE, 7),
                "normal": (psutil.IOPRIO_CLASS_BE, 4),
                "high": (psutil.IOPRIO_CLASS_RT, 0),
            }
            ioclass, value = mapping[level]
            if ioclass == psutil.IOPRIO_CLASS_IDLE:
                proc.ionice(ioclass)
            else:
                proc.ionice(ioclass, value)
            return

        if hasattr(psutil, "IOPRIO_LOW"):
            mapping = {
                "idle": psutil.IOPRIO_LOW,
                "low": psutil.IOPRIO_LOW,
                "normal": psutil.IOPRIO_NORMAL,
                "high": psutil.IOPRIO_HIGH,
            }
            proc.ionice(mapping[level])
            return

        raise RuntimeError("I/O priority not supported on this platform")

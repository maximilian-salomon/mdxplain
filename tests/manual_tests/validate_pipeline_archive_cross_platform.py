#!/usr/bin/env python
# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from OpenAI Codex (GPT-5.3-Codex High).
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

"""Validate cross-platform compatibility of PipelineManager archives.

This manual test script executes deterministic load/validate/save cycles and
checks that runtime-bound resources (memmap/zarr) and sampled array signatures
remain stable across roundtrips.

Execution overview
------------------
1. Read the input archive and record its file members.
2. Load the pipeline into a fresh runtime cache per cycle.
3. Validate runtime-bound paths and sampled array signatures.
4. Optionally run feature analysis checks.
5. Re-export the archive for the next cycle and compare members.
6. Print a clear pass/fail result and write a JSON report.

Usage
-----
This script is designed to run without parameters.

Defaults:
- archive path (project root): ``pipeline_from_data.tar.zst``
- cycles: ``3``

No-parameter example:

``python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py``

Examples with parameters:

``python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py --cycles 5``

``python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py --roundtrip-compression gz --cycles 3``

``python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py other_archive.tar.zst --cycles 2 --skip-analysis-checks``

Result markers:
- ``TEST RESULT: PASS`` and ``TEST_RESULT=PASS`` for success.
- ``TEST RESULT: FAIL`` and ``TEST_RESULT=FAIL`` for failure.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import json
import re
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import numpy as np
from mdxplain.pipeline.manager.pipeline_manager import PipelineManager

SCOPED_CACHE_PATTERN = re.compile(r"cache_[0-9a-f]{32}_\d{8}_\d{6}$")
ArraySignatureMap = Dict[str, Dict[str, Any]]
DEFAULT_ARCHIVE_PATH = Path("pipeline_from_data.tar.zst")


def _require(condition: bool, message: str) -> None:
    """Raise an assertion with a stable error message.

    Parameters
    ----------
    condition : bool
        Condition that must evaluate to ``True``.
    message : str
        Message to include in the raised assertion.

    Raises
    ------
    AssertionError
        Raised when ``condition`` is ``False``.
    """
    if not condition:
        raise AssertionError(message)


def _normalize_archive_name(name: str) -> str:
    """Normalize archive member names to POSIX-style separators.

    Parameters
    ----------
    name : str
        Raw member path from a tar archive entry.

    Returns
    -------
    str
        Normalized member path with ``/`` separators.
    """
    return name.replace("\\", "/")


def _list_archive_file_members(archive_path: Path) -> List[str]:
    """Return sorted file members from an archive.

    Parameters
    ----------
    archive_path : pathlib.Path
        Archive to inspect.

    Returns
    -------
    list of str
        Sorted list of normalized file-entry names.
    """
    members: List[str] = []
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.isfile():
                members.append(_normalize_archive_name(member.name))
    return sorted(members)


def _to_jsonable(value: Any) -> Any:
    """Convert values into JSON-serializable primitives.

    Parameters
    ----------
    value : Any
        Arbitrary value extracted from runtime objects.

    Returns
    -------
    Any
        Primitive JSON-compatible value or a ``repr`` fallback.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sample_flat_array(flat: np.ndarray, sample_points: int) -> np.ndarray:
    """Sample a flattened array using evenly spaced deterministic indices.

    Parameters
    ----------
    flat : numpy.ndarray
        Flattened source array.
    sample_points : int
        Maximum number of points to sample.

    Returns
    -------
    numpy.ndarray
        Deterministic sample view or the full array when empty.
    """
    size = int(flat.size)
    if size == 0:
        return flat
    sample_size = min(sample_points, size)
    sample_indices = np.linspace(0, size - 1, num=sample_size, dtype=np.int64)
    return flat[sample_indices]


def _hash_array_sample(sample: np.ndarray) -> str:
    """Compute a stable SHA-256 hash for a sampled array.

    Parameters
    ----------
    sample : numpy.ndarray
        Sampled values used for signature comparison.

    Returns
    -------
    str
        SHA-256 hex digest of the sample payload.
    """
    if sample.dtype.hasobject:
        payload = "|".join(repr(item) for item in sample.tolist()).encode("utf-8")
    else:
        payload = np.ascontiguousarray(sample).view(np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _sample_numeric_stats(sample: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics for numeric samples.

    Parameters
    ----------
    sample : numpy.ndarray
        Numeric sample values.

    Returns
    -------
    dict
        Numeric summary statistics and finite-value flag.
    """
    if sample.size == 0:
        return {}
    sample_float = np.asarray(sample, dtype=np.float64)
    return {
        "sample_mean": float(np.mean(sample_float)),
        "sample_std": float(np.std(sample_float)),
        "sample_min": float(np.min(sample_float)),
        "sample_max": float(np.max(sample_float)),
        "sample_all_finite": bool(np.all(np.isfinite(sample_float))),
    }


def _collect_numpy_arrays_recursive(root: object) -> List[Tuple[str, np.ndarray]]:
    """Collect unique numpy arrays from an object graph.

    Parameters
    ----------
    root : object
        Root object that contains nested runtime data.

    Returns
    -------
    list of tuple of (str, numpy.ndarray)
        Array paths and array objects discovered during traversal.

    Notes
    -----
    Traversal de-duplicates both container objects and array objects by ID to
    avoid repeated visits in cyclic graphs.
    """
    arrays: List[Tuple[str, np.ndarray]] = []
    seen_objects = set()
    seen_arrays = set()
    stack: List[Tuple[str, Any]] = [("root", root)]

    while stack:
        path, value = stack.pop()
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            value_id = id(value)
            if value_id in seen_arrays:
                continue
            seen_arrays.add(value_id)
            arrays.append((path, value))
            continue
        if isinstance(value, (str, bytes, int, float, bool, np.generic)):
            continue

        value_id = id(value)
        if value_id in seen_objects:
            continue
        seen_objects.add(value_id)

        if isinstance(value, dict):
            for key, sub_value in value.items():
                stack.append((f"{path}[{key!r}]", sub_value))
            continue
        if isinstance(value, (list, tuple)):
            for index, sub_value in enumerate(value):
                stack.append((f"{path}[{index}]", sub_value))
            continue
        if isinstance(value, set):
            for index, sub_value in enumerate(sorted(value, key=repr)):
                stack.append((f"{path}{{{index}}}", sub_value))
            continue
        if hasattr(value, "__dict__"):
            for attr_name, attr_value in vars(value).items():
                stack.append((f"{path}.{attr_name}", attr_value))

    return arrays


def _array_signature(array: np.ndarray, sample_points: int = 128) -> Dict[str, Any]:
    """Build a stable comparison signature for one array.

    Parameters
    ----------
    array : numpy.ndarray
        Source array to summarize.
    sample_points : int, default=128
        Maximum number of deterministic sample points.

    Returns
    -------
    dict
        Signature including shape, dtype, sample hash, edge values, and
        numeric sample statistics where applicable.
    """
    arr = np.asarray(array)
    flat = arr.reshape(-1)
    size = int(flat.size)
    sample = _sample_flat_array(flat, sample_points=sample_points)
    sample_hash = _hash_array_sample(sample)
    sample_size = int(sample.size)

    numeric = np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)
    stats = _sample_numeric_stats(sample) if numeric else {}

    first_value: Optional[Any] = None
    last_value: Optional[Any] = None
    if size > 0:
        first_value = _to_jsonable(flat[0])
        last_value = _to_jsonable(flat[-1])

    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": size,
        "sample_size": sample_size,
        "sample_hash_sha256": sample_hash,
        "first": first_value,
        "last": last_value,
        **stats,
    }


def _capture_array_signatures(root: object) -> Tuple[ArraySignatureMap, int, int]:
    """Capture signatures and array-type counters for a root object.

    Parameters
    ----------
    root : object
        Root object to traverse for arrays.

    Returns
    -------
    tuple
        ``(signatures, memmap_count, ndarray_count)``.
    """
    signatures: ArraySignatureMap = {}
    memmap_count = 0
    ndarray_count = 0

    for path, array in _collect_numpy_arrays_recursive(root):
        signatures[path] = _array_signature(array)
        if isinstance(array, np.memmap):
            memmap_count += 1
        else:
            ndarray_count += 1

    return signatures, memmap_count, ndarray_count


def _assert_signature_dicts_equal(
    reference: ArraySignatureMap,
    current: ArraySignatureMap,
    *,
    context: str,
) -> None:
    """Assert that two array-signature mappings are equal.

    Parameters
    ----------
    reference : dict
        Baseline signatures.
    current : dict
        Current-cycle signatures.
    context : str
        Context identifier included in assertion messages.

    Raises
    ------
    AssertionError
        Raised when keys or values do not match.
    """
    ref_keys = set(reference)
    cur_keys = set(current)
    if ref_keys != cur_keys:
        missing = sorted(ref_keys - cur_keys)
        extra = sorted(cur_keys - ref_keys)
        raise AssertionError(
            f"{context}: array key mismatch. missing={missing[:5]} extra={extra[:5]}"
        )

    for key in sorted(ref_keys):
        if reference[key] != current[key]:
            raise AssertionError(
                f"{context}: array signature mismatch at path={key}"
            )


def _validate_runtime_cache(runtime_cache: Path) -> None:
    """Validate runtime-cache existence and naming constraints.

    Parameters
    ----------
    runtime_cache : pathlib.Path
        Runtime cache path resolved from pipeline config.
    """
    _require(runtime_cache.exists(), f"Runtime cache does not exist: {runtime_cache}")
    _require(runtime_cache.is_dir(), f"Runtime cache is not a directory: {runtime_cache}")
    _require(
        SCOPED_CACHE_PATTERN.fullmatch(runtime_cache.name) is not None,
        (
            "Runtime cache directory does not match expected scoped pattern "
            f"'cache_<uuid>_<timestamp>': {runtime_cache.name}"
        ),
    )


def _validate_memmap_bindings(
    arrays_with_path: List[Tuple[str, np.ndarray]],
    runtime_cache: Path,
) -> None:
    """Validate memmap binding paths and read access.

    Parameters
    ----------
    arrays_with_path : list of tuple
        Collected arrays together with traversal paths.
    runtime_cache : pathlib.Path
        Runtime cache root that must contain all memmap files.
    """
    for _, array in arrays_with_path:
        if not isinstance(array, np.memmap):
            continue
        memmap_path = Path(array.filename).resolve()
        _require(
            runtime_cache in memmap_path.parents,
            (
                "Memmap path is not bound to runtime cache. "
                f"runtime_cache={runtime_cache}, memmap={memmap_path}"
            ),
        )
        _require(memmap_path.exists(), f"Memmap file missing: {memmap_path}")

        flat_view = np.asarray(array).reshape(-1)
        if flat_view.size > 0:
            _ = flat_view[0]


def _validate_trajectory_bindings(
    data: Any,
    runtime_cache: Path,
) -> Tuple[int, int, int]:
    """Validate trajectory zarr bindings and xyz read access.

    Parameters
    ----------
    data : Any
        Pipeline runtime data object.
    runtime_cache : pathlib.Path
        Runtime cache root that must contain zarr caches.

    Returns
    -------
    tuple
        ``(trajectory_count, zarr_path_count, zarr_xyz_samples)``.
    """
    trajectories = list(getattr(data.trajectory_data, "trajectories", []))
    zarr_path_count = 0
    zarr_xyz_samples = 0

    for traj in trajectories:
        zarr_cache_path = getattr(traj, "zarr_cache_path", None)
        if not zarr_cache_path:
            continue

        zarr_path = Path(zarr_cache_path).resolve()
        zarr_path_count += 1
        _require(zarr_path.exists(), f"zarr cache path missing: {zarr_path}")
        _require(
            runtime_cache in zarr_path.parents,
            (
                "zarr cache path is not bound to runtime cache. "
                f"runtime_cache={runtime_cache}, zarr_path={zarr_path}"
            ),
        )

        xyz = getattr(traj, "xyz", None)
        if xyz is None:
            continue
        try:
            xyz_first_frame = np.asarray(xyz[0]).reshape(-1)
        except Exception:
            xyz_first_frame = np.asarray(xyz).reshape(-1)
        if xyz_first_frame.size > 0:
            _ = xyz_first_frame[0]
        zarr_xyz_samples += 1

    return len(trajectories), zarr_path_count, zarr_xyz_samples


def _iter_feature_entries(feature_data: Any) -> Iterator[Tuple[str, Any, Any]]:
    """Iterate over non-empty feature entries.

    Parameters
    ----------
    feature_data : Any
        ``pipeline._data.feature_data`` container.

    Yields
    ------
    tuple
        ``(feature_name, entry_key, entry_obj)`` for each non-``None`` entry.
    """
    if not isinstance(feature_data, dict):
        return

    for feature_name, per_traj in feature_data.items():
        if isinstance(per_traj, dict):
            iterable: Iterable[Tuple[Any, Any]] = per_traj.items()
        elif isinstance(per_traj, (list, tuple)):
            iterable = enumerate(per_traj)
        else:
            continue

        for entry_key, entry_obj in iterable:
            if entry_obj is None:
                continue
            yield feature_name, entry_key, entry_obj


def _validate_feature_analysis(pipeline: PipelineManager) -> Dict[str, int]:
    """Run finite-value checks on feature-analysis outputs.

    Parameters
    ----------
    pipeline : PipelineManager
        Loaded pipeline under validation.

    Returns
    -------
    dict
        Counter summary for executed ``mean`` and ``std`` checks.
    """
    checked_mean = 0
    checked_std = 0
    feature_data = getattr(pipeline._data, "feature_data", {})

    for feature_name, entry_key, entry in _iter_feature_entries(feature_data):
        analysis = getattr(entry, "analysis", None)
        if analysis is None:
            continue

        if hasattr(analysis, "compute_mean"):
            mean_value = analysis.compute_mean()
            mean_array = np.asarray(mean_value)
            if mean_array.size > 0:
                _require(
                    np.all(np.isfinite(mean_array)),
                    (
                        f"Non-finite mean values in feature={feature_name}, "
                        f"entry={entry_key}"
                    ),
                )
            checked_mean += 1

        if hasattr(analysis, "compute_std"):
            std_value = analysis.compute_std()
            std_array = np.asarray(std_value)
            if std_array.size > 0:
                _require(
                    np.all(np.isfinite(std_array)),
                    (
                        f"Non-finite std values in feature={feature_name}, "
                        f"entry={entry_key}"
                    ),
                )
            checked_std += 1

    return {
        "analysis_mean_checks": checked_mean,
        "analysis_std_checks": checked_std,
    }


def _validate_loaded_pipeline(
    pipeline: PipelineManager,
    run_analysis_checks: bool,
) -> Dict[str, Any]:
    """Validate one loaded pipeline instance.

    Parameters
    ----------
    pipeline : PipelineManager
        Loaded pipeline instance.
    run_analysis_checks : bool
        If ``True``, run per-feature ``compute_mean`` and ``compute_std`` checks.

    Returns
    -------
    dict
        Validation summary including counts and array signatures.
    """
    runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
    _validate_runtime_cache(runtime_cache)
    data = pipeline._data
    arrays_with_path = _collect_numpy_arrays_recursive(data)
    array_signatures, memmap_count, ndarray_count = _capture_array_signatures(data)
    _validate_memmap_bindings(arrays_with_path, runtime_cache)
    trajectory_count, zarr_path_count, zarr_xyz_samples = _validate_trajectory_bindings(
        data,
        runtime_cache,
    )

    analysis_summary = {
        "analysis_mean_checks": 0,
        "analysis_std_checks": 0,
    }
    if run_analysis_checks:
        analysis_summary = _validate_feature_analysis(pipeline)

    return {
        "runtime_cache": str(runtime_cache),
        "trajectory_count": trajectory_count,
        "zarr_path_count": zarr_path_count,
        "zarr_xyz_samples": zarr_xyz_samples,
        "array_count": len(array_signatures),
        "memmap_count": memmap_count,
        "ndarray_count": ndarray_count,
        "array_signatures": array_signatures,
        **analysis_summary,
    }


def _create_roundtrip_archive(
    pipeline: PipelineManager,
    output_dir: Path,
    cycle: int,
    compression: str,
    baseline_members: List[str],
) -> Tuple[Path, int]:
    """Create a roundtrip archive and verify member parity.

    Parameters
    ----------
    pipeline : PipelineManager
        Pipeline to serialize.
    output_dir : pathlib.Path
        Output directory for archive base path.
    cycle : int
        Current cycle number.
    compression : str
        Compression algorithm passed to archive creation.
    baseline_members : list of str
        Baseline archive members used for equality checks.

    Returns
    -------
    tuple
        ``(created_archive_path, created_member_count)``.
    """
    roundtrip_base = output_dir / f"roundtrip_cycle_{cycle}"
    created_archive = Path(
        pipeline.create_sharable_archive(
            str(roundtrip_base),
            compression=compression,
        )
    ).resolve()
    created_members = _list_archive_file_members(created_archive)
    _require(
        created_members == baseline_members,
        (
            "Archive member mismatch after roundtrip. "
            f"cycle={cycle}, created={created_archive}"
        ),
    )
    return created_archive, len(created_members)


@dataclass
class CycleResult:
    """Result payload for a single validation cycle.

    Attributes
    ----------
    report : dict
        Cycle-level report payload.
    runtime_cache : pathlib.Path
        Scoped runtime cache used in this cycle.
    array_signatures : dict
        Captured array signatures for this cycle.
    created_archive : pathlib.Path or None
        Roundtrip archive created for the next cycle, if any.
    """

    report: Dict[str, Any]
    runtime_cache: Path
    array_signatures: ArraySignatureMap
    created_archive: Optional[Path]


def _run_single_cycle(
    *,
    cycle: int,
    cycles_total: int,
    current_archive: Path,
    cache_root: Path,
    output_dir: Path,
    roundtrip_compression: str,
    run_analysis_checks: bool,
    baseline_members: List[str],
) -> CycleResult:
    """Execute one load/validate/(optional)save cycle.

    Parameters
    ----------
    cycle : int
        Current cycle number (1-based).
    cycles_total : int
        Total requested cycle count.
    current_archive : pathlib.Path
        Input archive for this cycle.
    cache_root : pathlib.Path
        Root directory in which runtime caches are created.
    output_dir : pathlib.Path
        Output directory for roundtrip archives.
    roundtrip_compression : str
        Compression for roundtrip archive creation.
    run_analysis_checks : bool
        Whether to execute feature-analysis checks.
    baseline_members : list of str
        Baseline member list for roundtrip archive comparison.

    Returns
    -------
    CycleResult
        Completed cycle result with report and signatures.
    """
    print(f"[cycle {cycle}/{cycles_total}] loading archive: {current_archive}")
    cycle_report: Dict[str, Any] = {
        "cycle": cycle,
        "input_archive": str(current_archive),
    }
    created_archive: Optional[Path] = None
    pipeline = None

    try:
        # Load archive into a scoped runtime cache for this cycle.
        pipeline = PipelineManager.load_from_archive(
            str(current_archive),
            cache_dir=str(cache_root),
            show_progress=False,
        )
        # Run runtime validation checks and collect signatures.
        validation = _validate_loaded_pipeline(
            pipeline=pipeline,
            run_analysis_checks=run_analysis_checks,
        )

        runtime_cache = Path(validation["runtime_cache"]).resolve()
        current_array_signatures = validation.pop("array_signatures")
        cycle_report["validation"] = validation
        cycle_report["array_signature_count"] = len(current_array_signatures)

        if cycle < cycles_total:
            # Persist a roundtrip artifact used as input for the next cycle.
            created_archive, created_member_count = _create_roundtrip_archive(
                pipeline=pipeline,
                output_dir=output_dir,
                cycle=cycle,
                compression=roundtrip_compression,
                baseline_members=baseline_members,
            )
            cycle_report["created_archive"] = str(created_archive)
            cycle_report["created_member_count"] = created_member_count

        cycle_report["status"] = "ok"
        print(
            f"[cycle {cycle}/{cycles_total}] ok "
            f"(zarr={validation['zarr_path_count']}, "
            f"memmaps={validation['memmap_count']}, "
            f"arrays={validation['array_count']})"
        )
        return CycleResult(
            report=cycle_report,
            runtime_cache=runtime_cache,
            array_signatures=current_array_signatures,
            created_archive=created_archive,
        )
    finally:
        if pipeline is not None:
            pipeline.close()
        # Runtime resource cleanup is handled by PipelineManager.close().


@dataclass
class ValidationRunResult:
    """Aggregate result for the full validation run.

    Attributes
    ----------
    cycle_reports : list of dict
        Per-cycle report entries.
    generated_archives : list of pathlib.Path
        Intermediate roundtrip archives produced during the run.
    runtime_caches : list of pathlib.Path
        Runtime cache directories used per cycle.
    reference_array_signatures : dict or None
        Baseline signatures from first successful cycle.
    success : bool
        Global run status.
    error : str or None
        Error message if the run failed.
    """

    cycle_reports: List[Dict[str, Any]]
    generated_archives: List[Path]
    runtime_caches: List[Path]
    reference_array_signatures: Optional[ArraySignatureMap]
    success: bool
    error: Optional[str]


def _execute_validation_cycles(
    *,
    archive_path: Path,
    cycles: int,
    cache_root: Path,
    output_dir: Path,
    roundtrip_compression: str,
    skip_analysis_checks: bool,
    baseline_members: List[str],
) -> ValidationRunResult:
    """Run all validation cycles and aggregate run state.

    Parameters
    ----------
    archive_path : pathlib.Path
        Initial input archive.
    cycles : int
        Number of cycles to execute.
    cache_root : pathlib.Path
        Root directory for runtime caches.
    output_dir : pathlib.Path
        Output directory for generated artifacts.
    roundtrip_compression : str
        Compression mode for roundtrip archives.
    skip_analysis_checks : bool
        If ``True``, skip feature-analysis checks.
    baseline_members : list of str
        Baseline archive members used for parity validation.

    Returns
    -------
    ValidationRunResult
        Aggregate run result.
    """
    cycle_reports: List[Dict[str, Any]] = []
    generated_archives: List[Path] = []
    runtime_caches: List[Path] = []
    reference_array_signatures: Optional[ArraySignatureMap] = None
    current_archive = archive_path

    try:
        for cycle in range(1, cycles + 1):
            cycle_result = _run_single_cycle(
                cycle=cycle,
                cycles_total=cycles,
                current_archive=current_archive,
                cache_root=cache_root,
                output_dir=output_dir,
                roundtrip_compression=roundtrip_compression,
                run_analysis_checks=not skip_analysis_checks,
                baseline_members=baseline_members,
            )

            if reference_array_signatures is None:
                reference_array_signatures = cycle_result.array_signatures
            else:
                _assert_signature_dicts_equal(
                    reference_array_signatures,
                    cycle_result.array_signatures,
                    context=f"cycle={cycle}",
                )

            runtime_caches.append(cycle_result.runtime_cache)
            cycle_reports.append(cycle_result.report)

            if cycle < cycles:
                _require(
                    cycle_result.created_archive is not None,
                    f"Roundtrip archive not created in cycle {cycle}",
                )
                generated_archives.append(cycle_result.created_archive)
                current_archive = cycle_result.created_archive

        _require(
            len(set(runtime_caches)) == len(runtime_caches),
            "Runtime cache directory should be unique for each load cycle.",
        )
        return ValidationRunResult(
            cycle_reports=cycle_reports,
            generated_archives=generated_archives,
            runtime_caches=runtime_caches,
            reference_array_signatures=reference_array_signatures,
            success=True,
            error=None,
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"Validation failed: {error_message}")
        return ValidationRunResult(
            cycle_reports=cycle_reports,
            generated_archives=generated_archives,
            runtime_caches=runtime_caches,
            reference_array_signatures=reference_array_signatures,
            success=False,
            error=error_message,
        )


def _cleanup_generated_archives(generated_archives: List[Path], keep_artifacts: bool) -> None:
    """Delete generated roundtrip archives if retention is disabled.

    Parameters
    ----------
    generated_archives : list of pathlib.Path
        Archives produced during validation cycles.
    keep_artifacts : bool
        When ``True``, skip cleanup.
    """
    if keep_artifacts:
        return
    for archive in generated_archives:
        with contextlib.suppress(FileNotFoundError, OSError):
            archive.unlink()


def _build_report(
    *,
    archive_path: Path,
    cycles_requested: int,
    roundtrip_compression: str,
    skip_analysis_checks: bool,
    keep_artifacts: bool,
    baseline_members: List[str],
    run_result: ValidationRunResult,
) -> Dict[str, Any]:
    """Build final report payload for JSON serialization.

    Parameters
    ----------
    archive_path : pathlib.Path
        Original archive requested by the user.
    cycles_requested : int
        Requested number of validation cycles.
    roundtrip_compression : str
        Compression mode used for generated archives.
    skip_analysis_checks : bool
        Whether analysis checks were skipped.
    keep_artifacts : bool
        Whether generated artifacts were retained.
    baseline_members : list of str
        Baseline archive members.
    run_result : ValidationRunResult
        Aggregate run result with all counters and cycle reports.

    Returns
    -------
    dict
        Final report payload.
    """
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_archive": str(archive_path),
        "cycles_requested": cycles_requested,
        "roundtrip_compression": roundtrip_compression,
        "member_equality_check": True,
        "skip_analysis_checks": skip_analysis_checks,
        "keep_artifacts": keep_artifacts,
        "baseline_member_count": len(baseline_members),
        "reference_array_signature_count": (
            len(run_result.reference_array_signatures)
            if run_result.reference_array_signatures is not None
            else 0
        ),
        "runtime_cache_count": len(run_result.runtime_caches),
        "runtime_cache_unique_count": len(set(run_result.runtime_caches)),
        "success": run_result.success,
        "error": run_result.error,
        "cycles": run_result.cycle_reports,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for archive validation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    examples = (
        "Examples:\n"
        "  python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py\n"
        "  python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py --cycles 5\n"
        "  python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py "
        "--roundtrip-compression gz --cycles 3\n"
        "  python3 tests/manual_tests/validate_pipeline_archive_cross_platform.py "
        "other_archive.tar.zst --cycles 2 --skip-analysis-checks\n"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Cross-platform archive validation: load -> test -> save -> "
            "load -> test (repeatable via --cycles)."
        ),
        epilog=examples,
    )
    parser.add_argument(
        "archive",
        type=Path,
        nargs="?",
        default=DEFAULT_ARCHIVE_PATH,
        help=(
            "Path to pipeline archive (e.g. pipeline.tar.zst). "
            f"Default: {DEFAULT_ARCHIVE_PATH}"
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help=(
            "Number of load/test cycles. Default=3 gives three full "
            "load/test cycles with roundtrip save/load between cycles."
        ),
    )
    parser.add_argument(
        "--roundtrip-compression",
        choices=("zst", "gz", "bz2"),
        default="zst",
        help="Compression for roundtrip archives created during validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/archive_validation"),
        help="Directory for temporary roundtrip artifacts and final report.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional explicit path for JSON validation report.",
    )
    parser.add_argument(
        "--skip-analysis-checks",
        action="store_true",
        help="Skip per-feature compute_mean/compute_std checks.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep generated roundtrip archives.",
    )
    return parser.parse_args()


def _prepare_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """Resolve output/report paths and create required directories.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    tuple
        ``(output_dir, cache_root, report_path)``.
    """
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "runtime_cache_root"
    cache_root.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.report_path.resolve()
        if args.report_path
        else output_dir / f"archive_validation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    return output_dir, cache_root, report_path


def _print_startup_explanation(args: argparse.Namespace, archive_path: Path) -> None:
    """Print an explicit execution explanation at startup.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    archive_path : pathlib.Path
        Resolved input archive path.
    """
    print("=== Archive Validation: Detailed Execution Explanation ===")
    print("Goal: verify that archive load/save works consistently across cycles.")
    print(
        "Input: "
        f"archive={archive_path}, cycles={args.cycles}, "
        f"compression={args.roundtrip_compression}"
    )
    print(f"Default archive path: {DEFAULT_ARCHIVE_PATH}")
    print(
        "Checks per cycle: runtime cache binding, memmap/zarr readability, "
        "array-signature equality, optional analysis checks."
    )
    print(
        "Output: JSON report plus final test marker "
        "(TEST RESULT: PASS/FAIL)."
    )
    print(
        "How to run: python3 "
        "tests/manual_tests/validate_pipeline_archive_cross_platform.py "
        "[archive.tar.zst]"
    )
    print(
        "How to use help: python3 "
        "tests/manual_tests/validate_pipeline_archive_cross_platform.py --help"
    )


def _validate_inputs(archive_path: Path, cycles: int) -> bool:
    """Validate archive path and cycle arguments.

    Parameters
    ----------
    archive_path : pathlib.Path
        Archive path provided by the user.
    cycles : int
        Requested number of validation cycles.

    Returns
    -------
    bool
        ``True`` when inputs are valid, otherwise ``False``.
    """
    if not archive_path.exists():
        print(f"Archive not found: {archive_path}")
        if archive_path == DEFAULT_ARCHIVE_PATH.resolve():
            print(
                "Create it first with: "
                "python3 tests/manual_tests/create_pipeline_archive_from_data.py"
            )
        return False
    if cycles < 1:
        print("--cycles must be >= 1")
        return False
    return True


def _load_baseline_members(archive_path: Path) -> List[str]:
    """Load baseline archive members and validate required files.

    Parameters
    ----------
    archive_path : pathlib.Path
        Input archive to inspect.

    Returns
    -------
    list of str
        Baseline sorted archive members.
    """
    baseline_members = _list_archive_file_members(archive_path)
    _require(
        "pipeline.pkl" in baseline_members,
        "Input archive does not contain required member: pipeline.pkl",
    )
    return baseline_members


def _print_console_summary(report: Dict[str, Any], report_path: Path) -> None:
    """Print a concise end-of-run summary to stdout.

    Parameters
    ----------
    report : dict
        Final validation report payload.
    report_path : pathlib.Path
        Written JSON report path.
    """
    cycles = report.get("cycles", [])
    cycles_requested = int(report.get("cycles_requested", 0))
    cycles_executed = len(cycles)
    cycles_ok = sum(1 for cycle in cycles if cycle.get("status") == "ok")
    status = "SUCCESS" if bool(report.get("success")) else "FAILED"

    print("=== Validation Summary ===")
    print(f"Status: {status}")
    print(
        "Cycles: "
        f"{cycles_ok}/{cycles_requested} ok "
        f"(executed={cycles_executed})"
    )
    print(
        "Runtime caches: "
        f"{report.get('runtime_cache_unique_count', 0)} unique / "
        f"{report.get('runtime_cache_count', 0)} total"
    )

    if cycles:
        last_validation = cycles[-1].get("validation", {})
        print(
            "Last cycle metrics: "
            f"arrays={last_validation.get('array_count', 0)}, "
            f"memmaps={last_validation.get('memmap_count', 0)}, "
            f"zarr={last_validation.get('zarr_path_count', 0)}"
        )

    error = report.get("error")
    if error:
        print(f"Error: {error}")

    print(f"JSON report: {report_path}")
    _print_test_result(bool(report.get("success")))


def _print_test_result(success: bool) -> None:
    """Print a clear and machine-readable final test result line.

    Parameters
    ----------
    success : bool
        Final validation status.
    """
    label = "PASS" if success else "FAIL"
    print(f"TEST RESULT: {label}")
    print(f"TEST_RESULT={label}")


def main() -> int:
    """Run the archive validation CLI workflow.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on failure).
    """
    # Phase 1: Parse input arguments.
    args = _parse_args()
    archive_path = args.archive.resolve()

    # Phase 2: Validate user-provided inputs.
    if not _validate_inputs(archive_path=archive_path, cycles=args.cycles):
        _print_test_result(False)
        return 1

    # Phase 3: Print startup explanation before execution starts.
    _print_startup_explanation(args, archive_path)

    try:
        # Phase 4: Resolve output locations and baseline archive members.
        output_dir, cache_root, report_path = _prepare_paths(args)
        baseline_members = _load_baseline_members(archive_path)
    except Exception as exc:
        print(f"Validation failed before cycle execution: {type(exc).__name__}: {exc}")
        _print_test_result(False)
        return 1

    # Phase 5: Execute load/validate/save cycles.
    run_result = _execute_validation_cycles(
        archive_path=archive_path,
        cycles=args.cycles,
        cache_root=cache_root,
        output_dir=output_dir,
        roundtrip_compression=args.roundtrip_compression,
        skip_analysis_checks=args.skip_analysis_checks,
        baseline_members=baseline_members,
    )
    _cleanup_generated_archives(
        generated_archives=run_result.generated_archives,
        keep_artifacts=args.keep_artifacts,
    )
    # Phase 6: Build and persist final report.
    report = _build_report(
        archive_path=archive_path,
        cycles_requested=args.cycles,
        roundtrip_compression=args.roundtrip_compression,
        skip_analysis_checks=args.skip_analysis_checks,
        keep_artifacts=args.keep_artifacts,
        baseline_members=baseline_members,
        run_result=run_result,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Phase 7: Print concise summary and explicit pass/fail marker.
    _print_console_summary(report, report_path)

    return 0 if run_result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

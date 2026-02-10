#!/usr/bin/env python
"""Validate cross-platform compatibility of a PipelineManager archive."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
import tarfile
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - runtime environment dependent
    np = None  # type: ignore[assignment]

try:
    from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
except ModuleNotFoundError:  # pragma: no cover - runtime environment dependent
    PipelineManager = None  # type: ignore[assignment]


SCOPED_CACHE_PATTERN = re.compile(r"cache_[0-9a-f]{32}_\d{8}_\d{6}$")


def _require(condition: bool, message: str) -> None:
    """Raise AssertionError with a clear message when check fails."""
    if not condition:
        raise AssertionError(message)


def _normalize_archive_name(name: str) -> str:
    return name.replace("\\", "/")


def _list_archive_file_members(archive_path: Path) -> List[str]:
    """Return sorted normalized file member names in archive."""
    members: List[str] = []
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.isfile():
                members.append(_normalize_archive_name(member.name))
    return sorted(members)


def _to_jsonable(value: Any) -> Any:
    """Convert values to JSON-compatible primitives."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _collect_numpy_arrays_recursive(root: object) -> List[Tuple[str, np.ndarray]]:
    """Collect all numpy arrays (including memmaps) from object graph."""
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
    """Create a stable signature for array correctness comparison."""
    arr = np.asarray(array)
    flat = arr.reshape(-1)
    size = int(flat.size)
    sample_size = min(sample_points, size)
    sample_indices = (
        np.linspace(0, size - 1, num=sample_size, dtype=np.int64)
        if size > 0
        else np.array([], dtype=np.int64)
    )
    sample = flat[sample_indices] if sample_size > 0 else flat

    if sample.dtype.hasobject:
        sample_payload = "|".join(repr(item) for item in sample.tolist()).encode("utf-8")
    else:
        sample_payload = np.ascontiguousarray(sample).view(np.uint8).tobytes()
    sample_hash = hashlib.sha256(sample_payload).hexdigest()

    numeric = np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)
    stats: Dict[str, Any] = {}
    if numeric and sample_size > 0:
        sample_float = np.asarray(sample, dtype=np.float64)
        stats = {
            "sample_mean": float(np.mean(sample_float)),
            "sample_std": float(np.std(sample_float)),
            "sample_min": float(np.min(sample_float)),
            "sample_max": float(np.max(sample_float)),
            "sample_all_finite": bool(np.all(np.isfinite(sample_float))),
        }

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


def _capture_array_signatures(root: object) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """Capture signatures for all numpy arrays in the object graph."""
    signatures: Dict[str, Dict[str, Any]] = {}
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
    reference: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
    *,
    context: str,
) -> None:
    """Assert that two array signature dictionaries are identical."""
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


def _iter_feature_entries(feature_data: Any) -> Iterator[Tuple[str, Any, Any]]:
    """Yield (feature_name, entry_key, entry_obj) for feature entries."""
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
    """Run analysis checks on all feature entries and return counters."""
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
    """Validate runtime bindings, zarr/memmap/array access and analysis outputs."""
    runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
    _require(runtime_cache.exists(), f"Runtime cache does not exist: {runtime_cache}")
    _require(runtime_cache.is_dir(), f"Runtime cache is not a directory: {runtime_cache}")
    _require(
        SCOPED_CACHE_PATTERN.fullmatch(runtime_cache.name) is not None,
        (
            "Runtime cache directory does not match expected scoped pattern "
            f"'cache_<uuid>_<timestamp>': {runtime_cache.name}"
        ),
    )

    data = pipeline._data
    arrays_with_path = _collect_numpy_arrays_recursive(data)
    array_signatures, memmap_count, ndarray_count = _capture_array_signatures(data)

    for _, array in arrays_with_path:
        if not isinstance(array, np.memmap):
            continue
        memmap_array = array
        memmap_path = Path(memmap_array.filename).resolve()
        _require(
            runtime_cache in memmap_path.parents,
            (
                "Memmap path is not bound to runtime cache. "
                f"runtime_cache={runtime_cache}, memmap={memmap_path}"
            ),
        )
        _require(memmap_path.exists(), f"Memmap file missing: {memmap_path}")
        flat_view = np.asarray(memmap_array).reshape(-1)
        if flat_view.size > 0:
            _ = flat_view[0]

    trajectories = list(getattr(data.trajectory_data, "trajectories", []))
    zarr_paths: List[str] = []
    zarr_xyz_samples = 0
    for traj in trajectories:
        zarr_cache_path = getattr(traj, "zarr_cache_path", None)
        if not zarr_cache_path:
            continue
        zarr_path = Path(zarr_cache_path).resolve()
        zarr_paths.append(str(zarr_path))
        _require(zarr_path.exists(), f"zarr cache path missing: {zarr_path}")
        _require(
            runtime_cache in zarr_path.parents,
            (
                "zarr cache path is not bound to runtime cache. "
                f"runtime_cache={runtime_cache}, zarr_path={zarr_path}"
            ),
        )
        xyz = getattr(traj, "xyz", None)
        if xyz is not None:
            try:
                xyz_first_frame = np.asarray(xyz[0]).reshape(-1)
            except Exception:
                xyz_first_frame = np.asarray(xyz).reshape(-1)
            if xyz_first_frame.size > 0:
                _ = xyz_first_frame[0]
            zarr_xyz_samples += 1

    analysis_summary = {
        "analysis_mean_checks": 0,
        "analysis_std_checks": 0,
    }
    if run_analysis_checks:
        analysis_summary = _validate_feature_analysis(pipeline)

    return {
        "runtime_cache": str(runtime_cache),
        "trajectory_count": len(trajectories),
        "zarr_path_count": len(zarr_paths),
        "zarr_xyz_samples": zarr_xyz_samples,
        "array_count": len(array_signatures),
        "memmap_count": memmap_count,
        "ndarray_count": ndarray_count,
        "array_signatures": array_signatures,
        **analysis_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-platform archive validation: load -> test -> save -> "
            "load -> test (repeatable via --cycles)."
        )
    )
    parser.add_argument(
        "archive",
        type=Path,
        help="Path to pipeline archive (e.g. pipeline.tar.xz).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help=(
            "Number of load/test cycles. Default=2 gives "
            "load->test->save->load->test."
        ),
    )
    parser.add_argument(
        "--roundtrip-compression",
        choices=("xz", "gz", "bz2"),
        default="gz",
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


def main() -> int:
    args = parse_args()

    if np is None:
        print("Missing dependency: numpy. Please run this script in the mdxplain environment.")
        return 1
    if PipelineManager is None:
        print(
            "Missing dependency: mdxplain package import failed. "
            "Run this script from the project environment where mdxplain is importable."
        )
        return 1

    archive_path = args.archive.resolve()
    if not archive_path.exists():
        print(f"Archive not found: {archive_path}")
        return 1
    if args.cycles < 1:
        print("--cycles must be >= 1")
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "runtime_cache_root"
    cache_root.mkdir(parents=True, exist_ok=True)

    report_path = (
        args.report_path.resolve()
        if args.report_path
        else output_dir / f"archive_validation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    )

    baseline_members = _list_archive_file_members(archive_path)
    _require(
        "pipeline.pkl" in baseline_members,
        "Input archive does not contain required member: pipeline.pkl",
    )

    cycle_reports: List[Dict[str, Any]] = []
    generated_archives: List[Path] = []
    runtime_caches: List[Path] = []
    current_archive = archive_path
    reference_array_signatures: Optional[Dict[str, Dict[str, Any]]] = None

    success = False
    error_message = None

    try:
        for cycle in range(1, args.cycles + 1):
            print(f"[cycle {cycle}/{args.cycles}] loading archive: {current_archive}")
            pipeline = None
            runtime_cache = None
            created_archive = None
            cycle_report: Dict[str, Any] = {
                "cycle": cycle,
                "input_archive": str(current_archive),
            }

            try:
                pipeline = PipelineManager.load_from_archive(
                    str(current_archive),
                    cache_dir=str(cache_root),
                    show_progress=False,
                )
                validation = _validate_loaded_pipeline(
                    pipeline=pipeline,
                    run_analysis_checks=not args.skip_analysis_checks,
                )
                runtime_cache = Path(validation["runtime_cache"]).resolve()
                runtime_caches.append(runtime_cache)
                current_array_signatures = validation.pop("array_signatures")
                if reference_array_signatures is None:
                    reference_array_signatures = current_array_signatures
                else:
                    _assert_signature_dicts_equal(
                        reference_array_signatures,
                        current_array_signatures,
                        context=f"cycle={cycle}",
                    )

                cycle_report["validation"] = validation
                cycle_report["array_signature_count"] = len(current_array_signatures)

                if cycle < args.cycles:
                    roundtrip_base = output_dir / f"roundtrip_cycle_{cycle}"
                    created_archive = Path(
                        pipeline.create_sharable_archive(
                            str(roundtrip_base),
                            compression=args.roundtrip_compression,
                        )
                    ).resolve()
                    generated_archives.append(created_archive)
                    cycle_report["created_archive"] = str(created_archive)

                    created_members = _list_archive_file_members(created_archive)
                    cycle_report["created_member_count"] = len(created_members)
                    _require(
                        created_members == baseline_members,
                        (
                            "Archive member mismatch after roundtrip. "
                            f"cycle={cycle}, created={created_archive}"
                        ),
                    )

                cycle_report["status"] = "ok"
                print(
                    f"[cycle {cycle}/{args.cycles}] ok "
                    f"(zarr={validation['zarr_path_count']}, "
                    f"memmaps={validation['memmap_count']}, "
                    f"arrays={validation['array_count']})"
                )
            finally:
                if pipeline is not None:
                    pipeline.close()
                # Runtime resource cleanup is handled by PipelineManager.close().

            cycle_reports.append(cycle_report)

            if cycle < args.cycles:
                if created_archive is None:
                    raise RuntimeError(
                        f"Roundtrip archive not created in cycle {cycle}"
                    )
                current_archive = created_archive

        _require(
            len(set(runtime_caches)) == len(runtime_caches),
            "Runtime cache directory should be unique for each load cycle.",
        )
        success = True
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"Validation failed: {error_message}")

    if (not args.keep_artifacts) and generated_archives:
        for archive in generated_archives:
            with contextlib.suppress(FileNotFoundError, OSError):
                archive.unlink()

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_archive": str(archive_path),
        "cycles_requested": args.cycles,
        "roundtrip_compression": args.roundtrip_compression,
        "member_equality_check": True,
        "skip_analysis_checks": args.skip_analysis_checks,
        "keep_artifacts": args.keep_artifacts,
        "baseline_member_count": len(baseline_members),
        "reference_array_signature_count": (
            len(reference_array_signatures)
            if reference_array_signatures is not None
            else 0
        ),
        "runtime_cache_count": len(runtime_caches),
        "runtime_cache_unique_count": len(set(runtime_caches)),
        "success": success,
        "error": error_message,
        "cycles": cycle_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to: {report_path}")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

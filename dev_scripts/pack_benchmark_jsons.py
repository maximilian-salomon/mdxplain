#!/usr/bin/env python
"""Package benchmark JSON outputs from benchmark scripts into a zip archive."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import re
import zipfile


DEFAULT_BENCHMARK_SCRIPTS = (
    Path("dev_scripts/benchmark_fast_standard.py"),
    Path("dev_scripts/benchmark_iterative.py"),
    Path("dev_scripts/benchmark_standard_full.py"),
)
RESULTS_DIR_PATTERN = re.compile(r'results_dir\s*=\s*Path\("([^"]+)"\)')


def _extract_results_dir_from_script(script_path: Path) -> Path:
    """Extract `results_dir = Path(\"...\")` from a benchmark script."""
    content = script_path.read_text(encoding="utf-8")
    match = RESULTS_DIR_PATTERN.search(content)
    if match is None:
        raise ValueError(
            f"Could not find results_dir assignment in benchmark script: {script_path}"
        )
    return Path(match.group(1))


def _discover_results_dirs(project_root: Path, benchmark_scripts: list[Path]) -> list[Path]:
    """Resolve benchmark result directories from benchmark scripts."""
    result_dirs: list[Path] = []
    seen_dirs = set()
    for script in benchmark_scripts:
        script_abs = (project_root / script).resolve()
        if not script_abs.exists():
            raise FileNotFoundError(f"Benchmark script not found: {script_abs}")

        results_dir_rel = _extract_results_dir_from_script(script_abs)
        results_dir_abs = (project_root / results_dir_rel).resolve()
        if results_dir_abs in seen_dirs:
            continue
        seen_dirs.add(results_dir_abs)
        result_dirs.append(results_dir_abs)
    return result_dirs


def _discover_json_files(project_root: Path, result_dirs: list[Path]) -> list[Path]:
    """Collect JSON files from resolved benchmark result directories."""
    json_files: list[Path] = []
    seen_files = set()
    for result_dir in result_dirs:
        if not result_dir.exists():
            continue
        for json_path in sorted(result_dir.rglob("*.json")):
            if not json_path.is_file():
                continue
            resolved = json_path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            json_files.append(json_path)
    return json_files


def _default_output_path(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / f"benchmark_jsons_{timestamp}.zip"


def _build_manifest(
    project_root: Path,
    benchmark_scripts: list[Path],
    result_dirs: list[Path],
    json_files: list[Path],
) -> dict:
    """Create metadata for reproducible packaging."""
    entries = []
    total_size_bytes = 0
    for path in json_files:
        size = path.stat().st_size
        total_size_bytes += size
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size_bytes": size,
            }
        )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "benchmark_scripts": [str(path) for path in benchmark_scripts],
        "result_dirs": [path.relative_to(project_root).as_posix() for path in result_dirs],
        "file_count": len(entries),
        "total_size_bytes": total_size_bytes,
        "files": entries,
    }


def _write_zip(
    project_root: Path,
    output_zip: Path,
    json_files: list[Path],
    manifest: dict,
) -> None:
    """Write JSON files and manifest into a zip archive."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for path in json_files:
            arcname = path.relative_to(project_root).as_posix()
            zf.write(path, arcname=arcname)
        zf.writestr("benchmark_json_manifest.json", json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect benchmark JSON outputs from benchmark scripts and pack "
            "them into one zip file."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing benchmark result directories (default: .)",
    )
    parser.add_argument(
        "--benchmark-script",
        action="append",
        dest="benchmark_scripts",
        help=(
            "Path to benchmark script that defines `results_dir = Path(...)`. "
            "Can be set multiple times. Defaults to the 3 benchmark scripts in dev_scripts."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output zip path. Default: benchmark_jsons_<timestamp>.zip in "
            "project root."
        ),
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success even if no JSON files are found.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be packed; do not write zip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    benchmark_scripts = (
        [Path(p) for p in args.benchmark_scripts]
        if args.benchmark_scripts
        else list(DEFAULT_BENCHMARK_SCRIPTS)
    )
    output_zip = args.output.resolve() if args.output else _default_output_path(project_root)

    result_dirs = _discover_results_dirs(
        project_root=project_root,
        benchmark_scripts=benchmark_scripts,
    )
    json_files = _discover_json_files(
        project_root=project_root,
        result_dirs=result_dirs,
    )
    if not json_files:
        message = (
            "No benchmark JSON files found.\n"
            f"Project root: {project_root}\n"
            f"Benchmark scripts: {[str(p) for p in benchmark_scripts]}\n"
            f"Resolved result dirs: {[str(p) for p in result_dirs]}"
        )
        if args.allow_empty:
            print(message)
            return 0
        print(message)
        return 1

    manifest = _build_manifest(
        project_root=project_root,
        benchmark_scripts=benchmark_scripts,
        result_dirs=result_dirs,
        json_files=json_files,
    )

    print(f"Found {len(json_files)} JSON files.")
    if args.dry_run:
        for entry in manifest["files"]:
            print(f"  - {entry['path']} ({entry['size_bytes']} bytes)")
        print(f"Dry run complete. Would create: {output_zip}")
        return 0

    _write_zip(
        project_root=project_root,
        output_zip=output_zip,
        json_files=json_files,
        manifest=manifest,
    )
    archive_size = output_zip.stat().st_size
    print(f"Created zip: {output_zip}")
    print(f"Archive size: {archive_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

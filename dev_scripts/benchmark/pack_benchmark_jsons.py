# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Benchmark JSON packaging script.
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

"""Package benchmark JSON outputs into one reproducible zip archive.

The script discovers benchmark result directories, collects all ``*.json``
files, and writes a compressed archive including a machine-readable manifest.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/pack_benchmark_jsons.py``
- ``python dev_scripts/benchmark/pack_benchmark_jsons.py --output benchmark/results.zip``
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import zipfile


RESULTS_DIR_PATTERN = re.compile(r'results_dir\s*=\s*Path\("([^"]+)"\)')
DEFAULT_BENCHMARK_SCRIPTS = [
    Path("dev_scripts/benchmark/benchmark_approx_memmap.py"),
    Path("dev_scripts/benchmark/benchmark_exact_memmap.py"),
    Path("dev_scripts/benchmark/benchmark_exact_ram.py"),
]


def _extract_results_dir_from_script(script_path: Path) -> Path:
    """Extract results directory assignment from one benchmark script file.

    Parameters
    ----------
    script_path : Path
        Path to a benchmark profile script containing ``results_dir``.

    Returns
    -------
    Path
        Relative results directory parsed from script source.

    Notes
    -----
    The parser expects the assignment form ``results_dir = Path("...")``.
    """
    # Read script source and extract configured results directory.
    content = script_path.read_text(encoding="utf-8")
    match = RESULTS_DIR_PATTERN.search(content)

    # Fail early when scripts do not expose a parseable results directory.
    if match is None:
        raise ValueError(f"Could not find results_dir assignment in {script_path}")
    return Path(match.group(1))


def _discover_result_dirs(project_root: Path, benchmark_scripts: list[Path]) -> list[Path]:
    """Resolve unique absolute result directories from benchmark script paths.

    Parameters
    ----------
    project_root : Path
        Project root used to resolve relative script and results paths.
    benchmark_scripts : list[Path]
        Benchmark scripts expected to define ``results_dir`` assignments.

    Returns
    -------
    list[Path]
        Unique absolute result directories in stable insertion order.

    Notes
    -----
    Missing script files raise ``FileNotFoundError``.
    """
    # Resolve directories once and keep insertion order stable.
    result_dirs: list[Path] = []
    seen_dirs: set[Path] = set()

    for script in benchmark_scripts:
        script_abs = (project_root / script).resolve()
        if not script_abs.exists():
            raise FileNotFoundError(f"Benchmark script not found: {script_abs}")

        results_rel = _extract_results_dir_from_script(script_abs)
        results_abs = (project_root / results_rel).resolve()
        if results_abs in seen_dirs:
            continue

        seen_dirs.add(results_abs)
        result_dirs.append(results_abs)

    return result_dirs


def _discover_json_files(result_dirs: list[Path]) -> list[Path]:
    """Collect unique JSON files recursively from resolved result directories.

    Parameters
    ----------
    result_dirs : list[Path]
        Absolute result directories discovered from benchmark scripts.

    Returns
    -------
    list[Path]
        Sorted JSON file paths with duplicates removed.

    Notes
    -----
    Non-existing result directories are skipped silently.
    """
    # Walk each directory recursively and collect unique JSON paths.
    json_files: list[Path] = []
    seen_files: set[Path] = set()

    for result_dir in result_dirs:
        if not result_dir.exists():
            continue
        for json_path in sorted(result_dir.rglob("*.json")):
            resolved = json_path.resolve()
            if (not json_path.is_file()) or (resolved in seen_files):
                continue
            seen_files.add(resolved)
            json_files.append(json_path)

    return json_files


def _default_output_zip(project_root: Path) -> Path:
    """Build timestamped default output zip path under project root.

    Parameters
    ----------
    project_root : Path
        Root directory where default output archive should be created.

    Returns
    -------
    Path
        Default archive path using current local timestamp.

    Notes
    -----
    Filename format is ``benchmark_jsons_YYYYMMDD_HHMMSS.zip``.
    """
    # Build deterministic timestamp pattern used across benchmark utilities.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / f"benchmark_jsons_{timestamp}.zip"


def _build_manifest(
    project_root: Path,
    benchmark_scripts: list[Path],
    result_dirs: list[Path],
    json_files: list[Path],
) -> dict[str, object]:
    """Create archive manifest with file metadata and provenance information.

    Parameters
    ----------
    project_root : Path
        Project root used for relative path generation.
    benchmark_scripts : list[Path]
        Benchmark scripts that defined the packed result directories.
    result_dirs : list[Path]
        Resolved result directories searched for JSON files.
    json_files : list[Path]
        Discovered JSON files included in the output archive.

    Returns
    -------
    dict[str, object]
        Manifest dictionary serialized into the zip archive.

    Notes
    -----
    All file paths in the manifest are stored relative to ``project_root``.
    """
    # Summarize file-level metadata for reproducible archive introspection.
    entries: list[dict[str, object]] = []
    total_size_bytes = 0
    for path in json_files:
        size_bytes = path.stat().st_size
        total_size_bytes += size_bytes
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size_bytes": size_bytes,
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


def _write_archive(output_zip: Path, project_root: Path, json_files: list[Path], manifest: dict[str, object]) -> None:
    """Write JSON files and manifest into output zip archive.

    Parameters
    ----------
    output_zip : Path
        Target archive path.
    project_root : Path
        Project root used for relative archive member names.
    json_files : list[Path]
        JSON files added to the archive.
    manifest : dict[str, object]
        Manifest metadata written as ``benchmark_json_manifest.json``.

    Returns
    -------
    None
        Archive is created on disk.

    Notes
    -----
    Compression mode is ``ZIP_DEFLATED`` with ``compresslevel=9``.
    """
    # Ensure output folder exists before creating compressed archive.
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # Write discovered JSON files and manifest in one archive transaction.
    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in json_files:
            arcname = path.relative_to(project_root).as_posix()
            archive.write(path, arcname=arcname)
        archive.writestr("benchmark_json_manifest.json", json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark JSON packaging.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed CLI options controlling discovery and archive output.

    Notes
    -----
    When ``--benchmark-script`` is omitted, default profile scripts are used.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/pack_benchmark_jsons.py --dry-run
    """
    # Configure CLI for benchmark JSON discovery and zip packaging.
    parser = argparse.ArgumentParser(description="Package benchmark JSON outputs into one zip archive.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--benchmark-script", action="append", dest="benchmark_scripts", help="Benchmark script path defining results_dir.")
    parser.add_argument("--output", type=Path, default=None, help="Output zip path. Defaults to timestamped archive.")
    parser.add_argument("--allow-empty", action="store_true", help="Exit with success when no JSON files are found.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovered files without writing archive.")
    return parser.parse_args()


def _resolve_benchmark_scripts(args: argparse.Namespace) -> list[Path]:
    """Resolve benchmark scripts from CLI arguments or defaults.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace from :func:`parse_args`.

    Returns
    -------
    list[Path]
        Benchmark script paths used for result directory discovery.

    Notes
    -----
    Returned paths are relative, resolved later against project root.
    """
    # Choose user-supplied benchmark scripts or known defaults.
    if args.benchmark_scripts:
        return [Path(value) for value in args.benchmark_scripts]
    return list(DEFAULT_BENCHMARK_SCRIPTS)


def _print_empty_message(project_root: Path, benchmark_scripts: list[Path], result_dirs: list[Path]) -> None:
    """Print standardized message for empty benchmark JSON discovery.

    Parameters
    ----------
    project_root : Path
        Resolved project root path.
    benchmark_scripts : list[Path]
        Benchmark scripts used for discovery.
    result_dirs : list[Path]
        Result directories resolved from benchmark scripts.

    Returns
    -------
    None
        Message is printed to stdout.

    Notes
    -----
    Caller decides whether this state is treated as success or failure.
    """
    # Keep diagnostics explicit to simplify troubleshooting in CI runs.
    print("No benchmark JSON files found.")
    print(f"Project root: {project_root}")
    print(f"Benchmark scripts: {[str(path) for path in benchmark_scripts]}")
    print(f"Resolved result dirs: {[str(path) for path in result_dirs]}")


def main() -> int:
    """Collect benchmark JSON files and write them into a zip archive.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Exit code ``0`` on success, non-zero on failure.

    Notes
    -----
    ``--dry-run`` prints file lists and skips archive creation.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/pack_benchmark_jsons.py --output benchmark/archive/results.zip
    """
    # Parse CLI configuration and normalize path-like settings.
    args = parse_args()
    project_root = args.project_root.resolve()
    benchmark_scripts = _resolve_benchmark_scripts(args)

    # Discover result directories and JSON files for packaging.
    result_dirs = _discover_result_dirs(project_root, benchmark_scripts)
    json_files = _discover_json_files(result_dirs)
    if not json_files:
        _print_empty_message(project_root, benchmark_scripts, result_dirs)
        return 0 if args.allow_empty else 1

    # Build manifest once so dry-run and archive path share exact metadata.
    output_zip = args.output.resolve() if args.output else _default_output_zip(project_root)
    manifest = _build_manifest(project_root, benchmark_scripts, result_dirs, json_files)
    print(f"Found {len(json_files)} JSON files.")

    # Support non-writing dry run mode for CI previews and debugging.
    if args.dry_run:
        for entry in manifest["files"]:
            print(f"  - {entry['path']} ({entry['size_bytes']} bytes)")
        print(f"Dry run complete. Would create: {output_zip}")
        return 0

    # Write archive and print resulting artifact details.
    _write_archive(output_zip, project_root, json_files, manifest)
    print(f"Created zip: {output_zip}")
    print(f"Archive size: {output_zip.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

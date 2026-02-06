#!/usr/bin/env python
# Benchmark: create stacked trajectories + run standard pipeline (with plots).

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mdxplain import PipelineManager


def _get_rss_bytes() -> int:
    try:
        import psutil  # noqa: WPS433

        process = psutil.Process(os.getpid())
        total_rss = int(process.memory_info().rss)
        for child in process.children(recursive=True):
            try:
                total_rss += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_rss
    except Exception:
        pass

    if os.name == "nt":
        import ctypes  # noqa: WPS433
        import ctypes.wintypes as wt  # noqa: WPS433

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", wt.SIZE_T),
                ("WorkingSetSize", wt.SIZE_T),
                ("QuotaPeakPagedPoolUsage", wt.SIZE_T),
                ("QuotaPagedPoolUsage", wt.SIZE_T),
                ("QuotaPeakNonPagedPoolUsage", wt.SIZE_T),
                ("QuotaNonPagedPoolUsage", wt.SIZE_T),
                ("PagefileUsage", wt.SIZE_T),
                ("PeakPagefileUsage", wt.SIZE_T),
            ]

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wt.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wt.DWORD,
        ]
        get_process_memory_info.restype = wt.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        if not get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.WorkingSetSize)

    return 0


def _read_smaps_rollup_kb(pid: int) -> dict[str, int]:
    path = Path(f"/proc/{pid}/smaps_rollup")
    if not path.exists():
        return {}

    stats: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", maxsplit=1)
            value_str = rest.strip().split()[0]
            stats[key] = int(value_str)
    except (OSError, ValueError):
        return {}
    return stats


def _get_non_cache_bytes() -> int:
    # Linux-only approximation of "pressure memory":
    # subtract clean file-backed pages (reclaimable cache) from RSS.
    if not sys.platform.startswith("linux"):
        return _get_rss_bytes()

    try:
        import psutil  # noqa: WPS433

        process = psutil.Process(os.getpid())
        all_processes = [process] + process.children(recursive=True)
    except Exception:
        all_processes = []

    total_bytes = 0
    found_any = False
    for proc in all_processes:
        try:
            stats = _read_smaps_rollup_kb(proc.pid)
        except Exception:
            continue
        if not stats:
            continue
        found_any = True
        rss_kb = int(stats.get("Rss", 0))
        clean_file_kb = int(stats.get("Shared_Clean", 0)) + int(stats.get("Private_Clean", 0))
        non_cache_kb = max(0, rss_kb - clean_file_kb)
        total_bytes += non_cache_kb * 1024

    if found_any:
        return total_bytes
    return _get_rss_bytes()


_CGROUP_MEMORY_FILES: Optional[Tuple[Optional[Path], Optional[Path]]] = None


def _read_int_from_file(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
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


def _get_mem_available_bytes() -> Optional[int]:
    if not sys.platform.startswith("linux"):
        return None

    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None

    try:
        for line in meminfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith("MemAvailable:"):
                continue
            value_kb = int(line.split()[1])
            return value_kb * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _resolve_cgroup_memory_files() -> Tuple[Optional[Path], Optional[Path]]:
    if not sys.platform.startswith("linux"):
        return None, None

    cgroup_file = Path("/proc/self/cgroup")
    if not cgroup_file.exists():
        return None, None

    try:
        lines = cgroup_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None, None

    # cgroup v2
    rel_path_v2: Optional[str] = None
    for line in lines:
        parts = line.split(":")
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            rel_path_v2 = parts[2].strip()
            break
    if rel_path_v2 is not None:
        base = Path("/sys/fs/cgroup")
        group_dir = base / rel_path_v2.lstrip("/")
        current_candidates = [group_dir / "memory.current", base / "memory.current"]
        limit_candidates = [group_dir / "memory.max", base / "memory.max"]
        current = next((p for p in current_candidates if p.exists()), None)
        limit = next((p for p in limit_candidates if p.exists()), None)
        if current is not None or limit is not None:
            return current, limit

    # cgroup v1 (memory controller)
    rel_path_v1: Optional[str] = None
    for line in lines:
        parts = line.split(":")
        if len(parts) != 3:
            continue
        controllers = parts[1].split(",")
        if "memory" in controllers:
            rel_path_v1 = parts[2].strip()
            break
    if rel_path_v1 is not None:
        base = Path("/sys/fs/cgroup/memory")
        group_dir = base / rel_path_v1.lstrip("/")
        current_candidates = [group_dir / "memory.usage_in_bytes", base / "memory.usage_in_bytes"]
        limit_candidates = [group_dir / "memory.limit_in_bytes", base / "memory.limit_in_bytes"]
        current = next((p for p in current_candidates if p.exists()), None)
        limit = next((p for p in limit_candidates if p.exists()), None)
        if current is not None or limit is not None:
            return current, limit

    return None, None


def _get_cgroup_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    global _CGROUP_MEMORY_FILES

    if _CGROUP_MEMORY_FILES is None:
        _CGROUP_MEMORY_FILES = _resolve_cgroup_memory_files()

    current_path, limit_path = _CGROUP_MEMORY_FILES
    current = _read_int_from_file(current_path)
    limit = _read_int_from_file(limit_path)

    # Some kernels expose a very large sentinel value as "unlimited".
    if limit is not None and limit >= (1 << 60):
        limit = None
    return current, limit


def _bytes_to_mb(value: int) -> float:
    return value / (1024 * 1024)


def _bytes_to_mb_optional(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return _bytes_to_mb(value)


def _fmt_optional_mb(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


class MemorySampler:
    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.max_rss = 0
        self.max_non_cache = 0
        self.min_mem_available: Optional[int] = None
        self.max_cgroup_current = 0
        self.cgroup_limit: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _get_rss_bytes()
            if rss > self.max_rss:
                self.max_rss = rss
            non_cache = _get_non_cache_bytes()
            if non_cache > self.max_non_cache:
                self.max_non_cache = non_cache
            mem_available = _get_mem_available_bytes()
            if mem_available is not None:
                if self.min_mem_available is None or mem_available < self.min_mem_available:
                    self.min_mem_available = mem_available
            cgroup_current, cgroup_limit = _get_cgroup_memory_bytes()
            if cgroup_current is not None and cgroup_current > self.max_cgroup_current:
                self.max_cgroup_current = cgroup_current
            if cgroup_limit is not None:
                self.cgroup_limit = cgroup_limit
            time.sleep(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        self.max_rss = _get_rss_bytes()
        self.max_non_cache = _get_non_cache_bytes()
        self.min_mem_available = _get_mem_available_bytes()
        cgroup_current, cgroup_limit = _get_cgroup_memory_bytes()
        self.max_cgroup_current = cgroup_current or 0
        self.cgroup_limit = cgroup_limit
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _close_fig(obj) -> None:
    if isinstance(obj, Figure):
        plt.close(obj)


@dataclass
class StepResult:
    name: str
    seconds: float
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    non_cache_start_mb: float
    non_cache_end_mb: float
    non_cache_peak_mb: float
    mem_available_start_mb: Optional[float]
    mem_available_end_mb: Optional[float]
    mem_available_min_mb: Optional[float]
    cgroup_current_start_mb: Optional[float]
    cgroup_current_end_mb: Optional[float]
    cgroup_current_peak_mb: Optional[float]
    cgroup_limit_mb: Optional[float]
    cache_size_mb: float


def _run_pipeline(dataset_dir: Path, cache_dir: Path, results_path: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PipelineManager(
        use_memmap=True,
        chunk_size=2000,
        show_progress=False,
        cache_dir=str(cache_dir),
    )

    output_root = results_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    steps: List[tuple[str, Callable[[], None]]] = []

    def add_step(name: str, func: Callable[[], None]) -> None:
        steps.append((name, func))

    # Standard pipeline (from tutorials/02_VillinHeadpiece_Full_Analysis.ipynb)
    add_step("load_trajectories", lambda: pipeline.trajectory.load_trajectories(str(dataset_dir)))
    add_step("add_labels", lambda: pipeline.trajectory.add_labels(traj_selection="all"))
    add_step("add_distances", lambda: pipeline.feature.add.distances())
    add_step("add_contacts", lambda: pipeline.feature.add.contacts(cutoff=4.5))
    add_step("create_selector", lambda: pipeline.feature_selector.create("contacts_only"))
    add_step("select_contacts", lambda: pipeline.feature_selector.add.contacts("contacts_only", "all"))
    add_step("apply_selector", lambda: pipeline.feature_selector.select("contacts_only"))
    add_step(
        "contact_kernel_pca",
        lambda: pipeline.decomposition.add.contact_kernel_pca(
            n_components=4,
            selection_name="contacts_only",
            decomposition_name="ContactKernelPCA",
            use_nystrom=True,
            n_landmarks=2000,
        ),
    )
    add_step(
        "dpa",
        lambda: pipeline.clustering.add.dpa(
            "ContactKernelPCA",
            Z=2.5,
            metric="euclidean",
            affinity="nearest_neighbors",
            cluster_name="DPA_ContactKernelPCA",
            method="knn_sampling",
        ),
    )
    add_step(
        "data_selector",
        lambda: pipeline.data_selector.create_from_clusters(
            group_name="cluster", clustering_name="DPA_ContactKernelPCA"
        ),
    )
    add_step(
        "comparison",
        lambda: pipeline.comparison.create_comparison(
            name="cluster_comparison",
            mode="one_vs_rest",
            feature_selector="contacts_only",
            data_selector_groups="cluster",
        ),
    )
    add_step(
        "feature_importance",
        lambda: pipeline.feature_importance.add.decision_tree(
            comparison_name="cluster_comparison",
            analysis_name="feature_importance",
            max_samples=100000,
        ),
    )
    add_step(
        "plot_decision_trees",
        lambda: pipeline.plots.feature_importance.decision_trees(
            feature_importance_name="feature_importance",
            save_fig=True,
            render=False,
            short_layout=True,
            hide_path=False,
            separate_trees=True,
            width_scale_factor=0.4,
            height_scale_factor=0.5,
            edge_symbol_fontsize=18,
            file_format="svg",
        ),
    )
    add_step(
        "print_top_features",
        lambda: pipeline.feature_importance.print_top_n_features(
            analysis_name="feature_importance"
        ),
    )
    add_step(
        "plot_violins",
        lambda: _close_fig(
            pipeline.plots.feature_importance.violins(
                feature_importance_name="feature_importance",
                n_top=2,
                save_fig=True,
                max_cols=7,
                file_format="svg",
                tick_fontsize=26,
                ylabel_fontsize=26,
                subplot_title_fontsize=26,
            )
        ),
    )
    add_step(
        "plot_densities",
        lambda: _close_fig(
            pipeline.plots.feature_importance.densities(
                feature_importance_name="feature_importance",
                n_top=2,
                save_fig=True,
                max_cols=7,
                file_format="svg",
                tick_fontsize=18,
                ylabel_fontsize=18,
                subplot_title_fontsize=18,
            )
        ),
    )
    add_step(
        "plot_time_series",
        lambda: _close_fig(
            pipeline.plots.feature_importance.time_series(
                feature_importance_name="feature_importance",
                n_top=2,
                save_fig=True,
                max_cols=3,
                membership_per_feature=True,
                clustering_name="DPA_ContactKernelPCA",
                file_format="svg",
                tick_fontsize=26,
                ylabel_fontsize=26,
                xlabel_fontsize=26,
                subplot_title_fontsize=26,
            )
        ),
    )
    add_step(
        "plot_membership",
        lambda: _close_fig(
            pipeline.plots.clustering.membership(
                clustering_name="DPA_ContactKernelPCA",
                save_fig=True,
                file_format="svg",
                tick_fontsize=18,
                xlabel_fontsize=18,
                ylabel_fontsize=18,
            )
        ),
    )
    add_step(
        "plot_landscape",
        lambda: _close_fig(
            pipeline.plots.landscape(
                decomposition_name="ContactKernelPCA",
                dimensions=[0, 1, 2, 3],
                clustering_name="DPA_ContactKernelPCA",
                save_fig=True,
                file_format="svg",
                tick_fontsize=26,
                xlabel_fontsize=26,
                ylabel_fontsize=26,
                contour_label_fontsize=26,
            )
        ),
    )
    add_step(
        "beta_factor_pdb",
        lambda: pipeline.structure_visualization.feature_importance.create_pdb_with_beta_factors(
            structure_viz_name="structure_viz",
            feature_importance_name="feature_importance",
        ),
    )
    add_step(
        "create_archive",
        lambda: pipeline.create_sharable_archive(str(output_root / "pipeline.tar.xz")),
    )

    results: List[StepResult] = []

    for name, func in steps:
        rss_start = _get_rss_bytes()
        non_cache_start = _get_non_cache_bytes()
        mem_available_start = _get_mem_available_bytes()
        cgroup_current_start, cgroup_limit_start = _get_cgroup_memory_bytes()
        t0 = time.perf_counter()
        with MemorySampler() as sampler:
            func()
        elapsed = time.perf_counter() - t0
        rss_end = _get_rss_bytes()
        non_cache_end = _get_non_cache_bytes()
        mem_available_end = _get_mem_available_bytes()
        cgroup_current_end, cgroup_limit_end = _get_cgroup_memory_bytes()
        cache_size = _dir_size_bytes(cache_dir)
        cgroup_limit = cgroup_limit_start or cgroup_limit_end or sampler.cgroup_limit

        results.append(
            StepResult(
                name=name,
                seconds=elapsed,
                rss_start_mb=_bytes_to_mb(rss_start),
                rss_end_mb=_bytes_to_mb(rss_end),
                rss_peak_mb=_bytes_to_mb(sampler.max_rss),
                non_cache_start_mb=_bytes_to_mb(non_cache_start),
                non_cache_end_mb=_bytes_to_mb(non_cache_end),
                non_cache_peak_mb=_bytes_to_mb(sampler.max_non_cache),
                mem_available_start_mb=_bytes_to_mb_optional(mem_available_start),
                mem_available_end_mb=_bytes_to_mb_optional(mem_available_end),
                mem_available_min_mb=_bytes_to_mb_optional(sampler.min_mem_available),
                cgroup_current_start_mb=_bytes_to_mb_optional(cgroup_current_start),
                cgroup_current_end_mb=_bytes_to_mb_optional(cgroup_current_end),
                cgroup_current_peak_mb=_bytes_to_mb_optional(
                    sampler.max_cgroup_current if sampler.max_cgroup_current > 0 else None
                ),
                cgroup_limit_mb=_bytes_to_mb_optional(cgroup_limit),
                cache_size_mb=_bytes_to_mb(cache_size),
            )
        )

        results_path.write_text(
            json.dumps([asdict(r) for r in results], indent=2),
            encoding="utf-8",
        )
        print(
            f"[{results_path.parent.name}] step={name} "
            f"seconds={elapsed:.2f} "
            f"rss_peak_mb={_bytes_to_mb(sampler.max_rss):.2f} "
            f"non_cache_peak_mb={_bytes_to_mb(sampler.max_non_cache):.2f} "
            f"mem_available_min_mb={_fmt_optional_mb(_bytes_to_mb_optional(sampler.min_mem_available))} "
            f"cgroup_peak_mb={_fmt_optional_mb(_bytes_to_mb_optional(sampler.max_cgroup_current if sampler.max_cgroup_current > 0 else None))}",
            flush=True,
        )

    total_seconds = sum(step.seconds for step in results)
    peak_rss_mb = max((step.rss_peak_mb for step in results), default=0.0)
    peak_non_cache_mb = max((step.non_cache_peak_mb for step in results), default=0.0)
    mem_available_values = [step.mem_available_min_mb for step in results if step.mem_available_min_mb is not None]
    min_mem_available_mb = min(mem_available_values) if mem_available_values else None
    cgroup_peak_values = [
        step.cgroup_current_peak_mb for step in results if step.cgroup_current_peak_mb is not None
    ]
    peak_cgroup_current_mb = max(cgroup_peak_values) if cgroup_peak_values else None
    cgroup_limit_mb = next((step.cgroup_limit_mb for step in results if step.cgroup_limit_mb is not None), None)
    cgroup_peak_of_limit_pct = None
    if cgroup_limit_mb and peak_cgroup_current_mb:
        cgroup_peak_of_limit_pct = (peak_cgroup_current_mb / cgroup_limit_mb) * 100.0
    cache_size_mb = _bytes_to_mb(_dir_size_bytes(cache_dir))

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "total_seconds": total_seconds,
                "peak_rss_mb": peak_rss_mb,
                "peak_non_cache_mb": peak_non_cache_mb,
                "min_mem_available_mb": min_mem_available_mb,
                "peak_cgroup_current_mb": peak_cgroup_current_mb,
                "cgroup_limit_mb": cgroup_limit_mb,
                "cgroup_peak_of_limit_pct": cgroup_peak_of_limit_pct,
                "cache_size_mb": cache_size_mb,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cgroup_pct_str = (
        f"{cgroup_peak_of_limit_pct:.2f}" if cgroup_peak_of_limit_pct is not None else "n/a"
    )
    print(
        f"[{output_root.name}] total_seconds={total_seconds:.2f}, "
        f"peak_rss_mb={peak_rss_mb:.2f}, "
        f"peak_non_cache_mb={peak_non_cache_mb:.2f}, "
        f"min_mem_available_mb={_fmt_optional_mb(min_mem_available_mb)}, "
        f"peak_cgroup_current_mb={_fmt_optional_mb(peak_cgroup_current_mb)}, "
        f"cgroup_limit_mb={_fmt_optional_mb(cgroup_limit_mb)}, "
        f"cgroup_peak_of_limit_pct={cgroup_pct_str}, "
        f"cache_size_mb={cache_size_mb:.2f}"
    )

    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".png", ".pdf"):
        for file in cache_dir.rglob(f"*{ext}"):
            shutil.copy2(file, plots_dir / file.name)

    structure_src = cache_dir / "structure_viz"
    structure_dst = output_root / "structure_viz"
    if structure_src.exists():
        if structure_dst.exists():
            shutil.rmtree(structure_dst, ignore_errors=True)
        shutil.copytree(structure_src, structure_dst)


def main() -> int:
    out_root = Path("data/benchmarks")
    results_dir = Path("benchmark_results")
    cache_root = Path("cache/benchmark")

    factors = [1, 2, 3, 5, 10, 30, 50]

    datasets: List[tuple[str, Path]] = []
    for factor in factors:
        if factor == 1:
            name = "2RJY"
            out_dir = Path("data/2RJY")
        else:
            name = f"2RJY_stack{factor}x"
            out_dir = out_root / name
        if not out_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset: {out_dir}. Run dev_scripts/benchmark_generate_data.py first."
            )
        datasets.append((name, out_dir))

    for name, dataset_dir in datasets:
        results_path = results_dir / name / "steps.json"
        run_cache = cache_root / name
        _run_pipeline(dataset_dir=dataset_dir, cache_dir=run_cache, results_path=results_path)

    summary = {}
    for name, _ in datasets:
        steps_path = results_dir / name / "steps.json"
        summary_path = results_dir / name / "summary.json"
        summary[name] = {
            "steps": json.loads(steps_path.read_text(encoding="utf-8")),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

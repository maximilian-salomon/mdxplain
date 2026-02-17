# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Benchmark analysis report script.
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

"""Generate benchmark analysis figures from benchmark JSON outputs.

File Description
----------------
This script loads benchmark result JSONs, computes derived metrics, and exports
publication-ready analysis figures as PNG and/or SVG, plus analysis tables as CSV.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/benchmark_analysis_report.py``
- ``python dev_scripts/benchmark/benchmark_analysis_report.py --filetype svg``
- ``python dev_scripts/benchmark/benchmark_analysis_report.py --filetype png --filetype svg``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd


PROFILE_DIRS = {
    "fast_standard": "benchmark_results",
    "iterative": "benchmark_results_iterative",
    "standard_full": "benchmark_results_standard_full",
}

PROFILE_LABELS = {
    "fast_standard": "Fast Standard",
    "iterative": "Iterative",
    "standard_full": "Standard Full",
}

PROFILE_COLOR_MAP = {
    "Fast Standard": "#1f77b4",
    "Iterative": "#2ca02c",
    "Standard Full": "#d62728",
}

GLOBAL_STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "lines.linewidth": 1.9,
    "lines.markersize": 3.8,
    "grid.alpha": 0.25,
}


@dataclass
class AnalysisContext:
    """Container for loaded benchmark data and derived plotting state.

    Parameters
    ----------
    totals_df : pd.DataFrame
        Run-level benchmark metrics.
    steps_df : pd.DataFrame
        Step-level benchmark metrics.
    profile_order : list[str]
        Ordered profile labels for plotting.
    profile_palette : dict[str, str]
        Profile-to-color mapping.
    ref_scale : int or None
        Reference stack factor used for step overlays.
    step_sub : pd.DataFrame
        Step subset selected for per-step profile comparisons.
    step_order : list[str]
        Stable step order for x-axis rendering.
    x_steps : np.ndarray
        Numeric x-axis positions for step plots.
    trade_points : pd.DataFrame
        Aggregated trade-off points for bubble rendering.
    bubble_transition_order : list[str]
        Transition order used for bubble arrows.
    bubble_profile_palette : dict[str, str]
        Color palette used by bubble plots.

    Returns
    -------
    None
        Dataclass instances are consumed by plotting helpers.

    Notes
    -----
    This object avoids global mutable state inside plotting logic.
    """

    totals_df: pd.DataFrame
    steps_df: pd.DataFrame
    profile_order: list[str]
    profile_palette: dict[str, str]
    ref_scale: Optional[int]
    step_sub: pd.DataFrame
    step_order: list[str]
    x_steps: np.ndarray
    trade_points: pd.DataFrame
    bubble_transition_order: list[str]
    bubble_profile_palette: dict[str, str]


def _set_plot_style() -> None:
    """Apply global plotting style and deterministic profile colors.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Matplotlib global style state is updated in place.

    Notes
    -----
    Style is configured once per report run.
    """
    # Configure baseline style first, then enforce profile color cycle.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(GLOBAL_STYLE)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[
            PROFILE_COLOR_MAP["Fast Standard"],
            PROFILE_COLOR_MAP["Iterative"],
            PROFILE_COLOR_MAP["Standard Full"],
        ]
    )


def _normalize_filetypes(raw_filetypes: list[str]) -> list[str]:
    """Normalize and validate requested figure export formats.

    Parameters
    ----------
    raw_filetypes : list[str]
        Raw user-provided format list.

    Returns
    -------
    list[str]
        Ordered unique normalized extensions.

    Notes
    -----
    Supported formats are ``png`` and ``svg``.
    """
    allowed = {"png", "svg"}
    normalized: list[str] = []
    for value in raw_filetypes:
        ext = str(value).strip().lower().lstrip(".")
        if not ext:
            continue
        if ext not in allowed:
            raise ValueError(f"Unsupported file type {value!r}. Allowed: png, svg")
        if ext not in normalized:
            normalized.append(ext)
    return normalized or ["png"]


def _slugify(text: str) -> str:
    """Convert arbitrary text into a safe filename stem.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Filesystem-safe lowercase slug.

    Notes
    -----
    Empty input is normalized to ``figure``.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return cleaned or "figure"


def _as_float(value) -> float:
    """Convert values to float with NaN fallback.

    Parameters
    ----------
    value : object
        Input value to convert.

    Returns
    -------
    float
        Parsed float or ``nan`` on conversion failure.

    Notes
    -----
    This helper keeps JSON parsing robust for missing fields.
    """
    try:
        return float(value)
    except Exception:
        return float("nan")


def _parse_scale_factor(run_name: str) -> int:
    """Extract stack factor from benchmark run name.

    Parameters
    ----------
    run_name : str
        Run folder key.

    Returns
    -------
    int
        Parsed scale factor, defaulting to ``1``.

    Notes
    -----
    Run names with suffix ``_stackNx`` map to factor ``N``.
    """
    match = re.search(r"_stack(\d+)x$", run_name)
    return int(match.group(1)) if match else 1


def _profile_dir_count(root: Path) -> int:
    """Count benchmark profile directories available under one root.

    Parameters
    ----------
    root : Path
        Candidate benchmark root.

    Returns
    -------
    int
        Number of known profile directories found.

    Notes
    -----
    Directory names are resolved from ``PROFILE_DIRS``.
    """
    return sum(1 for directory in PROFILE_DIRS.values() if (root / directory).is_dir())


def _iter_benchmark_candidates(preferred: Path) -> list[Path]:
    """Build ordered benchmark-root candidate list.

    Parameters
    ----------
    preferred : Path
        Preferred benchmark directory from CLI.

    Returns
    -------
    list[Path]
        Deduplicated candidate path list.

    Notes
    -----
    Search order prefers explicit paths before heuristic discovery.
    """
    # Build candidate list from preferred path and benchmark-like folders.
    ordered: list[Path] = []
    for hint in [preferred, Path("benchmark_jsons_latest")]:
        if hint.is_dir():
            ordered.append(hint)
    for path in sorted(Path(".").glob("benchmark*")):
        if path.is_dir():
            ordered.append(path)
    for path in sorted(Path(".").iterdir()):
        if path.is_dir():
            ordered.append(path)

    # Keep first occurrence only while preserving order.
    dedup: list[Path] = []
    seen: set[str] = set()
    for path in ordered:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup


def resolve_benchmark_root(preferred: Path) -> Path:
    """Resolve benchmark root containing benchmark result directories.

    Parameters
    ----------
    preferred : Path
        Preferred benchmark root from CLI.

    Returns
    -------
    Path
        Resolved benchmark root path.

    Notes
    -----
    The resolver scans top-level and one nested directory level.

    Examples
    --------
    >>> from pathlib import Path
    >>> # root = resolve_benchmark_root(Path("benchmark"))
    """
    if preferred.is_dir() and _profile_dir_count(preferred) > 0:
        return preferred.resolve()

    ranked: dict[str, tuple[int, float, Path]] = {}
    for candidate in _iter_benchmark_candidates(preferred):
        roots_to_check = [candidate]
        try:
            roots_to_check.extend([path for path in candidate.iterdir() if path.is_dir()])
        except Exception:
            pass
        _rank_candidate_roots(ranked, roots_to_check)

    if ranked:
        return _best_ranked_root(ranked)

    expected = ", ".join(PROFILE_DIRS.values())
    raise FileNotFoundError(
        "Could not auto-detect benchmark root. "
        f"Expected directories like: {expected}."
    )


def _rank_candidate_roots(ranked: dict[str, tuple[int, float, Path]], roots: list[Path]) -> None:
    """Update ranked root map using candidate root list.

    Parameters
    ----------
    ranked : dict[str, tuple[int, float, Path]]
        Mutable ranking map.
    roots : list[Path]
        Candidate roots to evaluate.

    Returns
    -------
    None
        Ranking map is updated in place.

    Notes
    -----
    Sorting key is ``(profile_count, mtime)``.
    """
    # Score every root and keep highest score per canonical path.
    for root in roots:
        score = _profile_dir_count(root)
        if score <= 0:
            continue
        try:
            mtime = root.stat().st_mtime
        except Exception:
            mtime = 0.0
        key = str(root.resolve())
        prev = ranked.get(key)
        if prev is None or (score, mtime) > (prev[0], prev[1]):
            ranked[key] = (score, mtime, root)


def _best_ranked_root(ranked: dict[str, tuple[int, float, Path]]) -> Path:
    """Return best root from ranked root mapping.

    Parameters
    ----------
    ranked : dict[str, tuple[int, float, Path]]
        Precomputed ranked roots.

    Returns
    -------
    Path
        Best ranked root path.

    Notes
    -----
    Highest profile count wins, then latest modification time.
    """
    best = sorted(ranked.values(), key=lambda item: (item[0], item[1]), reverse=True)[0][2]
    return best.resolve()


def _load_profile_runs(profile_dir: Path) -> dict[str, dict]:
    """Load run payloads for one profile directory.

    Parameters
    ----------
    profile_dir : Path
        Profile result directory.

    Returns
    -------
    dict[str, dict]
        Mapping from run name to summary/steps payload.

    Notes
    -----
    Root-level summary is preferred over per-run fallback loading.
    """
    runs: dict[str, dict] = {}
    root_summary = profile_dir / "summary.json"
    if root_summary.exists():
        payload = json.loads(root_summary.read_text(encoding="utf-8"))
        runs = _extract_runs_from_root_summary(payload)
    if runs:
        return runs
    return _load_profile_runs_from_subdirs(profile_dir)


def _extract_runs_from_root_summary(payload: object) -> dict[str, dict]:
    """Extract run payloads from profile-level summary JSON payload.

    Parameters
    ----------
    payload : object
        Parsed JSON root object.

    Returns
    -------
    dict[str, dict]
        Extracted run mapping.

    Notes
    -----
    Invalid entries are ignored silently.
    """
    # Keep only entries that provide both summary and steps sub-objects.
    runs: dict[str, dict] = {}
    if not isinstance(payload, dict):
        return runs
    for run_name, value in payload.items():
        if not isinstance(value, dict):
            continue
        if "summary" in value and "steps" in value:
            runs[run_name] = value
    return runs


def _load_profile_runs_from_subdirs(profile_dir: Path) -> dict[str, dict]:
    """Load run payloads from per-run subdirectories.

    Parameters
    ----------
    profile_dir : Path
        Profile result directory.

    Returns
    -------
    dict[str, dict]
        Run mapping loaded from ``summary.json`` and ``steps.json`` files.

    Notes
    -----
    Only complete run directories are included.
    """
    # Build fallback mapping from run subdirectories.
    runs: dict[str, dict] = {}
    for run_dir in sorted(path for path in profile_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        steps_path = run_dir / "steps.json"
        if not (summary_path.exists() and steps_path.exists()):
            continue
        runs[run_dir.name] = {
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "steps": json.loads(steps_path.read_text(encoding="utf-8")),
        }
    return runs


def _build_total_row(
    profile: str,
    run_name: str,
    run_key: str,
    run_label: str,
    scale_factor: int,
    summary: dict,
) -> dict:
    """Build one run-level total metrics row.

    Parameters
    ----------
    profile : str
        Profile identifier key.
    run_name : str
        Run name.
    run_key : str
        Unique run key.
    run_label : str
        Human-readable run label.
    scale_factor : int
        Parsed run scale factor.
    summary : dict
        Run summary JSON payload.

    Returns
    -------
    dict
        Row compatible with totals DataFrame construction.

    Notes
    -----
    Missing numeric fields are normalized via ``_as_float``.
    """
    return {
        "profile": profile,
        "profile_label": PROFILE_LABELS[profile],
        "run": run_name,
        "run_key": run_key,
        "run_label": run_label,
        "scale_factor": scale_factor,
        "total_seconds": _as_float(summary.get("total_seconds")),
        "peak_non_cache_mb": _as_float(summary.get("peak_non_cache_mb")),
        "min_mem_available_mb": _as_float(summary.get("min_mem_available_mb")),
        "peak_cgroup_current_mb": _as_float(summary.get("peak_cgroup_current_mb")),
        "cgroup_limit_mb": _as_float(summary.get("cgroup_limit_mb")),
        "cgroup_peak_of_limit_pct": _as_float(summary.get("cgroup_peak_of_limit_pct")),
        "cache_size_mb": _as_float(summary.get("cache_size_mb")),
    }


def _build_step_rows(
    profile: str,
    run_name: str,
    run_key: str,
    run_label: str,
    scale_factor: int,
    steps: list,
) -> list[dict]:
    """Build all step-level rows for one run.

    Parameters
    ----------
    profile : str
        Profile identifier key.
    run_name : str
        Run name.
    run_key : str
        Unique run key.
    run_label : str
        Human-readable run label.
    scale_factor : int
        Parsed run scale factor.
    steps : list
        Step payload list from run JSON.

    Returns
    -------
    list[dict]
        Step rows for DataFrame construction.

    Notes
    -----
    Step index order is preserved from source JSON.
    """
    rows: list[dict] = []
    for step_idx, step in enumerate(steps):
        rows.append(
            {
                "profile": profile,
                "profile_label": PROFILE_LABELS[profile],
                "run": run_name,
                "run_key": run_key,
                "run_label": run_label,
                "scale_factor": scale_factor,
                "step": step.get("name", f"step_{step_idx}"),
                "step_index": step_idx,
                "seconds": _as_float(step.get("seconds")),
                "non_cache_start_mb": _as_float(step.get("non_cache_start_mb")),
                "non_cache_end_mb": _as_float(step.get("non_cache_end_mb")),
                "non_cache_peak_mb": _as_float(step.get("non_cache_peak_mb")),
                "mem_available_start_mb": _as_float(step.get("mem_available_start_mb")),
                "mem_available_end_mb": _as_float(step.get("mem_available_end_mb")),
                "mem_available_min_mb": _as_float(step.get("mem_available_min_mb")),
                "cgroup_current_start_mb": _as_float(step.get("cgroup_current_start_mb")),
                "cgroup_current_end_mb": _as_float(step.get("cgroup_current_end_mb")),
                "cgroup_current_peak_mb": _as_float(step.get("cgroup_current_peak_mb")),
                "cgroup_limit_mb": _as_float(step.get("cgroup_limit_mb")),
                "cache_size_mb": _as_float(step.get("cache_size_mb")),
            }
        )
    return rows


def _derive_loaded_metrics(totals: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add derived columns to loaded totals and step DataFrames.

    Parameters
    ----------
    totals : pd.DataFrame
        Raw totals DataFrame.
    steps : pd.DataFrame
        Raw steps DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrames with derived metrics.

    Notes
    -----
    Returned DataFrames are sorted deterministically.
    """
    # Sort loaded data once to keep all downstream plots stable.
    totals = totals.sort_values(["profile", "scale_factor", "run"]).reset_index(drop=True)
    steps = steps.sort_values(["profile", "scale_factor", "run", "step_index"]).reset_index(drop=True)

    # Compute run-level and step-level derived metrics.
    totals["non_cache_per_second"] = totals["peak_non_cache_mb"] / totals["total_seconds"]
    steps["delta_cache_mb"] = steps.groupby("run_key")["cache_size_mb"].diff().fillna(steps["cache_size_mb"])
    steps["delta_non_cache_mb"] = steps["non_cache_end_mb"] - steps["non_cache_start_mb"]
    steps["non_cache_peak_over_start_mb"] = steps["non_cache_peak_mb"] - steps["non_cache_start_mb"]
    steps["seconds_share_pct"] = steps["seconds"] / steps.groupby("run_key")["seconds"].transform("sum") * 100.0
    return totals, steps


def load_all_benchmarks(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load benchmark totals and step metrics from benchmark root.

    Parameters
    ----------
    root : Path
        Resolved benchmark root directory.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Totals and steps DataFrames.

    Notes
    -----
    Missing profile directories are skipped automatically.

    Examples
    --------
    >>> from pathlib import Path
    >>> # totals, steps = load_all_benchmarks(Path("benchmark"))
    """
    totals_rows: list[dict] = []
    steps_rows: list[dict] = []

    for profile, directory_name in PROFILE_DIRS.items():
        profile_dir = root / directory_name
        if not profile_dir.exists():
            continue
        runs = _load_profile_runs(profile_dir)
        _append_profile_rows(profile, runs, totals_rows, steps_rows)

    totals = pd.DataFrame(totals_rows)
    steps = pd.DataFrame(steps_rows)
    if totals.empty or steps.empty:
        raise RuntimeError("No benchmark data loaded. Check benchmark JSON directories.")
    return _derive_loaded_metrics(totals, steps)


def _append_profile_rows(
    profile: str,
    runs: dict[str, dict],
    totals_rows: list[dict],
    steps_rows: list[dict],
) -> None:
    """Append all rows for one profile into row accumulators.

    Parameters
    ----------
    profile : str
        Profile identifier key.
    runs : dict[str, dict]
        Run payload mapping for this profile.
    totals_rows : list[dict]
        Mutable totals-row accumulator.
    steps_rows : list[dict]
        Mutable step-row accumulator.

    Returns
    -------
    None
        Row accumulators are updated in place.

    Notes
    -----
    One totals row and many step rows are added per run.
    """
    for run_name, run_data in runs.items():
        scale_factor = _parse_scale_factor(run_name)
        run_key = f"{profile}::{run_name}"
        run_label = f"{PROFILE_LABELS[profile]} | {scale_factor}x"
        summary = run_data.get("summary", {})
        steps = run_data.get("steps", [])

        totals_rows.append(
            _build_total_row(profile, run_name, run_key, run_label, scale_factor, summary)
        )
        steps_rows.extend(
            _build_step_rows(profile, run_name, run_key, run_label, scale_factor, steps)
        )


def _get_profile_palette(profiles: Sequence[str]) -> dict[str, str]:
    """Build profile-to-color mapping for ordered profile labels.

    Parameters
    ----------
    profiles : Sequence[str]
        Ordered profile labels.

    Returns
    -------
    dict[str, str]
        Profile palette mapping.

    Notes
    -----
    Unknown profile labels fall back to ``matplotlib.tab10`` colors.
    """
    fallback = plt.cm.tab10(np.linspace(0.0, 0.9, max(1, len(profiles))))
    palette: dict[str, str] = {}
    for idx, profile in enumerate(profiles):
        palette[profile] = PROFILE_COLOR_MAP.get(profile, fallback[idx])
    return palette


def _blend_towards_white(color: str, ratio: float) -> tuple[float, float, float]:
    """Blend one color with white using a stable scalar ratio.

    Parameters
    ----------
    color : str
        Matplotlib color string.
    ratio : float
        Blend ratio in ``[0, 1]`` where ``0`` keeps original color.

    Returns
    -------
    tuple[float, float, float]
        RGB tuple usable by matplotlib.

    Notes
    -----
    Ratios are clipped to ``[0, 1]`` before blending.
    """
    rgb = np.array(to_rgb(color), dtype=float)
    clipped = float(np.clip(ratio, 0.0, 1.0))
    mixed = (1.0 - clipped) * rgb + clipped * np.ones(3, dtype=float)
    return float(mixed[0]), float(mixed[1]), float(mixed[2])


def _get_shades(base_color: str, count: int, light: float = 0.30, dark: float = 0.95) -> list[tuple[float, float, float]]:
    """Create light-to-dark shades derived from one base color.

    Parameters
    ----------
    base_color : str
        Base profile color.
    count : int
        Number of shades to generate.
    light : float, default=0.30
        Relative intensity for lightest shade.
    dark : float, default=0.95
        Relative intensity for darkest shade.

    Returns
    -------
    list[tuple[float, float, float]]
        RGB color list in deterministic order.

    Notes
    -----
    Single-color requests return one medium shade.
    """
    if count <= 1:
        mid = (float(light) + float(dark)) / 2.0
        return [_blend_towards_white(base_color, 1.0 - mid)]

    factors = np.linspace(float(light), float(dark), int(count))
    return [_blend_towards_white(base_color, 1.0 - float(value)) for value in factors]


def _get_run_palette(runs: list[str], base_color: str) -> dict[str, tuple[float, float, float]]:
    """Map ordered run labels to deterministic shades of one color.

    Parameters
    ----------
    runs : list[str]
        Ordered run names.
    base_color : str
        Profile base color.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Run-to-RGB mapping.

    Notes
    -----
    Run ordering is preserved.
    """
    shades = _get_shades(base_color, max(1, len(runs)))
    return {run: shades[idx] for idx, run in enumerate(runs)}


def _select_profile_order(totals: pd.DataFrame) -> list[str]:
    """Select deterministic profile order for plotting.

    Parameters
    ----------
    totals : pd.DataFrame
        Totals DataFrame.

    Returns
    -------
    list[str]
        Ordered profile labels.

    Notes
    -----
    Preferred order is Fast Standard, Iterative, Standard Full.
    """
    order = [
        profile
        for profile in ["Fast Standard", "Iterative", "Standard Full"]
        if profile in totals["profile_label"].unique()
    ]
    return order or sorted(totals["profile_label"].unique())


def _select_reference_step_subset(
    totals: pd.DataFrame,
    steps: pd.DataFrame,
    profiles: list[str],
) -> tuple[Optional[int], pd.DataFrame]:
    """Select reference step subset used for per-step profile plots.

    Parameters
    ----------
    totals : pd.DataFrame
        Totals DataFrame.
    steps : pd.DataFrame
        Steps DataFrame.
    profiles : list[str]
        Ordered profile labels.

    Returns
    -------
    tuple[int or None, pd.DataFrame]
        Selected reference scale and step subset.

    Notes
    -----
    Highest shared scale is preferred when available.
    """
    scale_sets = [
        set(totals[totals["profile_label"] == profile]["scale_factor"].unique())
        for profile in profiles
    ]
    shared_scales = sorted(set.intersection(*scale_sets)) if scale_sets else []

    if shared_scales:
        ref_scale = shared_scales[-1]
        return ref_scale, steps[steps["scale_factor"] == ref_scale].copy()

    ref_scale = None
    max_scale_by_profile = (
        totals.groupby("profile_label", as_index=False)["scale_factor"]
        .max()
        .set_index("profile_label")["scale_factor"]
        .to_dict()
    )
    parts = [
        steps[(steps["profile_label"] == profile) & (steps["scale_factor"] == max_scale)]
        for profile, max_scale in max_scale_by_profile.items()
    ]
    return ref_scale, pd.concat(parts, ignore_index=True) if parts else steps.iloc[0:0].copy()


def _build_step_order(step_sub: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Build stable step order and numeric x-axis positions.

    Parameters
    ----------
    step_sub : pd.DataFrame
        Selected step subset.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Ordered steps and x positions.

    Notes
    -----
    Empty subsets return empty outputs.
    """
    if step_sub.empty:
        return [], np.array([])
    order = step_sub.groupby("step")["step_index"].median().sort_values().index.tolist()
    return order, np.arange(len(order))


def _prepare_tradeoff_points(totals: pd.DataFrame, profiles: list[str]) -> pd.DataFrame:
    """Aggregate one trade-off point per profile and stack factor.

    Parameters
    ----------
    totals : pd.DataFrame
        Totals DataFrame.
    profiles : list[str]
        Ordered profile labels.

    Returns
    -------
    pd.DataFrame
        Aggregated trade-off points.

    Notes
    -----
    Values are averaged over repeated runs per profile and scale.
    """
    trade_all = totals[totals["profile_label"].isin(profiles)].copy()
    if trade_all.empty:
        return trade_all
    return (
        trade_all.groupby(["scale_factor", "profile_label"], as_index=False)
        .agg(
            total_seconds=("total_seconds", "mean"),
            peak_non_cache_mb=("peak_non_cache_mb", "mean"),
            cache_size_mb=("cache_size_mb", "mean"),
        )
        .sort_values(["scale_factor", "profile_label"])
        .reset_index(drop=True)
    )


def _build_bubble_transition_order(profiles: list[str]) -> list[str]:
    """Build transition order for bubble overlays.

    Parameters
    ----------
    profiles : list[str]
        Ordered profile labels.

    Returns
    -------
    list[str]
        Bubble transition order.

    Notes
    -----
    Preferred sequence is Standard Full -> Iterative -> Fast Standard.
    """
    order = [
        profile
        for profile in ["Standard Full", "Iterative", "Fast Standard"]
        if profile in profiles
    ]
    return order or list(profiles)


def _build_bubble_palette(
    profiles: list[str],
    profile_palette: dict[str, str],
    transition_order: list[str],
) -> dict[str, str]:
    """Build bubble-specific profile palette.

    Parameters
    ----------
    profiles : list[str]
        Ordered profile labels.
    profile_palette : dict[str, str]
        Default profile palette.
    transition_order : list[str]
        Transition order for bubble overlays.

    Returns
    -------
    dict[str, str]
        Bubble palette mapping.

    Notes
    -----
    Transition profiles receive fixed high-contrast colors first.
    """
    fixed = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    palette = {
        profile: fixed[idx % len(fixed)]
        for idx, profile in enumerate(transition_order)
    }
    for profile in profiles:
        palette.setdefault(profile, profile_palette.get(profile, "#1f77b4"))
    return palette


def _build_context(totals: pd.DataFrame, steps: pd.DataFrame) -> AnalysisContext:
    """Build analysis context from loaded benchmark DataFrames.

    Parameters
    ----------
    totals : pd.DataFrame
        Totals DataFrame.
    steps : pd.DataFrame
        Steps DataFrame.

    Returns
    -------
    AnalysisContext
        Fully initialized plotting context.

    Notes
    -----
    Context fields are precomputed to keep plotting helpers concise.
    """
    profiles = _select_profile_order(totals)
    palette = _get_profile_palette(profiles)
    ref_scale, step_sub = _select_reference_step_subset(totals, steps, profiles)
    step_order, x_steps = _build_step_order(step_sub)
    trade_points = _prepare_tradeoff_points(totals, profiles)
    transition_order = _build_bubble_transition_order(profiles)
    bubble_palette = _build_bubble_palette(profiles, palette, transition_order)

    return AnalysisContext(
        totals_df=totals,
        steps_df=steps,
        profile_order=profiles,
        profile_palette=palette,
        ref_scale=ref_scale,
        step_sub=step_sub,
        step_order=step_order,
        x_steps=x_steps,
        trade_points=trade_points,
        bubble_transition_order=transition_order,
        bubble_profile_palette=bubble_palette,
    )


def _friendly_figure_basename(fig: Figure) -> str:
    """Build deterministic filename stem for one figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure instance.

    Returns
    -------
    str
        Normalized file stem.

    Notes
    -----
    ``_mdxplain_filename_hint`` is preferred when present.
    """
    hint = getattr(fig, "_mdxplain_filename_hint", None)
    if isinstance(hint, str) and hint.strip():
        return _slugify(hint)

    suptitle = getattr(getattr(fig, "_suptitle", None), "get_text", lambda: "")().strip()
    axis_title = fig.axes[0].get_title().strip() if fig.axes else ""
    return _slugify(suptitle or axis_title or f"figure_{fig.number}")


def _save_figure(fig: Figure, export_dir: Path, filetypes: list[str], dpi: int) -> None:
    """Save one figure in all requested export formats.

    Parameters
    ----------
    fig : Figure
        Figure to export.
    export_dir : Path
        Target export directory.
    filetypes : list[str]
        Requested file extensions.
    dpi : int
        Export DPI for raster formats.

    Returns
    -------
    None
        Figure files are written to disk.

    Notes
    -----
    SVG exports ignore DPI internally but keep API symmetric.
    """
    stem = _friendly_figure_basename(fig)
    for ext in filetypes:
        fig.savefig(export_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")


def _pretty_step_label(step_name: str, max_len: int = 16) -> str:
    """Create compact label text for step names.

    Parameters
    ----------
    step_name : str
        Raw step name.
    max_len : int, default=16
        Maximum label length.

    Returns
    -------
    str
        Compact step label.

    Notes
    -----
    Common words are shortened before truncation.
    """
    label = str(step_name).replace("_", " ")
    replacements = {
        "trajectories": "trajs",
        "trajectory": "traj",
        "selection": "select",
        "pressure": "press",
        "feature": "feat",
        "features": "feats",
        "distances": "dists",
        "distance": "dist",
        "contacts": "cont",
        "contact": "cont",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = " ".join(label.split())
    return label if len(label) <= max_len else label[: max_len - 3] + "..."


def _apply_step_ticks(ax, ctx: AnalysisContext) -> None:
    """Apply compact step ticks to a step-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ctx : AnalysisContext
        Analysis context with step ordering.

    Returns
    -------
    None
        Axis tick state is updated in place.

    Notes
    -----
    Tick labels are rotated to improve readability.
    """
    idx = np.arange(len(ctx.step_order))
    labels = [_pretty_step_label(step_name) for step_name in ctx.step_order]
    ax.set_xticks(idx, labels=labels, rotation=38, ha="right")
    ax.tick_params(axis="x", labelsize=7, pad=2)


def _step_metric_pivot(ctx: AnalysisContext, metric: str) -> pd.DataFrame:
    """Build profile-by-step pivot for one step metric.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    metric : str
        Step metric column name.

    Returns
    -------
    pd.DataFrame
        Pivot table aligned to context step/profile order.

    Notes
    -----
    Pivot values are averaged over matching runs.
    """
    return (
        ctx.step_sub.pivot_table(index="step", columns="profile_label", values=metric, aggfunc="mean")
        .reindex(index=ctx.step_order, columns=ctx.profile_order)
    )


def plot_scale_metric(
    ax,
    ctx: AnalysisContext,
    metric: str,
    title: str,
    ylabel: str,
    *,
    y_log: bool,
    show_legend: bool = False,
) -> None:
    """Plot profile scaling for one run-level metric.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ctx : AnalysisContext
        Analysis context.
    metric : str
        Totals metric column name.
    title : str
        Axis title.
    ylabel : str
        Axis y-label.
    y_log : bool
        Enables logarithmic y-axis.
    show_legend : bool, default=False
        Displays profile legend when true.

    Returns
    -------
    None
        Axis is updated with line plot.

    Notes
    -----
    X-axis is always logarithmic in stack factor.

    Examples
    --------
    >>> # plot_scale_metric(ax, ctx, "total_seconds", "Runtime", "s", y_log=True)
    """
    # Plot one line per profile in stable profile order.
    for profile in ctx.profile_order:
        sub = ctx.totals_df[ctx.totals_df["profile_label"] == profile].sort_values("scale_factor")
        if sub.empty:
            continue
        ax.plot(
            sub["scale_factor"].to_numpy(dtype=float),
            sub[metric].to_numpy(dtype=float),
            marker="o",
            linewidth=GLOBAL_STYLE["lines.linewidth"],
            markersize=GLOBAL_STYLE["lines.markersize"],
            color=ctx.profile_palette[profile],
            label=profile,
        )

    # Apply consistent axis style and optional reference marker.
    if ctx.ref_scale is not None:
        ax.axvline(ctx.ref_scale, linestyle="--", color="gray", alpha=0.45, linewidth=1.0)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Stack Factor")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=GLOBAL_STYLE["grid.alpha"])
    if show_legend:
        ax.legend(loc="upper left", frameon=False, title="Profiles")


def _plot_step_profile_lines(ax, ctx: AnalysisContext, pivot: pd.DataFrame, mark_max: bool) -> None:
    """Plot per-profile step lines for a precomputed step pivot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ctx : AnalysisContext
        Analysis context.
    pivot : pd.DataFrame
        Metric pivot from ``_step_metric_pivot``.
    mark_max : bool
        Highlights max point per profile when true.

    Returns
    -------
    None
        Lines are rendered on target axis.

    Notes
    -----
    Non-finite series are skipped automatically.
    """
    # Render one step trajectory per profile and optionally highlight peaks.
    for profile in ctx.profile_order:
        if profile not in pivot.columns:
            continue
        y_values = pivot[profile].to_numpy(dtype=float)
        ax.plot(
            ctx.x_steps,
            y_values,
            marker="o",
            linewidth=GLOBAL_STYLE["lines.linewidth"],
            markersize=GLOBAL_STYLE["lines.markersize"],
            color=ctx.profile_palette[profile],
            label=profile,
        )
        if mark_max and np.isfinite(y_values).any():
            peak_idx = int(np.nanargmax(y_values))
            ax.scatter(
                ctx.x_steps[peak_idx],
                y_values[peak_idx],
                s=50,
                color=ctx.profile_palette[profile],
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
            )


def plot_step_line(
    ax,
    ctx: AnalysisContext,
    metric: str,
    title: str,
    ylabel: str,
    *,
    mark_max: bool = False,
    zero_line: bool = False,
    y_log: bool = False,
) -> None:
    """Plot per-step profile comparison for one step metric.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ctx : AnalysisContext
        Analysis context.
    metric : str
        Step metric column name.
    title : str
        Axis title.
    ylabel : str
        Axis y-label.
    mark_max : bool, default=False
        Highlights max point per profile.
    zero_line : bool, default=False
        Draws horizontal zero reference line.
    y_log : bool, default=False
        Enables logarithmic y-axis.

    Returns
    -------
    None
        Axis is updated with line plot.

    Notes
    -----
    Empty step subsets produce a centered fallback message.

    Examples
    --------
    >>> # plot_step_line(ax, ctx, "seconds", "Per-Step Time", "s", y_log=True)
    """
    # Handle empty step subsets gracefully to avoid plotting errors.
    if ctx.step_sub.empty or not ctx.step_order:
        ax.text(0.5, 0.5, "No step data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # Render metric lines and finalize axis styling.
    pivot = _step_metric_pivot(ctx, metric)
    _plot_step_profile_lines(ax, ctx, pivot, mark_max)
    if zero_line:
        ax.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    if y_log:
        ax.set_yscale("log")
    _apply_step_ticks(ax, ctx)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=GLOBAL_STYLE["grid.alpha"])


def _compute_local_exponents(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute local log-log exponents between consecutive points.

    Parameters
    ----------
    x : np.ndarray
        X values.
    y : np.ndarray
        Y values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered x-current values and local exponents.

    Notes
    -----
    Non-positive or non-finite values are excluded.
    """
    x_curr = x[1:]
    x_prev = x[:-1]
    y_curr = y[1:]
    y_prev = y[:-1]

    valid = (
        np.isfinite(x_curr)
        & np.isfinite(x_prev)
        & np.isfinite(y_curr)
        & np.isfinite(y_prev)
        & (x_curr > 0)
        & (x_prev > 0)
        & (y_curr > 0)
        & (y_prev > 0)
        & (x_curr != x_prev)
    )
    if not valid.any():
        return np.array([]), np.array([])

    exponents = np.log(y_curr[valid] / y_prev[valid]) / np.log(x_curr[valid] / x_prev[valid])
    return x_curr[valid], exponents


def _annotate_exponent_means(ax, annotations: list[tuple[str, float]], palette: dict[str, str]) -> None:
    """Add in-plot exponent summary annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    annotations : list[tuple[str, float]]
        ``(profile, mean_exponent)`` pairs.
    palette : dict[str, str]
        Profile-to-color mapping.

    Returns
    -------
    None
        Annotation text artists are added to axis.

    Notes
    -----
    Annotation vertical offsets are fixed for stable placement.
    """
    for idx, (profile, mean_exp) in enumerate(annotations):
        ax.text(
            0.02,
            0.97 - 0.09 * idx,
            f"{profile}: O(n^{mean_exp:.2f})",
            transform=ax.transAxes,
            fontsize=8,
            color=palette[profile],
            va="top",
        )


def plot_local_exponent(
    ax,
    ctx: AnalysisContext,
    metric: str,
    title: str,
    *,
    ylabel: str,
) -> None:
    """Plot local scaling exponents for one run-level metric.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ctx : AnalysisContext
        Analysis context.
    metric : str
        Totals metric column name.
    title : str
        Axis title.
    ylabel : str
        Axis y-label.

    Returns
    -------
    None
        Axis is updated with exponent curves.

    Notes
    -----
    Local exponent is defined on consecutive scale-factor intervals.

    Examples
    --------
    >>> # plot_local_exponent(ax, ctx, "total_seconds", "Runtime Exponent", ylabel="k")
    """
    # Plot exponent series for each profile and collect mean annotations.
    annotations: list[tuple[str, float]] = []
    for profile in ctx.profile_order:
        sub = ctx.totals_df[ctx.totals_df["profile_label"] == profile].sort_values("scale_factor")
        x = sub["scale_factor"].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        x_valid, exponents = _compute_local_exponents(x, y)
        if len(x_valid) == 0:
            continue

        ax.plot(
            x_valid,
            exponents,
            marker="o",
            linewidth=GLOBAL_STYLE["lines.linewidth"],
            markersize=GLOBAL_STYLE["lines.markersize"],
            color=ctx.profile_palette[profile],
            label=profile,
        )
        mean_exp = float(np.nanmean(exponents))
        if np.isfinite(mean_exp):
            annotations.append((profile, mean_exp))

    # Finalize axis appearance and annotation block.
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6)
    if ctx.ref_scale is not None:
        ax.axvline(ctx.ref_scale, linestyle="--", color="gray", alpha=0.35)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Current Scale")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=GLOBAL_STYLE["grid.alpha"])
    _annotate_exponent_means(ax, annotations, ctx.profile_palette)


def _target_profile_scale_pairs(ctx: AnalysisContext) -> list[tuple[str, int]]:
    """Build profile/scale pairs used by summary table.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[tuple[str, int]]
        Requested profile-scale pairs.

    Notes
    -----
    Pair list mirrors historical analysis table selection.
    """
    pairs: list[tuple[str, int]] = []
    for scale in [1, 3, 5]:
        for profile in ctx.profile_order:
            pairs.append((profile, scale))
    pairs.extend([("Iterative", 10), ("Fast Standard", 50)])
    return pairs


def _summarize_top_steps(sub_step: pd.DataFrame) -> tuple[str, float, str, float]:
    """Summarize dominant time and RAM-pressure steps for one subset.

    Parameters
    ----------
    sub_step : pd.DataFrame
        Step subset for one profile/scale pair.

    Returns
    -------
    tuple[str, float, str, float]
        Top time-step name/value and top pressure-step name/value.

    Notes
    -----
    Empty subsets return ``n/a`` with ``nan`` values.
    """
    if sub_step.empty:
        return "n/a", np.nan, "n/a", np.nan

    top_time = (
        sub_step.groupby("step", as_index=False)["seconds_share_pct"]
        .mean()
        .sort_values("seconds_share_pct", ascending=False)
        .head(1)
    )
    top_pressure = (
        sub_step.groupby("step", as_index=False)["non_cache_peak_mb"]
        .mean()
        .sort_values("non_cache_peak_mb", ascending=False)
        .head(1)
    )

    time_step = top_time.iloc[0]["step"] if not top_time.empty else "n/a"
    time_pct = float(top_time.iloc[0]["seconds_share_pct"]) if not top_time.empty else np.nan
    pressure_step = top_pressure.iloc[0]["step"] if not top_pressure.empty else "n/a"
    pressure_mb = float(top_pressure.iloc[0]["non_cache_peak_mb"]) if not top_pressure.empty else np.nan
    return time_step, time_pct, pressure_step, pressure_mb


def _format_summary_table_numeric(out: pd.DataFrame) -> pd.DataFrame:
    """Format summary table numeric columns as strings.

    Parameters
    ----------
    out : pd.DataFrame
        Unformatted summary table.

    Returns
    -------
    pd.DataFrame
        Formatted summary table.

    Notes
    -----
    ``nan`` values are rendered as literal text ``nan``.
    """
    numeric_cols = [
        "Runtime [s]",
        "Peak RAM Pressure [MB]",
        "Cache [MB]",
        "Top Time [%]",
        "Top Pressure [MB]",
    ]
    for col in numeric_cols:
        out[col] = out[col].map(lambda value: f"{value:.2f}" if pd.notna(value) else "nan")
    return out


def build_requested_table(ctx: AnalysisContext) -> pd.DataFrame:
    """Build summary table used by analysis figures.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    pd.DataFrame
        Summary table with selected profile/scale rows.

    Notes
    -----
    Row selection follows the legacy analysis convention.

    Examples
    --------
    >>> # table = build_requested_table(ctx)
    """
    rows: list[dict[str, object]] = []
    for profile, scale in _target_profile_scale_pairs(ctx):
        sub_tot = ctx.totals_df[
            (ctx.totals_df["profile_label"] == profile)
            & (ctx.totals_df["scale_factor"] == scale)
        ]
        if sub_tot.empty:
            continue

        sub_step = ctx.steps_df[
            (ctx.steps_df["profile_label"] == profile)
            & (ctx.steps_df["scale_factor"] == scale)
        ]
        time_step, time_pct, pressure_step, pressure_mb = _summarize_top_steps(sub_step)

        rows.append(
            {
                "Profile": f"{profile} | {int(scale)}x",
                "Runtime [s]": float(sub_tot["total_seconds"].mean()),
                "Peak RAM Pressure [MB]": float(sub_tot["peak_non_cache_mb"].mean()),
                "Cache [MB]": float(sub_tot["cache_size_mb"].mean()),
                "Top Time Step": time_step,
                "Top Time [%]": time_pct,
                "Top Pressure Step": pressure_step,
                "Top Pressure [MB]": pressure_mb,
            }
        )

    columns = [
        "Profile",
        "Runtime [s]",
        "Peak RAM Pressure [MB]",
        "Cache [MB]",
        "Top Time Step",
        "Top Time [%]",
        "Top Pressure Step",
        "Top Pressure [MB]",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return _format_summary_table_numeric(pd.DataFrame(rows))


def _build_totals_overview_table(ctx: AnalysisContext) -> pd.DataFrame:
    """Build run-level totals overview table from analysis context.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context with totals metrics.

    Returns
    -------
    pd.DataFrame
        Sorted totals overview table.

    Notes
    -----
    Column order follows the notebook's first overview display.
    """
    columns = [
        "profile_label",
        "run",
        "scale_factor",
        "total_seconds",
        "peak_non_cache_mb",
        "cache_size_mb",
        "non_cache_per_second",
        "min_mem_available_mb",
        "peak_cgroup_current_mb",
        "cgroup_peak_of_limit_pct",
    ]
    return (
        ctx.totals_df[columns]
        .sort_values(["profile_label", "scale_factor", "run"])
        .reset_index(drop=True)
    )


def _build_summary_by_profile_table(ctx: AnalysisContext) -> pd.DataFrame:
    """Build per-profile aggregate summary table.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context with totals metrics.

    Returns
    -------
    pd.DataFrame
        Aggregated profile summary table.

    Notes
    -----
    Aggregations match legacy notebook profile summary.
    """
    return (
        ctx.totals_df.groupby("profile_label")
        .agg(
            runs=("run", "count"),
            max_scale=("scale_factor", "max"),
            min_total_seconds=("total_seconds", "min"),
            max_total_seconds=("total_seconds", "max"),
            max_peak_non_cache_mb=("peak_non_cache_mb", "max"),
            max_cache_size_mb=("cache_size_mb", "max"),
            max_non_cache_per_second=("non_cache_per_second", "max"),
        )
        .reset_index()
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Safely divide two floats and return ``nan`` for invalid denominators.

    Parameters
    ----------
    numerator : float
        Dividend value.
    denominator : float
        Divisor value.

    Returns
    -------
    float
        Division result or ``nan`` when division is invalid.

    Notes
    -----
    Zero and non-finite denominators are treated as invalid.
    """
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return np.nan
    return float(numerator) / float(denominator)


def _build_normalized_scaling_table(ctx: AnalysisContext) -> pd.DataFrame:
    """Build scale-normalized runtime/RAM/cache table relative to baseline.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context with totals metrics.

    Returns
    -------
    pd.DataFrame
        Normalized scaling table.

    Notes
    -----
    Baseline per profile is ``1x`` when present, else first sorted run.
    """
    rows: list[dict[str, object]] = []
    for profile, profile_df in ctx.totals_df.groupby("profile_label"):
        ordered = profile_df.sort_values("scale_factor")
        baseline_df = ordered[ordered["scale_factor"] == 1]
        baseline = baseline_df.iloc[0] if not baseline_df.empty else ordered.iloc[0]
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "profile_label": profile,
                    "run": row["run"],
                    "scale_factor": row["scale_factor"],
                    "total_seconds_x": _safe_ratio(row["total_seconds"], baseline["total_seconds"]),
                    "peak_non_cache_mb_x": _safe_ratio(row["peak_non_cache_mb"], baseline["peak_non_cache_mb"]),
                    "cache_size_mb_x": _safe_ratio(row["cache_size_mb"], baseline["cache_size_mb"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["profile_label", "scale_factor", "run"]).reset_index(drop=True)


def _build_scaling_exponent_table(ctx: AnalysisContext) -> pd.DataFrame:
    """Build global scaling exponent table for key run-level metrics.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context with totals metrics.

    Returns
    -------
    pd.DataFrame
        Pivoted exponent table with one row per profile.

    Notes
    -----
    Exponents use log-log linear regression ``y ~ n^k``.
    """
    rows: list[dict[str, object]] = []
    metrics = ["total_seconds", "peak_non_cache_mb", "cache_size_mb"]
    for profile, profile_df in ctx.totals_df.groupby("profile_label"):
        x_vals = profile_df.sort_values("scale_factor")["scale_factor"].to_numpy(dtype=float)
        for metric in metrics:
            y_vals = profile_df.sort_values("scale_factor")[metric].to_numpy(dtype=float)
            mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
            slope = np.polyfit(np.log10(x_vals[mask]), np.log10(y_vals[mask]), 1)[0] if mask.sum() >= 2 else np.nan
            rows.append({"profile_label": profile, "metric": metric, "scaling_exponent_k": float(slope)})
    pivot = pd.DataFrame(rows).pivot(index="profile_label", columns="metric", values="scaling_exponent_k")
    return pivot.reset_index()


def _build_run_wise_totals_table(ctx: AnalysisContext) -> pd.DataFrame:
    """Build run-wise totals table with canonical benchmark columns.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context with totals metrics.

    Returns
    -------
    pd.DataFrame
        Sorted run-wise totals table.

    Notes
    -----
    This table is used by run-wise bar plots and ranking tables.
    """
    columns = [
        "profile_label",
        "run",
        "run_label",
        "scale_factor",
        "total_seconds",
        "peak_non_cache_mb",
        "cache_size_mb",
        "non_cache_per_second",
        "min_mem_available_mb",
        "peak_cgroup_current_mb",
        "cgroup_peak_of_limit_pct",
    ]
    return (
        ctx.totals_df[columns]
        .sort_values(["profile_label", "scale_factor", "run"])
        .reset_index(drop=True)
    )


def _build_run_wise_rank_table(run_wise_totals: pd.DataFrame) -> pd.DataFrame:
    """Build run-wise ranking table for runtime and memory pressure metrics.

    Parameters
    ----------
    run_wise_totals : pd.DataFrame
        Run-wise totals table.

    Returns
    -------
    pd.DataFrame
        Table with additional rank columns.

    Notes
    -----
    Lower rank is better for runtime and smaller memory/cache values.
    """
    ranked = run_wise_totals.copy()
    ranked["rank_time"] = ranked["total_seconds"].rank(method="min")
    ranked["rank_non_cache_peak"] = ranked["peak_non_cache_mb"].rank(method="min")
    ranked["rank_cache_size"] = ranked["cache_size_mb"].rank(method="min")
    return ranked


def _build_top_fastest_runs_table(run_wise_rank: pd.DataFrame) -> pd.DataFrame:
    """Build table containing the eight fastest benchmark runs.

    Parameters
    ----------
    run_wise_rank : pd.DataFrame
        Run-wise ranking table.

    Returns
    -------
    pd.DataFrame
        Top-8 fastest runs table.

    Notes
    -----
    Sorted ascending by total runtime.
    """
    return run_wise_rank.sort_values("total_seconds").head(8).reset_index(drop=True)


def _build_top_non_cache_peak_table(run_wise_rank: pd.DataFrame) -> pd.DataFrame:
    """Build table containing the eight highest non-cache RAM pressure runs.

    Parameters
    ----------
    run_wise_rank : pd.DataFrame
        Run-wise ranking table.

    Returns
    -------
    pd.DataFrame
        Top-8 highest non-cache peak runs table.

    Notes
    -----
    Sorted descending by ``peak_non_cache_mb``.
    """
    return run_wise_rank.sort_values("peak_non_cache_mb", ascending=False).head(8).reset_index(drop=True)


def _build_jump_rows_for_metric(profile: str, profile_df: pd.DataFrame, metric: str) -> list[dict[str, object]]:
    """Build per-run jump diagnostics rows for one profile and metric.

    Parameters
    ----------
    profile : str
        Profile label.
    profile_df : pd.DataFrame
        Profile subset sorted by ``scale_factor``.
    metric : str
        Metric name to evaluate.

    Returns
    -------
    list[dict[str, object]]
        Jump diagnostics row list.

    Notes
    -----
    First scale entry naturally contains ``nan`` for delta-based columns.
    """
    ordered = profile_df.sort_values("scale_factor").copy()
    base = ordered.iloc[0]
    scales = ordered["scale_factor"].astype(float)
    values = ordered[metric].astype(float)
    prev_scales = scales.shift(1)
    prev_values = values.shift(1)
    jump_abs = values - prev_values
    jump_pct = (values / prev_values - 1.0) * 100.0
    delta_scale = scales - prev_scales
    marginal = jump_abs / delta_scale
    exponent = np.log(values / prev_values) / np.log(scales / prev_scales)
    linear_expected = base[metric] * (scales / float(base["scale_factor"]))
    overhead = values / linear_expected
    rows: list[dict[str, object]] = []
    for idx in range(len(ordered)):
        rows.append(
            {
                "profile_label": profile,
                "run": ordered.iloc[idx]["run"],
                "scale_factor": scales.iloc[idx],
                "metric": metric,
                "value": values.iloc[idx],
                "prev_scale": prev_scales.iloc[idx],
                "prev_value": prev_values.iloc[idx],
                "jump_abs": jump_abs.iloc[idx],
                "jump_pct": jump_pct.iloc[idx],
                "marginal_per_scale": marginal.iloc[idx],
                "local_exponent": exponent.iloc[idx],
                "linear_expected_from_1x": linear_expected.iloc[idx],
                "overhead_vs_linear": overhead.iloc[idx],
            }
        )
    return rows


def _build_jump_table(ctx: AnalysisContext) -> pd.DataFrame:
    """Build jump diagnostics table across key totals metrics.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context with totals metrics.

    Returns
    -------
    pd.DataFrame
        Jump diagnostics table sorted by metric/profile/scale.

    Notes
    -----
    Metrics include runtime, RAM pressure peak, and cache size.
    """
    rows: list[dict[str, object]] = []
    for profile, profile_df in ctx.totals_df.groupby("profile_label"):
        for metric in ["total_seconds", "peak_non_cache_mb", "cache_size_mb"]:
            rows.extend(_build_jump_rows_for_metric(profile, profile_df, metric))
    out = pd.DataFrame(rows)
    return out.sort_values(["metric", "profile_label", "scale_factor"]).reset_index(drop=True)


def _build_largest_jumps_table(jump_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-metric/profile table of largest relative jumps.

    Parameters
    ----------
    jump_df : pd.DataFrame
        Jump diagnostics table.

    Returns
    -------
    pd.DataFrame
        Largest jump rows for each metric/profile pair.

    Notes
    -----
    Rows are selected by maximum ``jump_pct``.
    """
    selected = (
        jump_df.dropna(subset=["jump_pct"])
        .sort_values("jump_pct", ascending=False)
        .groupby(["metric", "profile_label"], as_index=False)
        .first()
    )
    cols = ["metric", "profile_label", "scale_factor", "jump_pct", "jump_abs", "marginal_per_scale"]
    return selected[cols].reset_index(drop=True)


def _collect_all_tables(
    ctx: AnalysisContext,
    normalized_scaling: pd.DataFrame,
    run_wise_totals: pd.DataFrame,
    jump_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Collect all notebook-equivalent tables for CSV export.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context with totals and steps data.
    normalized_scaling : pd.DataFrame
        Normalized scaling table.
    run_wise_totals : pd.DataFrame
        Run-wise totals table.
    jump_df : pd.DataFrame
        Jump diagnostics table.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from table export stem to DataFrame.

    Notes
    -----
    Output includes both pre-master tables and final summary table.
    """
    run_wise_rank = _build_run_wise_rank_table(run_wise_totals)
    largest_jumps = _build_largest_jumps_table(jump_df)
    return {
        "totals_overview": _build_totals_overview_table(ctx),
        "summary_by_profile": _build_summary_by_profile_table(ctx),
        "normalized_scaling": normalized_scaling,
        "scaling_exponents": _build_scaling_exponent_table(ctx),
        "run_wise_totals": run_wise_totals,
        "run_wise_rank": run_wise_rank,
        "top8_fastest_runs": _build_top_fastest_runs_table(run_wise_rank),
        "top8_highest_non_cache_peak": _build_top_non_cache_peak_table(run_wise_rank),
        "jump_diagnostics": jump_df,
        "largest_jumps": largest_jumps,
        "summary_table_old_columns_score_removed": build_requested_table(ctx),
    }


def _csv_ready_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize one table for stable CSV export.

    Parameters
    ----------
    table : pd.DataFrame
        Source table.

    Returns
    -------
    pd.DataFrame
        Table with index columns materialized when needed.

    Notes
    -----
    This avoids losing named indices in CSV output.
    """
    has_named_index = table.index.name is not None
    has_multi_index = isinstance(table.index, pd.MultiIndex)
    return table.reset_index() if (has_named_index or has_multi_index) else table


def _export_tables_csv(tables: dict[str, pd.DataFrame], export_dir: Path) -> None:
    """Export all analysis tables as CSV files.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Mapping of table name to DataFrame.
    export_dir : Path
        Root export directory for figures and tables.

    Returns
    -------
    None
        CSV files are written to ``export_dir / "tables"``.

    Notes
    -----
    File names are slugified for cross-platform safety.
    """
    table_dir = export_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        stem = _slugify(name)
        csv_path = table_dir / f"{stem}.csv"
        _csv_ready_table(table).to_csv(csv_path, index=False)


def _overall_common_columns() -> list[str]:
    """Return shared identity columns used by overall data export.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Shared run identity columns.

    Notes
    -----
    Column order is stable and used across summary and step records.
    """
    return ["profile", "profile_label", "run", "run_key", "run_label", "scale_factor"]


def _overall_summary_metric_columns() -> list[str]:
    """Return raw summary metric columns from benchmark JSON payloads.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Run-level summary metric columns.

    Notes
    -----
    Metrics map directly to ``summary.json`` values loaded per run.
    """
    return [
        "total_seconds",
        "peak_non_cache_mb",
        "min_mem_available_mb",
        "peak_cgroup_current_mb",
        "cgroup_limit_mb",
        "cgroup_peak_of_limit_pct",
        "cache_size_mb",
    ]


def _overall_step_metric_columns() -> list[str]:
    """Return raw step metric columns from benchmark step payloads.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Step-level metric columns.

    Notes
    -----
    Metrics map directly to values loaded from ``steps.json``.
    """
    return [
        "seconds",
        "non_cache_start_mb",
        "non_cache_end_mb",
        "non_cache_peak_mb",
        "mem_available_start_mb",
        "mem_available_end_mb",
        "mem_available_min_mb",
        "cgroup_current_start_mb",
        "cgroup_current_end_mb",
        "cgroup_current_peak_mb",
        "cgroup_limit_mb",
        "cache_size_mb",
    ]


def _overall_data_columns() -> list[str]:
    """Return canonical column order for ``overall_data.csv`` export.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Overall data CSV columns in deterministic order.

    Notes
    -----
    Summary and step metrics are namespace-prefixed for clarity.
    """
    summary_cols = [f"summary_{name}" for name in _overall_summary_metric_columns()]
    step_cols = [f"step_{name}" for name in _overall_step_metric_columns()]
    return [
        *_overall_common_columns(),
        "record_type",
        "step_name",
        "step_index",
        *summary_cols,
        *step_cols,
    ]


def _build_overall_summary_frame(totals: pd.DataFrame) -> pd.DataFrame:
    """Build summary-record rows for overall benchmark data CSV.

    Parameters
    ----------
    totals : pd.DataFrame
        Totals DataFrame loaded from benchmark results.

    Returns
    -------
    pd.DataFrame
        Summary records aligned to overall CSV schema.

    Notes
    -----
    Step-specific columns are kept empty for summary rows.
    """
    common = _overall_common_columns()
    summary_metrics = _overall_summary_metric_columns()
    frame = totals[common + summary_metrics].copy()
    frame["record_type"] = "summary"
    frame["step_name"] = pd.NA
    frame["step_index"] = pd.Series([pd.NA] * len(frame.index), dtype="Int64")
    frame = frame.rename(columns={name: f"summary_{name}" for name in summary_metrics})
    for name in _overall_step_metric_columns():
        frame[f"step_{name}"] = np.nan
    return frame.reindex(columns=_overall_data_columns())


def _build_overall_step_frame(steps: pd.DataFrame) -> pd.DataFrame:
    """Build step-record rows for overall benchmark data CSV.

    Parameters
    ----------
    steps : pd.DataFrame
        Steps DataFrame loaded from benchmark results.

    Returns
    -------
    pd.DataFrame
        Step records aligned to overall CSV schema.

    Notes
    -----
    Summary-specific columns are kept empty for step rows.
    """
    common = _overall_common_columns()
    step_metrics = _overall_step_metric_columns()
    frame = steps[common + ["step", "step_index"] + step_metrics].copy()
    frame["record_type"] = "step"
    frame = frame.rename(columns={"step": "step_name", **{name: f"step_{name}" for name in step_metrics}})
    frame["step_index"] = pd.to_numeric(frame["step_index"], errors="coerce").astype("Int64")
    for name in _overall_summary_metric_columns():
        frame[f"summary_{name}"] = np.nan
    return frame.reindex(columns=_overall_data_columns())


def _build_overall_data_table(totals: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Build complete raw-data table for ``overall_data.csv`` export.

    Parameters
    ----------
    totals : pd.DataFrame
        Totals DataFrame loaded from benchmark results.
    steps : pd.DataFrame
        Steps DataFrame loaded from benchmark results.

    Returns
    -------
    pd.DataFrame
        Combined summary+step rows in one table.

    Notes
    -----
    Output is sorted by profile, scale, run, record type, and step index.
    """
    summary_frame = _build_overall_summary_frame(totals)
    step_frame = _build_overall_step_frame(steps)
    combined = pd.concat([summary_frame, step_frame], ignore_index=True)
    return combined.sort_values(["profile", "scale_factor", "run", "record_type", "step_index"]).reset_index(drop=True)


def _export_overall_data_csv(totals: pd.DataFrame, steps: pd.DataFrame, export_dir: Path) -> None:
    """Export one complete benchmark raw-data CSV as ``overall_data.csv``.

    Parameters
    ----------
    totals : pd.DataFrame
        Totals DataFrame loaded from benchmark results.
    steps : pd.DataFrame
        Steps DataFrame loaded from benchmark results.
    export_dir : Path
        Root export directory.

    Returns
    -------
    None
        ``overall_data.csv`` is written for side effects.

    Notes
    -----
    This file contains all loaded summary and step records in one CSV and is
    written under ``export_dir / "tables"``.
    """
    overall_data = _build_overall_data_table(totals, steps)
    table_dir = export_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    overall_data.to_csv(table_dir / "overall_data.csv", index=False)
    legacy_path = export_dir / "overall_data.csv"
    if legacy_path.exists():
        legacy_path.unlink()


def _table_plot_title(table_name: str) -> str:
    """Create human-readable title from one internal table name.

    Parameters
    ----------
    table_name : str
        Internal table key.

    Returns
    -------
    str
        Readable title string.

    Notes
    -----
    Underscores are converted to spaces and words are title-cased.
    """
    words = str(table_name).replace("_", " ").strip().split()
    return " ".join(word.capitalize() for word in words) if words else "Table"


def _truncate_text(value: object, max_chars: int = 42) -> str:
    """Convert one table cell value to bounded display text.

    Parameters
    ----------
    value : object
        Arbitrary cell value.
    max_chars : int, default=42
        Maximum output character count.

    Returns
    -------
    str
        Bounded cell text.

    Notes
    -----
    Truncation keeps the start of long values and appends ``...``.
    """
    text = str(value)
    if len(text) <= int(max_chars):
        return text
    return f"{text[: max(0, int(max_chars) - 3)]}..."


def _table_display_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Build display-optimized table frame for figure rendering.

    Parameters
    ----------
    table : pd.DataFrame
        Source table.

    Returns
    -------
    pd.DataFrame
        String-formatted table for plotting.

    Notes
    -----
    Long cell values are shortened for readability.
    """
    frame = _csv_ready_table(table).copy()
    for column in frame.columns:
        frame[column] = frame[column].map(_truncate_text)
    return frame


def _table_figure_size(row_count: int, col_count: int) -> tuple[float, float]:
    """Compute deterministic figure size for one table image.

    Parameters
    ----------
    row_count : int
        Number of table rows.
    col_count : int
        Number of table columns.

    Returns
    -------
    tuple[float, float]
        Width and height in inches.

    Notes
    -----
    Sizes are bounded to avoid oversized or unreadably small images.
    """
    width = min(30.0, max(8.8, 1.8 + 1.7 * max(1, int(col_count))))
    height = min(40.0, max(3.8, 1.8 + 0.34 * max(1, int(row_count))))
    return float(width), float(height)


def _table_fontsize(row_count: int) -> float:
    """Select table font size from row count.

    Parameters
    ----------
    row_count : int
        Number of table rows.

    Returns
    -------
    float
        Font size in points.

    Notes
    -----
    Larger tables get smaller text to keep cells visible.
    """
    if row_count <= 12:
        return 8.0
    if row_count <= 28:
        return 7.0
    if row_count <= 60:
        return 6.2
    return 5.4


def _render_table_plot_figure(table_name: str, table: pd.DataFrame) -> Figure:
    """Render one data table as a standalone figure.

    Parameters
    ----------
    table_name : str
        Internal table key used for titles and filename hints.
    table : pd.DataFrame
        Source table.

    Returns
    -------
    Figure
        Table figure.

    Notes
    -----
    Empty tables render a centered fallback message.
    """
    frame = _table_display_frame(table)
    width, height = _table_figure_size(len(frame.index), len(frame.columns))
    fig, ax = plt.subplots(1, 1, figsize=(width, height), constrained_layout=True)
    ax.axis("off")
    if frame.empty:
        ax.text(0.5, 0.5, f"{_table_plot_title(table_name)}\n(no rows)", ha="center", va="center", fontsize=11)
    else:
        artist = ax.table(cellText=frame.values, colLabels=frame.columns, loc="center", cellLoc="center")
        artist.auto_set_font_size(False)
        artist.set_fontsize(_table_fontsize(len(frame.index)))
        artist.scale(1.02, 1.18)
    ax.set_title(_table_plot_title(table_name), fontsize=11, pad=8)
    fig._mdxplain_filename_hint = f"table_{_slugify(table_name)}"
    return fig


def _export_table_plot_figures(
    tables: dict[str, pd.DataFrame],
    export_dir: Path,
    filetypes: list[str],
    dpi: int,
) -> None:
    """Export all tables as standalone plot figures.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Mapping from table key to DataFrame.
    export_dir : Path
        Root export directory.
    filetypes : list[str]
        Requested export file types.
    dpi : int
        Raster export DPI.

    Returns
    -------
    None
        Table plot figures are written and closed for side effects.

    Notes
    -----
    Figures are exported one-by-one to minimize memory footprint.
    """
    for table_name, table in tables.items():
        figure = _render_table_plot_figure(table_name, table)
        _export_figures([figure], export_dir, filetypes, dpi)


def _compute_bubble_sizes(cache_vals: np.ndarray, min_size: float = 520.0, max_size: float = 4200.0) -> np.ndarray:
    """Scale cache values to bubble marker areas.

    Parameters
    ----------
    cache_vals : np.ndarray
        Cache values in MB.
    min_size : float, default=520.0
        Minimum marker area in pt^2.
    max_size : float, default=4200.0
        Maximum marker area in pt^2.

    Returns
    -------
    np.ndarray
        Marker areas in pt^2.

    Notes
    -----
    Linear min-max scaling is used on finite values.
    """
    vals = np.asarray(cache_vals, dtype=float)
    if vals.size == 0:
        return vals

    finite = np.isfinite(vals)
    out = np.full(vals.shape, min_size)
    if not finite.any():
        return out

    clean = vals[finite]
    vmin, vmax = float(np.nanmin(clean)), float(np.nanmax(clean))
    if np.isclose(vmin, vmax):
        out[finite] = (min_size + max_size) / 2.0
        return out

    spread = np.clip((clean - vmin) / (vmax - vmin), 0.0, 1.0)
    out[finite] = min_size + spread * (max_size - min_size)
    return out


def _marker_radius_points(area_points2: float) -> float:
    """Convert marker area to marker radius in points.

    Parameters
    ----------
    area_points2 : float
        Marker area in points squared.

    Returns
    -------
    float
        Marker radius in points.

    Notes
    -----
    Radius uses circular-equivalent conversion.
    """
    return float(np.sqrt(max(area_points2, 0.0) / np.pi))


def _build_bubble_geometry(ax, points_df: pd.DataFrame) -> list[dict[str, float]]:
    """Build bubble geometry in display coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for coordinate transforms.
    points_df : pd.DataFrame
        Trade-off points including ``bubble_size``.

    Returns
    -------
    list[dict[str, float]]
        Bubble center/radius descriptors.

    Notes
    -----
    Display coordinates are used for robust collision checks.
    """
    px_per_point = ax.figure.dpi / 72.0
    bubbles: list[dict[str, float]] = []
    for row in points_df.itertuples(index=False):
        center_x, center_y = ax.transData.transform((row.total_seconds, row.peak_non_cache_mb))
        radius_px = _marker_radius_points(float(row.bubble_size)) * px_per_point
        bubbles.append({"center_x": float(center_x), "center_y": float(center_y), "radius_px": float(radius_px)})
    return bubbles


def _bbox_overlaps_bubbles(bbox, bubbles: list[dict[str, float]], pad_px: float = 2.0) -> bool:
    """Check whether text bounding box intersects any bubble.

    Parameters
    ----------
    bbox : Bbox
        Candidate text bounding box.
    bubbles : list[dict[str, float]]
        Bubble geometry descriptors.
    pad_px : float, default=2.0
        Extra collision padding in pixels.

    Returns
    -------
    bool
        True when a collision is detected.

    Notes
    -----
    Distance check uses nearest-point-to-rectangle logic.
    """
    for bubble in bubbles:
        center_x = bubble["center_x"]
        center_y = bubble["center_y"]
        radius = bubble["radius_px"] + pad_px
        near_x = min(max(center_x, bbox.x0), bbox.x1)
        near_y = min(max(center_y, bbox.y0), bbox.y1)
        if (near_x - center_x) ** 2 + (near_y - center_y) ** 2 < radius**2:
            return True
    return False


def _bbox_overlaps_label_boxes(bbox, label_boxes, pad_px: float = 2.0) -> bool:
    """Check collision against already placed label boxes.

    Parameters
    ----------
    bbox : Bbox
        Candidate text bounding box.
    label_boxes : list[dict[str, float]]
        Previously placed label boxes.
    pad_px : float, default=2.0
        Extra collision padding in pixels.

    Returns
    -------
    bool
        True when a collision is detected.

    Notes
    -----
    Boxes are expanded by ``pad_px`` before overlap test.
    """
    if not label_boxes:
        return False

    x0 = float(bbox.x0) - pad_px
    y0 = float(bbox.y0) - pad_px
    x1 = float(bbox.x1) + pad_px
    y1 = float(bbox.y1) + pad_px

    for other in label_boxes:
        if x1 < other["x0"] or x0 > other["x1"]:
            continue
        if y1 < other["y0"] or y0 > other["y1"]:
            continue
        return True
    return False


def _point_in_rect(point: tuple[float, float], rect: tuple[float, float, float, float]) -> bool:
    """Check whether a point lies inside an axis-aligned rectangle.

    Parameters
    ----------
    point : tuple[float, float]
        Point coordinates.
    rect : tuple[float, float, float, float]
        Rectangle bounds ``(x0, y0, x1, y1)``.

    Returns
    -------
    bool
        True when point lies inside rectangle.

    Notes
    -----
    Rectangle bounds are treated as inclusive.
    """
    x, y = point
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
    """Compute orientation cross-product helper for segment tests.

    Parameters
    ----------
    p : tuple[float, float]
        Segment point A.
    q : tuple[float, float]
        Segment point B.
    r : tuple[float, float]
        Test point.

    Returns
    -------
    float
        Signed orientation value.

    Notes
    -----
    Positive values indicate counter-clockwise turn.
    """
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_segment(
    p: tuple[float, float],
    q: tuple[float, float],
    r: tuple[float, float],
    eps: float,
) -> bool:
    """Check whether point ``r`` lies on segment ``pq``.

    Parameters
    ----------
    p : tuple[float, float]
        Segment start point.
    q : tuple[float, float]
        Segment end point.
    r : tuple[float, float]
        Candidate on-segment point.
    eps : float
        Tolerance.

    Returns
    -------
    bool
        True when ``r`` lies on segment bounds.

    Notes
    -----
    Bounding-box check uses epsilon-expanded limits.
    """
    return (
        min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps
        and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps
    )


def _segments_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
    eps: float = 1e-9,
) -> bool:
    """Check whether two line segments intersect.

    Parameters
    ----------
    a0 : tuple[float, float]
        Segment A start.
    a1 : tuple[float, float]
        Segment A end.
    b0 : tuple[float, float]
        Segment B start.
    b1 : tuple[float, float]
        Segment B end.
    eps : float, default=1e-9
        Numeric tolerance.

    Returns
    -------
    bool
        True when segments intersect.

    Notes
    -----
    Proper and colinear intersection cases are both handled.
    """
    o1 = _orient(a0, a1, b0)
    o2 = _orient(a0, a1, b1)
    o3 = _orient(b0, b1, a0)
    o4 = _orient(b0, b1, a1)

    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    if abs(o1) <= eps and _on_segment(a0, a1, b0, eps):
        return True
    if abs(o2) <= eps and _on_segment(a0, a1, b1, eps):
        return True
    if abs(o3) <= eps and _on_segment(b0, b1, a0, eps):
        return True
    if abs(o4) <= eps and _on_segment(b0, b1, a1, eps):
        return True
    return False


def _segment_intersects_rect(
    p0: tuple[float, float],
    p1: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> bool:
    """Check whether a line segment intersects an axis-aligned rectangle.

    Parameters
    ----------
    p0 : tuple[float, float]
        Segment start point.
    p1 : tuple[float, float]
        Segment end point.
    rect : tuple[float, float, float, float]
        Rectangle bounds ``(x0, y0, x1, y1)``.

    Returns
    -------
    bool
        True when segment intersects rectangle.

    Notes
    -----
    Endpoint containment and edge intersections are tested.
    """
    x0, y0, x1, y1 = rect
    if max(p0[0], p1[0]) < x0 or min(p0[0], p1[0]) > x1:
        return False
    if max(p0[1], p1[1]) < y0 or min(p0[1], p1[1]) > y1:
        return False
    if _point_in_rect(p0, rect) or _point_in_rect(p1, rect):
        return True

    c0 = (x0, y0)
    c1 = (x1, y0)
    c2 = (x1, y1)
    c3 = (x0, y1)
    edges = [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]
    return any(_segments_intersect(p0, p1, e0, e1) for e0, e1 in edges)


def _bbox_overlaps_segments(bbox, segments, pad_px: float = 2.5) -> bool:
    """Check whether text bbox intersects any arrow segment.

    Parameters
    ----------
    bbox : Bbox
        Candidate text bounding box.
    segments : list[dict[str, float]]
        Arrow segments in display coordinates.
    pad_px : float, default=2.5
        Extra collision padding in pixels.

    Returns
    -------
    bool
        True when collision is detected.

    Notes
    -----
    Segment checks are performed against expanded rectangle bounds.
    """
    if not segments:
        return False

    rect = (
        float(bbox.x0) - pad_px,
        float(bbox.y0) - pad_px,
        float(bbox.x1) + pad_px,
        float(bbox.y1) + pad_px,
    )
    for seg in segments:
        p0 = (float(seg["x0"]), float(seg["y0"]))
        p1 = (float(seg["x1"]), float(seg["y1"]))
        if _segment_intersects_rect(p0, p1, rect):
            return True
    return False


def _format_log_tick_general(value: float) -> str:
    """Format positive log tick as ``Mx10^n`` string.

    Parameters
    ----------
    value : float
        Tick value.

    Returns
    -------
    str
        Formatted tick label text.

    Notes
    -----
    Non-positive or non-finite values return empty labels.
    """
    if not np.isfinite(value) or value <= 0.0:
        return ""

    exponent = int(np.floor(np.log10(value)))
    base = 10.0**exponent
    mantissa = value / base
    rounded = int(np.round(mantissa))

    if 1 <= rounded <= 9 and abs(mantissa - rounded) < 0.08:
        return f"10^{exponent}" if rounded == 1 else f"{rounded}x10^{exponent}"
    mtxt = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    return f"{mtxt}x10^{exponent}"


def _build_fixed_ticks(
    min_value: float,
    max_value: float,
    *,
    count: int,
    mantissas: list[int],
) -> list[float]:
    """Build fixed logarithmic ticks within axis limits.

    Parameters
    ----------
    min_value : float
        Lower axis limit.
    max_value : float
        Upper axis limit.
    count : int
        Target number of ticks.
    mantissas : list[int]
        Candidate mantissas for powers of ten.

    Returns
    -------
    list[float]
        Tick positions inside current limits.

    Notes
    -----
    Falls back to geometric spacing when discrete candidates are insufficient.
    """
    count = max(2, int(count))
    vmin = float(min_value)
    vmax = float(max_value)
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return []
    if vmin <= 0.0 or vmax <= 0.0:
        return []
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    log_min = float(np.log10(vmin))
    log_max = float(np.log10(vmax))
    exp_min = int(np.floor(log_min)) - 2
    exp_max = int(np.ceil(log_max)) + 2

    candidates = []
    for exp in range(exp_min, exp_max + 1):
        scale = 10.0**exp
        for mantissa in mantissas:
            value = float(mantissa * scale)
            if vmin <= value <= vmax:
                candidates.append(value)
    candidates = sorted(set(candidates))

    if len(candidates) >= count:
        targets = np.linspace(log_min, log_max, count)
        return _pick_nearest_ticks(candidates, targets, count)
    return [float(value) for value in np.geomspace(vmin, vmax, count)]


def _pick_nearest_ticks(candidates: list[float], targets: np.ndarray, count: int) -> list[float]:
    """Pick nearest unique candidate ticks for target log-space points.

    Parameters
    ----------
    candidates : list[float]
        Candidate tick values.
    targets : np.ndarray
        Target positions in log10 space.
    count : int
        Desired result count.

    Returns
    -------
    list[float]
        Selected tick values.

    Notes
    -----
    Selection preserves uniqueness by removing each chosen candidate.
    """
    remaining = candidates[:]
    chosen: list[float] = []
    for target in targets:
        if not remaining:
            break
        idx = min(range(len(remaining)), key=lambda i: abs(np.log10(remaining[i]) - target))
        chosen.append(remaining.pop(idx))
    chosen = sorted(set(chosen))
    if len(chosen) == count:
        return chosen
    return [float(value) for value in np.geomspace(min(candidates), max(candidates), count)]


def _spread_tradeoff_points(points_df: pd.DataFrame, spread_log10: float = 0.0, order: Optional[list[str]] = None) -> pd.DataFrame:
    """Spread same-scale tradeoff points in log-space to reduce overlaps.

    Parameters
    ----------
    points_df : pd.DataFrame
        Tradeoff points.
    spread_log10 : float, default=0.0
        Spread magnitude in log10-space.
    order : list[str] or None, default=None
        Profile ordering for deterministic spread direction.

    Returns
    -------
    pd.DataFrame
        Possibly adjusted tradeoff points.

    Notes
    -----
    Spreading is disabled when magnitude is non-positive.
    """
    spread = float(spread_log10)
    if points_df.empty or (not np.isfinite(spread)) or spread <= 0.0:
        return points_df

    out = points_df.copy()
    ranking = {profile: idx for idx, profile in enumerate(order or [])}
    for _scale, idxs in out.groupby("scale_factor").groups.items():
        ordered = sorted(list(idxs), key=lambda idx: ranking.get(out.at[idx, "profile_label"], 999))
        if len(ordered) < 2:
            continue
        _spread_scale_group(out, ordered, spread)
    return out


def _spread_scale_group(points: pd.DataFrame, ordered_indices: list[int], spread: float) -> None:
    """Apply spread transform to one scale-group of tradeoff points.

    Parameters
    ----------
    points : pd.DataFrame
        Mutable points DataFrame.
    ordered_indices : list[int]
        Ordered row indices for one scale factor.
    spread : float
        Spread magnitude in log10-space.

    Returns
    -------
    None
        DataFrame rows are modified in place.

    Notes
    -----
    Isotropic fallback directions are used for zero-norm vectors.
    """
    log_x = np.log10(points.loc[ordered_indices, "total_seconds"].to_numpy(dtype=float))
    log_y = np.log10(points.loc[ordered_indices, "peak_non_cache_mb"].to_numpy(dtype=float))
    center_x = float(np.nanmean(log_x))
    center_y = float(np.nanmean(log_y))

    for pos, idx in enumerate(ordered_indices):
        dx = float(log_x[pos] - center_x)
        dy = float(log_y[pos] - center_y)
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-9:
            angle = 2.0 * np.pi * float(pos) / float(max(len(ordered_indices), 1))
            ux, uy = float(np.cos(angle)), float(np.sin(angle))
        else:
            ux, uy = dx / norm, dy / norm

        points.at[idx, "total_seconds"] = 10.0 ** (log_x[pos] + spread * ux)
        points.at[idx, "peak_non_cache_mb"] = 10.0 ** (log_y[pos] + spread * uy)


def _format_speed_factor(value: float) -> str:
    """Format speed transition factor text.

    Parameters
    ----------
    value : float
        Speed factor value.

    Returns
    -------
    str
        Compact factor text.

    Notes
    -----
    Near-integer values are rendered without decimal places.
    """
    if not np.isfinite(value):
        return "xnan"
    rounded = int(np.round(value))
    if abs(value - rounded) < 0.18:
        return f"x{rounded}"
    return f"x{value:.1f}".rstrip("0").rstrip(".")


def _format_ram_factor(value: float) -> str:
    """Format RAM transition factor text.

    Parameters
    ----------
    value : float
        RAM factor value.

    Returns
    -------
    str
        Compact factor text.

    Notes
    -----
    Higher values use one decimal place for readability.
    """
    if not np.isfinite(value):
        return "xnan"
    if value >= 10.0:
        return f"x{value:.1f}".rstrip("0").rstrip(".")
    return f"x{value:.2f}".rstrip("0").rstrip(".")


def _line_unit_vectors(src_px: np.ndarray, dst_px: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute unit vectors along and perpendicular to one arrow segment.

    Parameters
    ----------
    src_px : np.ndarray
        Source point in display coordinates.
    dst_px : np.ndarray
        Destination point in display coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Along-unit vector, perpendicular-unit vector, and segment norm.

    Notes
    -----
    Norm is returned to allow short-segment filtering by callers.
    """
    vec = dst_px - src_px
    norm = float(np.hypot(vec[0], vec[1]))
    if norm <= 1e-9:
        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), norm
    along = vec / norm
    perp = np.asarray([-along[1], along[0]], dtype=float)
    return along, perp, norm


def _edge_to_edge_arrow(
    src_center_px: np.ndarray,
    dst_center_px: np.ndarray,
    src_radius_px: float,
    dst_radius_px: float,
    edge_pad_px: float = 1.6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build edge-to-edge arrow endpoints between bubble circles.

    Parameters
    ----------
    src_center_px : np.ndarray
        Source bubble center in display coordinates.
    dst_center_px : np.ndarray
        Destination bubble center in display coordinates.
    src_radius_px : float
        Source bubble radius in pixels.
    dst_radius_px : float
        Destination bubble radius in pixels.
    edge_pad_px : float, default=1.6
        Extra edge offset padding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        Arrow start/end points, or ``None`` for degenerate segments.

    Notes
    -----
    Arrows shorter than two pixels are suppressed.
    """
    along, _perp, norm = _line_unit_vectors(src_center_px, dst_center_px)
    if norm <= 1e-6:
        return None

    start_px = src_center_px + along * (src_radius_px + edge_pad_px)
    end_px = dst_center_px - along * (dst_radius_px + edge_pad_px)
    if float(np.hypot(*(end_px - start_px))) <= 2.0:
        return None
    return start_px, end_px


def _candidate_label_offsets(along: np.ndarray, perp: np.ndarray) -> list[np.ndarray]:
    """Build candidate label offset vectors in display points.

    Parameters
    ----------
    along : np.ndarray
        Along-arrow unit vector.
    perp : np.ndarray
        Perpendicular-arrow unit vector.

    Returns
    -------
    list[np.ndarray]
        Candidate offset vectors.

    Notes
    -----
    Near-arrow candidates are prioritized before wider fallback offsets.
    """
    offsets: list[np.ndarray] = []
    for dist in [10.0, 13.0, 17.0, 22.0]:
        for side in [1.0, -1.0]:
            base = perp * (dist * side)
            offsets.extend([base, base + along * 2.2, base - along * 2.2])
    for dist in [26.0, 32.0, 40.0, 52.0]:
        for angle_deg in [90, -90, 65, -65, 115, -115, 45, -45]:
            theta = np.deg2rad(float(angle_deg))
            offsets.append(np.asarray([np.cos(theta) * dist, np.sin(theta) * dist], dtype=float))
    return offsets


def _inside_axes_bbox(bbox, axes_bbox, pad_px: float = 2.0) -> bool:
    """Check whether a text bbox lies fully inside the axis bbox.

    Parameters
    ----------
    bbox : Bbox
        Candidate text bounding box.
    axes_bbox : Bbox
        Axis bounding box.
    pad_px : float, default=2.0
        Inner padding from axis edges.

    Returns
    -------
    bool
        True when bbox lies inside padded axis bounds.

    Notes
    -----
    Strict containment avoids clipping near axes borders.
    """
    return (
        bbox.x0 >= axes_bbox.x0 + pad_px
        and bbox.y0 >= axes_bbox.y0 + pad_px
        and bbox.x1 <= axes_bbox.x1 - pad_px
        and bbox.y1 <= axes_bbox.y1 - pad_px
    )


def _append_label_bbox(label_boxes: list[dict[str, float]], bbox) -> None:
    """Append normalized bbox coordinates to occupied label list.

    Parameters
    ----------
    label_boxes : list[dict[str, float]]
        Mutable label bbox list.
    bbox : Bbox
        Label bounding box.

    Returns
    -------
    None
        Label box list is updated in place.

    Notes
    -----
    Coordinates are stored in display pixel space.
    """
    label_boxes.append(
        {
            "x0": float(bbox.x0),
            "y0": float(bbox.y0),
            "x1": float(bbox.x1),
            "y1": float(bbox.y1),
        }
    )


def _label_artist_is_valid(
    artist,
    axes_bbox,
    bubbles: list[dict[str, float]],
    arrow_segments,
    label_boxes,
    renderer,
) -> bool:
    """Validate candidate label artist against overlap constraints.

    Parameters
    ----------
    artist : Annotation
        Candidate annotation artist.
    axes_bbox : Bbox
        Axis display bounding box.
    bubbles : list[dict[str, float]]
        Bubble geometry descriptors.
    arrow_segments : list[dict[str, float]]
        Arrow segment descriptors.
    label_boxes : list[dict[str, float]]
        Placed label boxes.
    renderer : RendererBase
        Active canvas renderer.

    Returns
    -------
    bool
        True when candidate is collision-free.

    Notes
    -----
    Candidate bbox is expanded slightly to include label padding.
    """
    bbox = artist.get_window_extent(renderer=renderer).expanded(1.03, 1.11)
    if not _inside_axes_bbox(bbox, axes_bbox, pad_px=2.0):
        return False
    if _bbox_overlaps_bubbles(bbox, bubbles, pad_px=10.0):
        return False
    if _bbox_overlaps_segments(bbox, arrow_segments, pad_px=3.0):
        return False
    if _bbox_overlaps_label_boxes(bbox, label_boxes, pad_px=2.0):
        return False
    _append_label_bbox(label_boxes, bbox)
    return True


def _create_transition_label_artist(ax, text: str, xy_data: tuple[float, float], offset: np.ndarray):
    """Create one transition label artist at given offset.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    text : str
        Label text.
    xy_data : tuple[float, float]
        Label anchor in data coordinates.
    offset : np.ndarray
        Offset vector in points.

    Returns
    -------
    Annotation
        Created annotation artist.

    Notes
    -----
    Styling is shared across regular and fallback transition labels.
    """
    return ax.annotate(
        text,
        xy=xy_data,
        xytext=(float(offset[0]), float(offset[1])),
        textcoords="offset points",
        annotation_clip=False,
        fontsize=6.5,
        color="dimgray",
        ha="center",
        va="center",
        linespacing=0.92,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="lightgray", linewidth=0.35, pad=0.65),
        zorder=9,
    )


def _fallback_transition_label(ax, text: str, xy_data: tuple[float, float]):
    """Create fallback transition label artist.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    text : str
        Label text.
    xy_data : tuple[float, float]
        Label anchor in data coordinates.

    Returns
    -------
    Annotation
        Fallback annotation artist.

    Notes
    -----
    Fallback offset is placed below the arrow midpoint.
    """
    return _create_transition_label_artist(ax, text, xy_data, np.asarray([0.0, -24.0], dtype=float))


def _place_transition_label(
    ax,
    text: str,
    mid_px: np.ndarray,
    *,
    along: np.ndarray,
    perp: np.ndarray,
    bubbles: list[dict[str, float]],
    arrow_segments,
    renderer,
    label_boxes,
):
    """Place one transition label near arrow while avoiding overlaps.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    text : str
        Label text.
    mid_px : np.ndarray
        Arrow midpoint in display coordinates.
    along : np.ndarray
        Along-arrow unit vector.
    perp : np.ndarray
        Perpendicular-arrow unit vector.
    bubbles : list[dict[str, float]]
        Bubble geometry descriptors.
    arrow_segments : list[dict[str, float]]
        Arrow segment descriptors.
    renderer : RendererBase
        Active canvas renderer.
    label_boxes : list[dict[str, float]]
        Placed label box descriptors.

    Returns
    -------
    Annotation
        Placed label artist.

    Notes
    -----
    Fallback placement is used when all candidates collide.
    """
    axes_bbox = ax.get_window_extent(renderer=renderer)
    xy_data = tuple(ax.transData.inverted().transform(mid_px))
    for vec in _candidate_label_offsets(along, perp):
        artist = _create_transition_label_artist(ax, text, xy_data, vec)
        if _label_artist_is_valid(artist, axes_bbox, bubbles, arrow_segments, label_boxes, renderer):
            return artist
        artist.remove()
    fallback = _fallback_transition_label(ax, text, xy_data)
    _append_label_bbox(label_boxes, fallback.get_window_extent(renderer=renderer).expanded(1.03, 1.11))
    return fallback


def _transition_centers(ax, src_row, dst_row) -> tuple[np.ndarray, np.ndarray]:
    """Resolve source/destination transition centers in display coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    src_row : Any
        Source tradeoff row.
    dst_row : Any
        Destination tradeoff row.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Source and destination display points.

    Notes
    -----
    Display coordinates are required for edge-to-edge arrow construction.
    """
    src_center = np.asarray(ax.transData.transform((src_row.total_seconds, src_row.peak_non_cache_mb)), dtype=float)
    dst_center = np.asarray(ax.transData.transform((dst_row.total_seconds, dst_row.peak_non_cache_mb)), dtype=float)
    return src_center, dst_center


def _transition_endpoint_pixels(ax, src_row, dst_row, src_center: np.ndarray, dst_center: np.ndarray):
    """Resolve edge-to-edge transition arrow endpoints in display coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    src_row : Any
        Source tradeoff row.
    dst_row : Any
        Destination tradeoff row.
    src_center : np.ndarray
        Source center in display coordinates.
    dst_center : np.ndarray
        Destination center in display coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        Arrow endpoints in display coordinates or ``None``.

    Notes
    -----
    Bubble radii are computed from marker areas and current figure DPI.
    """
    px_per_point = ax.figure.dpi / 72.0
    src_radius = _marker_radius_points(float(src_row.bubble_size)) * px_per_point
    dst_radius = _marker_radius_points(float(dst_row.bubble_size)) * px_per_point
    return _edge_to_edge_arrow(src_center, dst_center, src_radius, dst_radius)


def _draw_transition_arrow_artist(ax, start_px: np.ndarray, end_px: np.ndarray) -> None:
    """Draw one transition arrow artist.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    start_px : np.ndarray
        Arrow start in display coordinates.
    end_px : np.ndarray
        Arrow end in display coordinates.

    Returns
    -------
    None
        Arrow annotation is added to axis.

    Notes
    -----
    Coordinate conversion is performed from display to data space.
    """
    start_xy = tuple(ax.transData.inverted().transform(start_px))
    end_xy = tuple(ax.transData.inverted().transform(end_px))
    ax.annotate(
        "",
        xy=end_xy,
        xytext=start_xy,
        arrowprops=dict(
            arrowstyle="->",
            color="lightgray",
            lw=1.25,
            alpha=0.96,
            shrinkA=0.0,
            shrinkB=0.0,
            mutation_scale=10.0,
        ),
        zorder=4,
    )


def _append_arrow_segment(arrow_segments: list[dict[str, float]], start_px: np.ndarray, end_px: np.ndarray) -> None:
    """Append one transition arrow segment descriptor.

    Parameters
    ----------
    arrow_segments : list[dict[str, float]]
        Mutable arrow-segment descriptor list.
    start_px : np.ndarray
        Arrow start in display coordinates.
    end_px : np.ndarray
        Arrow end in display coordinates.

    Returns
    -------
    None
        Segment descriptor is appended to list.

    Notes
    -----
    Segment coordinates are stored as primitive floats.
    """
    arrow_segments.append(
        {
            "x0": float(start_px[0]),
            "y0": float(start_px[1]),
            "x1": float(end_px[0]),
            "y1": float(end_px[1]),
        }
    )


def _transition_label_text(src_row, dst_row) -> str:
    """Build transition label text for one profile-to-profile arrow.

    Parameters
    ----------
    src_row : Any
        Source tradeoff row.
    dst_row : Any
        Destination tradeoff row.

    Returns
    -------
    str
        Formatted two-line transition label text.

    Notes
    -----
    Labels include RAM factor and speed factor.
    """
    ram_factor = float(dst_row.peak_non_cache_mb / src_row.peak_non_cache_mb) if src_row.peak_non_cache_mb else np.nan
    speed_factor = float(src_row.total_seconds / dst_row.total_seconds) if dst_row.total_seconds else np.nan
    return f"RAM {_format_ram_factor(ram_factor)}\nSpeed {_format_speed_factor(speed_factor)}"


def _draw_transition_label_artist(
    ax,
    src_row,
    dst_row,
    start_px: np.ndarray,
    end_px: np.ndarray,
    bubbles: list[dict[str, float]],
    renderer,
    label_boxes,
    arrow_segments,
) -> None:
    """Draw one transition label artist for an existing arrow segment.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    src_row : Any
        Source tradeoff row.
    dst_row : Any
        Destination tradeoff row.
    start_px : np.ndarray
        Arrow start in display coordinates.
    end_px : np.ndarray
        Arrow end in display coordinates.
    bubbles : list[dict[str, float]]
        Bubble geometry descriptors.
    renderer : RendererBase
        Active canvas renderer.
    label_boxes : list[dict[str, float]]
        Placed label box descriptors.
    arrow_segments : list[dict[str, float]]
        Arrow segment descriptors.

    Returns
    -------
    None
        Label annotation is added to axis.

    Notes
    -----
    Label placement uses iterative collision-aware offsets.
    """
    along, perp, _ = _line_unit_vectors(start_px, end_px)
    _place_transition_label(
        ax,
        _transition_label_text(src_row, dst_row),
        0.5 * (start_px + end_px),
        along=along,
        perp=perp,
        bubbles=bubbles,
        arrow_segments=arrow_segments,
        renderer=renderer,
        label_boxes=label_boxes,
    )


def _draw_single_transition(
    ax,
    src_row,
    dst_row,
    bubbles: list[dict[str, float]],
    renderer,
    label_boxes,
    arrow_segments,
    *,
    draw_arrows: bool,
    draw_labels: bool,
) -> None:
    """Draw one transition arrow and optional label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    src_row : Any
        Source point row tuple.
    dst_row : Any
        Destination point row tuple.
    bubbles : list[dict[str, float]]
        Bubble geometry descriptors.
    renderer : RendererBase
        Active canvas renderer.
    label_boxes : list[dict[str, float]]
        Placed label box descriptors.
    arrow_segments : list[dict[str, float]]
        Mutable arrow segment list.
    draw_arrows : bool
        Enables arrow drawing.
    draw_labels : bool
        Enables label drawing.

    Returns
    -------
    None
        Axis artists are added in place.

    Notes
    -----
    Arrows are drawn edge-to-edge between bubble boundaries.
    """
    src_center, dst_center = _transition_centers(ax, src_row, dst_row)
    endpoints = _transition_endpoint_pixels(ax, src_row, dst_row, src_center, dst_center)
    if endpoints is None:
        return
    start_px, end_px = endpoints

    if draw_arrows:
        _draw_transition_arrow_artist(ax, start_px, end_px)
    _append_arrow_segment(arrow_segments, start_px, end_px)

    if not draw_labels:
        return
    _draw_transition_label_artist(
        ax,
        src_row,
        dst_row,
        start_px,
        end_px,
        bubbles=bubbles,
        renderer=renderer,
        label_boxes=label_boxes,
        arrow_segments=arrow_segments,
    )


def _draw_transition_overlays(
    ax,
    points_df: pd.DataFrame,
    ctx: AnalysisContext,
    bubbles: list[dict[str, float]],
    renderer,
    *,
    draw_arrows: bool,
    draw_labels: bool,
) -> None:
    """Draw all per-scale profile transition overlays.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    points_df : pd.DataFrame
        Tradeoff points with bubble size.
    ctx : AnalysisContext
        Analysis context.
    bubbles : list[dict[str, float]]
        Bubble geometry descriptors.
    renderer : RendererBase
        Active canvas renderer.
    draw_arrows : bool
        Enables arrow drawing.
    draw_labels : bool
        Enables label drawing.

    Returns
    -------
    None
        Overlay artists are added in place.

    Notes
    -----
    Missing profile pairs for a scale are skipped automatically.
    """
    if points_df.empty or len(ctx.bubble_transition_order) < 2:
        return

    label_boxes: list[dict[str, float]] = []
    arrow_segments: list[dict[str, float]] = []
    pairs = list(zip(ctx.bubble_transition_order[:-1], ctx.bubble_transition_order[1:]))

    for scale in sorted(points_df["scale_factor"].unique()):
        sub = points_df[points_df["scale_factor"] == scale]
        point_map = {row.profile_label: row for row in sub.itertuples(index=False)}
        for src_profile, dst_profile in pairs:
            if src_profile not in point_map or dst_profile not in point_map:
                continue
            _draw_single_transition(
                ax,
                point_map[src_profile],
                point_map[dst_profile],
                bubbles,
                renderer,
                label_boxes,
                arrow_segments,
                draw_arrows=draw_arrows,
                draw_labels=draw_labels,
            )


def _draw_tradeoff_legend(ax, ctx: AnalysisContext) -> None:
    """Draw profile legend for bubble tradeoff chart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    None
        Legend artist is added to axis.

    Notes
    -----
    Legend order follows bubble transition order.
    """
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="none",
            markerfacecolor=ctx.bubble_profile_palette.get(profile, "#1f77b4"),
            markeredgecolor="black",
            markersize=8,
            label=profile,
        )
        for profile in ctx.bubble_transition_order
    ]
    ax.legend(handles=handles, title="Profiles", loc="upper left", frameon=False)


def _annotate_tradeoff_scale_labels(ax, points_df: pd.DataFrame) -> None:
    """Annotate bubble centers with scale-factor labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    points_df : pd.DataFrame
        Tradeoff points.

    Returns
    -------
    None
        Text annotations are added to axis.

    Notes
    -----
    Labels are centered over bubble positions.
    """
    for row in points_df.itertuples(index=False):
        ax.text(
            row.total_seconds,
            row.peak_non_cache_mb,
            f"{int(row.scale_factor)}x",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="black",
            zorder=10,
        )


def _scatter_tradeoff_points(ax, points_df: pd.DataFrame, ctx: AnalysisContext, emphasize: bool) -> None:
    """Render tradeoff bubble scatter points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    points_df : pd.DataFrame
        Tradeoff points with bubble sizes.
    ctx : AnalysisContext
        Analysis context.
    emphasize : bool
        Enables emphasized bubble border/alpha style.

    Returns
    -------
    None
        Scatter markers are added to axis.

    Notes
    -----
    Color mapping uses bubble-specific profile palette.
    """
    alpha = 0.74 if emphasize else 0.84
    edge_width = 1.0 if emphasize else 0.7
    for row in points_df.itertuples(index=False):
        ax.scatter(
            row.total_seconds,
            row.peak_non_cache_mb,
            s=float(row.bubble_size),
            alpha=alpha,
            color=ctx.bubble_profile_palette.get(row.profile_label, "#1f77b4"),
            edgecolor="black",
            linewidth=edge_width,
            zorder=6,
        )


def _apply_tradeoff_axis_style(ax, title: str) -> None:
    """Apply standard axis settings for tradeoff bubble charts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    title : str
        Axis title.

    Returns
    -------
    None
        Axis state is updated in place.

    Notes
    -----
    Minor grid/ticks are disabled to keep chart readable.
    """
    ax.set_title(title, pad=8)
    ax.set_xlabel("Runtime [s]")
    ax.set_ylabel("Peak RAM Pressure [MB]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", alpha=GLOBAL_STYLE["grid.alpha"])


def _apply_tradeoff_ticks(ax, emphasize: bool, y_tick_count: Optional[int]) -> None:
    """Apply custom major ticks for tradeoff bubble chart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    emphasize : bool
        Enables dense custom ticks for emphasized mode.
    y_tick_count : int or None
        Explicit y-axis major tick count.

    Returns
    -------
    None
        Axis tick locators/formatters are updated.

    Notes
    -----
    Original axis limits are restored after custom tick assignment.
    """
    use_custom = bool(emphasize) or (y_tick_count is not None)
    if not use_custom:
        return

    # Keep current limits stable while applying custom tick locators.
    ax.figure.canvas.draw()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    if emphasize:
        x_ticks = _build_fixed_ticks(min(x_limits), max(x_limits), count=10, mantissas=[1, 5])
        if x_ticks:
            ax.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _format_log_tick_general(value)))

    y_count = max(2, int(y_tick_count)) if y_tick_count is not None else 10
    y_ticks = _build_fixed_ticks(min(y_limits), max(y_limits), count=y_count, mantissas=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    if y_ticks:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _format_log_tick_general(value)))

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)


def plot_tradeoff_bubbles(
    ax,
    ctx: AnalysisContext,
    title: str,
    *,
    show_legend: bool = False,
    transition_labels: bool = False,
    emphasize_transitions: bool = False,
    point_spread_log10: float = 0.0,
    bubble_size_scale: float = 1.0,
    y_tick_count: Optional[int] = None,
) -> None:
    """Plot runtime-vs-RAM tradeoff bubbles with optional transitions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ctx : AnalysisContext
        Analysis context.
    title : str
        Axis title.
    show_legend : bool, default=False
        Displays profile legend.
    transition_labels : bool, default=False
        Draws transition arrows and labels.
    emphasize_transitions : bool, default=False
        Enables emphasized visual style and denser ticks.
    point_spread_log10 : float, default=0.0
        Log-space spread to separate overlapping points.
    bubble_size_scale : float, default=1.0
        Scale factor applied to bubble areas.
    y_tick_count : int or None, default=None
        Explicit number of y-axis major ticks.

    Returns
    -------
    None
        Axis is updated with bubble chart.

    Notes
    -----
    Arrows are rendered edge-to-edge between bubble boundaries.

    Examples
    --------
    >>> # plot_tradeoff_bubbles(ax, ctx, "Trade-off", show_legend=True)
    """
    points = ctx.trade_points.copy()
    if points.empty:
        ax.text(0.5, 0.5, "No trade-off data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # Apply optional point spreading and bubble scaling before plotting.
    points = _spread_tradeoff_points(points, spread_log10=point_spread_log10, order=ctx.bubble_transition_order)
    points["bubble_size"] = _compute_bubble_sizes(points["cache_size_mb"].to_numpy(dtype=float))
    size_scale = float(bubble_size_scale) if np.isfinite(float(bubble_size_scale)) else 1.0
    if size_scale > 0.0:
        points["bubble_size"] = np.maximum(12.0, points["bubble_size"] * size_scale)

    # Render bubbles, configure axis, then draw optional transition overlays.
    _scatter_tradeoff_points(ax, points, ctx, emphasize_transitions)
    _apply_tradeoff_axis_style(ax, title)
    _apply_tradeoff_ticks(ax, emphasize_transitions, y_tick_count)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bubbles = _build_bubble_geometry(ax, points)
    _draw_transition_overlays(
        ax,
        points,
        ctx,
        bubbles,
        renderer,
        draw_arrows=bool(transition_labels),
        draw_labels=bool(transition_labels),
    )

    # Final annotation pass for scale labels and optional legend.
    _annotate_tradeoff_scale_labels(ax, points)
    if show_legend:
        _draw_tradeoff_legend(ax, ctx)


def _shared_scales(ctx: AnalysisContext) -> list[int]:
    """Compute scales shared by all profiles in current analysis context.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[int]
        Sorted list of shared stack factors.

    Notes
    -----
    Empty profile sets return an empty list.
    """
    scale_sets = [
        set(ctx.totals_df[ctx.totals_df["profile_label"] == profile]["scale_factor"].unique())
        for profile in ctx.profile_order
    ]
    return sorted(set.intersection(*scale_sets)) if scale_sets else []


def _ordered_runs_for_profile(ctx: AnalysisContext, profile: str) -> list[str]:
    """Return deterministic run order for one profile.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    profile : str
        Profile label.

    Returns
    -------
    list[str]
        Sorted run names.

    Notes
    -----
    Runs are sorted by scale factor and then run name.
    """
    profile_totals = ctx.totals_df[ctx.totals_df["profile_label"] == profile]
    return profile_totals.sort_values(["scale_factor", "run"])["run"].tolist()


def _step_order_for_profile(ctx: AnalysisContext, profile: str, ordered_runs: list[str]) -> list[str]:
    """Build step order for one profile based on its first ordered run.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    profile : str
        Profile label.
    ordered_runs : list[str]
        Ordered run names for profile.

    Returns
    -------
    list[str]
        Ordered step names.

    Notes
    -----
    Empty run lists return no steps.
    """
    if not ordered_runs:
        return []
    profile_steps = ctx.steps_df[ctx.steps_df["profile_label"] == profile]
    base_run = ordered_runs[0]
    return (
        profile_steps[profile_steps["run"] == base_run]
        .sort_values("step_index")["step"]
        .tolist()
    )


def _format_step_labels(step_order: Sequence[str]) -> list[str]:
    """Format compact x-axis labels for a given step ordering.

    Parameters
    ----------
    step_order : Sequence[str]
        Step names in display order.

    Returns
    -------
    list[str]
        Compact step labels.

    Notes
    -----
    Uses the same label compaction as all step-oriented plots.
    """
    return [_pretty_step_label(step_name) for step_name in step_order]


def _profile_step_pivot(
    profile_steps: pd.DataFrame,
    ordered_runs: list[str],
    step_order: list[str],
    metric: str,
) -> pd.DataFrame:
    """Build run-by-step pivot for one profile metric panel.

    Parameters
    ----------
    profile_steps : pd.DataFrame
        Step rows from one profile.
    ordered_runs : list[str]
        Ordered run names.
    step_order : list[str]
        Ordered step names.
    metric : str
        Step metric name.

    Returns
    -------
    pd.DataFrame
        Run-by-step matrix aligned to provided order.

    Notes
    -----
    Missing values remain ``nan`` after reindexing.
    """
    return (
        profile_steps.pivot_table(index="run", columns="step", values=metric, aggfunc="mean")
        .reindex(index=ordered_runs, columns=step_order)
    )


def _render_pre_master_overall_metrics_figure(ctx: AnalysisContext) -> Figure:
    """Render pre-master 2x2 global metrics figure from notebook section.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    Figure
        Rendered figure.

    Notes
    -----
    This reproduces the pre-master overview panel set.
    """
    specs = [
        ("total_seconds", "Total Runtime [s]", "Value"),
        ("peak_non_cache_mb", "Peak Non-Cache RAM [MB]", "Value"),
        ("cache_size_mb", "Final Cache Size [MB]", "Value"),
        ("non_cache_per_second", "Non-Cache Pressure Intensity [MB/s]", "Value"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for idx, (metric, title, ylabel) in enumerate(specs):
        ax = axes.ravel()[idx]
        plot_scale_metric(ax, ctx, metric, title, ylabel, y_log=True, show_legend=(idx == 0))
    fig.suptitle("Overall Non-Cache-Focused Metrics per Profile (log-log)", fontsize=14)
    fig._mdxplain_filename_hint = "overall_non_cache_focused_metrics_per_profile_log_log"
    return fig


def _render_pre_master_normalized_scaling_figure(
    ctx: AnalysisContext,
    normalized_scaling: pd.DataFrame,
) -> Figure:
    """Render pre-master normalized scaling figure (runtime/RAM/cache).

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    normalized_scaling : pd.DataFrame
        Normalized scaling table.

    Returns
    -------
    Figure
        Rendered figure.

    Notes
    -----
    Relative metrics are plotted against stack factor in log-log scale.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    specs = [
        ("total_seconds_x", "Runtime Relative to 1x"),
        ("peak_non_cache_mb_x", "Peak Non-Cache Relative to 1x"),
        ("cache_size_mb_x", "Cache Size Relative to 1x"),
    ]
    max_scale = float(normalized_scaling["scale_factor"].max()) if not normalized_scaling.empty else np.nan
    for ax, (metric, title) in zip(axes, specs):
        for profile in ctx.profile_order:
            profile_df = normalized_scaling[normalized_scaling["profile_label"] == profile].sort_values("scale_factor")
            if profile_df.empty:
                continue
            ax.plot(profile_df["scale_factor"], profile_df[metric], marker="o", linewidth=2, label=profile)
        if np.isfinite(max_scale) and max_scale > 1.0:
            ax.plot([1.0, max_scale], [1.0, max_scale], "--", color="gray", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Stack Factor")
        ax.set_ylabel("Relative factor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=GLOBAL_STYLE["grid.alpha"])
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Profile")
    fig._mdxplain_filename_hint = "normalized_scaling_relative_to_1x"
    return fig


def _render_pre_master_runwise_totals_bars_figure(
    ctx: AnalysisContext,
    run_wise_totals: pd.DataFrame,
) -> Figure:
    """Render pre-master run-wise totals bar overview figure.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    run_wise_totals : pd.DataFrame
        Run-wise totals table.

    Returns
    -------
    Figure
        Rendered bar figure.

    Notes
    -----
    Runtime is shown on log scale, memory/cache on linear scale.
    """
    plot_df = run_wise_totals.sort_values(["scale_factor", "profile_label", "run"]).copy()
    plot_df["x_idx"] = np.arange(len(plot_df))
    bar_colors = [ctx.profile_palette.get(profile, "#4c72b0") for profile in plot_df["profile_label"]]
    fig, axes = plt.subplots(3, 1, figsize=(21, 11), constrained_layout=True)
    specs = [
        ("total_seconds", "Run-wise Total Runtime [s]", "Runtime [s]", "log"),
        ("peak_non_cache_mb", "Run-wise RAM Pressure [MB]", "RAM Pressure [MB]", "linear"),
        ("cache_size_mb", "Run-wise Cache Size [MB]", "Cache Size [MB]", "linear"),
    ]
    for ax, (metric, title, ylabel, yscale) in zip(axes, specs):
        ax.bar(plot_df["x_idx"], plot_df[metric], color=bar_colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(plot_df["x_idx"], labels=plot_df["run_label"], rotation=40, ha="right")
        if yscale == "log":
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=GLOBAL_STYLE["grid.alpha"])
    handles = [
        plt.Line2D([0], [0], color=ctx.profile_palette[profile], linewidth=8, label=profile)
        for profile in ctx.profile_order
        if profile in set(plot_df["profile_label"])
    ]
    axes[0].legend(handles=handles, title="Profile", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig._mdxplain_filename_hint = "runwise_totals_bars"
    return fig


def _pre_master_step_metric_specs() -> list[tuple[str, str, str]]:
    """Return metric specifications for pre-master step run comparison.

    Parameters
    ----------
    None

    Returns
    -------
    list[tuple[str, str, str]]
        Metric name, panel title, and y-scale mode tuples.

    Notes
    -----
    The order matches the legacy notebook layout.
    """
    return [
        ("seconds", "Step Absolute Time [s]", "log"),
        ("seconds_share_pct", "Step Time Share [%]", "linear"),
        ("non_cache_peak_mb", "Step RAM Pressure Peak [MB]", "linear"),
        ("non_cache_peak_over_start_mb", "Step RAM Pressure Peak over Start [MB]", "linear"),
    ]


def _plot_pre_master_profile_step_panels(
    axes_2d,
    ctx: AnalysisContext,
    profile: str,
    profile_index: int,
    specs: list[tuple[str, str, str]],
) -> None:
    """Plot all per-step run panels for one profile into provided axes grid.

    Parameters
    ----------
    axes_2d : np.ndarray
        Two-dimensional axis array.
    ctx : AnalysisContext
        Analysis context.
    profile : str
        Profile label.
    profile_index : int
        Profile index in outer loop.
    specs : list[tuple[str, str, str]]
        Metric specifications for panel rendering.

    Returns
    -------
    None
        Axes grid is updated in place.

    Notes
    -----
    Panels are skipped when run or step data is unavailable.
    """
    profile_steps = ctx.steps_df[ctx.steps_df["profile_label"] == profile].copy()
    ordered_runs = _ordered_runs_for_profile(ctx, profile)
    step_order = _step_order_for_profile(ctx, profile, ordered_runs)
    if (not ordered_runs) or (not step_order):
        return
    x_vals = np.arange(len(step_order))
    step_labels = _format_step_labels(step_order)
    run_palette = _get_run_palette(ordered_runs, ctx.profile_palette.get(profile, "#4c72b0"))
    for metric_index, (metric, title, yscale) in enumerate(specs):
        row = 2 * profile_index + (metric_index // 2)
        col = metric_index % 2
        axis = axes_2d[row, col]
        pivot = _profile_step_pivot(profile_steps, ordered_runs, step_order, metric)
        for run_name in ordered_runs:
            y_vals = pivot.loc[run_name].to_numpy(dtype=float)
            axis.plot(x_vals, y_vals, marker="o", linewidth=GLOBAL_STYLE["lines.linewidth"], markersize=GLOBAL_STYLE["lines.markersize"], color=run_palette[run_name], label=run_name)
        axis.set_title(f"{profile} | {title}", fontsize=11)
        axis.set_xticks(x_vals, labels=step_labels, rotation=38, ha="right")
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(True, alpha=GLOBAL_STYLE["grid.alpha"])
        if yscale == "log":
            axis.set_yscale("log")
    axes_2d[2 * profile_index, 0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title=f"{profile} runs", fontsize=7, title_fontsize=8, frameon=False)


def _render_pre_master_per_step_run_comparison_figure(ctx: AnalysisContext) -> Figure:
    """Render pre-master per-step run comparison across profiles.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    Figure
        Rendered figure.

    Notes
    -----
    Layout uses two rows per profile and two metrics per row.
    """
    profiles = [profile for profile in ctx.profile_order if profile in set(ctx.steps_df["profile_label"].unique())]
    specs = _pre_master_step_metric_specs()
    n_rows = max(1, 2 * len(profiles))
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, max(10, 3.2 * n_rows)), constrained_layout=True)
    axes_2d = np.asarray(axes, dtype=object).reshape(n_rows, 2)
    if not profiles:
        axes_2d[0, 0].text(0.5, 0.5, "No step data available", ha="center", va="center", transform=axes_2d[0, 0].transAxes)
        axes_2d[0, 0].set_axis_off()
        axes_2d[0, 1].set_axis_off()
    for profile_index, profile in enumerate(profiles):
        _plot_pre_master_profile_step_panels(axes_2d, ctx, profile, profile_index, specs)
    fig.suptitle("Per-Step Run Comparison Across Profiles", fontsize=14)
    fig._mdxplain_filename_hint = "per_step_run_comparison_across_profiles"
    return fig


def _render_pre_master_shared_scale_step_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render pre-master step comparison figures for shared scales.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Shared-scale figure list.

    Notes
    -----
    One figure is generated per scale shared by all profiles.
    """
    figures: list[Figure] = []
    step_metrics = [
        ("seconds", "Step Time [s]", "log"),
        ("seconds_share_pct", "Step Time Share [%]", "linear"),
        ("non_cache_peak_mb", "Step RAM Pressure Peak [MB]", "linear"),
    ]
    for scale in _shared_scales(ctx):
        sub = ctx.steps_df[ctx.steps_df["scale_factor"] == scale].copy()
        step_order = sub.groupby("step")["step_index"].median().sort_values().index.tolist()
        if not step_order:
            continue
        profiles = [profile for profile in ctx.profile_order if profile in set(sub["profile_label"].unique())]
        palette = _get_profile_palette(profiles)
        x_vals = np.arange(len(step_order))
        step_labels = _format_step_labels(step_order)
        fig, axes = plt.subplots(1, len(step_metrics), figsize=(19.5, 4.6), constrained_layout=True)
        for ax, (metric, title, yscale) in zip(axes, step_metrics):
            pivot = (
                sub.pivot_table(index="step", columns="profile_label", values=metric, aggfunc="mean")
                .reindex(index=step_order, columns=profiles)
            )
            for profile in profiles:
                ax.plot(x_vals, pivot[profile].to_numpy(dtype=float), marker="o", linewidth=1.9, markersize=4, color=palette[profile], label=profile)
            ax.set_title(f"{title} | {scale}x")
            ax.set_xlabel("Step")
            ax.set_xticks(x_vals, labels=step_labels, rotation=38, ha="right")
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(True, alpha=GLOBAL_STYLE["grid.alpha"])
            if yscale == "log":
                ax.set_yscale("log")
        axes[0].set_ylabel("Value")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Profile", frameon=False)
        fig.suptitle(f"Step-wise profile comparison at shared scale {scale}x", fontsize=13)
        fig._mdxplain_filename_hint = f"shared_scale_step_comparison_{scale}x"
        figures.append(fig)
    return figures


def _render_pre_master_jump_diagnostics_figure(jump_df: pd.DataFrame) -> Figure:
    """Render pre-master 3x3 jump diagnostics grid.

    Parameters
    ----------
    jump_df : pd.DataFrame
        Jump diagnostics table.

    Returns
    -------
    Figure
        Rendered diagnostics figure.

    Notes
    -----
    Rows correspond to metrics and columns to diagnostic views.
    """
    metric_labels = {
        "total_seconds": "Total Runtime",
        "peak_non_cache_mb": "Peak Non-Cache",
        "cache_size_mb": "Cache Size",
    }
    fig, axes = plt.subplots(3, 3, figsize=(22, 14), constrained_layout=True)
    for row_idx, metric in enumerate(["total_seconds", "peak_non_cache_mb", "cache_size_mb"]):
        metric_df = jump_df[jump_df["metric"] == metric]
        ax = axes[row_idx, 0]
        for profile, profile_df in metric_df.groupby("profile_label"):
            ordered = profile_df.sort_values("scale_factor")
            ax.plot(ordered["scale_factor"], ordered["jump_pct"], marker="o", linewidth=2, label=profile)
        ax.set_title(f"{metric_labels[metric]}: Relative Jump [%]")
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("%")
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=GLOBAL_STYLE["grid.alpha"])

        ax = axes[row_idx, 1]
        for profile, profile_df in metric_df.groupby("profile_label"):
            ordered = profile_df.sort_values("scale_factor")
            ax.plot(ordered["scale_factor"], ordered["local_exponent"], marker="o", linewidth=2, label=profile)
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(f"{metric_labels[metric]}: Local Exponent k")
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("k_local")
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=GLOBAL_STYLE["grid.alpha"])

        ax = axes[row_idx, 2]
        for profile, profile_df in metric_df.groupby("profile_label"):
            ordered = profile_df.sort_values("scale_factor")
            ax.plot(ordered["scale_factor"], ordered["overhead_vs_linear"], marker="o", linewidth=2, label=profile)
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(f"{metric_labels[metric]}: Overhead vs Linear")
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("actual / linear_expected")
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=GLOBAL_STYLE["grid.alpha"])
    axes[0, 2].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Profile")
    fig._mdxplain_filename_hint = "jump_diagnostics_grid"
    return fig


def _ram_pressure_metric_specs() -> list[tuple[str, str]]:
    """Return metric specifications for RAM pressure diagnostics plot.

    Parameters
    ----------
    None

    Returns
    -------
    list[tuple[str, str]]
        Metric name and title tuples.

    Notes
    -----
    Order matches the corresponding notebook diagnostics section.
    """
    return [
        ("non_cache_peak_mb", "RAM Pressure Peak per Step [MB]"),
        ("delta_non_cache_mb", "RAM Pressure Delta End-Start per Step [MB]"),
        ("non_cache_peak_over_start_mb", "RAM Pressure Peak over Start per Step [MB]"),
    ]


def _plot_profile_ram_pressure_rows(
    axes_1d,
    ctx: AnalysisContext,
    profile: str,
    start_row: int,
    metrics: list[tuple[str, str]],
) -> int:
    """Plot RAM pressure diagnostic rows for one profile.

    Parameters
    ----------
    axes_1d : np.ndarray
        One-dimensional axis array.
    ctx : AnalysisContext
        Analysis context.
    profile : str
        Profile label.
    start_row : int
        First row index for this profile.
    metrics : list[tuple[str, str]]
        Metric specifications.

    Returns
    -------
    int
        Next free row index after plotted panels.

    Notes
    -----
    Returns ``start_row`` unchanged when no data is available.
    """
    profile_steps = ctx.steps_df[ctx.steps_df["profile_label"] == profile].copy()
    ordered_runs = _ordered_runs_for_profile(ctx, profile)
    step_order = _step_order_for_profile(ctx, profile, ordered_runs)
    if (not ordered_runs) or (not step_order):
        return start_row
    x_vals = np.arange(len(step_order))
    step_labels = _format_step_labels(step_order)
    run_palette = _get_run_palette(ordered_runs, ctx.profile_palette.get(profile, "#4c72b0"))
    row_idx = start_row
    for metric, title in metrics:
        axis = axes_1d[row_idx]
        pivot = _profile_step_pivot(profile_steps, ordered_runs, step_order, metric)
        for run_name in ordered_runs:
            y_vals = pivot.loc[run_name].to_numpy(dtype=float)
            axis.plot(x_vals, y_vals, marker="o", linewidth=GLOBAL_STYLE["lines.linewidth"], markersize=GLOBAL_STYLE["lines.markersize"], color=run_palette[run_name], label=run_name)
        axis.set_title(f"{profile} | {title}", fontsize=11)
        axis.set_ylabel("MB")
        axis.set_xticks(x_vals, labels=step_labels, rotation=38, ha="right")
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(True, alpha=GLOBAL_STYLE["grid.alpha"])
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Run", fontsize=7, title_fontsize=8, frameon=False)
        row_idx += 1
    return row_idx


def _render_pre_master_ram_pressure_diagnostics_figure(ctx: AnalysisContext) -> Figure:
    """Render pre-master per-step RAM pressure diagnostics figure.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    Figure
        Rendered diagnostics figure.

    Notes
    -----
    Each profile contributes one row per RAM-pressure metric.
    """
    profiles = [profile for profile in ctx.profile_order if profile in set(ctx.totals_df["profile_label"].unique())]
    metrics = _ram_pressure_metric_specs()
    n_rows = max(1, len(profiles) * len(metrics))
    fig, axes = plt.subplots(n_rows, 1, figsize=(22, 3.45 * n_rows), constrained_layout=True)
    axes_1d = np.asarray(axes, dtype=object).reshape(n_rows)
    row_idx = 0
    for profile in profiles:
        row_idx = _plot_profile_ram_pressure_rows(axes_1d, ctx, profile, row_idx, metrics)
    if row_idx == 0:
        axes_1d[0].text(0.5, 0.5, "No step RAM diagnostics available", ha="center", va="center", transform=axes_1d[0].transAxes)
    for idx in range(row_idx, n_rows):
        axes_1d[idx].set_axis_off()
    axes_1d[max(0, min(row_idx, n_rows) - 1)].set_xlabel("Step")
    fig.suptitle("Per-Step RAM Pressure Diagnostics", fontsize=14)
    fig._mdxplain_filename_hint = "per_step_ram_pressure_diagnostics"
    return fig


def _render_pre_master_stacked_step_breakdown_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render pre-master stacked step contribution figures per profile.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Stacked contribution figure list.

    Notes
    -----
    Each figure contains stacked time and cache delta bars.
    """
    figures: list[Figure] = []
    for profile in ctx.profile_order:
        profile_totals = ctx.totals_df[ctx.totals_df["profile_label"] == profile].copy()
        profile_steps = ctx.steps_df[ctx.steps_df["profile_label"] == profile].copy()
        ordered_runs = profile_totals.sort_values(["scale_factor", "run"])["run"].tolist()
        step_order = _step_order_for_profile(ctx, profile, ordered_runs)
        if (not ordered_runs) or (not step_order):
            continue
        time_pivot = _profile_step_pivot(profile_steps, ordered_runs, step_order, "seconds")
        cache_pivot = _profile_step_pivot(profile_steps, ordered_runs, step_order, "delta_cache_mb")
        step_colors = _get_shades(ctx.profile_palette.get(profile, "#4c72b0"), max(1, len(step_order)), light=0.30, dark=0.95)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
        time_pivot.plot(kind="bar", stacked=True, ax=axes[0], color=step_colors)
        axes[0].set_title(f"{profile} | Step Time Contributions")
        axes[0].set_ylabel("Seconds")
        axes[0].tick_params(axis="x", rotation=35)
        cache_pivot.plot(kind="bar", stacked=True, ax=axes[1], color=step_colors)
        axes[1].set_title(f"{profile} | Step Cache Growth (delta cache)")
        axes[1].set_ylabel("MB")
        axes[1].tick_params(axis="x", rotation=35)
        for axis in axes:
            axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, title="Step")
        fig._mdxplain_filename_hint = f"{profile.lower().replace(' ', '_')}_stacked_step_breakdown"
        figures.append(fig)
    return figures


def _render_pre_master_profile_scaling_trend_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render pre-master profile-specific scaling trend figures.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Per-profile scaling trend figures.

    Notes
    -----
    One log-log panel is created per metric.
    """
    figures: list[Figure] = []
    scaling_metrics = [
        ("total_seconds", "Total Runtime [s]"),
        ("peak_non_cache_mb", "Peak Non-Cache RAM [MB]"),
        ("cache_size_mb", "Final Cache Size [MB]"),
    ]
    for profile in ctx.profile_order:
        profile_totals = ctx.totals_df[ctx.totals_df["profile_label"] == profile].sort_values("scale_factor")
        if profile_totals.empty:
            continue
        x_vals = profile_totals["scale_factor"].to_numpy(dtype=float)
        color = ctx.profile_palette.get(profile, "#4c72b0")
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), constrained_layout=True)
        for ax, (metric, title) in zip(axes, scaling_metrics):
            y_vals = profile_totals[metric].to_numpy(dtype=float)
            ax.loglog(x_vals, y_vals, marker="o", linewidth=2, color=color)
            ax.set_title(f"{profile} | {title}")
            ax.set_xlabel("Stack Factor")
            ax.set_ylabel("Value")
            ax.grid(True, which="major", alpha=GLOBAL_STYLE["grid.alpha"])
        fig.suptitle(f"{profile} | Scaling Trends", fontsize=14)
        fig._mdxplain_filename_hint = f"{profile.lower().replace(' ', '_')}_scaling_trends"
        figures.append(fig)
    return figures


def _render_pre_master_figures(
    ctx: AnalysisContext,
    normalized_scaling: pd.DataFrame,
    run_wise_totals: pd.DataFrame,
    jump_df: pd.DataFrame,
) -> list[Figure]:
    """Render all pre-master notebook figures in deterministic order.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    normalized_scaling : pd.DataFrame
        Normalized scaling table.
    run_wise_totals : pd.DataFrame
        Run-wise totals table.
    jump_df : pd.DataFrame
        Jump diagnostics table.

    Returns
    -------
    list[Figure]
        Pre-master figure list.

    Notes
    -----
    Figure order follows historical notebook execution order.
    """
    figures: list[Figure] = []
    figures.append(_render_pre_master_overall_metrics_figure(ctx))
    figures.append(_render_pre_master_normalized_scaling_figure(ctx, normalized_scaling))
    figures.append(_render_pre_master_runwise_totals_bars_figure(ctx, run_wise_totals))
    figures.append(_render_pre_master_per_step_run_comparison_figure(ctx))
    figures.extend(_render_pre_master_shared_scale_step_figures(ctx))
    figures.append(_render_pre_master_jump_diagnostics_figure(jump_df))
    figures.append(_render_pre_master_ram_pressure_diagnostics_figure(ctx))
    figures.extend(_render_pre_master_stacked_step_breakdown_figures(ctx))
    figures.extend(_render_pre_master_profile_scaling_trend_figures(ctx))
    return figures


def _render_master_top_row(fig: Figure, grid, ctx: AnalysisContext) -> None:
    """Render top row of master figure.

    Parameters
    ----------
    fig : Figure
        Master figure.
    grid : GridSpec
        Master gridspec.
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    None
        Top-row axes are added to figure.

    Notes
    -----
    Top row contains global scaling and tradeoff overview.
    """
    ax = fig.add_subplot(grid[0, 0])
    plot_scale_metric(ax, ctx, "total_seconds", "Global Scaling: Total Runtime", "Runtime [s]", y_log=True, show_legend=True)

    ax = fig.add_subplot(grid[0, 1])
    plot_scale_metric(ax, ctx, "peak_non_cache_mb", "Global Scaling: Peak RAM Pressure", "Peak RAM Pressure [MB]", y_log=True)

    ax = fig.add_subplot(grid[0, 2])
    plot_scale_metric(ax, ctx, "cache_size_mb", "Global Scaling: Cache Size", "Cache Size [MB]", y_log=True)

    ax = fig.add_subplot(grid[0, 3])
    plot_tradeoff_bubbles(ax, ctx, "Trade-off (all runs): Runtime vs Peak RAM Pressure", transition_labels=True)


def _render_master_middle_row(fig: Figure, grid, ctx: AnalysisContext) -> None:
    """Render middle row of master figure.

    Parameters
    ----------
    fig : Figure
        Master figure.
    grid : GridSpec
        Master gridspec.
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    None
        Middle-row axes are added to figure.

    Notes
    -----
    Middle row contains per-step diagnostics.
    """
    ax = fig.add_subplot(grid[1, 0])
    plot_step_line(ax, ctx, "seconds", "Per-Step Absolute Time", "Time [s]", y_log=True)

    ax = fig.add_subplot(grid[1, 1])
    plot_step_line(ax, ctx, "non_cache_peak_over_start_mb", "Per-Step RAM Pressure Peak over Start", "Peak over Start [MB]")

    ax = fig.add_subplot(grid[1, 2])
    plot_step_line(ax, ctx, "non_cache_peak_mb", "Per-Step RAM Pressure Peak (total)", "Peak RAM Pressure [MB]")

    ax = fig.add_subplot(grid[1, 3])
    plot_step_line(ax, ctx, "delta_cache_mb", "Per-Step Cache Growth", "Cache Delta [MB]", zero_line=True)


def _render_master_bottom_row(fig: Figure, grid, ctx: AnalysisContext) -> None:
    """Render bottom row of master figure.

    Parameters
    ----------
    fig : Figure
        Master figure.
    grid : GridSpec
        Master gridspec.
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    None
        Bottom-row axes are added to figure.

    Notes
    -----
    Bottom row contains scaling exponents and summary table.
    """
    ax = fig.add_subplot(grid[2, 0])
    plot_local_exponent(ax, ctx, "total_seconds", "Scaling Exponent: Runtime (O(n^k))", ylabel="k_runtime")

    ax = fig.add_subplot(grid[2, 1])
    plot_local_exponent(ax, ctx, "peak_non_cache_mb", "Scaling Exponent: RAM Pressure (O(n^k))", ylabel="k_ram_pressure")

    ax = fig.add_subplot(grid[2, 2:4])
    ax.axis("off")
    summary = build_requested_table(ctx)
    if summary.empty:
        ax.text(0.5, 0.5, "No table data available", ha="center", va="center", fontsize=11)
        return
    table = ax.table(cellText=summary.values, colLabels=summary.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.05, 1.28)
    ax.set_title("Summary Table (old columns, Score removed)", fontsize=10.5, pad=8)


def _render_master_figure(ctx: AnalysisContext) -> Figure:
    """Render benchmark master overview figure.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    Figure
        Master figure.

    Notes
    -----
    Figure uses a 3x4 grid layout with compact spacing.
    """
    fig = plt.figure(figsize=(24, 14), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.03, wspace=0.04, hspace=0.05)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.1, 1.1])

    _render_master_top_row(fig, grid, ctx)
    _render_master_middle_row(fig, grid, ctx)
    _render_master_bottom_row(fig, grid, ctx)

    fig.suptitle("Benchmark Master Overview | RAM Pressure Focus", fontsize=14)
    fig._mdxplain_filename_hint = "benchmark_master_overview_ram_pressure"
    return fig


def _render_single_scaling_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render single-plot global scaling figures.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Scaling figure list.

    Notes
    -----
    Includes runtime, RAM pressure, and cache scaling plots.
    """
    figures: list[Figure] = []

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.2), constrained_layout=True)
    plot_scale_metric(ax, ctx, "total_seconds", "Global Scaling: Total Runtime", "Runtime [s]", y_log=True, show_legend=True)
    fig._mdxplain_filename_hint = "single_global_scaling_total_runtime"
    figures.append(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.2), constrained_layout=True)
    plot_scale_metric(ax, ctx, "peak_non_cache_mb", "Global Scaling: Peak RAM Pressure", "Peak RAM Pressure [MB]", y_log=True)
    fig._mdxplain_filename_hint = "single_global_scaling_peak_ram_pressure"
    figures.append(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.2), constrained_layout=True)
    plot_scale_metric(ax, ctx, "cache_size_mb", "Global Scaling: Cache Size", "Cache Size [MB]", y_log=True)
    fig._mdxplain_filename_hint = "single_global_scaling_cache_size"
    figures.append(fig)
    return figures


def _render_single_tradeoff_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render single-plot tradeoff bubble figures.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Tradeoff figure list.

    Notes
    -----
    Includes baseline and transition-emphasized variants.
    """
    figures: list[Figure] = []

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 6.2), constrained_layout=True)
    plot_tradeoff_bubbles(ax, ctx, title="Trade-off: Runtime vs Peak RAM Pressure", show_legend=True)
    fig._mdxplain_filename_hint = "single_tradeoff_bubbles_runtime_vs_ram"
    figures.append(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9.8, 6.6), constrained_layout=True)
    plot_tradeoff_bubbles(
        ax,
        ctx,
        title="Trade-off: Runtime vs Peak RAM Pressure",
        show_legend=True,
        transition_labels=True,
        point_spread_log10=0.026,
        bubble_size_scale=0.78,
        y_tick_count=5,
    )
    fig._mdxplain_filename_hint = "single_tradeoff_bubbles_spread_transitions"
    figures.append(fig)
    return figures


def _render_single_step_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render single-plot per-step diagnostic figures.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Step figure list.

    Notes
    -----
    Four per-step metrics are rendered.
    """
    specs = [
        ("seconds", "Per-Step Absolute Time", "Time [s]", "single_per_step_absolute_time", {"y_log": True}),
        (
            "non_cache_peak_over_start_mb",
            "Per-Step RAM Pressure Peak over Start",
            "Peak over Start [MB]",
            "single_per_step_ram_pressure_peak_over_start",
            {},
        ),
        (
            "non_cache_peak_mb",
            "Per-Step RAM Pressure Peak (total)",
            "Peak RAM Pressure [MB]",
            "single_per_step_ram_pressure_peak_total",
            {},
        ),
        ("delta_cache_mb", "Per-Step Cache Growth", "Cache Delta [MB]", "single_per_step_cache_growth", {"zero_line": True}),
    ]

    figures: list[Figure] = []
    for metric, title, ylabel, hint, kwargs in specs:
        fig, ax = plt.subplots(1, 1, figsize=(10.2, 5.4), constrained_layout=True)
        plot_step_line(ax, ctx, metric, title, ylabel, **kwargs)
        fig._mdxplain_filename_hint = hint
        figures.append(fig)
    return figures


def _render_single_exponent_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render single-plot scaling exponent figures.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Exponent figure list.

    Notes
    -----
    Runtime and RAM-pressure exponents are plotted separately.
    """
    figures: list[Figure] = []

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.2), constrained_layout=True)
    plot_local_exponent(ax, ctx, "total_seconds", "Scaling Exponent: Runtime (O(n^k))", ylabel="k_runtime")
    fig._mdxplain_filename_hint = "single_scaling_exponent_runtime"
    figures.append(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.2), constrained_layout=True)
    plot_local_exponent(ax, ctx, "peak_non_cache_mb", "Scaling Exponent: RAM Pressure (O(n^k))", ylabel="k_ram_pressure")
    fig._mdxplain_filename_hint = "single_scaling_exponent_ram_pressure"
    figures.append(fig)
    return figures


def _render_single_table_figure(ctx: AnalysisContext) -> Figure:
    """Render single-plot summary table figure.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    Figure
        Table figure.

    Notes
    -----
    Empty data renders a centered fallback text.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14.0, 4.8), constrained_layout=True)
    ax.axis("off")
    summary = build_requested_table(ctx)
    if summary.empty:
        ax.text(0.5, 0.5, "No table data available", ha="center", va="center", fontsize=11)
    else:
        table = ax.table(cellText=summary.values, colLabels=summary.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.03, 1.30)
        ax.set_title("Summary Table (old columns, Score removed)", fontsize=11, pad=8)
    fig._mdxplain_filename_hint = "single_summary_table_old_columns_score_removed"
    return fig


def _render_single_figures(ctx: AnalysisContext) -> list[Figure]:
    """Render all single-subplot analysis figures.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.

    Returns
    -------
    list[Figure]
        Combined single-subplot figure list.

    Notes
    -----
    Figure order is deterministic for reproducible exports.
    """
    figures: list[Figure] = []
    figures.extend(_render_single_scaling_figures(ctx))
    figures.extend(_render_single_tradeoff_figures(ctx))
    figures.extend(_render_single_step_figures(ctx))
    figures.extend(_render_single_exponent_figures(ctx))
    figures.append(_render_single_table_figure(ctx))
    return figures


def _export_figures(figures: list[Figure], export_dir: Path, filetypes: list[str], dpi: int) -> None:
    """Export and close all generated figures.

    Parameters
    ----------
    figures : list[Figure]
        Figure list to export.
    export_dir : Path
        Target export directory.
    filetypes : list[str]
        Requested export filetypes.
    dpi : int
        Export DPI.

    Returns
    -------
    None
        Figures are written to disk and closed.

    Notes
    -----
    Closing figures avoids memory accumulation in repeated runs.
    """
    for fig in figures:
        _save_figure(fig, export_dir, filetypes, dpi)
        plt.close(fig)


def _export_pre_master_figure_groups(
    ctx: AnalysisContext,
    normalized_scaling: pd.DataFrame,
    run_wise_totals: pd.DataFrame,
    jump_df: pd.DataFrame,
    export_dir: Path,
    filetypes: list[str],
    dpi: int,
) -> None:
    """Export pre-master figure groups and close each group immediately.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    normalized_scaling : pd.DataFrame
        Normalized scaling table.
    run_wise_totals : pd.DataFrame
        Run-wise totals table.
    jump_df : pd.DataFrame
        Jump diagnostics table.
    export_dir : Path
        Target export directory.
    filetypes : list[str]
        Requested export file types.
    dpi : int
        Raster export DPI.

    Returns
    -------
    None
        Figures are exported and closed for side effects.

    Notes
    -----
    Group-wise export avoids having too many open figure handles.
    """
    # Export each pre-master section immediately to keep open figure count low.
    _export_figures([_render_pre_master_overall_metrics_figure(ctx)], export_dir, filetypes, dpi)
    _export_figures([_render_pre_master_normalized_scaling_figure(ctx, normalized_scaling)], export_dir, filetypes, dpi)
    _export_figures([_render_pre_master_runwise_totals_bars_figure(ctx, run_wise_totals)], export_dir, filetypes, dpi)
    _export_figures([_render_pre_master_per_step_run_comparison_figure(ctx)], export_dir, filetypes, dpi)
    _export_figures(_render_pre_master_shared_scale_step_figures(ctx), export_dir, filetypes, dpi)
    _export_figures([_render_pre_master_jump_diagnostics_figure(jump_df)], export_dir, filetypes, dpi)
    _export_figures([_render_pre_master_ram_pressure_diagnostics_figure(ctx)], export_dir, filetypes, dpi)
    _export_figures(_render_pre_master_stacked_step_breakdown_figures(ctx), export_dir, filetypes, dpi)
    _export_figures(_render_pre_master_profile_scaling_trend_figures(ctx), export_dir, filetypes, dpi)


def _export_single_figure_groups(
    ctx: AnalysisContext,
    export_dir: Path,
    filetypes: list[str],
    dpi: int,
) -> None:
    """Export single-plot figure groups and close each group immediately.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context.
    export_dir : Path
        Target export directory.
    filetypes : list[str]
        Requested export file types.
    dpi : int
        Raster export DPI.

    Returns
    -------
    None
        Figures are exported and closed for side effects.

    Notes
    -----
    Single-plot groups are exported in deterministic order.
    """
    # Export singles by group to prevent figure accumulation.
    _export_figures(_render_single_scaling_figures(ctx), export_dir, filetypes, dpi)
    _export_figures(_render_single_tradeoff_figures(ctx), export_dir, filetypes, dpi)
    _export_figures(_render_single_step_figures(ctx), export_dir, filetypes, dpi)
    _export_figures(_render_single_exponent_figures(ctx), export_dir, filetypes, dpi)
    _export_figures([_render_single_table_figure(ctx)], export_dir, filetypes, dpi)


def run_report(
    benchmark_root: Path,
    export_dir: Path,
    filetypes: list[str],
    dpi: int,
) -> None:
    """Run standalone benchmark analysis and export figures plus CSV tables.

    Parameters
    ----------
    benchmark_root : Path
        Preferred benchmark root directory.
    export_dir : Path
        Target directory for exported figures and tables.
    filetypes : list[str]
        Requested export formats.
    dpi : int
        Export DPI for raster outputs.

    Returns
    -------
    None
        Figures and tables are generated and exported for side effects.

    Notes
    -----
    When ``filetypes`` is empty, default export type is PNG.
    All analysis tables are exported as CSV under ``export_dir/tables``.
    All tables are also exported as standalone table-figure images.
    One additional ``overall_data.csv`` with all loaded raw records is exported.

    Examples
    --------
    >>> from pathlib import Path
    >>> # run_report(Path("benchmark"), Path("benchmark/export"), ["png"], 220)
    """
    plt.close("all")
    try:
        # Load benchmark data and build immutable plotting context.
        _set_plot_style()
        resolved_root = resolve_benchmark_root(benchmark_root.resolve())
        totals, steps = load_all_benchmarks(resolved_root)
        ctx = _build_context(totals, steps)

        # Build all table artifacts once for both CSV export and pre-master plots.
        normalized_scaling = _build_normalized_scaling_table(ctx)
        run_wise_totals = _build_run_wise_totals_table(ctx)
        jump_df = _build_jump_table(ctx)
        tables = _collect_all_tables(ctx, normalized_scaling, run_wise_totals, jump_df)

        # Prepare export settings and export figures in small groups.
        export_root = export_dir.resolve()
        export_root.mkdir(parents=True, exist_ok=True)
        normalized_types = _normalize_filetypes(filetypes or ["png"])
        export_dpi = max(50, int(dpi))

        _export_pre_master_figure_groups(
            ctx,
            normalized_scaling,
            run_wise_totals,
            jump_df,
            export_root,
            normalized_types,
            export_dpi,
        )
        _export_figures([_render_master_figure(ctx)], export_root, normalized_types, export_dpi)
        _export_single_figure_groups(ctx, export_root, normalized_types, export_dpi)
        _export_tables_csv(tables, export_root)
        _export_table_plot_figures(tables, export_root, normalized_types, export_dpi)
        _export_overall_data_csv(totals, steps, export_root)
    finally:
        # Ensure no matplotlib figures remain open after report execution.
        plt.close("all")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark analysis report generation.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.

    Notes
    -----
    Default export format is PNG when ``--filetype`` is omitted.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/benchmark_analysis_report.py --filetype svg
    """
    parser = argparse.ArgumentParser(description="Generate benchmark analysis figures and CSV tables from benchmark JSON outputs.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"), help="Benchmark root directory.")
    parser.add_argument("--export-dir", type=Path, default=Path("benchmark/export"), help="Figure/table export directory.")
    parser.add_argument(
        "--filetype",
        action="append",
        default=None,
        choices=["png", "svg"],
        help="Export file type. Repeat for multiple formats. Default: png.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Export DPI for raster outputs.")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for benchmark analysis report generation.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Process exit code.

    Notes
    -----
    Returns ``0`` on successful analysis and export.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/benchmark_analysis_report.py
    """
    args = parse_args()
    run_report(
        benchmark_root=args.benchmark_root,
        export_dir=args.export_dir,
        filetypes=args.filetype or ["png"],
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
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
Discrete rendering helper for time-series feature plots.

This helper contains rendering behavior that only applies to discrete features
(axis mapping, discrete styles, offset layout, and occupancy curves).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..time_series_plot_config import TimeSeriesPlotConfig

from ....helper.color_resolution_helper import ColorResolutionHelper
from ....helper.discrete_feature_helper import DiscreteFeatureHelper


class TimeSeriesDiscretePlotHelper:
    """
    Discrete feature utilities used by time-series plotting.

    The helper centralizes axis mapping and rendering details that are specific
    to discrete features.
    """

    @staticmethod
    def build_axis_config(
        config: TimeSeriesPlotConfig,
        feat_idx: int,
        viz: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build discrete axis mapping for one feature.

        The method uses metadata labels when available and falls back to
        observed trajectory values when metadata labels are missing.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        feat_idx : int
            Feature index in selected matrix.
        viz : Dict[str, Any]
            Visualization configuration.

        Returns
        ------
        Dict[str, Any]
            Axis configuration dictionary.
        """
        if TimeSeriesDiscretePlotHelper._has_metadata_tick_labels(
            viz=viz,
            long_labels=config.long_labels
        ):
            return DiscreteFeatureHelper.build_axis_config(
                selector_data=None,
                viz=viz,
                long_labels=config.long_labels,
                x_padding=0.3,
                fallback_from_data=True
            )

        frame_indices_by_traj = TimeSeriesDiscretePlotHelper._build_frame_indices_by_trajectory(
            config.frame_mapping
        )
        selector_data = TimeSeriesDiscretePlotHelper._collect_selector_data_for_axis(
            config=config,
            feat_idx=feat_idx,
            frame_indices_by_traj=frame_indices_by_traj
        )

        return DiscreteFeatureHelper.build_axis_config(
            selector_data=selector_data,
            viz=viz,
            long_labels=config.long_labels,
            x_padding=0.3,
            fallback_from_data=True
        )

    @staticmethod
    def plot_discrete_overlay_or_offset(
        ax: plt.Axes,
        feat_idx: int,
        config: TimeSeriesPlotConfig,
        axis_config: Optional[Dict[str, Any]],
        apply_offsets: bool
    ) -> None:
        """
        Plot discrete traces as overlay or as vertically offset traces.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        feat_idx : int
            Feature index in selected matrix.
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        axis_config : Dict[str, Any], optional
            Discrete value mapping produced by `build_axis_config`.
        apply_offsets : bool
            Whether traces should be shifted by small vertical offsets.
        """
        if axis_config is None:
            return

        value_to_position = axis_config.get("value_to_position", {})
        frame_indices_by_traj = TimeSeriesDiscretePlotHelper._build_frame_indices_by_trajectory(
            config.frame_mapping
        )
        n_traces = TimeSeriesDiscretePlotHelper._count_overlay_traces(
            config=config,
            frame_indices_by_traj=frame_indices_by_traj
        )
        if n_traces <= 0:
            return

        thickness = TimeSeriesDiscretePlotHelper._scale_discrete_thickness(
            base_thickness=config.thickness,
            n_traces=n_traces,
            threshold=config.discrete_auto_offset_threshold
        )
        if apply_offsets:
            offsets = TimeSeriesDiscretePlotHelper.compute_symmetric_offsets(
                n_series=n_traces,
                span=config.discrete_offset_span
            )
        else:
            offsets = np.zeros(n_traces, dtype=float)

        rendered_trace_idx = 0
        for traj_idx in sorted(config.tag_map.keys()):
            traj_frame_indices = frame_indices_by_traj.get(traj_idx, [])
            if not traj_frame_indices:
                continue
            x_values, y_values = TimeSeriesDiscretePlotHelper._get_trajectory_xy(
                config=config,
                traj_idx=traj_idx,
                feat_idx=feat_idx,
                traj_frame_indices=traj_frame_indices
            )
            if y_values.size == 0:
                continue

            mapped = DiscreteFeatureHelper.prepare_discrete_data(y_values, value_to_position)
            if mapped.size == 0:
                continue

            mapped_float = mapped.astype(float)
            if config.use_tag_coloring:
                tags = config.tag_map.get(traj_idx, [])
                for tag in tags:
                    offset = offsets[rendered_trace_idx]
                    rendered_trace_idx += 1
                    TimeSeriesDiscretePlotHelper.render_discrete_trace(
                        ax=ax,
                        x_values=x_values,
                        y_values=mapped_float + offset,
                        color=config.tag_colors.get(tag, "black"),
                        plot_style=config.discrete_plot_style,
                        thickness=thickness,
                        alpha=0.8
                    )
            else:
                traj_name = config.pipeline_data.trajectory_data.trajectory_names[traj_idx]
                offset = offsets[rendered_trace_idx]
                rendered_trace_idx += 1
                TimeSeriesDiscretePlotHelper.render_discrete_trace(
                    ax=ax,
                    x_values=x_values,
                    y_values=mapped_float + offset,
                    color=config.traj_colors.get(traj_name, "black"),
                    plot_style=config.discrete_plot_style,
                    thickness=thickness,
                    alpha=0.8
                )

    @staticmethod
    def plot_discrete_occupancy(
        ax: plt.Axes,
        feat_idx: int,
        config: TimeSeriesPlotConfig,
        axis_config: Optional[Dict[str, Any]]
    ) -> None:
        """
        Plot per-state occupancy probabilities over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        feat_idx : int
            Feature index in selected matrix.
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        axis_config : Dict[str, Any], optional
            Discrete value mapping produced by `build_axis_config`.
        """
        if axis_config is None:
            return

        n_states = len(axis_config.get("positions", []))
        if n_states <= 0:
            return

        value_to_position = axis_config.get("value_to_position", {})
        frame_indices_by_traj = TimeSeriesDiscretePlotHelper._build_frame_indices_by_trajectory(
            config.frame_mapping
        )
        mapped_series: List[np.ndarray] = []
        reference_traj_idx: Optional[int] = None
        reference_frame_indices: List[int] = []

        for traj_idx in sorted(config.tag_map.keys()):
            traj_frame_indices = frame_indices_by_traj.get(traj_idx, [])
            if not traj_frame_indices:
                continue

            y_values = TimeSeriesDiscretePlotHelper._get_trajectory_y_values(
                config=config,
                traj_frame_indices=traj_frame_indices,
                feat_idx=feat_idx
            )
            if y_values.size == 0:
                continue

            mapped = DiscreteFeatureHelper.prepare_discrete_data(y_values, value_to_position)
            if mapped.size == 0:
                continue

            mapped_series.append(mapped)
            if reference_traj_idx is None:
                reference_traj_idx = traj_idx
                reference_frame_indices = traj_frame_indices

        if not mapped_series or reference_traj_idx is None:
            return

        probabilities = TimeSeriesDiscretePlotHelper.calculate_time_resolved_probabilities(
            mapped_series=mapped_series,
            n_positions=n_states
        )
        if probabilities.shape[1] == 0:
            return

        x_ref = TimeSeriesDiscretePlotHelper._get_trajectory_x_values(
            config=config,
            traj_idx=reference_traj_idx,
            traj_frame_indices=reference_frame_indices,
            n_points=probabilities.shape[1]
        )
        state_labels = [str(label) for label in axis_config.get("tick_labels", [])]
        if not state_labels:
            state_labels = [str(i) for i in range(n_states)]

        state_colors = dict(config.discrete_state_colors)
        if isinstance(config.colors, dict):
            for label in state_labels:
                if label in config.colors:
                    state_colors[label] = config.colors[label]

        missing_labels = [label for label in state_labels if label not in state_colors]
        if missing_labels:
            state_colors.update(
                ColorResolutionHelper.resolve_label_colors(
                    labels=missing_labels,
                    colors=config.colors
                )
            )

        for label in state_labels:
            config.discrete_state_colors[label] = state_colors[label]

        for state_idx, state_label in enumerate(state_labels):
            if state_idx >= probabilities.shape[0]:
                break
            ax.plot(
                x_ref,
                probabilities[state_idx],
                color=state_colors[state_label],
                linewidth=max(1.0, config.thickness),
                alpha=0.9
            )

    @staticmethod
    def configure_discrete_y_axis(
        ax: plt.Axes,
        viz: Dict[str, Any],
        long_labels: bool,
        tick_fontsize: Optional[int] = None,
        axis_config: Optional[Dict[str, Any]] = None,
        resolved_discrete_layout: str = "overlay",
        discrete_offset_span: float = 0.28
    ) -> None:
        """
        Configure y-axis ticks and limits for discrete features.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        viz : Dict[str, Any]
            Visualization metadata (used if `axis_config` is missing).
        long_labels : bool
            Whether long labels should be used.
        tick_fontsize : int, optional
            Tick label font size.
        axis_config : Dict[str, Any], optional
            Prepared axis configuration from `build_axis_config`.
        resolved_discrete_layout : str, default="overlay"
            Effective discrete layout mode.
        discrete_offset_span : float, default=0.28
            Offset half-span used to adjust y-margins in offset mode.
        """
        if axis_config is not None:
            positions = list(axis_config.get("positions", []))
            tick_labels = list(axis_config.get("tick_labels", []))
        else:
            tick_labels_dict = viz.get("tick_labels", {})
            label_key = "long" if long_labels else "short"
            tick_labels = list(tick_labels_dict.get(label_key, []))
            positions = list(range(len(tick_labels)))

        if not tick_labels or not positions:
            return

        margin = discrete_offset_span + 0.12 if resolved_discrete_layout == "offset" else 0.3
        min_pos = float(min(positions))
        max_pos = float(max(positions))
        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels, fontsize=tick_fontsize or 10)
        ax.set_ylim(min_pos - margin, max_pos + margin)

    @staticmethod
    def compute_symmetric_offsets(n_series: int, span: float) -> np.ndarray:
        """
        Return symmetric offsets for discrete traces.

        Parameters
        ----------
        n_series : int
            Number of traces that need separation.
        span : float
            Half-width of the offset interval.

        Returns
        -------
        np.ndarray
            Offset values in `[-span, +span]`.
        """
        if n_series <= 1:
            return np.array([0.0], dtype=float)
        return np.linspace(-span, span, n_series, dtype=float)

    @staticmethod
    def calculate_time_resolved_probabilities(
        mapped_series: List[np.ndarray],
        n_positions: int
    ) -> np.ndarray:
        """
        Convert mapped discrete trajectories into per-time state probabilities.

        Parameters
        ----------
        mapped_series : List[np.ndarray]
            List of mapped state arrays (one per trajectory).
        n_positions : int
            Number of discrete states.
        """
        if n_positions <= 0:
            return np.zeros((0, 0), dtype=float)

        valid_series = [
            np.asarray(series).ravel()
            for series in mapped_series
            if np.asarray(series).size > 0
        ]
        if not valid_series:
            return np.zeros((n_positions, 0), dtype=float)

        min_length = min(series.size for series in valid_series)
        if min_length <= 0:
            return np.zeros((n_positions, 0), dtype=float)

        probabilities = np.zeros((n_positions, min_length), dtype=float)
        for time_idx in range(min_length):
            values_t = []
            for series in valid_series:
                value = int(series[time_idx])
                if 0 <= value < n_positions:
                    values_t.append(value)
            if not values_t:
                continue
            counts = np.bincount(values_t, minlength=n_positions)
            probabilities[:, time_idx] = counts / float(len(values_t))

        return probabilities

    @staticmethod
    def render_discrete_trace(
        ax: plt.Axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        color: str,
        plot_style: str,
        thickness: float,
        alpha: float = 0.8
    ) -> None:
        """
        Render one discrete trace with the configured style.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        x_values : np.ndarray
            X-axis values.
        y_values : np.ndarray
            Y-axis values.
        color : str
            Trace color.
        plot_style : str
            One of `line`, `step`, `segments`, `scatter`.
        thickness : float
            Line width or scatter scale.
        alpha : float, default=0.8
            Alpha value for the trace.
        """
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        if x_values.size == 0 or y_values.size == 0:
            return

        if plot_style == "scatter":
            ax.scatter(x_values, y_values, color=color, s=12.0 * thickness, alpha=alpha, linewidths=0)
            return

        if plot_style == "segments":
            if x_values.size >= 2:
                ax.hlines(
                    y=y_values[:-1],
                    xmin=x_values[:-1],
                    xmax=x_values[1:],
                    colors=color,
                    linewidth=thickness,
                    alpha=alpha
                )
            else:
                ax.scatter(x_values, y_values, color=color, s=12.0 * thickness, alpha=alpha, linewidths=0)
            return

        if plot_style == "step":
            ax.step(x_values, y_values, where="post", color=color, linewidth=thickness, alpha=alpha)
            return

        ax.plot(x_values, y_values, color=color, linewidth=thickness, alpha=alpha)

    @staticmethod
    def _get_trajectory_xy(
        config: TimeSeriesPlotConfig,
        traj_idx: int,
        feat_idx: int,
        traj_frame_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract x/y data for one trajectory and one feature.

        The function respects selected frames and `use_time` mode.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        traj_idx : int
            Trajectory index.
        feat_idx : int
            Feature index.
        traj_frame_indices : List[int], optional
            Precomputed global frame indices for this trajectory. When omitted,
            indices are derived from `config.frame_mapping`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x_values, y_values)
        """
        if traj_frame_indices is None:
            frame_indices_by_traj = TimeSeriesDiscretePlotHelper._build_frame_indices_by_trajectory(
                config.frame_mapping
            )
            traj_frame_indices = frame_indices_by_traj.get(traj_idx, [])
        if not traj_frame_indices:
            return np.array([], dtype=float), np.array([], dtype=float)

        y_values = TimeSeriesDiscretePlotHelper._get_trajectory_y_values(
            config=config,
            traj_frame_indices=traj_frame_indices,
            feat_idx=feat_idx
        )
        x_values = TimeSeriesDiscretePlotHelper._get_trajectory_x_values(
            config=config,
            traj_idx=traj_idx,
            traj_frame_indices=traj_frame_indices,
            n_points=y_values.size
        )
        return x_values, y_values

    @staticmethod
    def _has_metadata_tick_labels(viz: Dict[str, Any], long_labels: bool) -> bool:
        """
        Check whether discrete metadata already provides usable tick labels.

        Parameters
        ----------
        viz : Dict[str, Any]
            Visualization metadata dictionary.
        long_labels : bool
            Whether long labels should be used.

        Returns
        -------
        bool
            True when metadata contains at least one tick label for the chosen
            label mode.
        """
        tick_labels_dict = viz.get("tick_labels", {})
        label_key = "long" if long_labels else "short"
        tick_labels = list(tick_labels_dict.get(label_key, []))
        return len(tick_labels) > 0

    @staticmethod
    def _collect_selector_data_for_axis(
        config: TimeSeriesPlotConfig,
        feat_idx: int,
        frame_indices_by_traj: Dict[int, List[int]]
    ) -> Dict[str, np.ndarray]:
        """
        Collect per-trajectory feature values for data-driven axis fallback.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        feat_idx : int
            Feature index in selected matrix.
        frame_indices_by_traj : Dict[int, List[int]]
            Precomputed global frame indices grouped by trajectory.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping `traj_<idx>` -> flattened feature values.
        """
        selector_data: Dict[str, np.ndarray] = {}
        for traj_idx in sorted(config.tag_map.keys()):
            traj_frame_indices = frame_indices_by_traj.get(traj_idx, [])
            if not traj_frame_indices:
                continue
            y_values = TimeSeriesDiscretePlotHelper._get_trajectory_y_values(
                config=config,
                traj_frame_indices=traj_frame_indices,
                feat_idx=feat_idx
            )
            if y_values.size > 0:
                selector_data[f"traj_{traj_idx}"] = y_values
        return selector_data

    @staticmethod
    def _count_overlay_traces(
        config: TimeSeriesPlotConfig,
        frame_indices_by_traj: Dict[int, List[int]]
    ) -> int:
        """
        Count drawable traces for overlay/offset layout without loading data.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        frame_indices_by_traj : Dict[int, List[int]]
            Precomputed global frame indices grouped by trajectory.

        Returns
        -------
        int
            Number of traces that will be rendered.
        """
        n_traces = 0
        for traj_idx in sorted(config.tag_map.keys()):
            if not frame_indices_by_traj.get(traj_idx):
                continue

            if config.use_tag_coloring:
                n_traces += len(config.tag_map.get(traj_idx, []))
            else:
                n_traces += 1
        return n_traces

    @staticmethod
    def _build_frame_indices_by_trajectory(
        frame_mapping: Dict[int, Tuple[int, int]]
    ) -> Dict[int, List[int]]:
        """
        Build trajectory-indexed global frame lists from frame mapping.

        Parameters
        ----------
        frame_mapping : Dict[int, Tuple[int, int]]
            Mapping `global_frame_idx -> (trajectory_idx, local_frame_idx)`.

        Returns
        -------
        Dict[int, List[int]]
            Mapping `trajectory_idx -> sorted global frame indices`.
        """
        indices_by_traj: Dict[int, List[int]] = {}
        for global_idx, (traj_idx, _) in frame_mapping.items():
            indices_by_traj.setdefault(traj_idx, []).append(global_idx)

        for global_indices in indices_by_traj.values():
            global_indices.sort()
        return indices_by_traj

    @staticmethod
    def _get_trajectory_y_values(
        config: TimeSeriesPlotConfig,
        traj_frame_indices: List[int],
        feat_idx: int
    ) -> np.ndarray:
        """
        Extract only y-values for one trajectory/feature combination.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        traj_frame_indices : List[int]
            Global frame indices for the selected trajectory.
        feat_idx : int
            Feature index in selected matrix.

        Returns
        -------
        np.ndarray
            Flattened y-values.
        """
        if not traj_frame_indices:
            return np.array([], dtype=float)
        return np.asarray(config.selected_matrix[traj_frame_indices, feat_idx]).ravel()

    @staticmethod
    def _get_trajectory_x_values(
        config: TimeSeriesPlotConfig,
        traj_idx: int,
        traj_frame_indices: List[int],
        n_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Build x-values for one trajectory based on current x-axis mode.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Time-series plotting configuration.
        traj_idx : int
            Trajectory index.
        traj_frame_indices : List[int]
            Global frame indices for the selected trajectory.
        n_points : int, optional
            Optional output length cap applied from the beginning.

        Returns
        -------
        np.ndarray
            X-values in ns (`use_time=True`) or sequential frame index.
        """
        if not traj_frame_indices:
            return np.array([], dtype=float)

        if config.use_time:
            trajectory = config.pipeline_data.trajectory_data.trajectories[traj_idx]
            local_frames = [config.frame_mapping[i][1] for i in traj_frame_indices]
            x_values = np.asarray(trajectory.time[local_frames], dtype=float) / 1000.0
        else:
            x_values = np.arange(len(traj_frame_indices), dtype=float)

        if n_points is None:
            return x_values
        return x_values[:n_points]

    @staticmethod
    def _scale_discrete_thickness(base_thickness: float, n_traces: int, threshold: int) -> float:
        """
        Thin discrete traces adaptively when many traces are rendered.

        Parameters
        ----------
        base_thickness : float
            User-configured thickness.
        n_traces : int
            Number of traces being drawn.
        threshold : int
            Trace count where thinning starts.

        Returns
        -------
        float
            Effective drawing thickness.
        """
        if n_traces < threshold:
            return base_thickness
        scaled = base_thickness * (8.0 / float(max(1, n_traces)))
        return max(0.3, scaled)

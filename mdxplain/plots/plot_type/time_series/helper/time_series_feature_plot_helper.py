# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
#
# This program is free software under GNU LGPL v3.

"""Helper for feature-level orchestration in time-series plots."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from ..time_series_plot_config import TimeSeriesPlotConfig

from .time_series_data_preparer import TimeSeriesDataPreparer
from .time_series_discrete_plot_helper import TimeSeriesDiscretePlotHelper
from .time_series_tag_coloring_helper import TimeSeriesTagColoringHelper


class TimeSeriesFeaturePlotHelper:
    """
    Orchestrator for feature subplot creation and rendering in time-series plots.

    Discrete-specific rendering details are delegated to
    `TimeSeriesDiscretePlotHelper` to keep this helper focused on subplot flow.
    """

    @staticmethod
    def plot_all_features(
        fig: Figure,
        gs: GridSpec,
        config: TimeSeriesPlotConfig
    ) -> plt.Axes:
        """
        Plot all configured features in the prepared grid layout.

        Parameters
        ----------
        fig : Figure
            Figure object receiving the feature subplots.
        gs : GridSpec
            Grid specification used to position feature subplots.
        config : TimeSeriesPlotConfig
            Central time-series plotting configuration.

        Returns
        -------
        matplotlib.axes.Axes
            Rightmost axes in the first row, used for legend anchoring.
        """
        col_offset = TimeSeriesFeaturePlotHelper._get_column_offset(config)
        rightmost_ax_first_row = None

        for feature_idx, (feat_type, feat_name) in enumerate(config.all_features):
            ax, is_first_row = TimeSeriesFeaturePlotHelper._create_feature_subplot(
                fig=fig,
                gs=gs,
                config=config,
                feature_index=feature_idx,
                col_offset=col_offset
            )
            if is_first_row:
                rightmost_ax_first_row = ax

            feat_idx = TimeSeriesFeaturePlotHelper._find_feature_index(
                feature_indices=config.feature_indices,
                feat_name=feat_name
            )
            if feat_idx is None:
                continue

            TimeSeriesFeaturePlotHelper._plot_single_feature(
                ax=ax,
                feat_idx=feat_idx,
                feat_type=feat_type,
                feat_name=feat_name,
                config=config
            )

        return rightmost_ax_first_row

    @staticmethod
    def _get_column_offset(config: TimeSeriesPlotConfig) -> int:
        """
        Return grid column offset for optional membership label column.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Central plotting configuration.

        Returns
        -------
        int
            `1` when a dedicated label column is present, otherwise `0`.
        """
        has_label_column = config.clustering_name and config.membership_per_feature
        return 1 if has_label_column else 0

    @staticmethod
    def _create_feature_subplot(
        fig: Figure,
        gs: GridSpec,
        config: TimeSeriesPlotConfig,
        feature_index: int,
        col_offset: int
    ) -> tuple[plt.Axes, bool]:
        """
        Create subplot axes for one feature based on precomputed layout.

        Parameters
        ----------
        fig : Figure
            Target figure.
        gs : GridSpec
            Precomputed grid specification.
        config : TimeSeriesPlotConfig
            Central plotting configuration.
        feature_index : int
            Index into `config.all_features`.
        col_offset : int
            Horizontal offset for optional label column.

        Returns
        -------
        tuple[matplotlib.axes.Axes, bool]
            Tuple `(ax, is_first_row)`.
        """
        _, row, col, colspan = config.layout[feature_index]
        actual_row = row * 2 if config.membership_per_feature else row
        ax = fig.add_subplot(gs[actual_row, col + col_offset: col + col_offset + colspan])
        return ax, row == 0

    @staticmethod
    def _plot_single_feature(
        ax: plt.Axes,
        feat_idx: int,
        feat_type: str,
        feat_name: str,
        config: TimeSeriesPlotConfig
    ) -> None:
        """
        Render one feature subplot including traces and axis styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Subplot axes for this feature.
        feat_idx : int
            Feature index in the selected matrix.
        feat_type : str
            Feature type key.
        feat_name : str
            Feature display name.
        config : TimeSeriesPlotConfig
            Central plotting configuration.

        Returns
        -------
        None
            Modifies the axes in place.
        """
        ax.set_title(feat_name, fontsize=config.subplot_title_fontsize or 14, pad=8)

        feature_metadata = config.metadata_map.get(feat_type, {}).get(feat_name, {})
        type_metadata = feature_metadata.get("type_metadata", {})
        viz = type_metadata.get("visualization", {})
        is_discrete = bool(viz.get("is_discrete", False))

        discrete_axis_config = None
        if is_discrete:
            discrete_axis_config = TimeSeriesDiscretePlotHelper.build_axis_config(
                config=config,
                feat_idx=feat_idx,
                viz=viz
            )

        TimeSeriesFeaturePlotHelper._plot_feature_lines(
            ax=ax,
            feat_idx=feat_idx,
            config=config,
            is_discrete=is_discrete,
            discrete_axis_config=discrete_axis_config
        )
        TimeSeriesFeaturePlotHelper._plot_vertical_markers(
            ax=ax,
            config=config
        )

        x_values = TimeSeriesDataPreparer.get_x_values(
            config.pipeline_data,
            traj_idx=0,
            use_time=config.use_time
        )
        TimeSeriesFeaturePlotHelper._configure_axes(
            ax=ax,
            feat_type=feat_type,
            feature_metadata=feature_metadata,
            config=config,
            x_values=x_values,
            is_discrete=is_discrete,
            discrete_axis_config=discrete_axis_config
        )

    @staticmethod
    def _plot_feature_lines(
        ax: plt.Axes,
        feat_idx: int,
        config: TimeSeriesPlotConfig,
        is_discrete: bool = False,
        discrete_axis_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Render feature traces using continuous or discrete rendering path.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        feat_idx : int
            Feature index in the selected matrix.
        config : TimeSeriesPlotConfig
            Central plotting configuration.
        is_discrete : bool, default=False
            Whether this feature is discrete.
        discrete_axis_config : Dict[str, Any], optional
            Axis configuration prepared for discrete value mapping.

        Returns
        -------
        None
            Modifies the axes in place.
        """
        if is_discrete:
            if config.resolved_discrete_layout == "occupancy":
                TimeSeriesDiscretePlotHelper.plot_discrete_occupancy(
                    ax=ax,
                    feat_idx=feat_idx,
                    config=config,
                    axis_config=discrete_axis_config
                )
            else:
                TimeSeriesDiscretePlotHelper.plot_discrete_overlay_or_offset(
                    ax=ax,
                    feat_idx=feat_idx,
                    config=config,
                    axis_config=discrete_axis_config,
                    apply_offsets=(config.resolved_discrete_layout == "offset")
                )
            return

        if config.use_tag_coloring:
            TimeSeriesTagColoringHelper.plot_feature_with_tag_colors(
                ax=ax,
                pipeline_data=config.pipeline_data,
                feat_idx=feat_idx,
                tag_map=config.tag_map,
                tag_colors=config.tag_colors,
                matrix=config.selected_matrix,
                frame_mapping=config.frame_mapping,
                use_time=config.use_time,
                smoothing=config.smoothing,
                smoothing_method=config.smoothing_method,
                smoothing_window=config.smoothing_window,
                smoothing_polyorder=config.smoothing_polyorder,
                show_unsmoothed_background=config.show_unsmoothed_background,
                thickness=config.thickness
            )
            return

        TimeSeriesTagColoringHelper.plot_feature_with_trajectory_colors(
            ax=ax,
            pipeline_data=config.pipeline_data,
            feat_idx=feat_idx,
            tag_map=config.tag_map,
            traj_colors=config.traj_colors,
            matrix=config.selected_matrix,
            frame_mapping=config.frame_mapping,
            use_time=config.use_time,
            smoothing=config.smoothing,
            smoothing_method=config.smoothing_method,
            smoothing_window=config.smoothing_window,
            smoothing_polyorder=config.smoothing_polyorder,
            show_unsmoothed_background=config.show_unsmoothed_background,
            thickness=config.thickness
        )

    @staticmethod
    def _plot_vertical_markers(
        ax: plt.Axes,
        config: TimeSeriesPlotConfig
    ) -> None:
        """
        Plot pre-resolved vertical guide markers for one feature subplot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw the marker lines on.
        config : TimeSeriesPlotConfig
            Central plotting configuration containing
            `resolved_vertical_markers` tuples `(x, color, label)`.

        Returns
        -------
        None
            Modifies the axes in place.
        """
        if not config.resolved_vertical_markers:
            return

        line_width = max(1.0, config.thickness)
        for x_position, color, _ in config.resolved_vertical_markers:
            ax.axvline(
                x=x_position,
                color=color,
                linestyle="--",
                linewidth=line_width,
                alpha=0.85,
                zorder=6
            )

    @staticmethod
    def _configure_axes(
        ax: plt.Axes,
        feat_type: str,
        feature_metadata: Dict[str, Any],
        config: TimeSeriesPlotConfig,
        x_values,
        is_discrete: bool = False,
        discrete_axis_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Apply axis labels, limits, and feature-specific reference styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to configure.
        feat_type : str
            Feature type key.
        feature_metadata : Dict[str, Any]
            Metadata for the current feature.
        config : TimeSeriesPlotConfig
            Central plotting configuration.
        x_values : np.ndarray
            Reference x-values for x-limits.
        is_discrete : bool, default=False
            Whether this feature is discrete.
        discrete_axis_config : Dict[str, Any], optional
            Prepared discrete axis configuration.

        Returns
        -------
        None
            Modifies the axes in place.
        """
        y_label = TimeSeriesFeaturePlotHelper._get_feature_y_label(
            feat_type=feat_type,
            feature_metadata=feature_metadata,
            is_discrete=is_discrete,
            resolved_discrete_layout=config.resolved_discrete_layout
        )
        ax.set_ylabel(y_label, fontsize=config.ylabel_fontsize or 12)
        ax.set_xlabel("Time (ns)" if config.use_time else "Frame", fontsize=config.xlabel_fontsize or 12)
        ax.tick_params(axis="both", labelsize=config.tick_fontsize or 10)
        ax.grid(True, alpha=0.3)

        if len(x_values) >= 2:
            x_range = x_values[-1] - x_values[0]
            x_margin = x_range * 0.05
            ax.set_xlim(x_values[0] - x_margin, x_values[-1] + x_margin)
        elif len(x_values) == 1:
            ax.set_xlim(x_values[0] - 0.5, x_values[0] + 0.5)

        type_metadata = feature_metadata.get("type_metadata", {})
        viz = type_metadata.get("visualization", {})
        if is_discrete and config.resolved_discrete_layout == "occupancy":
            ax.set_ylim(-0.02, 1.02)
        elif bool(viz.get("is_discrete", False)):
            TimeSeriesDiscretePlotHelper.configure_discrete_y_axis(
                ax=ax,
                viz=viz,
                long_labels=config.long_labels,
                tick_fontsize=config.tick_fontsize,
                axis_config=discrete_axis_config,
                resolved_discrete_layout=config.resolved_discrete_layout,
                discrete_offset_span=config.discrete_offset_span
            )

        if feat_type == "distances" and config.contact_threshold is not None:
            ax.axhline(
                y=config.contact_threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=5
            )

    @staticmethod
    def _get_feature_y_label(
        feat_type: str,
        feature_metadata: Dict[str, Any],
        is_discrete: bool,
        resolved_discrete_layout: str
    ) -> str:
        """
        Resolve y-axis label from metadata and discrete rendering context.

        Parameters
        ----------
        feat_type : str
            Feature type key.
        feature_metadata : Dict[str, Any]
            Metadata for the current feature.
        is_discrete : bool
            Whether this feature is discrete.
        resolved_discrete_layout : str
            Effective discrete layout mode.

        Returns
        -------
        str
            Y-axis label for the subplot.
        """
        if is_discrete and resolved_discrete_layout == "occupancy":
            return "Probability"

        type_metadata = feature_metadata.get("type_metadata", {})
        viz = type_metadata.get("visualization", {})
        return viz.get("axis_label", feat_type.capitalize())

    @staticmethod
    def _find_feature_index(feature_indices: Dict[int, str], feat_name: str) -> Optional[int]:
        """
        Find matrix feature index for a given feature name.

        Parameters
        ----------
        feature_indices : Dict[int, str]
            Mapping from matrix index to feature name.
        feat_name : str
            Feature name to look up.

        Returns
        -------
        int or None
            Matching index if found, otherwise `None`.
        """
        for idx, name in feature_indices.items():
            if name == feat_name:
                return idx
        return None

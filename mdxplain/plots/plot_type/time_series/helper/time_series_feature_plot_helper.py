# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
#
# This program is free software under GNU LGPL v3.

"""Helper for feature plotting in time series."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from ..time_series_plot_config import TimeSeriesPlotConfig

from .time_series_tag_coloring_helper import TimeSeriesTagColoringHelper


class TimeSeriesFeaturePlotHelper:
    """Helper for feature plotting with tag/trajectory coloring."""

    @staticmethod
    def plot_all_features(
        fig: Figure,
        gs: GridSpec,
        config: TimeSeriesPlotConfig
    ) -> plt.Axes:
        """
        Plot all features in grid.

        Parameters
        ----------
        fig : Figure
            Figure to plot on
        gs : GridSpec
            GridSpec layout
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        plt.Axes
            Rightmost axes in first row

        Examples
        --------
        >>> ax = TimeSeriesFeaturePlotHelper.plot_all_features(fig, gs, config)
        """
        col_offset = TimeSeriesFeaturePlotHelper._get_column_offset(config)
        rightmost_ax_first_row = None

        for i, (feat_type, feat_name) in enumerate(config.all_features):
            ax, is_first_row = TimeSeriesFeaturePlotHelper._create_feature_subplot(
                fig, gs, config, i, col_offset
            )

            if is_first_row:
                rightmost_ax_first_row = ax

            feat_idx = TimeSeriesFeaturePlotHelper._find_feature_index(
                config.feature_indices, feat_name
            )

            if feat_idx is not None:
                TimeSeriesFeaturePlotHelper._plot_single_feature(
                    ax, feat_idx, feat_type, feat_name, config
                )

        return rightmost_ax_first_row

    @staticmethod
    def _get_column_offset(config: TimeSeriesPlotConfig) -> int:
        """
        Get column offset for label column.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Configuration object

        Returns
        -------
        int
            Column offset (1 if label column exists, 0 otherwise)

        Examples
        --------
        >>> offset = TimeSeriesFeaturePlotHelper._get_column_offset(config)
        """
        has_label_column = config.clustering_name and config.membership_per_feature
        return 1 if has_label_column else 0

    @staticmethod
    def _create_feature_subplot(fig, gs, config, feature_index, col_offset):
        """
        Create subplot for feature.

        Parameters
        ----------
        fig : Figure
            Figure object
        gs : GridSpec
            GridSpec layout
        config : TimeSeriesPlotConfig
            Configuration object
        feature_index : int
            Index in config.all_features
        col_offset : int
            Column offset for label column

        Returns
        -------
        tuple
            (axes, is_first_row)

        Examples
        --------
        >>> ax, is_first = TimeSeriesFeaturePlotHelper._create_feature_subplot(
        ...     fig, gs, config, 0, 1
        ... )
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
    ):
        """
        Plot single feature on axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        feat_idx : int
            Feature index
        feat_type : str
            Feature type
        feat_name : str
            Feature name
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesFeaturePlotHelper._plot_single_feature(
        ...     ax, 0, "distances", "CA-CB", config
        ... )
        """
        ax.set_title(feat_name, fontsize=12, pad=8)

        TimeSeriesFeaturePlotHelper._plot_feature_lines(ax, feat_idx, config)

        feature_metadata = config.metadata_map.get(feat_type, {}).get(feat_name, {})
        TimeSeriesFeaturePlotHelper._configure_axes(
            ax, feat_type, feat_name, feature_metadata, config
        )

    @staticmethod
    def _plot_feature_lines(ax: plt.Axes, feat_idx: int, config):
        """
        Plot feature lines with appropriate coloring.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        feat_idx : int
            Feature index
        config : TimeSeriesPlotConfig
            Configuration object

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesFeaturePlotHelper._plot_feature_lines(ax, 0, config)
        """
        if config.use_tag_coloring:
            TimeSeriesTagColoringHelper.plot_feature_with_tag_colors(
                ax, config.pipeline_data, feat_idx, config.tag_map,
                config.tag_colors, config.feature_selector_name, config.use_time
            )
        else:
            TimeSeriesTagColoringHelper.plot_feature_with_trajectory_colors(
                ax, config.pipeline_data, feat_idx, config.tag_map,
                config.traj_colors, config.feature_selector_name, config.use_time
            )

    @staticmethod
    def _configure_axes(ax, feat_type, feat_name, feature_metadata, config):
        """
        Configure axes labels and limits.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to configure
        feat_type : str
            Feature type
        feat_name : str
            Feature name
        feature_metadata : Dict
            Feature metadata
        config : TimeSeriesPlotConfig
            Configuration object

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesFeaturePlotHelper._configure_axes(
        ...     ax, "distances", "CA-CB", metadata, config
        ... )
        """
        y_label = TimeSeriesFeaturePlotHelper._get_feature_y_label(
            feat_type, feature_metadata
        )
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlabel("Time (ns)" if config.use_time else "Frame", fontsize=11)
        ax.grid(True, alpha=0.3)

        type_metadata = feature_metadata.get("type_metadata", {})
        viz = type_metadata.get("visualization", {})
        if viz.get("is_discrete", False):
            TimeSeriesFeaturePlotHelper._configure_discrete_y_axis(
                ax, viz, config.long_labels
            )

        if feat_type == "distances" and config.contact_threshold is not None:
            ax.axhline(y=config.contact_threshold, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7, zorder=5)

    @staticmethod
    def _get_feature_y_label(feat_type: str, feature_metadata: Dict) -> str:
        """
        Get Y-axis label from metadata.

        Parameters
        ----------
        feat_type : str
            Feature type
        feature_metadata : Dict
            Feature metadata

        Returns
        -------
        str
            Y-axis label

        Examples
        --------
        >>> label = TimeSeriesFeaturePlotHelper._get_feature_y_label(
        ...     "distances", {"type_metadata": {"visualization": {"axis_label": "Distance (Å)"}}}
        ... )
        >>> print(label)  # "Distance (Å)"

        Notes
        -----
        Falls back to capitalized feature type if no axis_label in metadata.
        """
        type_metadata = feature_metadata.get("type_metadata", {})
        viz = type_metadata.get("visualization", {})
        return viz.get("axis_label", feat_type.capitalize())

    @staticmethod
    def _find_feature_index(feature_indices: Dict[int, str], feat_name: str) -> int:
        """
        Find feature index by name.

        Parameters
        ----------
        feature_indices : Dict[int, str]
            Feature index to name mapping
        feat_name : str
            Feature name to search for

        Returns
        -------
        int or None
            Feature index if found, None otherwise

        Examples
        --------
        >>> idx = TimeSeriesFeaturePlotHelper._find_feature_index(
        ...     {0: "CA-CB", 1: "CA-CG"}, "CA-CB"
        ... )
        >>> print(idx)  # 0

        Notes
        -----
        Linear search - acceptable since feature_indices is typically small.
        """
        for idx, name in feature_indices.items():
            if name == feat_name:
                return idx
        return None

    @staticmethod
    def _configure_discrete_y_axis(ax, viz: Dict, long_labels: bool):
        """
        Configure Y-axis for discrete features.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to configure
        viz : Dict
            Visualization metadata from feature
        long_labels : bool
            Use long labels (True) or short labels (False)

        Returns
        -------
        None
            Modifies ax in place

        Examples
        --------
        >>> TimeSeriesFeaturePlotHelper._configure_discrete_y_axis(
        ...     ax, {"tick_labels": {"short": ["H", "E"], "long": ["Helix", "Sheet"]}}, True
        ... )

        Notes
        -----
        Sets Y-ticks, tick labels, and Y-limits based on metadata.
        Falls back silently if tick_labels are missing.
        """
        tick_labels_dict = viz.get("tick_labels", {})
        label_key = "long" if long_labels else "short"
        tick_labels = tick_labels_dict.get(label_key, [])

        if not tick_labels:
            return

        n_ticks = len(tick_labels)
        positions = list(range(n_ticks))
        ylim = (-0.3, n_ticks - 1 + 0.3)

        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(ylim)

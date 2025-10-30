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
Time series plotter for feature visualization.

Creates line plots showing feature values over time with optional
cluster membership bars.
"""

from typing import Optional, List, Union, Dict, Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .time_series_plot_config import TimeSeriesPlotConfig
from .helper import TimeSeriesDataPreparer
from .helper.time_series_tag_coloring_helper import TimeSeriesTagColoringHelper
from .helper.time_series_membership_plot_helper import TimeSeriesMembershipPlotHelper
from .helper.time_series_grid_layout_helper import TimeSeriesGridLayoutHelper
from .helper.time_series_feature_plot_helper import TimeSeriesFeaturePlotHelper
from ..feature_importance_base import FeatureImportanceBasePlotter
from ...helper.clustering_data_helper import ClusteringDataHelper
from ...helper.grid_layout_helper import GridLayoutHelper
from ...helper.contact_to_distances_converter import ContactToDistancesConverter
from ...helper.title_legend_helper import TitleLegendHelper


class TimeSeriesPlotter(FeatureImportanceBasePlotter):
    """
    Plotter for feature time series visualization.

    Creates line plots showing feature values over time for trajectories,
    with optional cluster membership visualization.

    Examples
    --------
    >>> plotter = TimeSeriesPlotter(pipeline_data, "cache/")
    >>> fig = plotter.plot(
    ...     feature_importance_name="analysis",
    ...     n_top=5
    ... )
    """

    def plot(
        self,
        feature_importance_name: Optional[str] = None,
        n_top: int = 5,
        feature_selector: Optional[str] = None,
        traj_selection: Union[int, str, List, "all"] = "all",
        use_time: bool = True,
        tags_for_coloring: Optional[List[str]] = None,
        allow_multi_tag_plotting: bool = False,
        clustering_name: Optional[str] = None,
        membership_per_feature: bool = False,
        membership_traj_selection: Union[str, int, List] = "all",
        contact_transformation: bool = True,
        max_cols: int = 2,
        long_labels: bool = False,
        subplot_height: float = 2.5,
        membership_bar_height: Optional[float] = None,
        show_legend: bool = True,
        contact_threshold: Optional[float] = 4.5,
        title: Optional[str] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300,
        smoothing: bool = True,
        smoothing_method: str = "savitzky",
        smoothing_window: int = 51,
        smoothing_polyorder: int = 3,
        show_unsmoothed_background: bool = True
    ) -> Figure:
        """
        Create time series plots from feature importance or manual selection.

        Supports two modes:
        1. Feature Importance mode: Automatic selection from feature importance
        2. Manual mode: User-defined feature selection

        Parameters
        ----------
        feature_importance_name : str, optional
            Name of feature importance analysis (Feature Importance mode)
        n_top : int, default=5
            Number of top features per comparison (Feature Importance mode)
        feature_selector : str, optional
            Name of feature selector (Manual mode)
        traj_selection : int, str, list, or "all", default="all"
            Trajectory selection for lines
        use_time : bool, default=True
            Use time (True) or frames (False)
        tags_for_coloring : List[str], optional
            Tags for coloring (auto-enables tag-based coloring)
        allow_multi_tag_plotting : bool, default=False
            Plot multi-tag trajectories multiple times
        clustering_name : str, optional
            Clustering for membership bars
        membership_per_feature : bool, default=False
            Show bar per feature (True) or once at bottom (False)
        membership_traj_selection : int, str, list, or "all", default="all"
            Trajectory selection for membership bars
        contact_transformation : bool, default=True
            Convert contacts to distances
        max_cols : int, default=2
            Maximum columns in grid layout
        long_labels : bool, default=False
            Use long labels for discrete features
        subplot_height : float, default=2.5
            Height per feature subplot
        membership_bar_height : float, optional
            Height per trajectory in membership bar.
            Default: 0.25 (membership_per_feature=True) or 0.5 (False)
        show_legend : bool, default=True
            Show legend
        contact_threshold : float, optional
            Contact threshold line (default: 4.5)
        title : str, optional
            Custom title
        save_fig : bool, default=False
            Save figure
        filename : str, optional
            Custom filename
        file_format : str, default="png"
            File format (png, pdf, svg, etc.)
        dpi : int, default=300
            Resolution
        smoothing : bool, default=True
            Enable or disable data smoothing
        smoothing_method : str, default="savitzky"
            Smoothing method ("moving_average" or "savitzky")
        smoothing_window : int, default=51
            Window size for smoothing in frames
        smoothing_polyorder : int, default=3
            Polynomial order for Savitzky-Golay filter (ignored for moving_average)
        show_unsmoothed_background : bool, default=True
            Show unsmoothed data as transparent background line when smoothing is enabled

        Returns
        -------
        matplotlib.figure.Figure
            Created figure

        Examples
        --------
        >>> # Feature Importance mode
        >>> fig = plotter.plot(
        ...     feature_importance_name="analysis",
        ...     n_top=5
        ... )

        >>> # Manual mode
        >>> fig = plotter.plot(
        ...     feature_selector="my_selector"
        ... )

        Notes
        -----
        Tag coloring is automatically enabled when tags_for_coloring is set.
        No need to manually set color_by_tags parameter.
        """
        # 1. Determine mode and validate parameters
        mode_type, mode_name = self._validate_and_determine_mode(
            feature_importance_name, feature_selector, None
        )

        # Apply mode-based defaults for membership_bar_height
        if membership_bar_height is None:
            membership_bar_height = 0.25 if membership_per_feature else 0.5

        # Create central configuration
        config = TimeSeriesPlotConfig(
            pipeline_data=self.pipeline_data,
            mode_type=mode_type,
            mode_name=mode_name,
            feature_importance_name=feature_importance_name,
            n_top=n_top,
            feature_selector=feature_selector,
            traj_selection=traj_selection,
            use_time=use_time,
            tags_for_coloring=tags_for_coloring,
            allow_multi_tag_plotting=allow_multi_tag_plotting,
            clustering_name=clustering_name,
            membership_per_feature=membership_per_feature,
            membership_traj_selection=membership_traj_selection,
            contact_transformation=contact_transformation,
            max_cols=max_cols,
            long_labels=long_labels,
            subplot_height=subplot_height,
            membership_bar_height=membership_bar_height,
            show_legend=show_legend,
            contact_threshold=contact_threshold,
            title=title,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi,
            smoothing=smoothing,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_polyorder=smoothing_polyorder,
            show_unsmoothed_background=show_unsmoothed_background
        )

        # Prepare plot data and update config
        plot_data = self._prepare_plot_data(config)
        config.feature_data = plot_data['feature_data']
        config.feature_indices = plot_data['feature_indices']
        config.metadata_map = plot_data['metadata_map']
        config.contact_cutoff = plot_data['contact_cutoff']
        config.feature_selector_name = plot_data['feature_selector_name']
        config.is_temporary = plot_data['is_temporary']
        config.tag_map = plot_data['tag_map']

        config.all_features = self._flatten_features(config.feature_data)

        # Prepare layout and dimensions
        self._prepare_layout_and_dimensions(config)

        # Create figure and grid
        self._prepare_grid_and_figure(config)

        # Prepare colors for legends
        self._prepare_colors(config)

        # Plot features
        config.rightmost_ax_first_row = TimeSeriesFeaturePlotHelper.plot_all_features(
            config.fig, config.gs, config
        )

        # Plot membership bars if clustering enabled
        if config.clustering_name:
            TimeSeriesMembershipPlotHelper.plot_membership_bars(
                config.fig, config.gs, config.n_rows, config
            )

        # Cleanup and finalize
        self._cleanup_and_finalize(config)

        return config.fig

    def _prepare_layout_and_dimensions(self, config: TimeSeriesPlotConfig):
        """
        Prepare layout and calculate dimensions.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        None
            Updates config in-place

        Examples
        --------
        >>> self._prepare_layout_and_dimensions(config)
        """
        # Compute layout
        config.layout, config.n_rows, config.n_cols = GridLayoutHelper.compute_uniform_grid_layout(
            len(config.all_features), config.max_cols
        )

        # Get n_frames from actually plotted trajectories
        config.n_frames = self._get_max_frames_from_plotted_trajectories(
            config.tag_map, config.pipeline_data
        )

        config.n_membership_rows = 0
        config.membership_row_height_inches = 0.0

        # Calculate membership dimensions if clustering enabled
        if config.clustering_name:
            membership_indices = self._get_trajectory_indices(
                config.membership_traj_selection, config.feature_selector_name
            )
            n_traj = len(membership_indices)

            spacing_factor = 1.2
            config.membership_row_height_inches = n_traj * config.membership_bar_height * spacing_factor
            config.n_membership_rows = config.n_rows if config.membership_per_feature else 1

    def _prepare_grid_and_figure(self, config: TimeSeriesPlotConfig):
        """
        Create figure and GridSpec.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        None
            Updates config in-place

        Examples
        --------
        >>> self._prepare_grid_and_figure(config)
        """
        config.fig = self._create_figure_dynamic(
            config.n_rows, config.n_cols, config.n_frames, config.n_membership_rows,
            config.membership_row_height_inches, config.subplot_height,
            config.membership_per_feature
        )

        wspace, hspace = self._configure_plot_spacing(config.long_labels)

        n_cols_grid, width_ratios = self._calculate_grid_columns_and_ratios(config)

        total_rows, height_ratios = TimeSeriesGridLayoutHelper.calculate_grid_dimensions(
            config.n_rows, config.clustering_name, config.membership_per_feature,
            config.membership_traj_selection, config.feature_selector_name,
            config.membership_bar_height, config.subplot_height, config.pipeline_data
        )

        config.gs, config.wrapped_title, config.top = TimeSeriesGridLayoutHelper.create_gridspec(
            config.fig, total_rows, n_cols_grid, wspace, hspace,
            height_ratios, width_ratios, config.title
        )

    def _prepare_colors(self, config: TimeSeriesPlotConfig):
        """
        Prepare color mappings.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        None
            Updates config in-place

        Examples
        --------
        >>> self._prepare_colors(config)
        """
        config.use_tag_coloring = len(config.tag_map) > 0 and any(
            len(tags) > 0 for tags in config.tag_map.values()
        )

        if config.use_tag_coloring:
            config.tag_colors = TimeSeriesTagColoringHelper.prepare_tag_legend_colors(
                config.tag_map
            )
            config.traj_colors = {}
        else:
            config.tag_colors = {}
            config.traj_colors = TimeSeriesTagColoringHelper.prepare_trajectory_legend_colors(
                config.pipeline_data, config.tag_map
            )

    def _prepare_plot_data(self, config: TimeSeriesPlotConfig):
        """
        Prepare all data for plotting.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        dict
            Plot data dictionary

        Examples
        --------
        >>> data = self._prepare_plot_data(config)
        """
        # Route to appropriate preparation method based on mode
        if config.mode_type == "feature_importance":
            feature_data, feature_indices, metadata_map, contact_cutoff, feature_selector_name, is_temporary = TimeSeriesDataPreparer.prepare(
                config.pipeline_data, config.mode_name, config.n_top, config.contact_transformation
            )
        else:
            # Manual mode
            feature_data, feature_indices, metadata_map, contact_cutoff, feature_selector_name, is_temporary = TimeSeriesDataPreparer.prepare_from_manual_selection(
                config.pipeline_data, config.mode_name, config.contact_transformation
            )

        traj_indices = self._get_trajectory_indices(config.traj_selection, feature_selector_name)
        tag_map = TimeSeriesTagColoringHelper.build_tag_map(
            config.pipeline_data, traj_indices, config.tags_for_coloring, config.allow_multi_tag_plotting
        )

        return {
            'feature_data': feature_data,
            'feature_indices': feature_indices,
            'metadata_map': metadata_map,
            'contact_cutoff': contact_cutoff,
            'feature_selector_name': feature_selector_name,
            'is_temporary': is_temporary,
            'tag_map': tag_map
        }

    def _create_figure_dynamic(
        self,
        n_rows: int,
        n_cols: int,
        n_frames: int,
        n_membership_rows: int,
        membership_row_height_inches: float,
        subplot_height: float,
        membership_per_feature: bool = False
    ):
        """
        Create figure with dynamic size based on data.

        Calculates width from n_frames and height from actual subplot
        requirements including membership rows.

        Parameters
        ----------
        n_rows : int
            Number of feature rows
        n_cols : int
            Number of columns
        n_frames : int
            Number of frames in trajectories
        n_membership_rows : int
            Number of membership rows (0 if no clustering)
        membership_row_height_inches : float
            Height of each membership row in inches
        subplot_height : float
            Height of each feature row in inches
        membership_per_feature : bool, default=False
            If True, uses higher overhead factor for more bottom space

        Returns
        -------
        Figure
            Matplotlib figure with correct dimensions

        Examples
        --------
        >>> fig = self._create_figure_dynamic(3, 2, 5000, 3, 2.4, 2.5, True)
        """
        # Dynamic width based on frames
        min_col_width = 5
        max_col_width = 12
        col_width = min(max_col_width, min_col_width + n_frames * 0.002)
        fig_width = max(10, n_cols * col_width)

        # Dynamic height based on actual content
        feature_height = n_rows * subplot_height
        membership_height = n_membership_rows * membership_row_height_inches

        # Use multiplicative factor to account for GridSpec overhead
        # (hspace, top margin, titles, labels, padding)
        content_height = feature_height + membership_height
        # More space for per-feature mode to avoid bottom cramping
        overhead_factor = 2.3 if membership_per_feature else 2.0

        fig_height = max(8, content_height * overhead_factor)

        return plt.figure(figsize=(fig_width, fig_height))

    def _cleanup_and_finalize(self, config: TimeSeriesPlotConfig):
        """
        Cleanup and finalize plot.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        None

        Examples
        --------
        >>> self._cleanup_and_finalize(config)
        """
        if config.is_temporary:
            ContactToDistancesConverter.cleanup_temporary_selector(
                config.pipeline_data, config.feature_selector_name
            )

        legend_colors = config.tag_colors if config.use_tag_coloring else config.traj_colors

        self._add_legend_or_title(
            config.fig, config.show_legend, legend_colors,
            config.wrapped_title, config.top, config.rightmost_ax_first_row,
            config.contact_threshold, config.clustering_name, config.membership_per_feature
        )

        if config.save_fig:
            self._save_figure(
                config.fig, config.filename, "feature_importance", config.feature_importance_name,
                config.n_top, config.file_format, config.dpi, "time_series"
            )

    def _add_legend_or_title(
        self, fig, show_legend, legend_colors,
        wrapped_title, top, rightmost_ax_first_row,
        contact_threshold, clustering_name, membership_per_feature
    ):
        """
        Add legend or title only.

        Parameters
        ----------
        fig : Figure
            Figure
        show_legend : bool
            Show legend
        legend_colors : Dict[str, str]
            Colors for legend
        wrapped_title : str
            Title
        top : float
            Top position
        rightmost_ax_first_row
            Rightmost axes
        contact_threshold : float or None
            Threshold
        clustering_name : str or None
            Clustering name
        membership_per_feature : bool
            Membership per feature flag

        Returns
        -------
        None

        Examples
        --------
        >>> self._add_legend_or_title(fig, True, {}, "Title", 0.9, ax, None, "dbscan", False)
        """
        if show_legend:
            # Use combined legend when clustering is active
            if clustering_name:
                self._add_combined_legend(
                    fig, wrapped_title, top, rightmost_ax_first_row,
                    legend_colors, clustering_name
                )
            else:
                self._add_title_and_legend_positioned(
                    fig, wrapped_title, top, rightmost_ax_first_row,
                    legend_colors, None, None, contact_threshold
                )
        else:
            title_offset_from_top = 0.15
            title_y = 1.0 - (title_offset_from_top / fig.get_figheight())
            TitleLegendHelper.add_title(fig, wrapped_title, title_y=title_y)

    def _get_trajectory_indices(
        self, traj_selection, feature_selector_name
    ) -> List[int]:
        """
        Get trajectory indices with data.

        Parameters
        ----------
        traj_selection : Union[int, str, List, "all"]
            Selection criteria
        feature_selector_name : str
            Feature selector name

        Returns
        -------
        List[int]
            Trajectory indices

        Examples
        --------
        >>> indices = self._get_trajectory_indices("all", "selector")
        """
        selected = self.pipeline_data.trajectory_data.get_trajectory_indices(
            traj_selection
        )
        _, frame_mapping = self.pipeline_data.get_selected_data(
            feature_selector_name, return_frame_mapping=True
        )
        traj_in_data = set(traj_idx for traj_idx, _ in frame_mapping.values())
        return [idx for idx in selected if idx in traj_in_data]

    def _add_combined_legend(
        self, fig, wrapped_title, _top, rightmost_ax_first_row,
        traj_colors, clustering_name
    ):
        """
        Add title and combined Trajectories + Clusters legend.

        Parameters
        ----------
        fig : Figure
            Figure
        wrapped_title : str
            Title text
        _top : float
            Top position (unused, kept for signature compatibility)
        rightmost_ax_first_row
            Rightmost axes
        traj_colors : Dict[str, str]
            Trajectory colors
        clustering_name : str
            Clustering name

        Returns
        -------
        None

        Examples
        --------
        >>> self._add_combined_legend(fig, "Title", 0.9, ax, colors, "dbscan")
        """
        # Add title
        title_offset_from_top = 0.15
        title_y = 1.0 - (title_offset_from_top / fig.get_figheight())
        TitleLegendHelper.add_title(fig, wrapped_title, title_y=title_y)

        # Create combined legend handles (inlined from _create_combined_legend_handles)
        handles = []

        # Section 1: Trajectories
        handles.append(mpatches.Patch(color='none', label='Trajectories:'))
        for label, color in traj_colors.items():
            handles.append(mpatches.Patch(color=color, label=f'  {label}'))

        # Spacer
        handles.append(mpatches.Patch(color='none', label=''))

        # Section 2: Clusters
        handles.append(mpatches.Patch(color='none', label='Clusters:'))

        _, cluster_colors = ClusteringDataHelper.load_clustering_data(
            self.pipeline_data, clustering_name
        )
        n_clusters = len(cluster_colors)

        for i in range(n_clusters):
            if i in cluster_colors:
                handles.append(
                    mpatches.Patch(color=cluster_colors[i], label=f'  Cluster {i}')
                )

        # Calculate legend position
        pos = rightmost_ax_first_row.get_position()
        gap_inches = 0.1
        legend_x = pos.x1 + (gap_inches / fig.get_figwidth())

        # Position legend relative to title, not GridSpec top
        gap_to_legend = 0.05
        legend_offset_from_top = title_offset_from_top + gap_to_legend
        legend_y = 1.0 - (legend_offset_from_top / fig.get_figheight())

        # Add legend
        fig.legend(
            handles=handles,
            loc='upper left',
            bbox_to_anchor=(legend_x, legend_y),
            bbox_transform=fig.transFigure,
            frameon=True,
            fontsize=10
        )

    def _add_title_and_legend_positioned(
        self,
        fig: Figure,
        wrapped_title: str,
        _top: float,
        rightmost_ax_first_row,
        data_selector_colors: Dict[str, str],
        legend_title: Optional[str],
        legend_labels: Optional[Dict[str, str]],
        active_threshold: Optional[float]
    ) -> None:
        """
        Add title and legend with calculated positions.

        Overrides base class to position legend relative to title.

        Parameters
        ----------
        fig : Figure
            Figure to add title/legend to
        wrapped_title : str
            Pre-wrapped title text
        _top : float
            Top position (unused, kept for base class signature compatibility)
        rightmost_ax_first_row
            Rightmost axes in first row
        data_selector_colors : Dict[str, str]
            DataSelector color mapping
        legend_title : str, optional
            Custom legend title
        legend_labels : Dict[str, str], optional
            Custom legend labels
        active_threshold : float, optional
            Threshold value for legend

        Returns
        -------
        None

        Examples
        --------
        >>> self._add_title_and_legend_positioned(
        ...     fig, "Title", 0.9, ax, colors, None, None, 4.5
        ... )
        """
        title_offset_from_top = 0.15
        title_y = 1.0 - (title_offset_from_top / fig.get_figheight())
        TitleLegendHelper.add_title(fig, wrapped_title, title_y=title_y)

        pos = rightmost_ax_first_row.get_position()
        gap_inches = 0.1
        legend_x = pos.x1 + (gap_inches / fig.get_figwidth())

        # Position legend relative to title, not GridSpec top
        gap_to_legend = 0.05
        legend_offset_from_top = title_offset_from_top + gap_to_legend
        legend_y = 1.0 - (legend_offset_from_top / fig.get_figheight())

        TitleLegendHelper.add_legend(
            fig, data_selector_colors,
            legend_title, legend_labels,
            contact_threshold=active_threshold,
            legend_x=legend_x, legend_y=legend_y
        )

    def _get_max_frames_from_plotted_trajectories(
        self, tag_map: Dict[int, List[str]], pipeline_data
    ) -> int:
        """
        Get maximum frame count from plotted trajectories.

        Parameters
        ----------
        tag_map : Dict[int, List[str]]
            Trajectory to tags mapping
        pipeline_data : PipelineData
            Pipeline data

        Returns
        -------
        int
            Maximum number of frames

        Examples
        --------
        >>> n_frames = self._get_max_frames_from_plotted_trajectories(
        ...     {0: ["test"], 1: []}, pipeline_data
        ... )
        """
        if not tag_map:
            return pipeline_data.trajectory_data.trajectories[0].n_frames

        return max(
            pipeline_data.trajectory_data.trajectories[traj_idx].n_frames
            for traj_idx in tag_map.keys()
        )

    @staticmethod
    def _configure_plot_spacing(long_labels: bool) -> Tuple[float, float]:
        """
        Configure plot spacing for time series plots.

        Parameters
        ----------
        long_labels : bool
            Whether using long labels for discrete features

        Returns
        -------
        Tuple[float, float]
            (wspace, hspace)

        Examples
        --------
        >>> wspace, hspace = TimeSeriesPlotter._configure_plot_spacing(True)
        >>> print(wspace)  # 0.25

        Notes
        -----
        Time series plots use 0.25 for long_labels, 0.15 otherwise.
        """
        wspace = 0.25 if long_labels else 0.15
        hspace = 0.4
        return wspace, hspace

    def _calculate_grid_columns_and_ratios(
        self, config: TimeSeriesPlotConfig
    ) -> Tuple[int, Optional[List[float]]]:
        """
        Calculate grid columns and width ratios.

        Parameters
        ----------
        config : TimeSeriesPlotConfig
            Configuration object

        Returns
        -------
        Tuple[int, Optional[List[float]]]
            (n_cols_grid, width_ratios)

        Examples
        --------
        >>> n_cols, ratios = self._calculate_grid_columns_and_ratios(config)

        Notes
        -----
        Adds label column when clustering_name and membership_per_feature are both set.
        """
        if config.clustering_name and config.membership_per_feature:
            label_width_ratio = 0.05
            n_cols_grid = config.n_cols + 1
            width_ratios = [label_width_ratio] + [1.0] * config.n_cols
            return n_cols_grid, width_ratios
        return config.n_cols, None

    def _validate_and_determine_mode(
        self,
        feature_importance_name: Optional[str],
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]],
    ) -> Tuple[str, str]:
        """
        Validate parameters and determine operational mode for time series.

        Time series specific validation that does not require data_selectors
        for manual mode (unlike violin/density plots).

        Parameters
        ----------
        feature_importance_name : str, optional
            Feature importance analysis name
        feature_selector : str, optional
            Feature selector name
        data_selectors : List[str], optional
            Ignored for time series (kept for API compatibility)

        Returns
        -------
        mode_type : str
            "feature_importance" or "manual"
        mode_name : str
            Name for this mode (fi_name or feature_selector_name)

        Raises
        ------
        ValueError
            If both modes specified or neither mode specified

        Examples
        --------
        >>> # Feature Importance mode
        >>> mode_type, mode_name = plotter._validate_and_determine_mode(
        ...     "tree_analysis", None, None
        ... )
        >>> print(mode_type, mode_name)
        feature_importance tree_analysis

        >>> # Manual mode (no data_selectors needed)
        >>> mode_type, mode_name = plotter._validate_and_determine_mode(
        ...     None, "my_selector", None
        ... )
        >>> print(mode_type, mode_name)
        manual my_selector
        """
        self._validate_exclusive_modes(feature_importance_name, feature_selector)

        if feature_importance_name:
            return "feature_importance", feature_importance_name

        return "manual", feature_selector

    @staticmethod
    def _validate_exclusive_modes(
        feature_importance_name: Optional[str],
        feature_selector: Optional[str]
    ) -> None:
        """
        Validate exclusive mode parameters.

        Parameters
        ----------
        feature_importance_name : str, optional
            Feature importance analysis name
        feature_selector : str, optional
            Feature selector name

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If both or neither mode specified

        Examples
        --------
        >>> TimeSeriesPlotter._validate_exclusive_modes("fi", None)  # OK
        >>> TimeSeriesPlotter._validate_exclusive_modes(None, "fs")  # OK
        >>> TimeSeriesPlotter._validate_exclusive_modes("fi", "fs")  # Raises
        >>> TimeSeriesPlotter._validate_exclusive_modes(None, None)  # Raises
        """
        if feature_importance_name and feature_selector:
            raise ValueError(
                "Cannot mix modes. "
                "Provide either feature_importance_name OR feature_selector."
            )

        if not feature_importance_name and not feature_selector:
            raise ValueError(
                "Must provide either feature_importance_name or feature_selector."
            )

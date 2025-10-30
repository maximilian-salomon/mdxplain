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
Facade for feature importance visualization.

Provides simplified interface for creating plots based on feature
importance analysis results. Coordinates between feature importance
data and specialized plotters.
"""

from typing import Optional, Union, List, Dict
from matplotlib.figure import Figure

from ..plot_type.violin.violin_plotter import ViolinPlotter
from ..plot_type.density.density_plotter import DensityPlotter
from ..plot_type.time_series.time_series_plotter import TimeSeriesPlotter


class FeatureImportanceFacade:
    """
    Facade for feature importance visualization.

    Provides high-level interface for creating visualizations from
    feature importance analysis results. Simplifies access to
    specialized plotters while managing pipeline data and configuration.

    Examples
    --------
    >>> # Access via plots manager
    >>> facade = plots_manager.feature_importance
    >>> fig = facade.violins("tree_analysis", n_top=10)
    >>> fig = facade.densities("tree_analysis", n_top=10)
    >>> fig = facade.time_series("tree_analysis", n_top=5)
    """

    def __init__(self, manager, pipeline_data) -> None:
        """
        Initialize feature importance facade.

        Parameters
        ----------
        manager : PlotsManager
            Plots manager instance
        pipeline_data : PipelineData
            Pipeline data container

        Returns
        -------
        None
            Initializes FeatureImportanceFacade instance
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = manager.cache_dir

    def violins(
        self,
        feature_importance_name: str,
        n_top: int = 10,
        contact_transformation: bool = True,
        max_cols: int = 4,
        long_labels: bool = False,
        contact_threshold: Optional[float] = 4.5,
        title: Optional[str] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300
    ) -> Figure:
        """
        Create violin plots from feature importance analysis.

        Visualizes the distribution of feature values showing separate
        violins for each DataSelector group with cluster-consistent colors.

        Parameters
        ----------
        feature_importance_name : str
            Name of feature importance analysis
        n_top : int, default=10
            Number of top features per comparison
        contact_transformation : bool, default=True
            If True, automatically convert contact features to distances.
            If False, plot contacts as binary values with Gaussian smoothing.
        max_cols : int, default=4
            Maximum number of columns in grid layout. Each (Feature, DataSelector)
            combination gets its own subplot arranged in a grid.
        long_labels : bool, default=False
            If True, use long descriptive labels for discrete features
            (e.g., "Contact"/"Non-Contact", "Alpha helix"/"Loop").
            If False, use short labels (e.g., "C"/"NC", "H"/"C").
            Automatically adjusts subplot spacing when True to prevent overlap.
        contact_threshold : float, optional
            Distance threshold in Angstrom for drawing contact threshold line
            on distance features. If provided, draws a red dashed horizontal line
            at this distance value. Common value: 4.5 Å (default cutoff for contacts).
        title : str, optional
            Custom plot title. Auto-generated if None.
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename. Auto-generated if None.
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.)
        dpi : int, default=300
            Resolution for saved figure in dots per inch

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing violin plots

        Raises
        ------
        ValueError
            If parameters invalid or required parameters missing for chosen mode

        Examples
        --------
        >>> # Basic violin plot
        >>> fig = facade.violins(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10
        ... )

        >>> # With long descriptive labels for discrete features
        >>> fig = facade.violins(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10,
        ...     long_labels=True
        ... )

        >>> # Save to file
        >>> fig = facade.violins(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10,
        ...     save_fig=True,
        ...     filename="important_features.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        - Each (Feature, DataSelector) combination gets its own subplot
        - contact_transformation=True: Converts boolean contacts to distances
        - contact_transformation=False: Visualizes binary contacts with
          Gaussian smoothing (tall+wide peaks for dominant states, short+narrow
          for rare states)
        - Uses DataSelector-based color mapping for cluster consistency
        - Y-axis shows feature values with units (Distance, Angle, etc.)
        - Each violin is centered in its subplot showing the full distribution
        - Grid layout controlled by max_cols parameter (default: 4 columns)
        """
        plotter = ViolinPlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_importance_name=feature_importance_name,
            n_top=n_top,
            contact_transformation=contact_transformation,
            max_cols=max_cols,
            long_labels=long_labels,
            contact_threshold=contact_threshold,
            title=title,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi
        )

    def densities(
        self,
        feature_importance_name: str,
        n_top: int = 10,
        contact_transformation: bool = True,
        max_cols: int = 4,
        long_labels: bool = False,
        kde_bandwidth: Union[str, float] = "scott",
        base_sigma: float = 0.05,
        max_sigma: float = 0.12,
        alpha: float = 0.3,
        line_width: float = 2.0,
        contact_threshold: Optional[float] = 4.5,
        title: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[Dict[str, str]] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300
    ) -> Figure:
        """
        Create density plots from feature importance analysis.

        Visualizes feature distributions as overlaid density curves in grid
        layout. Each feature gets one grid cell with curves for each
        DataSelector group using cluster-consistent colors.

        Parameters
        ----------
        feature_importance_name : str
            Name of feature importance analysis
        n_top : int, default=10
            Number of top features per comparison
        contact_transformation : bool, default=True
            If True, convert boolean contact features (0/1) to continuous
            distances for smoother visualization. If False, plot contacts
            using Gaussian smoothing with height-dependent widths.
        max_cols : int, default=4
            Maximum number of columns in grid layout. Actual layout may use
            fewer columns to maintain roughly square overall shape.
        long_labels : bool, default=False
            If True, use long descriptive labels for discrete features
            (e.g., "Contact"/"Non-Contact", "Alpha helix"/"Loop").
            If False, use short labels (e.g., "C"/"NC", "H"/"C").
        kde_bandwidth : str or float, default="scott"
            KDE bandwidth for continuous features:
            
            - "scott": Scott's rule (automatic bandwidth selection)
            - "silverman": Silverman's rule
            - float: Manual bandwidth value
        base_sigma : float, default=0.05
            Minimum Gaussian width for binary contact features (narrowest peak)
        max_sigma : float, default=0.12
            Maximum Gaussian width for binary contact features (widest peak)
        alpha : float, default=0.3
            Transparency for filled density curves (0=transparent, 1=opaque)
        line_width : float, default=2.0
            Width of density curve contour lines
        contact_threshold : float, optional
            Distance threshold in Angstrom for drawing contact threshold line.
        title : str, optional
            Custom plot title. Auto-generated if None.
        legend_title : str, optional
            Custom title for DataSelector legend. If None, uses "DataSelectors".
        legend_labels : Dict[str, str], optional
            Custom labels for DataSelectors in legend.
            Maps original names to display names.
            Example: {"cluster_0": "Inactive", "cluster_1": "Active"}
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename. Auto-generated if None.
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.)
        dpi : int, default=300
            Resolution for saved figure in dots per inch

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing density plots in grid layout

        Raises
        ------
        ValueError
            If parameters invalid or required parameters missing for chosen mode

        Examples
        --------
        >>> # Basic density plot
        >>> fig = facade.densities(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10
        ... )

        >>> # Plot binary contacts without distance transformation
        >>> fig = facade.densities(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10,
        ...     contact_transformation=False,
        ...     base_sigma=0.04,
        ...     max_sigma=0.15
        ... )

        >>> # Custom KDE bandwidth for continuous features
        >>> fig = facade.densities(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10,
        ...     kde_bandwidth=0.5
        ... )

        >>> # Save to file with custom layout
        >>> fig = facade.densities(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=15,
        ...     max_cols=5,
        ...     save_fig=True,
        ...     filename="density_plots.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        Binary Contact Features:

        - When contact_transformation=True: Converts to distances (default)
        - When contact_transformation=False: Uses Gaussian smoothing where:
        
          - Dominant states (high probability) → tall AND wide peaks
          - Rare states (low probability) → short AND narrow peaks
          - This prevents visual overlap when multiple DataSelectors plotted

        Continuous Features:

        - Uses standard Kernel Density Estimation (KDE)
        - Automatic bandwidth selection via Scott's or Silverman's rule
        - Manual bandwidth control available via kde_bandwidth parameter

        Grid Layout:

        - Features grouped by type where possible
        - max_cols controls maximum columns (default: 4)
        - Layout algorithm maintains roughly square overall shape
        - Each grid cell shows one feature with overlaid curves

        Color Mapping:
        
        - Uses DataSelector-based colors for cluster consistency
        - Same colors across all plots in pipeline for same DataSelectors
        - Filled curves with transparency (alpha) + solid contour lines
        """
        plotter = DensityPlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_importance_name=feature_importance_name,
            n_top=n_top,
            contact_transformation=contact_transformation,
            max_cols=max_cols,
            long_labels=long_labels,
            kde_bandwidth=kde_bandwidth,
            base_sigma=base_sigma,
            max_sigma=max_sigma,
            alpha=alpha,
            line_width=line_width,
            contact_threshold=contact_threshold,
            title=title,
            legend_title=legend_title,
            legend_labels=legend_labels,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi
        )

    def time_series(
        self,
        feature_importance_name: str,
        n_top: int = 5,
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
        smoothing_method: Optional[str] = None,
        smoothing_window: int = 51,
        smoothing_polyorder: int = 3,
        show_unsmoothed_background: bool = False
    ) -> Figure:
        """
        Create time series plots from feature importance analysis.

        Visualizes temporal evolution of important features as line plots
        with one subplot per feature. Each trajectory is shown as a separate
        line, optionally colored by trajectory number or tags. Can include
        cluster membership visualization as colored bars below plots.

        Parameters
        ----------
        feature_importance_name : str
            Name of feature importance analysis
        n_top : int, default=5
            Number of top features per sub-comparison. Union taken across
            all sub-comparisons to determine features to plot.
        traj_selection : int, str, list, or "all", default="all"
            Trajectories to plot. Can be indices, names, tags, or "all".
        use_time : bool, default=True
            If True, use Time (ns) for x-axis. If False, use frame numbers.
        tags_for_coloring : list of str, optional
            Tags to use for trajectory coloring. If set, automatically enables
            tag-based coloring. Trajectories grouped by shared tags from this list.
        allow_multi_tag_plotting : bool, default=False
            How to handle trajectories with multiple matching tags:

            - False: Exclude trajectories with multiple tags
            - True: Plot such trajectories multiple times (once per tag)
        clustering_name : str, optional
            Name of clustering analysis for membership visualization.
            If None, no membership bars shown.
        membership_per_feature : bool, default=False
            If True, show membership bar below each feature subplot.
            If False, show single membership bar at bottom of figure.
        membership_traj_selection : int, str, list, or "all", default="all"
            Trajectories to include in membership visualization.
            Can differ from main traj_selection.
        contact_transformation : bool, default=True
            If True, automatically convert contact features to distances.
        max_cols : int, default=2
            Maximum number of columns in grid layout. Each feature gets
            one grid cell arranged in rows and columns.
        long_labels : bool, default=False
            If True, use long descriptive labels for discrete features
            (e.g., "Contact"/"Non-Contact"). If False, use short labels
            (e.g., "C"/"NC"). Applies to binary/discrete features only.
        subplot_height : float, default=2.5
            Height per feature subplot in inches
        membership_bar_height : float, optional
            Height per trajectory in membership bar in inches.
            Default: 0.25 (membership_per_feature=True) or 0.5 (False)
        show_legend : bool, default=True
            Show legend for trajectory/tag colors
        contact_threshold : Optional[float], default=4.5
            Distance threshold in Angstrom for drawing contact threshold line
            on distance features. If provided, draws a red dashed horizontal line.
        title : str, optional
            Custom plot title. Auto-generated if None.
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename. Auto-generated if None.
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.)
        dpi : int, default=300
            Resolution for saved figure in dots per inch
        smoothing_method : str, optional
            Smoothing method ("moving_average", "savitzky", or None for no smoothing)
        smoothing_window : int, default=51
            Window size for smoothing in frames
        smoothing_polyorder : int, default=3
            Polynomial order for Savitzky-Golay filter (ignored for moving_average)
        show_unsmoothed_background : bool, default=False
            Show unsmoothed data as transparent background line

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing time series plots

        Raises
        ------
        ValueError
            If color_by_tags=True but tags_for_coloring not specified,
            or if no trajectories remain after filtering

        Examples
        --------
        >>> # Basic time series plot with top 5 features
        >>> fig = facade.time_series(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=5
        ... )

        >>> # Color by trajectory tags
        >>> fig = facade.time_series(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=5,
        ...     color_by_tags=True,
        ...     tags_for_coloring=["system_A", "system_B"]
        ... )

        >>> # Add cluster membership visualization
        >>> fig = facade.time_series(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=5,
        ...     clustering_name="dbscan_clustering",
        ...     membership_per_feature=True
        ... )

        >>> # Plot specific trajectories with frame numbers
        >>> fig = facade.time_series(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=5,
        ...     traj_selection=[0, 1, 2],
        ...     use_time=False
        ... )

        >>> # Custom layout with long labels
        >>> fig = facade.time_series(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10,
        ...     max_cols=3,
        ...     long_labels=True,
        ...     subplot_height=3.0
        ... )

        >>> # Save high-resolution PDF
        >>> fig = facade.time_series(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10,
        ...     save_fig=True,
        ...     filename="feature_timeseries.pdf",
        ...     file_format="pdf",
        ...     dpi=600
        ... )

        Notes
        -----
        - Each feature gets its own subplot with all trajectories overlaid
        - X-axis shows either Time (ns) or frame numbers
        - Contact features automatically converted to distances (default)
        - Cluster membership shown as colored horizontal bars
        - Memory-efficient rendering using block optimization
        - Supports trajectory filtering by index, name, or tags
        """
        plotter = TimeSeriesPlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_importance_name=feature_importance_name,
            n_top=n_top,
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
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_polyorder=smoothing_polyorder,
            show_unsmoothed_background=show_unsmoothed_background
        )

    def decision_trees(
        self,
        feature_importance_name: str,
        max_depth_display: Optional[int] = None,
        max_cols: int = 2,
        subplot_width: float = 10.0,
        subplot_height: float = 8.0,
        title: Optional[str] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300,
        render: bool = True,
        separate_trees: Union[bool, str] = "auto",
        width_scale_factor: float = 1.0,
        height_scale_factor: float = 1.0,
        short_labels: bool = False,
        short_naming: bool = False,
        short_layout: bool = False,
        short_edge_labels: bool = False,
        wrap_length: int = 40
    ) -> Union[Figure, List[str], None]:
        """
        Create decision tree visualizations from feature importance analysis.

        Plots the trained decision tree models from feature importance analysis
        in a grid layout, with one tree per sub-comparison. Only works with
        decision_tree analyzer type.

        Parameters
        ----------
        feature_importance_name : str
            Name of feature importance analysis (must use decision_tree analyzer)
        max_depth_display : int, optional
            Maximum tree depth to display for clarity. None shows full tree.
            Useful for limiting visualization of very deep trees.
        max_cols : int, default=2
            Maximum number of columns in grid layout
        subplot_width : float, default=10.0
            Width of each tree subplot in inches
        subplot_height : float, default=8.0
            Height of each tree subplot in inches
        title : str, optional
            Custom plot title. Auto-generated if None.
        save_fig : Union[bool, str], default="auto"
            Whether to save figure/trees to file(s):

            - "auto": True if render=False (prevents no output), else False
            - True: Always save
            - False: Never save (requires render=True)
        filename : str, optional
            Custom filename for grid mode. Auto-generated if None.
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.)
        dpi : int, default=300
            Resolution for saved figure(s) in dots per inch
        render : Union[bool, str], default="auto"
            Whether to display in Jupyter:

            - "auto": False if grid too large (>50"), True for separate trees
            - True: Always display
            - False: Never display (requires save_fig=True)
        separate_trees : Union[bool, str], default="auto"
            Tree layout mode:
            
            - "auto": True if depth > 5 OR comparisons > 4
            - True: Each tree as separate plot (prevents RAM issues)
            - False: Grid layout (all trees in one figure)
        width_scale_factor : float, default=1.0
            Multiplicative factor for figure width (use >1.0 for wider boxes)
        height_scale_factor : float, default=1.0
            Multiplicative factor for figure height (use >1.0 for taller boxes)
        short_labels : bool, default=False
            Use short discrete labels (NC vs Non-Contact) for feature values
        short_naming : bool, default=False
            Truncate class/selector names to 16 chars with [...] pattern
        short_layout : bool, default=False
            Minimal tree layout (no path display) + enables all short options
        short_edge_labels : bool, default=False
            Show only values/conditions on edges (e.g., 'Contact' or '≤ 3.50 Å')
            instead of full format 'contact: Leu13-ARG31 = Contact'
        wrap_length : int, default=40
            Maximum line length for text wrapping in node labels, class lines,
            feature lines, and edge labels. Text longer than this will wrap
            at spaces (colons, equals signs, etc.).

        Returns
        -------
        matplotlib.figure.Figure, List[str], or None
        
        - Figure: Grid mode with render=True
        - List[str]: Separate trees with save_fig=True (filenames)
        - None: render=False or separate trees without saving

        Raises
        ------
        ValueError
            If feature_importance_name not found, analyzer type is not
            "decision_tree", models not available in metadata, or
            both render and save_fig are False (no output method)

        Examples
        --------
        >>> # Basic decision tree visualization
        >>> fig = facade.decision_trees(
        ...     feature_importance_name="tree_analysis"
        ... )

        >>> # Limit tree depth for clarity
        >>> fig = facade.decision_trees(
        ...     feature_importance_name="tree_analysis",
        ...     max_depth_display=3
        ... )

        >>> # Custom layout with larger subplots
        >>> fig = facade.decision_trees(
        ...     feature_importance_name="tree_analysis",
        ...     max_cols=3,
        ...     subplot_width=12.0,
        ...     subplot_height=10.0
        ... )

        >>> # Save as PDF
        >>> fig = facade.decision_trees(
        ...     feature_importance_name="tree_analysis",
        ...     save_fig=True,
        ...     filename="decision_trees.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        - Red-highlighted node shows the split with maximum discriminative score
        - Edge labels are feature-type-specific (e.g., "Formed"/"Broken" for contacts)
        - Node sizes automatically adjusted to prevent overlap
        - Only available for decision_tree analyzer type
        """
        from ..plot_type.decision_trees.decision_tree_plotter import DecisionTreePlotter

        plotter = DecisionTreePlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_importance_name=feature_importance_name,
            max_depth_display=max_depth_display,
            max_cols=max_cols,
            subplot_width=subplot_width,
            subplot_height=subplot_height,
            title=title,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi,
            render=render,
            separate_trees=separate_trees,
            width_scale_factor=width_scale_factor,
            height_scale_factor=height_scale_factor,
            short_labels=short_labels,
            short_naming=short_naming,
            short_layout=short_layout,
            short_edge_labels=short_edge_labels,
            wrap_length=wrap_length
        )

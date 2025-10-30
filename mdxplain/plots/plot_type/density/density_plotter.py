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
Density plotter for feature importance visualization.

Creates density plots showing probability distributions as overlaid curves
for the most important features identified in feature importance analysis.
"""

from typing import Optional, Dict, List, Tuple
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

from ...helper.validation_helper import ValidationHelper
from ...helper.title_legend_helper import TitleLegendHelper
from .helper.density_data_preparer import DensityDataPreparer
from .helper.density_calculation_helper import DensityCalculationHelper
from ..feature_importance_base.feature_importance_base_plotter import FeatureImportanceBasePlotter


class DensityPlotter(FeatureImportanceBasePlotter):
    """
    Create density plots for feature importance visualization.

    Visualizes feature value distributions as overlaid probability density
    curves for top-ranked features from feature importance analysis. Each
    feature gets one subplot with multiple overlaid density curves (one per
    DataSelector), allowing visual comparison of distributions.

    Similar to violin plots but with overlaid curves instead of violin
    shapes, providing clearer visualization when many DataSelectors are
    present.

    Examples
    --------
    >>> # Basic density plot
    >>> plotter = DensityPlotter(pipeline_data, cache_dir="./cache")
    >>> fig = plotter.plot(
    ...     feature_importance_name="tree_analysis",
    ...     n_top=10
    ... )

    >>> # Custom styling with manual selection
    >>> fig = plotter.plot(
    ...     feature_selector="my_features",
    ...     data_selectors=["cluster_0", "cluster_1", "cluster_2"],
    ...     max_cols=3,
    ...     alpha=0.4
    ... )
    """

    def plot(
        self,
        feature_importance_name: Optional[str] = None,
        n_top: int = 10,
        feature_selector: Optional[str] = None,
        data_selectors: Optional[List[str]] = None,
        contact_transformation: bool = True,
        max_cols: int = 4,
        long_labels: bool = False,
        kde_bandwidth: str = "scott",
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
        dpi: int = 300,
    ) -> Figure:
        """
        Create density plots from feature importance or manual selection.

        Supports two modes:
        1. Feature Importance mode: Automatic selection from feature importance
        2. Manual mode: User-defined feature and DataSelector selection

        Creates intelligent grid layout with overlaid density curves per
        feature, automatic X-axis labels, cluster-consistent colors, and
        special Gaussian smoothing for discrete features (contacts, DSSP).

        Parameters
        ----------
        feature_importance_name : str, optional
            Name of feature importance analysis (Feature Importance mode)
        n_top : int, default=10
            Number of top features per comparison (Feature Importance mode)
        feature_selector : str, optional
            Name of feature selector (Manual mode)
        data_selectors : List[str], optional
            DataSelector names to plot (Manual mode)
        contact_transformation : bool, default=True
            If True, automatically convert contact features to distances.
            If False, plot contacts as discrete values with Gaussian smoothing.
        max_cols : int, default=4
            Maximum number of grid columns per row
        long_labels : bool, default=False
            If True, use long descriptive labels for discrete features
            (e.g., "Contact"/"Non-Contact", "Alpha helix"/"Loop").
            If False, use short labels (e.g., "C"/"NC", "H"/"C").
        kde_bandwidth : str, default="scott"
            KDE bandwidth for continuous features:
            
            - "scott": Scott's rule (automatic)
            - "silverman": Silverman's rule
            - float: Manual bandwidth value
        base_sigma : float, default=0.05
            Minimum Gaussian width for discrete features
        max_sigma : float, default=0.12
            Maximum Gaussian width for discrete features
        alpha : float, default=0.3
            Transparency of filled density curves (0=transparent, 1=opaque)
        line_width : float, default=2.0
            Width of density curve contour lines
        contact_threshold : float, optional
            Distance threshold in Angstrom for drawing contact threshold line
            on distance features. If provided, draws a red dashed vertical line
            at this distance value for features that were transformed from contacts.
            Default is 4.5 Å. None to disable.
        title : str, optional
            Custom plot title
        legend_title : str, optional
            Custom title for DataSelector legend
        legend_labels : Dict[str, str], optional
            Custom labels for DataSelectors in legend.
            Maps original names to display names.
            Example: {"cluster_0": "Inactive", "cluster_1": "Active"}
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename (auto-generated if None)
        file_format : str, default="png"
            File format for saving
        dpi : int, default=300
            Resolution for saved figure

        Returns
        -------
        matplotlib.figure.Figure
            Created figure object with grid layout

        Raises
        ------
        ValueError
            If parameters invalid or required parameters missing for chosen mode

        Examples
        --------
        >>> # Feature Importance mode
        >>> fig = plotter.plot(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10
        ... )

        >>> # Manual mode with discrete features (contacts)
        >>> fig = plotter.plot(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     contact_transformation=False,
        ...     base_sigma=0.04,
        ...     max_sigma=0.15
        ... )

        >>> # Custom styling
        >>> fig = plotter.plot(
        ...     feature_importance_name="tree",
        ...     n_top=5,
        ...     max_cols=3,
        ...     alpha=0.4,
        ...     line_width=2.5,
        ...     title="Feature Distributions",
        ...     legend_labels={"cluster_0": "State A", "cluster_1": "State B"}
        ... )

        Notes
        -----
        - Features displayed in grid layout with max_cols per row
        - Discrete features (contacts, DSSP) use height-dependent Gaussian bells
        - Continuous features use standard KDE
        - Grid layout optimized for quadratic figure shape
        - X-axis labels automatically set per feature type
        - Colors consistent with clustering via ColorMappingHelper
        """
        # 1. Validate and determine mode
        mode_type, mode_name = self._validate_and_determine_mode(
            feature_importance_name, feature_selector, data_selectors
        )

        # 2. Prepare density data
        density_data, metadata_map, data_selector_colors, contact_cutoff = self._prepare_density_data(
            mode_type, mode_name, n_top, feature_selector, data_selectors,
            contact_transformation
        )

        # 3. Flatten to feature list
        all_features = self._flatten_features(density_data)

        # 4. Setup layout and figure
        layout, n_rows, n_cols, fig = self._setup_layout_and_figure(
            all_features, max_cols
        )

        # 5. Configure spacing (density needs custom hspace)
        wspace = 0.4
        hspace = 0.7 if long_labels else 0.45

        # 6. Create GridSpec with title
        gs, wrapped_title, top = self._create_gridspec_with_title(
            fig, n_rows, n_cols, wspace, hspace,
            title, "Feature Importance Density Plot"
        )

        # 7. Plot all features
        rightmost_ax_first_row, active_threshold = self._plot_all_features(
            fig, gs, all_features, layout, density_data, metadata_map,
            data_selector_colors, long_labels, kde_bandwidth, base_sigma, max_sigma,
            alpha, line_width, contact_threshold, contact_cutoff
        )

        # 8. Add title and legend
        self._add_title_and_legend_positioned(
            fig, wrapped_title, top, rightmost_ax_first_row,
            data_selector_colors, legend_title, legend_labels, active_threshold
        )

        # 9. Save if requested
        if save_fig:
            self._save_figure(
                fig, filename, mode_type, mode_name,
                n_top, file_format, dpi, prefix="density"
            )

        return fig

    def _prepare_density_data(
        self,
        mode_type: str,
        mode_name: str,
        n_top: int,
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]],
        contact_transformation: bool,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]],
              Dict[str, Dict[str, Dict[str, any]]], Dict[str, str], Optional[float]]:
        """
        Prepare density plot data based on mode.

        Parameters
        ----------
        mode_type : str
            "feature_importance" or "manual"
        mode_name : str
            Name identifier for this mode
        n_top : int
            Number of top features
        feature_selector : str, optional
            Feature selector name
        data_selectors : List[str], optional
            DataSelector names
        contact_transformation : bool
            If True, convert contacts to distances (default ViolinDataPreparer behavior).
            If False, use raw contact data (0/1) with Gaussian smoothing.

        Returns
        -------
        density_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure
        metadata_map : Dict[str, Dict[str, Dict[str, any]]]
            Feature metadata (type_metadata structure per feature)
        data_selector_colors : Dict[str, str]
            DataSelector color mapping
        contact_cutoff : Optional[float]
            Contact cutoff value if converted from contacts, None otherwise
        """
        # Get data from DensityDataPreparer
        if mode_type == "feature_importance":
            ValidationHelper.validate_positive_integer(n_top, "n_top")
            density_data, metadata_map, data_selector_colors, contact_cutoff = (
                DensityDataPreparer.prepare_from_feature_importance(
                    self.pipeline_data, mode_name, n_top, contact_transformation
                )
            )
        else:
            density_data, metadata_map, data_selector_colors, contact_cutoff = (
                DensityDataPreparer.prepare_from_manual_selection(
                    self.pipeline_data, feature_selector, data_selectors,
                    contact_transformation
                )
            )

        return density_data, metadata_map, data_selector_colors, contact_cutoff

    def _plot_all_features(
        self,
        fig: Figure,
        gs: GridSpec,
        all_features: List[Tuple[str, str]],
        layout: List[Tuple[str, int, int, int]],
        density_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        metadata_map: Dict[str, Dict[str, Dict[str, any]]],
        data_selector_colors: Dict[str, str],
        long_labels: bool,
        kde_bandwidth: str,
        base_sigma: float,
        max_sigma: float,
        alpha: float,
        line_width: float,
        contact_threshold: Optional[float],
        contact_cutoff: Optional[float]
    ) -> Tuple:
        """
        Plot all features in grid layout.

        Parameters
        ----------
        fig : Figure
            Figure to plot on
        gs : GridSpec
            Grid specification
        all_features : List[Tuple[str, str]]
            List of (feature_type, feature_name)
        layout : List[Tuple[str, int, int, int]]
            Layout specification
        density_data : Dict
            Feature data
        metadata_map : Dict
            Feature metadata
        data_selector_colors : Dict
            Color mapping
        long_labels : bool
            Use long descriptive labels for discrete features
        kde_bandwidth : str
            KDE bandwidth
        base_sigma : float
            Min Gaussian width
        max_sigma : float
            Max Gaussian width
        alpha : float
            Fill transparency
        line_width : float
            Contour line width
        contact_threshold : float, optional
            Distance threshold for contact threshold line
        contact_cutoff : float, optional
            Contact cutoff from conversion

        Returns
        -------
        Tuple[Any, Optional[float]]
            (rightmost_ax_first_row, active_threshold)
            active_threshold is the threshold value that was actually drawn,
            or None if no threshold was drawn
        """
        rightmost_ax_first_row = None
        active_threshold = None

        for i, (feat_type, feat_name) in enumerate(all_features):
            if i >= len(layout):
                break

            _, row, col, colspan = layout[i]
            ax = fig.add_subplot(gs[row, col : col + colspan])

            if row == 0:
                rightmost_ax_first_row = ax

            feat_metadata = metadata_map.get(feat_type, {}).get(feat_name, {})

            # Only draw threshold for distances
            resolved_threshold = None
            if feat_type == "distances":
                resolved_threshold = contact_cutoff if contact_cutoff is not None else contact_threshold

            # Track first non-None threshold for legend
            if resolved_threshold is not None and active_threshold is None:
                active_threshold = resolved_threshold

            self._plot_feature_densities(
                ax,
                density_data[feat_type][feat_name],
                feat_metadata, data_selector_colors,
                kde_bandwidth, base_sigma, max_sigma,
                alpha, line_width, resolved_threshold
            )

            self._style_subplot(ax, feat_name, feat_metadata, long_labels)

        return rightmost_ax_first_row, active_threshold

    def _plot_feature_densities(
        self,
        ax,
        selector_data: Dict[str, np.ndarray],
        feat_metadata: Dict[str, any],
        data_selector_colors: Dict[str, str],
        kde_bandwidth: str,
        base_sigma: float,
        max_sigma: float,
        alpha: float,
        line_width: float,
        resolved_threshold: Optional[float]
    ):
        """
        Plot overlaid density curves for one feature.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Subplot axes
        selector_data : Dict[str, np.ndarray]
            {data_selector_name: values}
        feat_metadata : dict
            Feature metadata with type_metadata structure
        data_selector_colors : Dict[str, str]
            Color mapping
        kde_bandwidth : str
            KDE bandwidth
        base_sigma : float
            Min Gaussian width
        max_sigma : float
            Max Gaussian width
        alpha : float
            Fill transparency
        line_width : float
            Contour line width
        resolved_threshold : float, optional
            Resolved threshold value to draw (already calculated)

        Returns
        -------
        None
            Modifies ax in place
        """
        # Extract metadata components from type_metadata structure
        type_metadata = feat_metadata.get("type_metadata", {})

        for selector_name, data in selector_data.items():
            color = data_selector_colors[selector_name]

            # Calculate density (feature-type-aware with metadata)
            x_range, density = DensityCalculationHelper.calculate_density(
                data, type_metadata,
                kde_bandwidth, base_sigma, max_sigma
            )

            # Plot filled curve
            ax.fill_between(
                x_range, density,
                alpha=alpha,
                color=color,
                label=selector_name
            )

            # Plot contour line
            ax.plot(
                x_range, density,
                color=color,
                linewidth=line_width
            )

        # Draw contact threshold line for distances (X-axis = distances)
        if resolved_threshold is not None:
            ax.axvline(
                x=resolved_threshold,
                color='red',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                zorder=5
            )

    def _style_subplot(
        self,
        ax,
        feat_name: str,
        feat_metadata: Dict[str, any],
        long_labels: bool
    ):
        """
        Apply styling to subplot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Subplot to style
        feat_name : str
            Feature name
        feat_metadata : dict
            Feature metadata with type_metadata structure
        long_labels : bool
            Use long descriptive labels for discrete features

        Returns
        -------
        None
            Modifies ax in place
        """
        # Title with line wrapping for long names
        wrapped_title = TitleLegendHelper.wrap_title(feat_name, max_chars_per_line=40)
        ax.set_title(wrapped_title, fontsize=14, pad=10, fontweight='bold')

        # Get visualization metadata from feature
        type_metadata = feat_metadata.get("type_metadata", {})
        viz = type_metadata.get("visualization", {})

        # Set X-axis label
        xlabel = viz.get("axis_label", "Feature Value")
        ax.set_xlabel(xlabel, fontsize=13)

        # Configure discrete features with tick labels
        if viz.get("is_discrete", False):
            # Get tick labels from visualization metadata
            tick_labels_dict = viz.get("tick_labels", {})
            label_key = "long" if long_labels else "short"
            tick_labels = tick_labels_dict.get(label_key, [])

            if tick_labels:
                # Compute positions and limits from tick_labels length
                n_ticks = len(tick_labels)
                positions = list(range(n_ticks))
                xlim = (-0.3, n_ticks - 1 + 0.3)

                ax.set_xticks(positions)
                # Rotate labels vertically for long_labels to prevent overlap
                if long_labels:
                    ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=10)
                else:
                    ax.set_xticklabels(tick_labels)
                ax.set_xlim(xlim)

        # Y-axis label
        ax.set_ylabel("Probability Density", fontsize=13)

        # Grid
        ax.grid(True, alpha=0.3, axis='both')

        # Tick sizes
        ax.tick_params(axis='both', labelsize=12)

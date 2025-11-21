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
Violin plotter for feature importance visualization.

Creates violin plots showing the distribution of feature values for
the most important features identified in feature importance analysis.
"""

from typing import Optional, Dict, List, Tuple
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

from ...helper.title_legend_helper import TitleLegendHelper
from ...helper.validation_helper import ValidationHelper
from .helper.violin_data_preparer import ViolinDataPreparer
from ..feature_importance_base.feature_importance_base_plotter import FeatureImportanceBasePlotter


class ViolinPlotter(FeatureImportanceBasePlotter):
    """
    Create violin plots for feature importance visualization.

    Visualizes the distribution of feature values for top-ranked features
    from feature importance analysis. Each subplot shows side-by-side violins
    for all DataSelectors of one feature, allowing visual comparison of
    distributions across different states/clusters.

    Similar architecture to DensityPlotter but uses violin shapes instead of
    overlaid density curves.

    Also can be used in manual mode to plot user-selected features and DataSelectors.

    Examples
    --------
    >>> # Create violin plot
    >>> plotter = ViolinPlotter(pipeline_data, cache_dir="./cache")
    >>> fig = plotter.plot(
    ...     feature_importance_name="tree_analysis",
    ...     n_top=10,
    ...     max_cols=4
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
        contact_threshold: Optional[float] = 4.5,
        title: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[Dict[str, str]] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300,
        title_fontsize: Optional[int] = None,
        subplot_title_fontsize: Optional[int] = None,
        ylabel_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None,
        legend_fontsize: Optional[int] = None,
        legend_title_fontsize: Optional[int] = None,
    ) -> Figure:
        """
        Create violin plots from feature importance or manual selection.

        Supports two modes:
        1. Feature Importance mode: Automatic selection from feature importance
        2. Manual mode: User-defined feature and DataSelector selection

        Creates grid layout where each subplot shows side-by-side violins for
        all DataSelectors of one feature. Similar to density plots but with
        violin shapes instead of overlaid curves.

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
            Automatically increases wspace when True to prevent label overlap.
        contact_threshold : float, optional
            Distance threshold in Angstrom for drawing contact threshold line
            on distance features. If provided, draws a red dashed horizontal line
            at this distance value for features that were transformed from contacts.
            Default is None (no threshold line). Common value: 4.5 Å.
        title : str, optional
            Custom plot title
        legend_title : str, optional
            Custom title for DataSelector legend. If None, uses "DataSelectors".
        legend_labels : Dict[str, str], optional
            Custom labels for DataSelectors in legend.
            Maps original DataSelector names to display names.
            Example: {"cluster_0": "Inactive", "cluster_1": "Active"}
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename (auto-generated if None)
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.).
            When using 'svg', text elements remain editable in SVG editors.
        dpi : int, default=300
            Resolution for saved figure
        title_fontsize : int, optional
            Font size for main figure title (default: 18)
        subplot_title_fontsize : int, optional
            Font size for feature subplot titles (default: 14)
        ylabel_fontsize : int, optional
            Font size for Y-axis labels (default: 13)
        tick_fontsize : int, optional
            Font size for axis tick labels (default: 12)
        legend_fontsize : int, optional
            Font size for legend entries (default: 14)
        legend_title_fontsize : int, optional
            Font size for legend title (default: 16)

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
        ...     n_top=10,
        ...     max_cols=4
        ... )

        >>> # Manual mode
        >>> fig = plotter.plot(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     max_cols=5
        ... )

        >>> # With long descriptive labels for discrete features
        >>> fig = plotter.plot(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10,
        ...     long_labels=True
        ... )

        Notes
        -----
        - One subplot per feature
        - All DataSelectors shown side-by-side as separate violins in each subplot
        - Grid layout with max_cols columns per row
        - X-axis shows DataSelector names
        - Y-axis shows feature values with appropriate units
        - Colors consistent with clustering via ColorMappingHelper
        - Similar architecture to density plots for code consistency
        """
        # 1. Determine mode and validate parameters
        mode_type, mode_name = self._validate_and_determine_mode(
            feature_importance_name, feature_selector, data_selectors
        )

        # 2. Prepare violin plot data
        violin_data, metadata_map, data_selector_colors, contact_cutoff = self._prepare_violin_data(
            mode_type, mode_name, n_top,
            feature_selector, data_selectors, contact_transformation
        )

        # 3. Flatten to feature list
        all_features = self._flatten_features(violin_data)

        # 4. Setup layout and figure
        layout, n_rows, n_cols, fig = self._setup_layout_and_figure(
            all_features, max_cols
        )

        # 5. Configure spacing
        _, wspace, hspace = self._configure_plot_spacing(
            all_features, metadata_map, long_labels
        )

        # 6. Create GridSpec with title
        gs, wrapped_title, top = self._create_gridspec_with_title(
            fig, n_rows, n_cols, wspace, hspace,
            title, "Feature Importance Violin Plot"
        )

        # 7. Plot all features
        first_ax, rightmost_ax_first_row, active_threshold = self._plot_all_features(
            fig, gs, all_features, layout,
            violin_data, metadata_map, data_selector_colors,
            long_labels, contact_threshold, contact_cutoff,
            subplot_title_fontsize, ylabel_fontsize, tick_fontsize
        )

        # 8. Add title and legend
        self._add_title_and_legend_positioned(
            fig, wrapped_title, top, rightmost_ax_first_row,
            data_selector_colors, legend_title, legend_labels, active_threshold,
            title_fontsize, legend_fontsize, legend_title_fontsize
        )

        # 9. Save if requested
        if save_fig:
            self._save_figure(
                fig, filename, mode_type, mode_name,
                n_top, file_format, dpi, prefix="violin"
            )

        return fig
    def _prepare_violin_data(
        self,
        mode_type: str,
        mode_name: str,
        n_top: int,
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]],
        contact_transformation: bool,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Dict[str, Dict[str, any]]], Dict[str, str], Optional[float]]:
        """
        Prepare violin plot data based on mode.

        Routes to appropriate ViolinDataPreparer method based on mode_type.
        Validates mode-specific parameters.

        Parameters
        ----------
        mode_type : str
            "feature_importance" or "manual"
        mode_name : str
            Name identifier for this mode
        n_top : int
            Number of top features (Feature Importance mode)
        feature_selector : str, optional
            Feature selector name (Manual mode)
        data_selectors : List[str], optional
            DataSelector names (Manual mode)
        contact_transformation : bool
            If True, convert contacts to distances. If False, use raw contacts.

        Returns
        -------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure
        metadata_map : Dict[str, Dict[str, Dict[str, any]]]
            Feature metadata for discrete feature support
        data_selector_colors : Dict[str, str]
            DataSelector color mapping
        contact_cutoff : Optional[float]
            Contact cutoff value if converted from contacts, None otherwise
        """
        if mode_type == "feature_importance":
            ValidationHelper.validate_positive_integer(n_top, "n_top")
            violin_data, metadata_map, data_selector_colors, contact_cutoff = (
                ViolinDataPreparer.prepare_from_feature_importance(
                    self.pipeline_data, mode_name, n_top, contact_transformation
                )
            )
        else:
            violin_data, metadata_map, data_selector_colors, contact_cutoff = (
                ViolinDataPreparer.prepare_from_manual_selection(
                    self.pipeline_data, feature_selector, data_selectors,
                    contact_transformation
                )
            )

        return violin_data, metadata_map, data_selector_colors, contact_cutoff

    def _plot_all_features(
        self,
        fig: Figure,
        gs: GridSpec,
        all_features: List[Tuple[str, str]],
        layout: List,
        violin_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        metadata_map: Dict[str, Dict[str, Dict[str, any]]],
        data_selector_colors: Dict[str, str],
        long_labels: bool,
        contact_threshold: Optional[float],
        contact_cutoff: Optional[float],
        subplot_title_fontsize: Optional[int],
        ylabel_fontsize: Optional[int],
        tick_fontsize: Optional[int]
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
        layout : List
            Layout from GridLayoutHelper
        violin_data : Dict
            Feature data
        metadata_map : Dict
            Feature metadata
        data_selector_colors : Dict
            Color mapping
        long_labels : bool
            Use long descriptive labels
        contact_threshold : float, optional
            Distance threshold line
        contact_cutoff : float, optional
            Contact cutoff from conversion
        subplot_title_fontsize : int, optional
            Font size for subplot titles
        ylabel_fontsize : int, optional
            Font size for Y-axis labels
        tick_fontsize : int, optional
            Font size for axis ticks

        Returns
        -------
        Tuple
            (first_ax, rightmost_ax_first_row, active_threshold)
        """
        first_ax = None
        rightmost_ax_first_row = None
        active_threshold = None

        for i, (feat_type, feat_name) in enumerate(all_features):
            if i >= len(layout):
                raise ValueError(
                    "Layout does not have enough entries for all features."
                    "This indicates an internal error."
                )

            ax, first_ax, rightmost_ax_first_row = self._create_subplot_axes(
                fig, gs, layout[i], i, first_ax, rightmost_ax_first_row
            )

            active_threshold = self._plot_single_feature(
                ax, feat_type, feat_name, violin_data, metadata_map,
                data_selector_colors, long_labels,
                contact_threshold, active_threshold, contact_cutoff,
                subplot_title_fontsize, ylabel_fontsize, tick_fontsize
            )

        return first_ax, rightmost_ax_first_row, active_threshold

    def _create_subplot_axes(
        self,
        fig: Figure,
        gs: GridSpec,
        layout_item: Tuple,
        index: int,
        first_ax,
        rightmost_ax_first_row
    ) -> Tuple:
        """
        Create subplot and track special axes.

        Parameters
        ----------
        fig : Figure
            Figure object
        gs : GridSpec
            Grid specification
        layout_item : Tuple
            Layout item (idx, row, col, colspan)
        index : int
            Feature index
        first_ax
            Current first axes
        rightmost_ax_first_row
            Current rightmost axes

        Returns
        -------
        Tuple
            (ax, first_ax, rightmost_ax_first_row)
        """
        _, row, col, colspan = layout_item
        ax = fig.add_subplot(gs[row, col:col + colspan])

        if index == 0:
            first_ax = ax
        if row == 0:
            rightmost_ax_first_row = ax

        return ax, first_ax, rightmost_ax_first_row

    def _plot_single_feature(
        self,
        ax,
        feat_type: str,
        feat_name: str,
        violin_data: Dict,
        metadata_map: Dict,
        data_selector_colors: Dict[str, str],
        long_labels: bool,
        contact_threshold: Optional[float],
        active_threshold: Optional[float],
        contact_cutoff: Optional[float],
        subplot_title_fontsize: Optional[int],
        ylabel_fontsize: Optional[int],
        tick_fontsize: Optional[int]
    ) -> Optional[float]:
        """
        Plot single feature with violins.

        Parameters
        ----------
        ax
            Axes to plot on
        feat_type : str
            Feature type
        feat_name : str
            Feature name
        violin_data : Dict
            All violin data
        metadata_map : Dict
            All metadata
        data_selector_colors : Dict[str, str]
            Color mapping
        long_labels : bool
            Use long labels
        contact_threshold : float, optional
            User threshold
        active_threshold : float, optional
            Currently active threshold
        contact_cutoff : float, optional
            Contact cutoff from conversion
        subplot_title_fontsize : int, optional
            Font size for subplot title
        ylabel_fontsize : int, optional
            Font size for Y-axis label
        tick_fontsize : int, optional
            Font size for tick labels

        Returns
        -------
        Optional[float]
            Updated active threshold
        """
        feat_metadata = metadata_map.get(feat_type, {}).get(feat_name, {})

        # Only draw threshold for distances
        resolved_threshold = None
        if feat_type == "distances":
            resolved_threshold = contact_cutoff if contact_cutoff is not None else contact_threshold

        if resolved_threshold is not None and active_threshold is None:
            active_threshold = resolved_threshold

        self._plot_feature_violins(
            ax, violin_data[feat_type][feat_name],
            data_selector_colors, resolved_threshold
        )

        self._style_subplot(
            ax, feat_name, feat_metadata, long_labels,
            subplot_title_fontsize, ylabel_fontsize, tick_fontsize
        )

        return active_threshold

    def _plot_feature_violins(
        self,
        ax,
        selector_data: Dict[str, np.ndarray],
        data_selector_colors: Dict[str, str],
        resolved_threshold: Optional[float]
    ) -> None:
        """
        Plot side-by-side violins for one feature.

        Parameters
        ----------
        ax
            Subplot axes
        selector_data : Dict[str, np.ndarray]
            {data_selector_name: values}
        data_selector_colors : Dict[str, str]
            Color mapping
        resolved_threshold : float, optional
            Threshold value to draw

        Returns
        -------
        None
            Modifies ax in place
        """
        selector_names, data_arrays = self._prepare_violin_arrays(selector_data)
        positions = list(range(len(selector_names)))

        # Calculate 25% and 75% quantiles for each violin
        quantiles_list = []
        for data in data_arrays:
            quantiles_list.append([0.25, 0.75])

        parts = ax.violinplot(
            data_arrays, positions=positions,
            showmeans=False, showmedians=True,
            showextrema=False, quantiles=quantiles_list, widths=0.7
        )

        self._apply_violin_colors(
            parts, selector_names, data_selector_colors
        )
        self._draw_threshold_line(ax, resolved_threshold)

    @staticmethod
    def _prepare_violin_arrays(
        selector_data: Dict[str, np.ndarray]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Prepare data arrays for violin plot.

        Adds small noise to constant data to avoid flat violins.

        Parameters
        ----------
        selector_data : Dict[str, np.ndarray]
            {selector_name: values}

        Returns
        -------
        Tuple[List[str], List[np.ndarray]]
            (selector_names, data_arrays)
        """
        selector_names = sorted(selector_data.keys())
        data_arrays = []

        for name in selector_names:
            data = selector_data[name].copy()
            if np.var(data) == 0:
                noise = np.random.normal(0, 0.01, len(data))
                data = data + noise
            data_arrays.append(data)

        return selector_names, data_arrays

    @staticmethod
    def _apply_violin_colors(parts, selector_names: List[str],
                             data_selector_colors: Dict[str, str]) -> None:
        """
        Apply colors to violin bodies.

        Parameters
        ----------
        parts
            Matplotlib violin parts
        selector_names : List[str]
            DataSelector names
        data_selector_colors : Dict[str, str]
            Color mapping

        Returns
        -------
        None
            Modifies parts in place
        """
        for i, pc in enumerate(parts['bodies']):
            color = data_selector_colors[selector_names[i]]
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)

        # Apply same colors to quantile lines (median and 25%/75%)
        colors = [data_selector_colors[name] for name in selector_names]

        # Median (50% quantile)
        if 'cmedians' in parts and parts['cmedians']:
            parts['cmedians'].set_colors(colors)
            parts['cmedians'].set_linewidth(2)

        # 25% and 75% quantiles
        if 'cquantiles' in parts and parts['cquantiles']:
            parts['cquantiles'].set_colors(colors)
            parts['cquantiles'].set_linewidth(1.5)

    @staticmethod
    def _draw_threshold_line(ax, threshold: Optional[float]) -> None:
        """
        Draw threshold line if provided.

        Parameters
        ----------
        ax
            Axes to draw on
        threshold : float, optional
            Threshold value

        Returns
        -------
        None
            Modifies ax in place
        """
        if threshold is not None:
            ax.axhline(
                y=threshold, color='red',
                linestyle='--', linewidth=1.5,
                alpha=0.7, zorder=5
            )

    def _style_subplot(
        self,
        ax,
        feat_name: str,
        feat_metadata: Dict[str, any],
        long_labels: bool,
        subplot_title_fontsize: Optional[int],
        ylabel_fontsize: Optional[int],
        tick_fontsize: Optional[int]
    ) -> None:
        """
        Apply styling to subplot.

        Parameters
        ----------
        ax
            Subplot to style
        feat_name : str
            Feature name
        feat_metadata : dict
            Feature metadata
        long_labels : bool
            Use long labels for discrete features
        subplot_title_fontsize : int, optional
            Font size for subplot title
        ylabel_fontsize : int, optional
            Font size for Y-axis label
        tick_fontsize : int, optional
            Font size for tick labels

        Returns
        -------
        None
            Modifies ax in place
        """
        wrapped_title = TitleLegendHelper.wrap_title(
            feat_name, max_chars_per_line=40
        )
        ax.set_title(wrapped_title, fontsize=subplot_title_fontsize or 14, pad=10, fontweight='bold')

        # Get visualization metadata from feature
        type_metadata = feat_metadata.get("type_metadata", {})
        viz = type_metadata.get("visualization", {})

        # Set Y-axis label
        ylabel = viz.get("axis_label", "Feature Value")
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize or 13)

        # Configure discrete features with tick labels
        if viz.get("is_discrete", False):
            self._configure_discrete_y_axis(ax, viz, long_labels, tick_fontsize)

        ax.set_xlabel("")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=tick_fontsize or 12)

    @staticmethod
    def _configure_discrete_y_axis(
        ax, viz: Dict, long_labels: bool, tick_fontsize: Optional[int] = None
    ) -> None:
        """
        Configure Y-axis for discrete features using visualization metadata.

        Parameters
        ----------
        ax
            Axes to configure
        viz : dict
            Visualization metadata from feature
        long_labels : bool
            Use long labels
        tick_fontsize : int, optional
            Font size for tick labels (default: 12)

        Returns
        -------
        None
            Modifies ax in place
        """
        # Get tick labels from visualization metadata
        tick_labels_dict = viz.get("tick_labels", {})
        label_key = "long" if long_labels else "short"
        tick_labels = tick_labels_dict.get(label_key, [])

        if not tick_labels:
            return

        # Compute positions and limits from tick_labels length
        n_ticks = len(tick_labels)
        positions = list(range(n_ticks))
        ylim = (-0.3, n_ticks - 1 + 0.3)

        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels, fontsize=tick_fontsize or 12)
        ax.set_ylim(ylim)

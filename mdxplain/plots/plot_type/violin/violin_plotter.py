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

from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np

from ...helper.validation_helper import ValidationHelper
from ...helper.grid_layout_helper import GridLayoutHelper
from .helper.violin_data_preparer import ViolinDataPreparer


# Feature-Type Y-Axis Labels
FEATURE_TYPE_LABELS = {
    "distances": "Distance (Å)",
    # "contacts": "Distance (Å)",  # Not used due to conversion
    "torsions": "Angle (°)",
    "sasa": "SASA (Å²)",
    "coordinates": "Position (Å)",
    "dssp": "Secondary Structure Code",
}

# Default contact threshold for distance plots
DEFAULT_CONTACT_THRESHOLD = 4.5  # Å


class ViolinPlotter:
    """
    Create violin plots for feature importance visualization.

    Visualizes the distribution of feature values for top-ranked features
    from feature importance analysis. Shows separate violins for each
    comparison, allowing visual comparison of feature distributions.

    Examples
    --------
    >>> # Create violin plot
    >>> plotter = ViolinPlotter(pipeline_data, cache_dir="./cache")
    >>> fig = plotter.plot(
    ...     feature_importance_name="tree_analysis",
    ...     n_top=10,
    ...     split_features=False
    ... )
    """

    def __init__(self, pipeline_data, cache_dir: str) -> None:
        """
        Initialize violin plotter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        cache_dir : str
            Cache directory path

        Returns
        -------
        None
            Initializes ViolinPlotter instance
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = cache_dir

    def plot(
        self,
        feature_importance_name: Optional[str] = None,
        n_top: int = 10,
        feature_selector: Optional[str] = None,
        data_selectors: Optional[List[str]] = None,
        split_features: bool = False,
        contact_threshold: Optional[float] = None,
        title: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[Dict[str, str]] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300,
    ) -> Figure:
        """
        Create violin plots from feature importance or manual selection.

        Supports two modes:
        1. Feature Importance mode: Automatic selection from feature importance
        2. Manual mode: User-defined feature and DataSelector selection

        Creates intelligent grid layout with feature-type grouping,
        automatic Y-axis labels, cluster-consistent colors, and
        contact threshold lines for distance plots.

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
        split_features : bool, default=False
            If True, create separate subplot per feature.
            If False, group by feature type in grid layout.
        contact_threshold : float, optional
            Contact distance threshold for horizontal line in distance plots.
            If None, auto-detects from contacts feature metadata (default: 4.5 Å).
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
            File format for saving
        dpi : int, default=300
            Resolution for saved figure

        Returns
        -------
        matplotlib.figure.Figure
            Created figure object with intelligent grid layout

        Raises
        ------
        ValueError
            If parameters invalid or required parameters missing for chosen mode

        Examples
        --------
        >>> # Feature Importance mode
        >>> fig = plotter.plot(feature_importance_name="tree_analysis", n_top=10)

        >>> # Manual mode
        >>> fig = plotter.plot(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"]
        ... )

        Notes
        -----
        - Features grouped by type (distances, torsions, sasa, etc.)
        - Contacts automatically converted to distances
        - Grid layout optimized for quadratic figure shape
        - Y-axis labels automatically set per feature type
        - Colors consistent with clustering via ColorMappingHelper
        - Contact threshold line only shown for distance features
        """
        # 1. Determine mode and validate parameters
        mode_type, mode_name = self._validate_and_determine_mode(
            feature_importance_name, feature_selector, data_selectors
        )

        # 2. Prepare violin plot data
        violin_data, data_selector_colors = self._prepare_violin_data(
            mode_type, mode_name, n_top,
            feature_selector, data_selectors
        )

        # 3. Create plot
        fig = self._create_plot(
            violin_data,
            data_selector_colors,
            split_features,
            contact_threshold,
            title,
            legend_title,
            legend_labels,
        )

        # 4. Save if requested
        self._save_if_requested(
            fig,
            save_fig,
            filename,
            mode_type,
            mode_name,
            n_top,
            split_features,
            file_format,
            dpi,
        )

        return fig

    def _validate_and_determine_mode(
        self,
        feature_importance_name: Optional[str],
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]],
    ) -> Tuple[str, str]:
        """
        Validate parameters and determine operational mode.

        Checks for mode conflicts, validates required parameters,
        and determines whether Feature Importance or Manual mode.

        Parameters
        ----------
        feature_importance_name : str, optional
            Feature importance analysis name
        feature_selector : str, optional
            Feature selector name
        data_selectors : List[str], optional
            DataSelector names

        Returns
        -------
        mode_type : str
            "feature_importance" or "manual"
        mode_name : str
            Name for this mode (fi_name or feature_selector_name)

        Raises
        ------
        ValueError
            If both modes specified or required parameters missing
        """
        if feature_importance_name is not None:
            # Feature Importance mode
            if feature_selector is not None or data_selectors is not None:
                raise ValueError(
                    "Cannot mix Feature Importance and Manual modes. "
                    "Provide either feature_importance_name OR "
                    "(feature_selector + data_selectors)."
                )
            return "feature_importance", feature_importance_name

        # Manual mode
        if feature_selector is None or data_selectors is None:
            raise ValueError(
                "Manual mode requires both feature_selector and data_selectors. "
                "Provide these parameters or use feature_importance_name for "
                "Feature Importance mode."
            )
        return "manual", feature_selector

    def _prepare_violin_data(
        self,
        mode_type: str,
        mode_name: str,
        n_top: int,
        feature_selector: Optional[str],
        data_selectors: Optional[List[str]],
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, str]]:
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

        Returns
        -------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure
        data_selector_colors : Dict[str, str]
            DataSelector color mapping
        """
        if mode_type == "feature_importance":
            ValidationHelper.validate_positive_integer(n_top, "n_top")
            return ViolinDataPreparer.prepare_from_feature_importance(
                self.pipeline_data, mode_name, n_top
            )
        else:
            return ViolinDataPreparer.prepare_from_manual_selection(
                self.pipeline_data, feature_selector, data_selectors
            )

    def _create_plot(
        self,
        violin_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        data_selector_colors: Dict[str, str],
        split_features: bool,
        contact_threshold: Optional[float],
        title: Optional[str],
        legend_title: Optional[str],
        legend_labels: Optional[Dict[str, str]],
    ) -> Figure:
        """
        Create violin plot figure.

        Auto-detects contact threshold if not provided, then routes
        to split or combined plot creation based on split_features flag.

        Parameters
        ----------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure
        data_selector_colors : Dict[str, str]
            DataSelector color mapping
        split_features : bool
            Create separate subplot per feature
        contact_threshold : float, optional
            Contact distance threshold
        title : str, optional
            Custom plot title
        legend_title : str, optional
            Custom legend title
        legend_labels : Dict[str, str], optional
            Custom DataSelector labels for legend

        Returns
        -------
        Figure
            Created matplotlib figure
        """
        if contact_threshold is None:
            contact_threshold = self._get_contact_threshold(self.pipeline_data)

        if split_features:
            return self._plot_split(
                violin_data, data_selector_colors, title, contact_threshold,
                legend_title, legend_labels
            )
        else:
            return self._plot_combined(
                violin_data, data_selector_colors, title, contact_threshold,
                legend_title, legend_labels
            )

    def _save_if_requested(
        self,
        fig: Figure,
        save_fig: bool,
        filename: Optional[str],
        mode_type: str,
        mode_name: str,
        n_top: int,
        split_features: bool,
        file_format: str,
        dpi: int,
    ) -> None:
        """
        Save figure if requested.

        Generates filename if not provided, then saves figure
        with specified format and resolution.

        Parameters
        ----------
        fig : Figure
            Figure to save
        save_fig : bool
            Whether to save figure
        filename : str, optional
            Custom filename (auto-generated if None)
        mode_type : str
            "feature_importance" or "manual"
        mode_name : str
            Name identifier for mode
        n_top : int
            Number of top features (for filename generation)
        split_features : bool
            Whether plot is split (for filename generation)
        file_format : str
            File format extension
        dpi : int
            Resolution in dots per inch

        Returns
        -------
        None
            Saves figure to file if save_fig=True
        """
        if not save_fig:
            return

        if filename is None:
            split_str = "_split" if split_features else ""
            if mode_type == "feature_importance":
                filename = (
                    f"violin_{mode_name}_top{n_top}{split_str}.{file_format}"
                )
            else:
                filename = f"violin_{mode_name}{split_str}.{file_format}"

        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def _plot_combined(
        self,
        violin_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        data_selector_colors: Dict[str, str],
        title: Optional[str],
        contact_threshold: Optional[float],
        legend_title: Optional[str],
        legend_labels: Optional[Dict[str, str]],
    ) -> Figure:
        """
        Create combined plot with intelligent multi-row layout.

        Creates subplot grid with max 45 violins per row. Features are
        never split across rows - if a feature doesn't fit in current row,
        it starts on a new row.

        Parameters
        ----------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested dict: {feat_type: {feat_name: {data_selector: values}}}
        data_selector_colors : Dict[str, str]
            DataSelector name to color mapping (cluster-consistent)
        title : str, optional
            Custom overall title
        contact_threshold : float, optional
            Contact distance threshold for horizontal line
        legend_title : str, optional
            Custom legend title
        legend_labels : Dict[str, str], optional
            Custom DataSelector labels for legend

        Returns
        -------
        Figure
            Created figure with multi-row layout

        Notes
        -----
        - Max 45 violins per row, features never split across rows
        - Each feature type in separate subplot with appropriate Y-label
        - Contact threshold line only shown for distance features
        - Larger fonts for better readability
        - Single-line title
        - Colors consistent with cluster plots via data_selector_colors mapping
        """
        # 1. Collect DataSelector names for consistent ordering
        selector_names = []
        for features in violin_data.values():
            for selector_data in features.values():
                for selector_name in selector_data.keys():
                    if selector_name not in selector_names:
                        selector_names.append(selector_name)
        selector_names.sort()
        n_selectors = len(selector_names)

        # 2. Compute multi-row layout (max 45 violins per row)
        row_layouts = self._compute_multi_row_layout(
            violin_data, n_selectors, max_violins_per_row=45
        )
        n_rows = len(row_layouts)

        # 3. Create figure with appropriate size
        fig_width = min(40, max(12, n_rows * 20))  # Max 40 inches
        fig_height = min(20, n_rows * 8)  # Max 20 inches (safety limit)
        fig = plt.figure(figsize=(fig_width, fig_height))

        # 4. Create GridSpec for rows
        gs = GridSpec(n_rows, 1, figure=fig, hspace=0.6)

        # 5. Plot each row
        first_ax = None
        for row_idx, row_features in enumerate(row_layouts):
            ax = fig.add_subplot(gs[row_idx, 0])

            # Store first axes for legend positioning
            if row_idx == 0:
                first_ax = ax

            # Combine all features in this row
            row_data = {}
            for feat_type, feat_name in row_features:
                row_data[feat_name] = violin_data[feat_type][feat_name]

            # Draw violins for this row
            self._plot_feature_type_violins(
                ax, row_data, data_selector_colors, selector_names
            )

            # Y-axis label with units (use first feature type for label)
            first_feat_type = row_features[0][0]
            ylabel = FEATURE_TYPE_LABELS.get(first_feat_type, "Feature Value")
            ax.set_ylabel(ylabel, fontsize=32, fontweight="bold", labelpad=15)

            ax.tick_params(axis="both", which="major", labelsize=28)

            # Add contact threshold line for distances
            if first_feat_type == "distances" and contact_threshold is not None:
                self._add_contact_threshold(ax, contact_threshold)
                ax.legend(loc="upper right", fontsize=28)

        # Overall title (single line)
        title_text = title if title else "Feature Importance Violin Plot"
        fig.suptitle(title_text, fontsize=40, fontweight="bold", y=0.98)

        # Add figure-wide legend for DataSelector colors (larger)
        legend_handles = []
        for selector_name in sorted(selector_names):
            # Use custom label if provided, otherwise use original name
            display_name = (
                legend_labels.get(selector_name, selector_name) 
                if legend_labels 
                else selector_name
            )

            patch = Patch(
                facecolor=data_selector_colors[selector_name],
                alpha=0.7,
                label=display_name,
            )
            legend_handles.append(patch)

        # Use custom legend title if provided, otherwise use default
        legend_title_text = legend_title if legend_title else "DataSelectors"

        # Get y-position of first row's top edge for legend anchoring
        pos = first_ax.get_position()
        legend_y = pos.y1 + 0.01  # Slightly above first row top edge

        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, legend_y),
            fontsize=24,
            title=legend_title_text,
            title_fontsize=28,
            framealpha=0.9,
        )

        return fig

    def _compute_multi_row_layout(
        self,
        violin_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        n_selectors: int,
        max_violins_per_row: int = 45,
    ) -> List[List[Tuple[str, str]]]:
        """
        Compute multi-row layout for features.

        Distributes features across rows with max violins per row.
        Features are NEVER split across rows. If a feature doesn't fit
        in current row, it starts on a new row.

        Parameters
        ----------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure
        n_selectors : int
            Number of DataSelectors (violins per feature)
        max_violins_per_row : int, default=45
            Maximum violins per row

        Returns
        -------
        List[List[Tuple[str, str]]]
            List of rows, each row is list of (feat_type, feat_name) tuples

        Examples
        --------
        >>> row_layouts = self._compute_multi_row_layout(data, 6, 30)
        >>> # Returns: [
        >>> #   [("distances", "feat1"), ("distances", "feat2")],
        >>> #   [("distances", "feat3"), ("torsions", "feat4")]
        >>> # ]

        Notes
        -----
        Each feature has n_selectors violins. Features are added to rows
        until adding next feature would exceed max_violins_per_row.
        """
        rows = []
        current_row = []
        current_violins = 0

        for feat_type, features in violin_data.items():
            for feat_name in features.keys():
                feature_violins = n_selectors

                # Check if feature fits in current row
                if current_violins + feature_violins > max_violins_per_row:
                    # Start new row if current row not empty
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                        current_violins = 0

                # Add feature to current row
                current_row.append((feat_type, feat_name))
                current_violins += feature_violins

        # Add last row if not empty
        if current_row:
            rows.append(current_row)

        return rows

    def _plot_split(
        self,
        violin_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        data_selector_colors: Dict[str, str],
        title: Optional[str],
        contact_threshold: Optional[float],
        legend_title: Optional[str],
        legend_labels: Optional[Dict[str, str]],
    ) -> Figure:
        """
        Create split plot with intelligent grid layout per feature.

        Creates one subplot per feature with grid layout optimized
        for quadratic figure shape. Each subplot shows violins for
        all DataSelectors of that feature.

        Parameters
        ----------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested dict: {feat_type: {feat_name: {data_selector: values}}}
        data_selector_colors : Dict[str, str]
            DataSelector name to color mapping (cluster-consistent)
        title : str, optional
            Custom overall title
        contact_threshold : float, optional
            Contact distance threshold for horizontal line
        legend_title : str, optional
            Custom legend title
        legend_labels : Dict[str, str], optional
            Custom DataSelector labels for legend

        Returns
        -------
        Figure
            Created figure with grid layout

        Notes
        -----
        - One subplot per feature (not per feature type)
        - Grid layout optimized for quadratic overall shape
        - Y-axis labels automatically set per feature type
        - Contact threshold line shown for distance/contact features
        - Colors consistent with cluster plots via data_selector_colors mapping
        """
        # 1. Flatten to all features with type info
        all_features = []
        for feat_type, features in violin_data.items():
            for feat_name in features.keys():
                all_features.append((feat_type, feat_name))

        # 2. Compute grid layout (treat each feature as 1 unit)
        feature_counts = {f"{ft}_{fn}": 1 for ft, fn in all_features}
        layout, n_rows, n_cols = GridLayoutHelper.compute_grid_layout(
            feature_counts
        )

        # 3. Collect DataSelector names for consistent ordering
        selector_names = []
        for features in violin_data.values():
            for selector_data in features.values():
                for selector_name in selector_data.keys():
                    if selector_name not in selector_names:
                        selector_names.append(selector_name)
        selector_names.sort()

        # 4. Create figure (2.5x wider for better visibility)
        fig = plt.figure(figsize=(max(10, n_cols * 3) * 2.5, n_rows * 5))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.4)

        # 5. Place subplots (one per feature)
        for i, (feat_type, feat_name) in enumerate(all_features):
            if i >= len(layout):
                break

            _, row, col, colspan = layout[i]
            ax = fig.add_subplot(gs[row, col : col + colspan])

            # Draw violins for this single feature
            single_feature_data = {feat_name: violin_data[feat_type][feat_name]}
            self._plot_feature_type_violins(
                ax, single_feature_data, data_selector_colors, selector_names
            )

            # Y-axis label with units
            ylabel = FEATURE_TYPE_LABELS.get(feat_type, "Feature Value")
            ax.set_ylabel(ylabel, fontsize=11)

            # Title = feature name
            ax.set_title(feat_name, fontsize=12, pad=8)

            # Add contact threshold line for distances
            if feat_type == "distances" and contact_threshold is not None:
                self._add_contact_threshold(ax, contact_threshold)
                ax.legend(loc="upper right", fontsize=9)

        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, y=0.998)
        else:
            fig.suptitle("Violin Plot (Split)", fontsize=16, y=0.998)

        # Add figure-wide legend for DataSelector colors
        legend_handles = []
        for selector_name in sorted(selector_names):
            # Use custom label if provided, otherwise use original name
            display_name = (
                legend_labels.get(selector_name, selector_name)
                if legend_labels
                else selector_name
            )
            patch = Patch(
                facecolor=data_selector_colors[selector_name],
                alpha=0.7,
                label=display_name,
            )
            legend_handles.append(patch)

        # Use custom legend title if provided, otherwise use default
        legend_title_text = legend_title if legend_title else "DataSelectors"

        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
            title=legend_title_text,
            framealpha=0.9,
        )

        return fig

    def _get_contact_threshold(self, pipeline_data) -> float:
        """
        Extract contact cutoff from feature metadata.

        Attempts to find the cutoff value used for contact feature
        computation. Falls back to DEFAULT_CONTACT_THRESHOLD if not found.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container

        Returns
        -------
        float
            Contact threshold in Ångström

        Examples
        --------
        >>> threshold = plotter._get_contact_threshold(pipeline_data)
        >>> print(threshold)  # e.g., 4.5

        Notes
        -----
        Searches contacts feature metadata for cutoff parameter.
        If contacts feature not computed or cutoff not found,
        returns DEFAULT_CONTACT_THRESHOLD (4.5 Å).
        """
        if "contacts" not in pipeline_data.feature_data:
            return DEFAULT_CONTACT_THRESHOLD

        contacts_data = pipeline_data.feature_data["contacts"]

        # Search through trajectory-specific metadata
        # contacts_data is Dict[int, FeatureData]
        for traj_idx, feature_data in contacts_data.items():
            metadata = feature_data.feature_metadata
            if isinstance(metadata, dict) and "cutoff" in metadata:
                return metadata["cutoff"]

        return DEFAULT_CONTACT_THRESHOLD

    def _add_contact_threshold(self, ax, threshold: float):
        """
        Add horizontal line for contact threshold.

        Draws a horizontal dashed line at the contact cutoff distance
        to provide visual reference for distance values.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to add line to
        threshold : float
            Contact threshold value in Ångström

        Returns
        -------
        None
            Modifies axes in-place

        Notes
        -----
        Line is styled as red dashed line with 70% opacity.
        Only added to distance feature plots.
        """
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            zorder=100,
            label=f"Contact threshold ({threshold:.1f} Å)",
        )

    def _plot_feature_type_violins(
        self,
        ax,
        features_data: Dict[str, Dict[str, np.ndarray]],
        data_selector_colors: Dict[str, str],
        selector_names: List[str],
    ):
        """
        Plot violins for all features of a specific type in one subplot.

        Draws violin plots for all features of a given type, using consistent
        DataSelector colors across all plots. Features are arranged left-to-right
        with gaps between them.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        features_data : Dict[str, Dict[str, np.ndarray]]
            Nested dict: {feature_name: {data_selector_name: values}}
        data_selector_colors : Dict[str, str]
            Mapping of data_selector_name -> color hex code
        selector_names : List[str]
            Ordered list of DataSelector names for consistent ordering

        Returns
        -------
        None
            Modifies axes in-place

        Notes
        -----
        - Uses data_selector_colors for cluster-consistent coloring
        - Adds small gap (1 unit) between feature groups
        - X-ticks placed at center of each feature's violin group
        """
        position = 0
        xticks = []
        xlabels = []
        separator_positions = []  # Positions for vertical separator lines
        feature_names_list = list(features_data.keys())

        for idx, (feat_name, selector_data) in enumerate(features_data.items()):
            # Prepare data in consistent DataSelector order
            data_list = []
            colors_list = []

            for selector_name in selector_names:
                if selector_name in selector_data:
                    data_list.append(selector_data[selector_name])

                    # Get color for this DataSelector (cluster-consistent)
                    colors_list.append(data_selector_colors[selector_name])

            if not data_list:
                continue

            n_selectors = len(data_list)
            positions = list(range(position, position + n_selectors))

            # Plot violins (width 0.8 to prevent overlap at adjacent positions)
            parts = ax.violinplot(
                data_list, positions=positions, showmeans=True, widths=0.8
            )

            # Apply cluster colors
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.7)

            # Hide default min/max lines (we'll draw quartiles instead)
            for key in ["cmins", "cmaxes", "cbars"]:
                if key in parts:
                    parts[key].set_visible(False)

            # Style mean line
            if "cmeans" in parts:
                parts["cmeans"].set_edgecolor("black")
                parts["cmeans"].set_linewidth(1.5)

            # Add quartile lines (Q1=25%, Median=50%, Q3=75%)
            for i, (pos, data) in enumerate(zip(positions, data_list)):
                quartiles = np.percentile(data, [25, 50, 75])

                # Q1 and Q3 as shorter lines
                ax.hlines(
                    quartiles[0],
                    pos - 0.15,
                    pos + 0.15,
                    colors="black",
                    linewidth=1.5,
                    zorder=100,
                )
                ax.hlines(
                    quartiles[2],
                    pos - 0.15,
                    pos + 0.15,
                    colors="black",
                    linewidth=1.5,
                    zorder=100,
                )

                # Median as thicker line
                ax.hlines(
                    quartiles[1],
                    pos - 0.2,
                    pos + 0.2,
                    colors="black",
                    linewidth=2.0,
                    zorder=100,
                )

            # Store separator position (except after last feature)
            if idx < len(feature_names_list) - 1:
                separator_x = position + n_selectors  # Center of gap
                separator_positions.append(separator_x)

            # Update position and labels
            position += n_selectors + 1  # Gap between features
            xticks.append(
                positions[len(positions) // 2] if positions else position
            )
            xlabels.append(feat_name)

        # Format x-axis with larger fonts
        ax.set_xticks(xticks)
        rotation = 90 if any(len(label) > 40 for label in xlabels) else 45
        ax.set_xticklabels(xlabels, rotation=rotation, ha="right", fontsize=24)
        ax.grid(True, alpha=0.3, axis="y")

        # Draw vertical separator lines between features
        for sep_x in separator_positions:
            ax.axvline(
                x=sep_x,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.3,
                zorder=1,
            )

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
Decision tree plotter for feature importance visualization.

Creates grid layouts of decision tree visualizations for each sub-comparison
in feature importance analysis.
"""

from typing import Optional, Union, List
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import os

from .helper.decision_tree_visualizer import DecisionTreeVisualizer
from .helper.decision_tree_visualization_config import DecisionTreeVisualizationConfig
from .helper.separate_tree_mode_helper import SeparateTreeModeHelper
from .helper.plot_configuration_helper import PlotConfigurationHelper
from ....utils.data_utils import DataUtils

# Global config instance
TREE_CONFIG = DecisionTreeVisualizationConfig()


class DecisionTreePlotter:
    """
    Plot decision trees from feature importance analysis.

    Creates grid layout of decision trees, one for each sub-comparison,
    visualizing the trained sklearn DecisionTreeClassifier models.

    Examples
    --------
    >>> plotter = DecisionTreePlotter(pipeline_data, cache_dir="./cache")
    >>> fig = plotter.plot(
    ...     feature_importance_name="tree_analysis",
    ...     max_depth_display=3,
    ...     max_cols=2
    ... )
    """

    def __init__(self, pipeline_data, cache_dir: str):
        """
        Initialize decision tree plotter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        cache_dir : str
            Cache directory path

        Returns
        -------
        None
            Initializes DecisionTreePlotter instance
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = cache_dir

    def plot(
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
        Plot decision trees for feature importance analysis.

        Creates grid of decision tree visualizations, one per sub-comparison.
        Only works with decision_tree analyzer type.

        Parameters
        ----------
        feature_importance_name : str
            Name of feature importance analysis
        max_depth_display : int, optional
            Maximum tree depth to display (None for full tree)
        max_cols : int, default=2
            Maximum columns in grid layout
        subplot_width : float, default=10.0
            Width of each tree subplot in inches
        subplot_height : float, default=8.0
            Height of each tree subplot in inches
        title : str, optional
            Custom title for the figure
        save_fig : bool, default=False
            Whether to save figure/trees to file(s)
        filename : str, optional
            Filename for saved figure (grid mode only)
        file_format : str, default="png"
            File format for saved figure(s)
        dpi : int, default=300
            DPI resolution for saved figure(s)
        render : bool, default=True
            Whether to display in Jupyter
        separate_trees : Union[bool, str], default="auto"
            Tree layout mode:
            - "auto": depth ≤4: Grid, depth 5-6: Separate trees
            - True: Each tree as separate plot
            - False: Grid layout (all trees in one figure)
            Note: depth > 6 raises ValueError (not visualizable)
        width_scale_factor : float, default=1.0
            Multiplicative factor for figure width (use >1.0 for wider boxes)
        height_scale_factor : float, default=1.0
            Multiplicative factor for figure height (use >1.0 for taller boxes)
        short_labels : bool, default=False
            Use short discrete labels (NC vs Non-Contact)
        short_naming : bool, default=False
            Truncate class names to 16 chars with [...] pattern
        short_layout : bool, default=False
            Minimal tree layout (no path display) + enables all short options
        short_edge_labels : bool, default=False
            Show only values/conditions on edges (e.g., 'Contact' or '≤ 3.50 Å')
        wrap_length : int, default=40
            Maximum line length for text wrapping in node labels

        Returns
        -------
        Figure, List[str], or None
            - Figure: Grid mode with render=True
            - List[str]: Separate trees with save_fig=True (filenames)
            - None: render=False or separate trees without saving

        Raises
        ------
        ValueError
            If analyzer type is not "decision_tree", models not found,
            depth > 6 (too large for visualization), or
            both render and save_fig are False (no output method)
        """
        fi_data, feature_metadata = PlotConfigurationHelper.validate_and_get_data(
            self.pipeline_data, feature_importance_name
        )
        effective_depth = PlotConfigurationHelper.calculate_effective_depth(
            fi_data, max_depth_display
        )
        subplot_width, subplot_height = PlotConfigurationHelper.calculate_subplot_sizes(
            fi_data, feature_metadata, max_depth_display, effective_depth,
            subplot_width, subplot_height
        )
        n_comparisons, n_rows, n_cols, hspace_dynamic = PlotConfigurationHelper.calculate_grid_layout(
            fi_data, max_cols
        )
        use_separate_trees = PlotConfigurationHelper.determine_separate_trees_mode(
            separate_trees, effective_depth
        )
        subplot_width, subplot_height, dpi = PlotConfigurationHelper.apply_dimension_limits(
            use_separate_trees, effective_depth, subplot_width, subplot_height,
            width_scale_factor, height_scale_factor, dpi
        )
        PlotConfigurationHelper.validate_output_methods(render, save_fig)

        if use_separate_trees:
            return self._plot_separate_trees_mode(
                fi_data, feature_metadata, feature_importance_name,
                max_depth_display, subplot_width, subplot_height,
                n_comparisons, effective_depth, file_format, dpi,
                render, save_fig, short_labels, short_naming,
                short_layout, short_edge_labels, wrap_length
            )

        return self._plot_grid_mode(
            fi_data, feature_metadata, feature_importance_name,
            max_depth_display, subplot_width, subplot_height,
            n_rows, n_cols, hspace_dynamic, effective_depth,
            title, filename, file_format, dpi, render, save_fig,
            short_labels, short_naming, short_layout, short_edge_labels, wrap_length
        )

    def _plot_separate_trees_mode(self, fi_data, feature_metadata,
                                   feature_importance_name, max_depth_display,
                                   subplot_width, subplot_height, n_comparisons,
                                   effective_depth, file_format, dpi, render,
                                   save_fig, short_labels, short_naming,
                                   short_layout, short_edge_labels, wrap_length):
        """
        Plot trees in separate figures mode.

        Parameters
        ----------
        Various parameters for tree plotting

        Returns
        -------
        List[str] or None
            List of filenames if save_fig, else None
        """
        print(f"ℹ️  Creating {n_comparisons} separate trees "
              f"(depth={effective_depth}, comparisons={n_comparisons})")
        return SeparateTreeModeHelper.plot_separate_trees(
            fi_data, feature_metadata, feature_importance_name,
            max_depth_display, subplot_width, subplot_height,
            file_format, dpi, render, save_fig, self.cache_dir,
            short_labels, short_naming, short_layout, short_edge_labels,
            wrap_length
        )

    def _plot_grid_mode(self, fi_data, feature_metadata, feature_importance_name,
                        max_depth_display, subplot_width, subplot_height,
                        n_rows, n_cols, hspace_dynamic, effective_depth,
                        title, filename, file_format, dpi, render, save_fig,
                        short_labels, short_naming, short_layout, short_edge_labels, wrap_length):
        """
        Plot trees in grid mode.

        Parameters
        ----------
        Various parameters for grid plotting

        Returns
        -------
        Figure or None
            Figure if render=True, else None
        """
        old_backend = self._setup_backend_for_grid(render)
        fig, gs = self._create_grid_figure(n_rows, n_cols, subplot_width,
                                            subplot_height, hspace_dynamic)
        self._plot_trees_in_grid(fi_data, feature_metadata, max_depth_display,
                                  subplot_width, effective_depth, n_rows, n_cols,
                                  fig, gs, short_labels, short_naming,
                                  short_layout, short_edge_labels, wrap_length)
        self._add_main_title(fig, title, feature_importance_name, effective_depth)
        return self._save_and_return_figure(fig, save_fig, filename,
                                             feature_importance_name, file_format,
                                             dpi, render, old_backend)

    def _setup_backend_for_grid(self, render):
        """
        Set up matplotlib backend for grid mode.

        Parameters
        ----------
        render : bool
            Whether to render

        Returns
        -------
        str or None
            Old backend if changed
        """
        if not render:
            old_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            return old_backend
        return None

    def _create_grid_figure(self, n_rows, n_cols, subplot_width,
                            subplot_height, hspace_dynamic):
        """
        Create figure with GridSpec.

        Parameters
        ----------
        n_rows : int
            Number of rows
        n_cols : int
            Number of columns
        subplot_width : float
            Width per subplot
        subplot_height : float
            Height per subplot
        hspace_dynamic : float
            Vertical spacing

        Returns
        -------
        tuple
            (fig, gs)
        """
        fig = plt.figure(figsize=(n_cols * subplot_width, n_rows * subplot_height))
        gs = GridSpec(
            n_rows, n_cols, figure=fig,
            wspace=TREE_CONFIG.grid_wspace, hspace=hspace_dynamic,
            top=TREE_CONFIG.grid_top, bottom=TREE_CONFIG.grid_bottom,
            left=TREE_CONFIG.grid_left, right=TREE_CONFIG.grid_right
        )
        return fig, gs

    def _plot_trees_in_grid(self, fi_data, feature_metadata, max_depth_display,
                            subplot_width, effective_depth, n_rows, n_cols,
                            fig, gs, short_labels, short_naming,
                            short_layout, short_edge_labels, wrap_length):
        """
        Plot each tree in grid.

        Parameters
        ----------
        Various parameters for tree plotting

        Returns
        -------
        None
            Modifies fig in place
        """
        n_comparisons = len(fi_data.metadata)
        for idx, metadata in enumerate(fi_data.metadata):
            ax = self._create_subplot_axes(fig, gs, idx, n_cols)
            self._plot_single_tree_in_grid(
                ax, metadata, feature_metadata, max_depth_display,
                subplot_width, effective_depth, idx, short_labels,
                short_naming, short_layout, short_edge_labels, wrap_length
            )

        self._hide_unused_subplots(fig, gs, n_comparisons, n_rows, n_cols)

    def _create_subplot_axes(self, fig, gs, idx, n_cols):
        """
        Create axes for subplot.

        Parameters
        ----------
        fig : Figure
            Figure object
        gs : GridSpec
            GridSpec object
        idx : int
            Index
        n_cols : int
            Number of columns

        Returns
        -------
        Axes
            Subplot axes
        """
        row = idx // n_cols
        col = idx % n_cols
        return fig.add_subplot(gs[row, col])

    def _plot_single_tree_in_grid(self, ax, metadata, feature_metadata,
                                   max_depth_display, subplot_width,
                                   effective_depth, idx, short_labels,
                                   short_naming, short_layout, short_edge_labels, wrap_length):
        """
        Plot single tree in grid cell.

        Parameters
        ----------
        Various parameters

        Returns
        -------
        None
            Modifies ax in place
        """
        model = metadata.get("model")
        if model is None:
            raise ValueError(
                f"Decision tree model not found in metadata for sub-comparison {idx}. "
                "Re-run feature importance analysis."
            )

        sub_comp_name = metadata.get("sub_comparison_name", f"Comparison {idx}")
        class_names = PlotConfigurationHelper.extract_class_names(metadata)
        target_width_px = subplot_width * 100

        visualizer = DecisionTreeVisualizer(
            model, feature_metadata, class_names, max_depth_display,
            short_labels=short_labels, short_naming=short_naming,
            short_layout=short_layout, short_edge_labels=short_edge_labels,
            wrap_length=wrap_length
        )
        visualizer.visualize(ax, target_width=target_width_px)

        self._add_subplot_border(ax)
        self._add_subplot_title(ax, sub_comp_name, effective_depth)

    def _add_subplot_border(self, ax):
        """
        Add gray border to subplot.

        Parameters
        ----------
        ax : Axes
            Subplot axes

        Returns
        -------
        None
            Modifies ax in place
        """
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(TREE_CONFIG.subplot_border_width)
            spine.set_edgecolor('lightgray')

    def _add_subplot_title(self, ax, sub_comp_name, effective_depth):
        """
        Add title to subplot with depth-scaled font size.

        Parameters
        ----------
        ax : Axes
            Subplot axes
        sub_comp_name : str
            Subtitle text
        effective_depth : int
            Tree depth

        Returns
        -------
        None
            Modifies ax in place
        """
        title_fontsize = self._calculate_title_fontsize(effective_depth)
        ax.set_title(sub_comp_name, fontsize=title_fontsize,
                     fontweight='bold', pad=TREE_CONFIG.subplot_title_pad)

    def _calculate_title_fontsize(self, effective_depth):
        """
        Calculate title fontsize based on depth.

        Parameters
        ----------
        effective_depth : int
            Tree depth

        Returns
        -------
        int
            Font size
        """
        return TREE_CONFIG.subplot_title_fontsize.get(
            effective_depth, TREE_CONFIG.default_subplot_title_fontsize
        )

    def _hide_unused_subplots(self, fig, gs, n_comparisons, n_rows, n_cols):
        """
        Hide unused subplots in grid.

        Parameters
        ----------
        fig : Figure
            Figure object
        gs : GridSpec
            GridSpec object
        n_comparisons : int
            Number of actual plots
        n_rows : int
            Grid rows
        n_cols : int
            Grid columns

        Returns
        -------
        None
            Modifies fig in place
        """
        for idx in range(n_comparisons, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax_unused = fig.add_subplot(gs[row, col])
            ax_unused.axis('off')

    def _add_main_title(self, fig, title, feature_importance_name, effective_depth):
        """
        Add main title to figure.

        Parameters
        ----------
        fig : Figure
            Figure object
        title : str or None
            Custom title
        feature_importance_name : str
            Analysis name
        effective_depth : int
            Tree depth

        Returns
        -------
        None
            Modifies fig in place
        """
        if title is None:
            title = f"Decision Trees: {feature_importance_name}"

        main_title_fontsize = self._calculate_main_title_fontsize(effective_depth)
        fig.suptitle(title, fontsize=main_title_fontsize, fontweight='bold',
                     x=0.5, y=TREE_CONFIG.main_title_y, ha='center')

    def _calculate_main_title_fontsize(self, effective_depth):
        """
        Calculate main title fontsize based on depth.

        Parameters
        ----------
        effective_depth : int
            Tree depth

        Returns
        -------
        int
            Font size
        """
        return TREE_CONFIG.main_title_fontsize.get(
            effective_depth, TREE_CONFIG.default_main_title_fontsize
        )

    def _save_and_return_figure(self, fig, save_fig, filename,
                                 feature_importance_name, file_format,
                                 dpi, render, old_backend):
        """
        Save figure and return based on render setting.

        Parameters
        ----------
        fig : Figure
            Figure to save/return
        save_fig : bool
            Whether to save
        filename : str or None
            Filename
        feature_importance_name : str
            Analysis name
        file_format : str
            File format
        dpi : int
            DPI
        render : bool
            Whether to render
        old_backend : str or None
            Old backend to restore

        Returns
        -------
        Figure or None
            Figure if render=True, else None
        """
        if save_fig:
            if filename is None:
                filename = f"decision_trees_{feature_importance_name}.{file_format}"
            filepath = DataUtils.get_cache_file_path(filename, self.cache_dir)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {filepath}")

        if render:
            return fig

        plt.close(fig)
        if old_backend is not None:
            matplotlib.use(old_backend)
        return None

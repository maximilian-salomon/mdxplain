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
Separate tree mode helper for decision tree plotting.

Provides static methods for plotting decision trees in separate-figure mode.
All methods are stateless and receive required state as parameters.
"""

from typing import Optional, Union, List
import matplotlib
import matplotlib.pyplot as plt
import os
import uuid

from .decision_tree_visualizer import DecisionTreeVisualizer
from .decision_tree_visualization_config import DecisionTreeVisualizationConfig
from .plot_configuration_helper import PlotConfigurationHelper

TREE_CONFIG = DecisionTreeVisualizationConfig()

# IPython optional dependency for Jupyter display
try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class SeparateTreeModeHelper:
    """
    Stateless helper for separate tree plotting mode.

    All methods are static and receive required state as parameters.
    Code is copied 1:1 from DecisionTreePlotter for consistency.

    Examples
    --------
    >>> files = SeparateTreeModeHelper.plot_separate_trees(
    ...     fi_data, feature_metadata, "analysis", max_depth=3,
    ...     width=10, height=8, format="png", dpi=300,
    ...     render=True, save=True, cache_dir="./cache"
    ... )
    """

    @staticmethod
    def plot_separate_trees(
        fi_data,
        feature_metadata,
        feature_importance_name: str,
        max_depth_display: Optional[int],
        subplot_width: float,
        subplot_height: float,
        file_format: str,
        dpi: int,
        render: bool,
        save_fig: bool,
        cache_dir: str,
        short_labels: bool,
        short_naming: bool,
        short_layout: bool,
        short_edge_labels: bool,
        wrap_length: int
    ) -> Union[List[str], None]:
        """
        Plot each tree in a separate file.

        Creates individual figure files for each sub-comparison tree,
        preventing memory issues with large/deep trees.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data
        feature_metadata : list
            Feature metadata
        feature_importance_name : str
            Name for file naming
        max_depth_display : int, optional
            Maximum depth to display
        subplot_width : float
            Width of each tree figure
        subplot_height : float
            Height of each tree figure
        file_format : str
            File format (png, pdf, etc.)
        dpi : int
            Resolution for saved figures
        render : bool
            Whether to render in Jupyter
        save_fig : bool
            Whether to save files
        cache_dir : str
            Cache directory for saving
        short_labels : bool
            Use short discrete labels
        short_naming : bool
            Truncate class names
        short_layout : bool
            Minimal tree layout
        short_edge_labels : bool
            Show only values on edges
        wrap_length : int
            Maximum line length for text wrapping

        Returns
        -------
        List[str] or None
            List of saved filenames if save_fig, else None

        Examples
        --------
        >>> files = SeparateTreeModeHelper.plot_separate_trees(...)
        >>> print(f"Created {len(files)} files")
        Created 6 files
        """
        old_backend = SeparateTreeModeHelper._setup_backend_for_separate(render)
        saved_files, temp_files = SeparateTreeModeHelper._plot_all_separate_trees(
            fi_data, feature_metadata, feature_importance_name,
            max_depth_display, subplot_width, subplot_height,
            file_format, dpi, render, save_fig, cache_dir,
            short_labels, short_naming, short_layout, short_edge_labels,
            wrap_length
        )
        SeparateTreeModeHelper._cleanup_separate_trees(old_backend, temp_files)

        if save_fig:
            print(f"{len(saved_files)} tree figures saved successfully")
            return saved_files
        print(f"{len(temp_files)} trees displayed (not saved)")
        return None

    @staticmethod
    def _setup_backend_for_separate(render):
        """
        Set up backend for separate trees.

        Parameters
        ----------
        render : bool
            Whether to render

        Returns
        -------
        str or None
            Old backend if changed

        Examples
        --------
        >>> old = SeparateTreeModeHelper._setup_backend_for_separate(False)
        >>> print(old)
        'module://matplotlib_inline.backend_inline'
        """
        if not render:
            old_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            return old_backend
        return None

    @staticmethod
    def _plot_all_separate_trees(fi_data, feature_metadata,
                                   feature_importance_name, max_depth_display,
                                   subplot_width, subplot_height, file_format,
                                   dpi, render, save_fig, cache_dir, short_labels,
                                   short_naming, short_layout, short_edge_labels,
                                   wrap_length):
        """
        Plot all trees separately.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data
        feature_metadata : list
            Feature metadata
        feature_importance_name : str
            Analysis name
        max_depth_display : int or None
            Max depth
        subplot_width : float
            Width
        subplot_height : float
            Height
        file_format : str
            Format
        dpi : int
            DPI
        render : bool
            Render flag
        save_fig : bool
            Save flag
        cache_dir : str
            Cache directory
        short_labels : bool
            Use short labels
        short_naming : bool
            Use short naming
        short_layout : bool
            Use short layout
        short_edge_labels : bool
            Use short edge labels
        wrap_length : int
            Maximum line length for text wrapping

        Returns
        -------
        tuple
            (saved_files, temp_files)

        Examples
        --------
        >>> saved, temp = SeparateTreeModeHelper._plot_all_separate_trees(...)
        >>> print(f"Saved: {len(saved)}, Temp: {len(temp)}")
        Saved: 6, Temp: 0
        """
        saved_files = []
        temp_files = []

        for idx, metadata in enumerate(fi_data.metadata):
            model = metadata.get("model")
            if model is None:
                continue

            fig, _ = SeparateTreeModeHelper._create_separate_tree_figure(
                metadata, feature_metadata, max_depth_display,
                subplot_width, subplot_height, short_labels,
                short_naming, short_layout, short_edge_labels, wrap_length, idx
            )

            SeparateTreeModeHelper._save_or_display_separate_tree(
                fig, metadata, feature_importance_name, idx,
                file_format, dpi, render, save_fig, cache_dir,
                saved_files, temp_files
            )

            plt.close(fig)

        return saved_files, temp_files

    @staticmethod
    def _create_separate_tree_figure(metadata, feature_metadata,
                                       max_depth_display, subplot_width,
                                       subplot_height, short_labels,
                                       short_naming, short_layout,
                                       short_edge_labels, wrap_length, idx):
        """
        Create figure for single separate tree.

        Parameters
        ----------
        metadata : dict
            Tree metadata
        feature_metadata : list
            Feature metadata
        max_depth_display : int or None
            Max depth
        subplot_width : float
            Width
        subplot_height : float
            Height
        short_labels : bool
            Use short labels
        short_naming : bool
            Use short naming
        short_layout : bool
            Use short layout
        short_edge_labels : bool
            Use short edge labels
        wrap_length : int
            Maximum line length for text wrapping
        idx : int
            Tree index

        Returns
        -------
        tuple
            (fig, ax)

        Examples
        --------
        >>> fig, ax = SeparateTreeModeHelper._create_separate_tree_figure(...)
        >>> print(f"Figure size: {fig.get_size_inches()}")
        Figure size: [10.  8.]
        """
        model = metadata.get("model")
        class_names = PlotConfigurationHelper.extract_class_names(metadata)

        fig = plt.figure(figsize=(subplot_width, subplot_height))
        ax = fig.add_subplot(1, 1, 1)

        # Calculate target width for adaptive spacing
        target_width_px = subplot_width * 100  # Convert inches to tree coordinates
        visualizer = DecisionTreeVisualizer(
            model, feature_metadata, class_names, max_depth_display,
            short_labels=short_labels, short_naming=short_naming,
            short_layout=short_layout, short_edge_labels=short_edge_labels,
            wrap_length=wrap_length
        )
        visualizer.visualize(ax, target_width=target_width_px)

        SeparateTreeModeHelper._format_separate_tree_axes(ax, metadata, idx)
        return fig, ax

    @staticmethod
    def _format_separate_tree_axes(ax, metadata, idx):
        """
        Format axes for separate tree.

        Parameters
        ----------
        ax : Axes
            Axes to format
        metadata : dict
            Tree metadata
        idx : int
            Tree index

        Returns
        -------
        None
            Modifies ax in place

        Examples
        --------
        >>> SeparateTreeModeHelper._format_separate_tree_axes(ax, metadata, 0)
        """
        sub_title = metadata.get("sub_comparison_name", f"Comparison {idx+1}")
        ax.set_title(sub_title, fontsize=TREE_CONFIG.separate_tree_title_fontsize,
                     fontweight='bold', pad=TREE_CONFIG.separate_tree_title_pad)

        # Add border around tree
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(TREE_CONFIG.subplot_border_width)
            spine.set_edgecolor('lightgray')

        # Hide ticks and labels but keep spines for border
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    @staticmethod
    def _save_or_display_separate_tree(fig, metadata, feature_importance_name,
                                         idx, file_format, dpi, render, save_fig,
                                         cache_dir, saved_files, temp_files):
        """
        Save or display separate tree.

        Parameters
        ----------
        fig : Figure
            Figure to save/display
        metadata : dict
            Tree metadata
        feature_importance_name : str
            Analysis name
        idx : int
            Tree index
        file_format : str
            File format
        dpi : int
            DPI
        render : bool
            Render flag
        save_fig : bool
            Save flag
        cache_dir : str
            Cache directory
        saved_files : list
            List to track saved files
        temp_files : list
            List to track temp files

        Returns
        -------
        None
            Modifies saved_files or temp_files lists

        Examples
        --------
        >>> saved = []
        >>> temp = []
        >>> SeparateTreeModeHelper._save_or_display_separate_tree(
        ...     fig, metadata, "analysis", 0, "png", 300, True, True,
        ...     "./cache", saved, temp
        ... )
        """
        comparison_name = SeparateTreeModeHelper._get_sanitized_comparison_name(metadata, idx)

        if save_fig:
            SeparateTreeModeHelper._save_permanent_tree(
                fig, feature_importance_name, idx, comparison_name,
                file_format, dpi, render, cache_dir, saved_files
            )
        else:
            SeparateTreeModeHelper._save_temporary_tree(
                fig, file_format, dpi, cache_dir, temp_files
            )

    @staticmethod
    def _get_sanitized_comparison_name(metadata, idx):
        """
        Get sanitized comparison name for filename.

        Parameters
        ----------
        metadata : dict
            Tree metadata
        idx : int
            Tree index

        Returns
        -------
        str
            Sanitized name

        Examples
        --------
        >>> name = SeparateTreeModeHelper._get_sanitized_comparison_name(
        ...     {"sub_comparison_name": "State A vs B"}, 0
        ... )
        >>> print(name)
        State_A_vs_B
        """
        comparison_name = metadata.get("sub_comparison_name", f"comp{idx}")
        return comparison_name.replace(" ", "_").replace("/", "_")

    @staticmethod
    def _save_permanent_tree(fig, feature_importance_name, idx,
                              comparison_name, file_format, dpi,
                              render, cache_dir, saved_files):
        """
        Save tree permanently in cache directory.

        Parameters
        ----------
        fig : Figure
            Figure to save
        feature_importance_name : str
            Analysis name
        idx : int
            Tree index
        comparison_name : str
            Sanitized comparison name
        file_format : str
            File format
        dpi : int
            DPI
        render : bool
            Render flag
        cache_dir : str
            Cache directory
        saved_files : list
            List to track saved files

        Returns
        -------
        None
            Modifies saved_files list

        Examples
        --------
        >>> saved = []
        >>> SeparateTreeModeHelper._save_permanent_tree(
        ...     fig, "analysis", 0, "comp0", "png", 300, True, "./cache", saved
        ... )
        """
        tree_filename = (f"decision_trees_{feature_importance_name}_"
                        f"comparison_{idx:02d}_{comparison_name}.{file_format}")
        tree_path = os.path.join(cache_dir, tree_filename)
        fig.savefig(tree_path, dpi=dpi, bbox_inches='tight')
        saved_files.append(tree_path)
        print(f"Saved: {tree_path}")

        if render:
            SeparateTreeModeHelper._display_image_in_jupyter(tree_path)

    @staticmethod
    def _save_temporary_tree(fig, file_format, dpi, cache_dir, temp_files):
        """
        Save tree temporarily for display in cache directory.

        Parameters
        ----------
        fig : Figure
            Figure to save
        file_format : str
            File format
        dpi : int
            DPI setting
        cache_dir : str
            Cache directory
        temp_files : list
            List of temp files to track

        Returns
        -------
        None
            Modifies temp_files list

        Examples
        --------
        >>> temp = []
        >>> SeparateTreeModeHelper._save_temporary_tree(
        ...     fig, "png", 300, "./cache", temp
        ... )
        """
        temp_filename = f"tree_temp_{uuid.uuid4().hex}.{file_format}"
        temp_path = os.path.join(cache_dir, temp_filename)
        fig.savefig(temp_path, dpi=dpi, bbox_inches='tight')
        temp_files.append(temp_path)
        SeparateTreeModeHelper._display_image_in_jupyter(temp_path)

    @staticmethod
    def _display_image_in_jupyter(filename):
        """
        Display image in Jupyter if available.

        Parameters
        ----------
        filename : str
            Image filename

        Returns
        -------
        None
            Displays in Jupyter or silently fails

        Examples
        --------
        >>> SeparateTreeModeHelper._display_image_in_jupyter("tree.png")
        """
        if IPYTHON_AVAILABLE:
            display(Image(filename=filename))

    @staticmethod
    def _cleanup_separate_trees(old_backend, temp_files):
        """
        Cleanup after separate trees plotting.

        Parameters
        ----------
        old_backend : str or None
            Old backend to restore
        temp_files : list
            Temporary files to remove

        Returns
        -------
        None
            Cleans up backend and temp files

        Examples
        --------
        >>> SeparateTreeModeHelper._cleanup_separate_trees(
        ...     'module://matplotlib_inline.backend_inline', []
        ... )
        """
        if old_backend is not None:
            matplotlib.use(old_backend)

        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass

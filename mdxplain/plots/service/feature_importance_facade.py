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

from typing import Optional
from matplotlib.figure import Figure

from ..plot_type.violin.violin_plotter import ViolinPlotter


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
        feature_importance_name: Optional[str] = None,
        n_top: int = 10,
        feature_selector: Optional[str] = None,
        data_selectors: Optional[list] = None,
        split_features: bool = False,
        contact_threshold: Optional[float] = None,
        title: Optional[str] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300
    ) -> Figure:
        """
        Create violin plots from feature importance or manual selection.

        Supports two modes:
        1. Feature Importance mode: Automatic selection from feature importance
        2. Manual mode: User-defined feature and DataSelector selection

        Visualizes the distribution of feature values showing separate
        violins for each DataSelector group with cluster-consistent colors.

        Parameters
        ----------
        feature_importance_name : str, optional
            Name of feature importance analysis (Feature Importance mode)
        n_top : int, default=10
            Number of top features per comparison (Feature Importance mode)
        feature_selector : str, optional
            Name of feature selector (Manual mode)
        data_selectors : list of str, optional
            DataSelector names to plot (Manual mode)
        split_features : bool, default=False
            If True, create separate subplot for each feature.
            If False, group by feature type in grid layout.
        contact_threshold : float, optional
            Contact distance threshold for horizontal line in distance plots.
            If None, auto-detects from contacts feature metadata (default: 4.5 Å).
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
        >>> # Feature Importance mode
        >>> fig = facade.violins(feature_importance_name="tree_analysis", n_top=10)

        >>> # Manual mode
        >>> fig = facade.violins(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"]
        ... )

        >>> # Manual mode with customization
        >>> fig = facade.violins(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     split_features=True,
        ...     show_points=True
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
        - Automatically converts boolean contact features to continuous
          distances for proper violin visualization
        - Uses DataSelector-based color mapping for cluster consistency
        - X-axis shows feature names, Y-axis shows feature values with units
        - Each violin represents the distribution for one DataSelector group
        - Features grouped by type (distances, torsions, etc.) in combined view
        """
        plotter = ViolinPlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_importance_name=feature_importance_name,
            n_top=n_top,
            feature_selector=feature_selector,
            data_selectors=data_selectors,
            split_features=split_features,
            contact_threshold=contact_threshold,
            title=title,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi
        )

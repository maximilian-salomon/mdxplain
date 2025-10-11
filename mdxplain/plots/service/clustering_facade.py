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
Clustering plotting facade.

Provides simplified interface for creating clustering-focused plots,
particularly landscape visualizations with cluster coloring.
"""

from typing import List, Optional, Dict, Tuple, Union
from matplotlib.figure import Figure

from ..plot_type.landscape import LandscapePlotter


class ClusteringFacade:
    """
    Facade for clustering visualization methods.

    Provides high-level interface for creating plots of clustering
    results overlaid on decomposition landscapes.

    Accessible via: pipeline.plots.clustering

    Examples
    --------
    >>> # Clustering-focused landscape
    >>> pipeline.plots.clustering.landscape(
    ...     clustering_name="dbscan",
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1],
    ...     show_centers=True
    ... )

    >>> # Multi-dimensional with energy
    >>> pipeline.plots.clustering.landscape(
    ...     clustering_name="hdbscan",
    ...     decomposition_name="tica",
    ...     dimensions=[0, 1, 2, 3],
    ...     show_centers=True,
    ...     energy_values=True
    ... )
    """

    def __init__(self, manager, pipeline_data) -> None:
        """
        Initialize clustering plotting facade.

        Parameters
        ----------
        manager : PlotsManager
            Plots manager instance for accessing cache_dir
        pipeline_data : PipelineData
            Pipeline data container

        Returns
        -------
        None
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = manager.cache_dir

    def landscape(
        self,
        clustering_name: str,
        decomposition_name: str,
        dimensions: List[int],
        show_centers: bool = True,
        energy_values: bool = False,
        bins: int = 50,
        temperature: float = 310.15,
        alpha: float = 0.6,
        cluster_contour: bool = True,
        cluster_contour_voronoi: bool = False,
        data_scatter: bool = True,
        show_clusters: Union[str, List[int]] = "all",
        center_marker: str = 'X',
        center_size: int = 200,
        title: Optional[str] = None,
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        subplot_size: float = 6.0,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300
    ) -> Figure:
        """
        Create landscape plot with clustering overlay.

        Visualizes clustering results on decomposition landscape with
        cluster-colored points and optional cluster centers.

        Parameters
        ----------
        clustering_name : str
            Name of clustering to visualize (e.g., "dbscan", "hdbscan")
        decomposition_name : str
            Name of decomposition for landscape (e.g., "pca", "tica")
        dimensions : List[int]
            Dimension indices to plot (must be even number).
            Paired consecutively: [0,1,2,3] → [(0,1), (2,3)]
        show_centers : bool, default=True
            Show cluster centers (default True for clustering view)
        energy_values : bool, default=False
            Show free energy landscape instead of density
        bins : int, default=50
            Number of bins for histogram/energy calculation
        temperature : float, default=300.0
            Temperature in Kelvin for energy calculation
        alpha : float, default=0.6
            Transparency for scatter points (0=transparent, 1=opaque)
        cluster_contour : bool, default=False
            Show clusters as transparent contours instead of scatter points
        cluster_contour_voronoi : bool, default=True
            Use Voronoi-style contours (True) or KDE-based density contours (False).
            Only applies when cluster_contour=True
        data_scatter : bool, default=True
            Show gray scatter points when no clustering
        show_clusters : Union[str, List[int]], default="all"
            Which clusters to display: "all" or list of cluster IDs.
            Colors remain consistent regardless of selection
        center_marker : str, default='X'
            Marker style for cluster centers
        center_size : int, default=200
            Marker size for cluster centers
        title : Optional[str], default=None
            Custom overall title (overrides default)
        xaxis_label : Optional[str], default=None
            Custom X-axis label (default: "Component {dim_x}")
        yaxis_label : Optional[str], default=None
            Custom Y-axis label (default: "Component {dim_y}")
        xlim : Optional[Tuple[float, float]], default=None
            X-axis limits for all subplots
        ylim : Optional[Tuple[float, float]], default=None
            Y-axis limits for all subplots
        subplot_size : float, default=4.0
            Size of each subplot in inches
        save_fig : bool, default=False
            Save figure to file
        filename : Optional[str], default=None
            Custom filename (overrides auto-generated name)
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.)
        dpi : int, default=300
            Resolution for saved figure

        Returns
        -------
        matplotlib.figure.Figure
            Created figure object

        Raises
        ------
        ValueError
            If clustering not found
        ValueError
            If decomposition not found
        ValueError
            If dimensions invalid or odd number
        ValueError
            If clustering incompatible with decomposition

        Examples
        --------
        >>> # Basic clustering landscape with centers
        >>> fig = pipeline.plots.clustering.landscape(
        ...     clustering_name="dbscan",
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1]
        ... )

        >>> # Multi-dimensional grid
        >>> fig = pipeline.plots.clustering.landscape(
        ...     clustering_name="hdbscan",
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1, 2, 3],
        ...     show_centers=True
        ... )

        >>> # Energy landscape with custom styling
        >>> fig = pipeline.plots.clustering.landscape(
        ...     clustering_name="dpa",
        ...     decomposition_name="tica",
        ...     dimensions=[0, 1],
        ...     show_centers=True,
        ...     energy_values=True,
        ...     title="DPA Clustering on TICA",
        ...     save_fig=True
        ... )

        Notes
        -----
        - This method is clustering-focused, so show_centers defaults to True
        - Clustering and decomposition must have matching number of frames
        - Cluster colors use Tab20 palette for consistency
        - Auto-generated filenames include clustering name
        """
        plotter = LandscapePlotter(self.pipeline_data, cache_dir=self.cache_dir)

        return plotter.plot(
            decomposition_name=decomposition_name,
            dimensions=dimensions,
            clustering_name=clustering_name,
            show_centers=show_centers,
            energy_values=energy_values,
            bins=bins,
            temperature=temperature,
            alpha=alpha,
            cluster_contour=cluster_contour,
            cluster_contour_voronoi=cluster_contour_voronoi,
            data_scatter=data_scatter,
            show_clusters=show_clusters,
            center_marker=center_marker,
            center_size=center_size,
            title=title,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label,
            xlim=xlim,
            ylim=ylim,
            subplot_size=subplot_size,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi
        )

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
Plots manager for trajectory analysis visualizations.

Central manager for all plotting functionality, providing access to
decomposition-focused and clustering-focused plot methods, as well
as direct plotting capabilities.
"""

from typing import List, Optional, Tuple, Union
from matplotlib.figure import Figure

from ..service.decomposition_facade import DecompositionFacade
from ..service.clustering_facade import ClusteringFacade
from ..plot_type.landscape import LandscapePlotter


class PlotsManager:
    """
    Manager for all plotting operations.

    Provides three access patterns:
    1. Direct: pipeline.plots.landscape(...)
    2. Decomposition-focused: pipeline.plots.decomposition.landscape(...)
    3. Clustering-focused: pipeline.plots.clustering.landscape(...)

    Examples
    --------
    >>> # Direct access
    >>> pipeline.plots.landscape(
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1]
    ... )

    >>> # Decomposition-focused
    >>> pipeline.plots.decomposition.landscape(
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1]
    ... )

    >>> # Clustering-focused (show_centers=True by default)
    >>> pipeline.plots.clustering.landscape(
    ...     clustering_name="dbscan",
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1]
    ... )
    """

    def __init__(
        self,
        use_memmap: bool = True,
        chunk_size: int = 2000,
        cache_dir: str = "./cache"
    ) -> None:
        """
        Initialize plots manager.

        Parameters
        ----------
        use_memmap : bool, default=True
            Whether to use memory mapping for large datasets
        chunk_size : int, default=2000
            Chunk size for memory-efficient processing
        cache_dir : str, default="./cache"
            Directory for cache files

        Returns
        -------
        None
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir

    @property
    def decomposition(self) -> DecompositionFacade:
        """
        Access decomposition-focused plotting methods.

        Returns
        -------
        DecompositionFacade
            Decomposition plotting interface

        Note
        ----
        Pipeline data is passed as None here because it will be
        automatically injected later when the facade methods are called.

        Examples
        --------
        >>> # Create decomposition landscape
        >>> pipeline.plots.decomposition.landscape(
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1, 2, 3]
        ... )
        """
        return DecompositionFacade(self, None)

    @property
    def clustering(self) -> ClusteringFacade:
        """
        Access clustering-focused plotting methods.

        Returns
        -------
        ClusteringFacade
            Clustering plotting interface

        Note
        ----
        Pipeline data is passed as None here because it will be
        automatically injected later when the facade methods are called.

        Examples
        --------
        >>> # Create clustering landscape
        >>> pipeline.plots.clustering.landscape(
        ...     clustering_name="dbscan",
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1],
        ...     show_centers=True
        ... )
        """
        return ClusteringFacade(self, None)

    def landscape(
        self,
        pipeline_data,
        decomposition_name: str,
        dimensions: List[int],
        clustering_name: Optional[str] = None,
        show_centers: bool = False,
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
        Create landscape plot directly from plots manager.

        Convenience method for quick landscape visualization without
        going through decomposition or clustering facades.

        Parameters
        ----------
        decomposition_name : str
            Name of decomposition to plot (e.g., "pca", "tica")
        dimensions : List[int]
            Dimension indices to plot (must be even number).
            Paired consecutively: [0,1,2,3] → [(0,1), (2,3)]
        clustering_name : Optional[str], default=None
            Name of clustering for color overlay
        show_centers : bool, default=False
            Show cluster centers (requires clustering_name)
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
            If decomposition not found
        ValueError
            If dimensions invalid or odd number
        ValueError
            If show_centers=True but no clustering_name
        ValueError
            If clustering not compatible with decomposition

        Examples
        --------
        >>> # Simple landscape
        >>> fig = pipeline.plots.landscape(
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1]
        ... )

        >>> # With clustering overlay
        >>> fig = pipeline.plots.landscape(
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1],
        ...     clustering_name="dbscan",
        ...     show_centers=True
        ... )

        >>> # Energy landscape with saving
        >>> fig = pipeline.plots.landscape(
        ...     decomposition_name="tica",
        ...     dimensions=[0, 1, 2, 3],
        ...     energy_values=True,
        ...     save_fig=True,
        ...     filename="tica_energy_landscape.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        This is a direct convenience method. For more specialized workflows,
        consider using:
        - pipeline.plots.decomposition.landscape() for decomposition focus
        - pipeline.plots.clustering.landscape() for clustering focus
        """
        plotter = LandscapePlotter(pipeline_data, cache_dir=self.cache_dir)

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

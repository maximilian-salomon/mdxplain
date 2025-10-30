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

"""Service for adding clustering algorithms without explicit type instantiation."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.cluster_manager import ClusterManager
    from ...pipeline.entities.pipeline_data import PipelineData

from .dbscan_add_service import DBSCANAddService
from .hdbscan_add_service import HDBSCANAddService
from .dpa_add_service import DPAAddService


class ClusterAddService:
    """
    Service for adding clustering algorithms without explicit type instantiation.

    This service provides an intuitive interface for adding clustering algorithms
    without requiring users to import and instantiate cluster types directly.
    Provides both standard calls with default center method and explicit selection.

    Examples
    --------
    >>> # Standard calls with default centroid centers
    >>> pipeline.clustering.add.dbscan("my_features", eps=0.5, min_samples=5)
    >>> pipeline.clustering.add.hdbscan("my_features", min_cluster_size=10)
    >>> pipeline.clustering.add.dpa("my_features", Z=2.0)

    >>> # Explicit center method selection
    >>> pipeline.clustering.add.dbscan.with_median_centers("my_features", eps=0.5)
    >>> pipeline.clustering.add.hdbscan.with_density_peak_centers("my_features", min_cluster_size=10)
    >>> pipeline.clustering.add.dpa.with_rmsd_centroid_centers("my_features", Z=2.0)
    """

    def __init__(self, manager: ClusterManager, pipeline_data: PipelineData) -> None:
        """
        Initialize service with manager and pipeline data.

        Parameters
        ----------
        manager : ClusterManager
            Cluster manager instance
        pipeline_data : PipelineData
            Pipeline data container (injected by AutoInjectProxy)

        Returns
        -------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
        self._dbscan_service = DBSCANAddService(manager, pipeline_data)
        self._hdbscan_service = HDBSCANAddService(manager, pipeline_data)
        self._dpa_service = DPAAddService(manager, pipeline_data)

    @property
    def dbscan(self) -> DBSCANAddService:
        """
        Service for adding DBSCAN clustering.

        Uses centroid (medoid) center calculation by default. The centroid is the
        actual data point closest to the cluster mean, ensuring the cluster center
        is a real conformational state from the trajectory.

        For alternative center methods, use:

        - .with_mean_centers() - Arithmetic mean (may not be real state)
        - .with_median_centers() - Feature-wise median (robust to outliers)
        - .with_density_peak_centers() - Highest density point
        - .with_median_centroid_centers() - Medoid from median
        - .with_rmsd_centroid_centers() - RMSD-based centroid

        Returns
        -------
        DBSCANAddService
            Service for DBSCAN clustering with flexible center method selection

        Examples
        --------
        >>> # Standard call with default centroid centers
        >>> pipeline.clustering.add.dbscan("features", eps=0.5, min_samples=5)

        >>> # Explicit center method
        >>> pipeline.clustering.add.dbscan.with_median_centers("features", eps=0.5)
        """
        return self._dbscan_service

    @property
    def hdbscan(self) -> HDBSCANAddService:
        """
        Service for adding HDBSCAN clustering.

        Uses centroid (medoid) center calculation by default. The centroid is the
        actual data point closest to the cluster mean, ensuring the cluster center
        is a real conformational state from the trajectory.

        For alternative center methods, use:

        - .with_mean_centers() - Arithmetic mean (may not be real state)
        - .with_median_centers() - Feature-wise median (robust to outliers)
        - .with_density_peak_centers() - Highest density point
        - .with_median_centroid_centers() - Medoid from median
        - .with_rmsd_centroid_centers() - RMSD-based centroid

        Returns
        -------
        HDBSCANAddService
            Service for HDBSCAN clustering with flexible center method selection

        Examples
        --------
        >>> # Standard call with default centroid centers
        >>> pipeline.clustering.add.hdbscan("features", min_cluster_size=10)

        >>> # Explicit center method
        >>> pipeline.clustering.add.hdbscan.with_density_peak_centers("features", min_cluster_size=10)
        """
        return self._hdbscan_service

    @property
    def dpa(self) -> DPAAddService:
        """
        Service for adding DPA clustering.

        Uses centroid (medoid) center calculation by default. The centroid is the
        actual data point closest to the cluster mean, ensuring the cluster center
        is a real conformational state from the trajectory.

        For alternative center methods, use:
        
        - .with_mean_centers() - Arithmetic mean (may not be real state)
        - .with_median_centers() - Feature-wise median (robust to outliers)
        - .with_density_peak_centers() - Highest density point
        - .with_median_centroid_centers() - Medoid from median
        - .with_rmsd_centroid_centers() - RMSD-based centroid

        Returns
        -------
        DPAAddService
            Service for DPA clustering with flexible center method selection

        Examples
        --------
        >>> # Standard call with default centroid centers
        >>> pipeline.clustering.add.dpa("features", Z=2.0)

        >>> # Explicit center method
        >>> pipeline.clustering.add.dpa.with_rmsd_centroid_centers("features", Z=2.0)
        """
        return self._dpa_service

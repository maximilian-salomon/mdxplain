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
Central pipeline manager with automatic data injection.

This module provides the PipelineManager class that serves as the central
orchestration point for all analysis workflows. It uses AutoInjectProxy
to automatically inject PipelineData into manager methods that need it.
"""

from typing import Any, Dict, Optional, cast
import os

from ..entities.pipeline_data import PipelineData
from .auto_inject_proxy import AutoInjectProxy

from ...trajectory import TrajectoryManager
from ...feature import FeatureManager
from ...feature_selection.managers.feature_selector_manager import FeatureSelectorManager
from ...clustering import ClusterManager
from ...decomposition import DecompositionManager
from ...data_selector.managers.data_selector_manager import DataSelectorManager
from ...comparison.managers.comparison_manager import ComparisonManager
from ...feature_importance.managers.feature_importance_manager import FeatureImportanceManager


class PipelineManager:
    """
    Central pipeline manager with automatic PipelineData injection.

    This class provides a unified interface to all analysis modules through
    AutoInjectProxy instances that automatically inject PipelineData into
    methods that expect it while leaving utility methods unchanged.

    The PipelineManager is designed to simplify the usage of the
    mdxplain pipeline system by providing a single entry point for all
    analysis workflows. It manages the trajectory loading, feature computation,
    clustering, and decomposition processes in a cohesive manner.

    It is the single entry point for:
    - Trajectory loading and validation
    - Feature computation
    - Feature selection
    - Clustering analysis
    - Decomposition analysis

    Examples:
    ---------
    Basic pipeline workflow:

    >>> pipeline = PipelineManager()
    >>>
    >>> # Methods expecting pipeline_data get automatic injection
    >>> pipeline.trajectory.load_trajectories('../data')
    >>> pipeline.feature.compute_features('distances', 'res CA')
    >>> pipeline.feature_selector.create('my_selection')
    >>> pipeline.feature_selector.add('my_selection', 'distances', 'res ALA')
    >>> pipeline.feature_selector.select('my_selection')
    >>> pipeline.clustering.cluster('my_features', 'dbscan', eps=0.5)
    >>>
    >>> # Utility methods work without injection
    >>> valid = pipeline.trajectory.validate_selection('res CA')
    >>> formats = pipeline.trajectory.get_supported_formats()
    >>>
    >>> # Advanced: Direct data access
    >>> summary = pipeline.data.get_data_summary()
    """

    def __init__(
        self,
        # Trajectory parameters
        stride: int = 1,
        concat: bool = False,
        selection: Optional[str] = None,
        # Feature/Decomposition parameters
        use_memmap: bool = False,
        chunk_size: int = 10000,
        # Cache directory for all managers
        cache_dir: str = "./cache",
    ):
        """
        Initialize the pipeline manager with configuration for all managers.

        Parameters:
        -----------
        stride : int, default=1
            Default stride for trajectory loading
        concat : bool, default=False
            Default concatenation setting for trajectories
        selection : str, optional
            Default MDTraj selection string for trajectories
        use_memmap : bool, default=False
            Whether to use memory mapping for feature and decomposition data
        chunk_size : int, default=10000
            Processing chunk size for feature and decomposition computation
        cache_dir : str, default="./cache"
            Cache directory path for all managers

        Returns:
        --------
        None
            Initializes PipelineManager with automatic data injection
        """
        # Validate parameters
        if stride <= 0 and not isinstance(stride, int):
            raise ValueError("Stride must be a positive integer.")
        if chunk_size <= 0 and not isinstance(chunk_size, int):
            raise ValueError("Chunk size must be a positive integer.")

        os.makedirs(cache_dir, exist_ok=True)

        # Central data container
        self._data = PipelineData(
            use_memmap=use_memmap, cache_dir=cache_dir, chunk_size=chunk_size
        )

        # Create manager instances with their configurations
        self._trajectory_manager = TrajectoryManager(
            stride=stride, concat=concat, selection=selection, cache_dir=cache_dir,
            use_memmap=use_memmap, chunk_size=chunk_size
        )
        self._feature_manager = FeatureManager(
            use_memmap=use_memmap, chunk_size=chunk_size, cache_dir=cache_dir
        )
        self._cluster_manager = ClusterManager(cache_dir=cache_dir)
        self._decomposition_manager = DecompositionManager(
            use_memmap=use_memmap, chunk_size=chunk_size, cache_dir=cache_dir
        )
        self._feature_selector_manager = FeatureSelectorManager()
        
        self._data_selector_manager = DataSelectorManager()
        self._comparison_manager = ComparisonManager()
        self._feature_importance_manager = FeatureImportanceManager()

    @property
    def trajectory(self) -> TrajectoryManager:
        """
        Access trajectory management with automatic PipelineData injection.

        Returns:
        --------
        TrajectoryManager
            Trajectory manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            TrajectoryManager, AutoInjectProxy(self._trajectory_manager, self._data)
        )

    @property
    def feature(self) -> FeatureManager:
        """
        Access feature computation with automatic PipelineData injection.

        Returns:
        --------
        FeatureManager
            Feature manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(FeatureManager, AutoInjectProxy(self._feature_manager, self._data))

    @property
    def clustering(self) -> ClusterManager:
        """
        Access clustering analysis with automatic PipelineData injection.

        Returns:
        --------
        ClusterManager
            Cluster manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(ClusterManager, AutoInjectProxy(self._cluster_manager, self._data))

    @property
    def decomposition(self) -> DecompositionManager:
        """
        Access decomposition analysis with automatic PipelineData injection.

        Returns:
        --------
        DecompositionManager
            Decomposition manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            DecompositionManager,
            AutoInjectProxy(self._decomposition_manager, self._data),
        )

    @property
    def feature_selector(self) -> FeatureSelectorManager:
        """
        Access feature selector management with automatic PipelineData injection.

        Returns:
        --------
        FeatureSelectorManager
            Feature selector manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            FeatureSelectorManager,
            AutoInjectProxy(self._feature_selector_manager, self._data),
        )

    @property
    def data_selector(self) -> DataSelectorManager:
        """
        Access data selector management with automatic PipelineData injection.

        Returns:
        --------
        DataSelectorManager
            Data selector manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            DataSelectorManager,
            AutoInjectProxy(self._data_selector_manager, self._data),
        )

    @property
    def comparison(self) -> ComparisonManager:
        """
        Access comparison management with automatic PipelineData injection.

        Returns:
        --------
        ComparisonManager
            Comparison manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            ComparisonManager,
            AutoInjectProxy(self._comparison_manager, self._data),
        )

    @property
    def feature_importance(self) -> FeatureImportanceManager:
        """
        Access feature importance analysis with automatic PipelineData injection.

        Returns:
        --------
        FeatureImportanceManager
            Feature importance manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            FeatureImportanceManager,
            AutoInjectProxy(self._feature_importance_manager, self._data),
        )

    @property
    def data(self):
        """
        Direct access to pipeline data (advanced usage).

        Provides direct access to the central PipelineData container for
        advanced users who need to inspect or manipulate data directly.
        Normal usage should go through the manager properties.

        Returns:
        --------
        PipelineData
            Central pipeline data container with all analysis data
        """
        return self._data

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of all pipeline data.

        Returns:
        --------
        dict
            Summary information about all loaded and computed data
        """
        return self._data.get_data_summary()

    def clear_all(self) -> None:
        """
        Clear all pipeline data.

        Resets the entire pipeline to empty state, clearing all
        trajectories, features, clustering, and decomposition results.
        """
        self._data.clear_all_data()

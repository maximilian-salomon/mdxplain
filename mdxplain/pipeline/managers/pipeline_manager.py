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

from __future__ import annotations

from typing import Any, Dict, Optional, cast
import os
import shutil
import numpy as np
from pathlib import Path

from ..entities.pipeline_data import PipelineData
from .auto_inject_proxy import AutoInjectProxy
from ...utils.archive_utils import ArchiveUtils

from ...trajectory import TrajectoryManager
from ...feature import FeatureManager
from ...feature_selection.managers.feature_selector_manager import (
    FeatureSelectorManager,
)
from ...clustering import ClusterManager
from ...decomposition import DecompositionManager
from ...data_selector.managers.data_selector_manager import DataSelectorManager
from ...comparison.managers.comparison_manager import ComparisonManager
from ...feature_importance.managers.feature_importance_manager import (
    FeatureImportanceManager,
)
from ...analysis import AnalysisManager
from ...plots.manager.plots_manager import PlotsManager
from ...structure_visualization.managers.structure_visualization_manager import (
    StructureVisualizationManager,
)


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
    - Data selection
    - Comparison management
    - Feature importance analysis
    - General analysis operations
    - Visualization and plotting

    Examples
    --------
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
        use_memmap: bool = True,
        chunk_size: int = 2000,
        dtype: type = np.float32,
        # Cache directory for all managers
        cache_dir: str = "./cache",
    ):
        """
        Initialize the pipeline manager with configuration for all managers.

        Parameters
        ----------
        stride : int, default=1
            Default stride for trajectory loading
        concat : bool, default=False
            Default concatenation setting for trajectories
        selection : str, optional
            Default MDTraj selection string for trajectories
        use_memmap : bool, default=False
            Whether to use memory mapping for feature and decomposition data
        chunk_size : int, default=2000
            Processing chunk size for feature and decomposition computation
        dtype : type, default=np.float32
            Data type for feature matrices (float32 or float64).
            float32 saves 50% memory and is sufficient for most MD analysis.
            Use float64 only if extreme numerical precision required.
        cache_dir : str, default="./cache"
            Cache directory path for all managers

        Returns
        -------
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
            use_memmap=use_memmap,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            dtype=dtype,
        )

        # Create manager instances with their configurations
        self._trajectory_manager = TrajectoryManager(
            stride=stride,
            concat=concat,
            selection=selection,
            cache_dir=cache_dir,
            use_memmap=use_memmap,
            chunk_size=chunk_size,
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
        self._feature_importance_manager = FeatureImportanceManager(
            use_memmap=use_memmap, chunk_size=chunk_size, cache_dir=cache_dir
        )
        self._analysis_manager = AnalysisManager()
        self._plots_manager = PlotsManager(
            use_memmap=use_memmap, chunk_size=chunk_size, cache_dir=cache_dir
        )
        self._structure_visualization_manager = StructureVisualizationManager(
            use_memmap=use_memmap, chunk_size=chunk_size, cache_dir=cache_dir
        )

    @property
    def trajectory(self) -> TrajectoryManager:
        """
        Access trajectory management with automatic PipelineData injection.

        Returns
        -------
        TrajectoryManager
            Trajectory manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            TrajectoryManager,
            AutoInjectProxy(self._trajectory_manager, self._data),
        )

    @property
    def feature(self) -> FeatureManager:
        """
        Access feature computation with automatic PipelineData injection.

        Returns
        -------
        FeatureManager
            Feature manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            FeatureManager, AutoInjectProxy(self._feature_manager, self._data)
        )

    @property
    def clustering(self) -> ClusterManager:
        """
        Access clustering analysis with automatic PipelineData injection.

        Returns
        -------
        ClusterManager
            Cluster manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            ClusterManager, AutoInjectProxy(self._cluster_manager, self._data)
        )

    @property
    def decomposition(self) -> DecompositionManager:
        """
        Access decomposition analysis with automatic PipelineData injection.

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
        FeatureImportanceManager
            Feature importance manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            FeatureImportanceManager,
            AutoInjectProxy(self._feature_importance_manager, self._data),
        )

    @property
    def analysis(self) -> AnalysisManager:
        """
        Access analysis operations with automatic PipelineData injection.

        Returns
        -------
        AnalysisManager
            Analysis manager with automatic PipelineData injection.
            All methods that expect pipeline_data parameter will receive it automatically.
        """
        return cast(
            AnalysisManager, AutoInjectProxy(self._analysis_manager, self._data)
        )

    @property
    def plots(self) -> PlotsManager:
        """
        Access plotting and visualization operations.

        Returns
        -------
        PlotsManager
            Plots manager for creating visualizations.
            Provides three access patterns:
            - Direct: pipeline.plots.landscape(...)
            - Decomposition-focused: pipeline.plots.decomposition.landscape(...)
            - Clustering-focused: pipeline.plots.clustering.landscape(...)

        Examples
        --------
        >>> # Direct landscape plot
        >>> pipeline.plots.landscape("pca", [0, 1])

        >>> # Decomposition-focused
        >>> pipeline.plots.decomposition.landscape("pca", [0, 1])

        >>> # Clustering-focused with centers
        >>> pipeline.plots.clustering.landscape(
        ...     "dbscan", "pca", [0, 1], show_centers=True
        ... )
        """
        return cast(
            PlotsManager, AutoInjectProxy(self._plots_manager, self._data)
        )

    @property
    def structure_visualization(self) -> StructureVisualizationManager:
        """
        Access 3D structure visualization with automatic PipelineData injection.

        Returns
        -------
        StructureVisualizationManager
            Structure visualization manager with automatic PipelineData injection.

        Examples
        --------
        >>> # Beta-factor visualization
        >>> pipeline.structure_visualization.visualize_importance_beta_factors(
        ...     "dt_analysis", "cluster_0_vs_rest", n_top=10
        ... )
        """
        return cast(
            StructureVisualizationManager,
            AutoInjectProxy(self._structure_visualization_manager, self._data),
        )

    @property
    def data(self):
        """
        Direct access to pipeline data (advanced usage).

        Provides direct access to the central PipelineData container for
        advanced users who need to inspect or manipulate data directly.
        Normal usage should go through the manager properties.

        Returns
        -------
        PipelineData
            Central pipeline data container with all analysis data
        """
        return self._data

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of all pipeline data.

        Returns
        -------
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

    def save_to_single_file(self, save_path: str) -> None:
        """
        Save complete pipeline to single pickle file.

        This method saves the entire PipelineData object including all
        computed features, trajectories, clusterings, decompositions,
        and metadata to a single file. Memmap files remain in cache directory.

        Parameters
        ----------
        save_path : str
            Path where to save the complete pipeline

        Returns
        -------
        None
            Saves the complete pipeline to the specified path

        Examples
        --------
        >>> pipeline.save_to_single_file('complete_analysis.pkl')
        """
        self._data.save(save_path)

    @staticmethod
    def load_from_single_file(
        load_path: str,
        cache_dir: str = "./cache",
        chunk_size: int = 1000,
        stride: int = 1,
        concat: bool = False,
        selection: Optional[str] = None,
    ) -> PipelineManager:
        """
        Load complete pipeline from single pickle file.

        This static method creates a new PipelineManager instance with
        specified cache directory and loads pipeline state from file.
        Memmap files are expected in the cache directory.

        Parameters
        ----------
        load_path : str
            Path to the saved pipeline pickle file
        cache_dir : str, default="./cache"
            Cache directory where memmap files are located
        chunk_size : int, default=1000
            Default chunk size for future operations
        stride : int, default=1
            Default stride for future trajectory loading
        concat : bool, default=False
            Default concat mode for future trajectory loading
        selection : str, optional
            Default MDTraj selection string for trajectories

        Returns
        -------
        PipelineManager
            New PipelineManager instance with loaded pipeline state

        Examples
        --------
        >>> loaded_pipeline = PipelineManager.load_from_single_file('analysis.pkl')
        >>> loaded_pipeline.print_info()

        >>> # Load with custom cache directory
        >>> loaded_pipeline = PipelineManager.load_from_single_file(
        ...     'analysis.pkl',
        ...     cache_dir='./my_cache'
        ... )

        >>> # Load with custom trajectory defaults for adding more data
        >>> loaded_pipeline = PipelineManager.load_from_single_file(
        ...     'analysis.pkl',
        ...     stride=10
        ... )
        >>> # Now load additional trajectories with stride=10
        >>> loaded_pipeline.trajectory.load_trajectories('new_data/')
        """
        pipeline = PipelineManager(
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            stride=stride,
            concat=concat,
            selection=selection
        )
        pipeline._data.load(load_path)
        return pipeline

    def create_sharable_archive(
        self,
        archive_path: str,
        compression: str = "xz",
        exclude_visualizations: bool = True,
        include_structure_files: bool = True
    ) -> str:
        """
        Create sharable compressed archive with pipeline and essential data.

        Creates compressed tar archive containing pipeline pickle file
        and all necessary memmap files from cache directory. Excludes
        visualization outputs by default for smaller archive size.

        Parameters
        ----------
        archive_path : str
            Path for output archive (extension added automatically)
        compression : str, default="xz"
            Compression method: "xz", "bz2", or "gz"
        exclude_visualizations : bool, default=True
            If True, exclude PNG/PDF/SVG plot outputs
        include_structure_files : bool, default=True
            If True, include PDB/PML structure files

        Returns
        -------
        str
            Path to created archive file

        Examples
        --------
        >>> # Minimal archive (only data)
        >>> pipeline.create_sharable_archive("analysis.tar.xz")

        >>> # Full archive (with visualizations)
        >>> pipeline.create_sharable_archive(
        ...     "analysis_full.tar.xz",
        ...     exclude_visualizations=False
        ... )

        >>> # Data only (no structure files)
        >>> pipeline.create_sharable_archive(
        ...     "analysis_data.tar.xz",
        ...     include_structure_files=False
        ... )

        Notes
        -----
        - xz compression provides best compression ratio
        - With use_memmap=False: Archive contains only pipeline.pkl + optional PDB/PML
        - With use_memmap=True: Archive contains pipeline.pkl + .dat files + zarr directories
        - Paths are preserved relative to cache directory
        - Memmap and zarr only included when use_memmap=True
        """
        return ArchiveUtils.create_archive(
            self._data,
            archive_path,
            compression,
            exclude_visualizations,
            include_structure_files
        )

    @staticmethod
    def load_from_archive(
        archive_path: str,
        cache_dir: str = "./cache",
        chunk_size: int = 1000,
        stride: int = 1,
        concat: bool = False,
        selection: Optional[str] = None,
    ) -> PipelineManager:
        """
        Load pipeline from sharable archive.

        Extracts compressed archive, moves cache files to specified
        cache directory, and loads pipeline state. Automatically
        repairs memmap file paths to point to new cache location.

        Parameters
        ----------
        archive_path : str
            Path to archive file
        cache_dir : str, default="./cache"
            Target cache directory for extracted files
        chunk_size : int, default=1000
            Default chunk size for future operations
        stride : int, default=1
            Default stride for future trajectory loading
        concat : bool, default=False
            Default concatenation setting for future trajectory loading
        selection : str, optional
            Default MDTraj selection string for trajectories

        Returns
        -------
        PipelineManager
            Loaded pipeline instance

        Examples
        --------
        >>> # Load from archive (default cache_dir)
        >>> pipeline = PipelineManager.load_from_archive("analysis.tar.xz")
        >>> pipeline.print_info()

        >>> # Load with custom cache directory
        >>> pipeline = PipelineManager.load_from_archive(
        ...     "analysis.tar.xz",
        ...     cache_dir="./my_cache"
        ... )

        >>> # Load with trajectory defaults for adding more data
        >>> pipeline = PipelineManager.load_from_archive(
        ...     "analysis.tar.xz",
        ...     cache_dir="./cache",
        ...     chunk_size=500,
        ...     stride=10
        ... )

        Notes
        -----
        - Extracts to temporary directory
        - Moves cache files to specified cache_dir
        - Automatically repairs memmap paths for portability
        - Cache directory created if it doesn't exist
        """
        extract_dir = ArchiveUtils.extract_archive(archive_path)

        pkl_path = extract_dir / "pipeline.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"pipeline.pkl not found in extracted archive"
            )

        # Move cache files to target cache_dir
        extracted_cache = extract_dir / "cache"
        target_cache = Path(cache_dir)
        target_cache.mkdir(parents=True, exist_ok=True)

        if extracted_cache.exists():
            for item in extracted_cache.iterdir():
                target_item = target_cache / item.name
                if target_item.exists():
                    if target_item.is_dir():
                        shutil.rmtree(target_item)
                    else:
                        target_item.unlink()
                shutil.move(str(item), str(target_item))

        # Load with custom cache_dir, chunk_size and trajectory defaults
        pipeline = PipelineManager(
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            stride=stride,
            concat=concat,
            selection=selection
        )
        pipeline._data.load(str(pkl_path))

        return pipeline

    def print_info(self) -> None:
        """
        Print comprehensive pipeline information.

        This method prints information from ALL managers to provide
        a complete overview of the pipeline state.

        Returns
        -------
        None
            Prints comprehensive pipeline information to console

        Examples
        --------
        >>> pipeline.print_info()
        ======= PIPELINE INFORMATION =======

        --- Trajectory Data ---
        Loaded 3 trajectories::
        
            [0] system1_traj1: 1000 frames
            [1] system1_traj2: 1500 frames
            [2] system2_traj1: 800 frames

        --- Feature Data ---
        Feature Types: 2 (distances, contacts)

        --- Clustering Data ---
        Clustering Names: 1 (conformations)

        (... information from all managers ...)
        """
        print("======= PIPELINE INFORMATION =======")

        print("\n--- Trajectory Data ---")
        self.trajectory.print_info()

        print("\n--- Feature Data ---")
        self.feature.print_info()

        print("\n--- Feature Selection Data ---")
        self.feature_selector.print_info()

        print("\n--- Clustering Data ---")
        self.clustering.print_info()

        print("\n--- Decomposition Data ---")
        self.decomposition.print_info()

        print("\n--- Data Selector Data ---")
        self.data_selector.print_info()

        print("\n--- Comparison Data ---")
        self.comparison.print_info()

        print("\n--- Feature Importance Data ---")
        self.feature_importance.print_info()

        print("\n======= END PIPELINE INFORMATION =======")

        # Summary at the end
        summary = self.summary()
        print(
            f"\nPipeline Summary: "
            f"{summary['trajectories_loaded']} trajectories, "
            f"{summary['features_computed']} feature types, "
            f"{summary['feature_selections']} feature selections, "
            f"{summary['clusterings_performed']} clusterings, "
            f"{summary['decompositions_computed']} decompositions, "
            f"{summary['data_selectors_created']} data selectors, "
            f"{summary['comparisons_created']} comparisons, "
            f"{summary['feature_importance_analyses']} feature importance analyses"
        )

    def update_config(
        self,
        chunk_size: int = None,
        cache_dir: str = None,
        use_memmap: bool = None,
    ):
        """
        Update pipeline configuration parameters at runtime.

        Allows modification of key configuration parameters after pipeline
        initialization. Changes are propagated to all managers and the
        central PipelineData container.

        Parameters
        ----------
        chunk_size : int, optional
            New chunk size for processing operations. Must be positive integer.
        cache_dir : str, optional
            New cache directory path. Directory will be created if it doesn't exist.
        use_memmap : bool, optional
            Whether to use memory mapping for data storage operations.

        Returns
        -------
        None
            Updates configuration in all components

        Raises
        ------
        ValueError
            If chunk_size is not a positive integer
        OSError
            If cache_dir cannot be created

        Examples
        --------
        Update chunk size for better memory management:

        >>> pipeline.update_config(chunk_size=5000)

        Change cache directory and enable memory mapping:

        >>> pipeline.update_config(cache_dir="/tmp/mdx_cache", use_memmap=True)
        """
        # Validate parameters
        if chunk_size is not None:
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer")

        if cache_dir is not None:
            if not isinstance(cache_dir, str):
                raise ValueError("cache_dir must be a string")
            # Create directory if it doesn't exist
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as e:
                raise OSError(
                    f"Cannot create cache directory '{cache_dir}': {e}"
                )

        if use_memmap is not None:
            if not isinstance(use_memmap, bool):
                raise ValueError("use_memmap must be a boolean")

        # Update PipelineData
        if chunk_size is not None:
            self._data.chunk_size = chunk_size
        if cache_dir is not None:
            self._data.cache_dir = cache_dir
        if use_memmap is not None:
            self._data.use_memmap = use_memmap

        # Update TrajectoryManager
        if chunk_size is not None:
            self._trajectory_manager.chunk_size = chunk_size
        if cache_dir is not None:
            self._trajectory_manager.cache_dir = cache_dir
        if use_memmap is not None:
            self._trajectory_manager.use_memmap = use_memmap

        # Update FeatureManager
        if chunk_size is not None:
            self._feature_manager.chunk_size = chunk_size
        if cache_dir is not None:
            self._feature_manager.cache_dir = cache_dir
        if use_memmap is not None:
            self._feature_manager.use_memmap = use_memmap

        # Update DecompositionManager
        if chunk_size is not None:
            self._decomposition_manager.chunk_size = chunk_size
        if cache_dir is not None:
            self._decomposition_manager.cache_dir = cache_dir
        if use_memmap is not None:
            self._decomposition_manager.use_memmap = use_memmap

        # Update ClusterManager (only cache_dir)
        if cache_dir is not None:
            self._cluster_manager.cache_dir = cache_dir

        # Update FeatureImportanceManager
        if chunk_size is not None:
            self._feature_importance_manager.chunk_size = chunk_size
        if cache_dir is not None:
            self._feature_importance_manager.cache_dir = cache_dir
        if use_memmap is not None:
            self._feature_importance_manager.use_memmap = use_memmap

        # Update PlotsManager
        if chunk_size is not None:
            self._plots_manager.chunk_size = chunk_size
        if cache_dir is not None:
            self._plots_manager.cache_dir = cache_dir
        if use_memmap is not None:
            self._plots_manager.use_memmap = use_memmap

        print("Configuration updated successfully:")
        if chunk_size is not None:
            print(f"  chunk_size: {chunk_size}")
        if cache_dir is not None:
            print(f"  cache_dir: {cache_dir}")
        if use_memmap is not None:
            print(f"  use_memmap: {use_memmap}")

    def get_config(self) -> dict:
        """
        Get current pipeline configuration parameters.

        Returns the current configuration settings that are active
        across all pipeline components.

        Returns
        -------
        dict
            Dictionary containing current configuration values

        Examples
        --------
        Check current configuration:

        >>> pipeline = PipelineManager(chunk_size=1000, use_memmap=True)
        >>> config = pipeline.get_config()
        >>> print(f"Using chunk size: {config['chunk_size']}")
        >>> print(f"Memory mapping: {config['use_memmap']}")
        """
        return self._data.get_config()

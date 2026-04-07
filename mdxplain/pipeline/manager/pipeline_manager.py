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

from typing import Any, Dict, Optional, Union, cast
from types import SimpleNamespace
from datetime import datetime
import gc
import re
import os
import shutil
import tempfile
import sys
import warnings
import numpy as np
from pathlib import Path
import uuid

from ..entities.pipeline_data import PipelineData
from ..helper.cache_remap_helper import CacheRemapHelper
from .auto_inject_proxy import AutoInjectProxy
from .performance_config import PerformanceConfig
from ...utils.archive_utils import ArchiveUtils
from ...utils.archive_fetch_helper import ArchiveFetchHelper
from ...utils.cleanup_utils import CleanupUtils
from ...utils.helper.load_and_save_helper import LoadAndSaveHelper
from ...utils.memmap_utils import MemmapUtils
from ...utils.path_utils import PathUtils
from ...utils.progress_utils import ProgressUtils
from ...utils.resource_utils import ResourceUtils
from ...feature.helper.feature_binding_helper import FeatureBindingHelper

from ...trajectory import TrajectoryManager
from ...feature import FeatureManager
from ...feature_selection.manager.feature_selector_manager import (
    FeatureSelectorManager,
)
from ...clustering import ClusterManager
from ...decomposition import DecompositionManager
from ...data_selector.manager.data_selector_manager import DataSelectorManager
from ...comparison.manager.comparison_manager import ComparisonManager
from ...feature_importance.manager.feature_importance_manager import (
    FeatureImportanceManager,
)
from ...analysis import AnalysisManager
from ...plots.manager.plots_manager import PlotsManager
from ...structure_visualization.manager.structure_visualization_manager import (
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
        # Memory management
        max_memory_gb: float = 6.0,
        # Output control
        show_progress: bool = True,
        # Stability configuration toggle
        use_stability_config: Optional[bool] = None,
    ):
        """
        Initialize the pipeline manager with configuration for all managers.

        Parameters
        ----------
        stride : int, default=1
            Default stride for trajectory loading. Larger values reduce memory
            footprint and downstream compute at the cost of temporal resolution.
            For large datasets, consider stride > 1 to keep feature matrices
            smaller and more cache-friendly.

        concat : bool, default=False
            Default concatenation setting for trajectories. When True, multiple
            trajectories are concatenated into one, which simplifies indexing
            but may increase memory pressure for very long series.

        selection : str, optional
            Default MDTraj selection string for trajectories. Use this to
            restrict atom selection at load time to reduce downstream features.

        use_memmap : bool, default=True
            Whether to use memory mapping for feature and decomposition data.
            Enable this for large datasets that do not fit comfortably in RAM.
            When stability settings are enabled, the pipeline lowers I/O
            priority during large sequential scans to keep the system
            responsive.

        chunk_size : int, default=2000
            Processing chunk size for feature and decomposition computation.
            Larger chunks reduce overhead but increase peak memory and I/O
            burst size. For very large memmaps, a moderate chunk size tends
            to provide the best I/O behavior.

        dtype : type, default=np.float32
            Data type for feature matrices (float32 or float64). float32 saves
            50% memory and is sufficient for most MD analysis. Use float64
            only if you need extreme numerical precision or stable eigenvalues.

        cache_dir : str, default="./cache"
            Cache directory path for all managers. For memmap-heavy workloads,
            use a fast local SSD to avoid I/O contention.

        max_memory_gb : float, default=6.0
            Maximum memory in GB for dataset processing. Used for memory-aware
            sampling in algorithms like DecisionTree. Increase this on large
            workstations to reduce sampling, or keep it low to stay responsive.
        show_progress : bool, default=True
            Enable or disable progress bars globally (tqdm). When False, all
            progress bars are suppressed. This also disables resource limit
            reporting unless you call report_resource_limits explicitly.

        use_stability_config : bool or None, default=None
            Choose how resource limits are applied at initialization. The
            default (None) enables stability settings only when use_memmap is
            True; otherwise it preserves OS defaults. When True, the pipeline
            applies a stability policy immediately: it lowers CPU priority
            (nice=15), keeps two CPU cores free via affinity, reduces I/O
            priority to "low", and caps BLAS/OpenMP thread pools to the active
            CPU set. When False, no resource limits are applied at startup and
            the process keeps OS defaults. All settings are stored in
            pipeline.config.performance; any change there is applied
            immediately.

        Returns
        -------
        None
            Initializes PipelineManager with automatic data injection
        """
        self._validate_init_params(stride=stride, chunk_size=chunk_size)
        self._pipeline_uuid = uuid.uuid4().hex
        self._pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_dir = PathUtils.create_pipeline_cache_dir(
            cache_dir,
            pipeline_uuid=self._pipeline_uuid,
            pipeline_timestamp=self._pipeline_timestamp,
            purpose="cache directory",
        )

        self._init_performance_config(
            use_memmap=use_memmap,
            use_stability_config=use_stability_config,
        )
        self._init_data(
            use_memmap=use_memmap,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            dtype=dtype,
            max_memory_gb=max_memory_gb,
        )
        self._init_managers(
            stride=stride,
            concat=concat,
            selection=selection,
            cache_dir=cache_dir,
            use_memmap=use_memmap,
            chunk_size=chunk_size,
        )
        self._apply_show_progress(show_progress)
        self.report_resource_limits()
        self._closed = False

    def _validate_init_params(self, stride: int, chunk_size: int) -> None:
        """
        Validate initialization parameters for basic correctness.

        This helper keeps the constructor focused on wiring while still
        enforcing minimum input constraints. It currently validates stride
        and chunk size since these affect memory access patterns and can
        trigger confusing downstream errors if invalid.

        Parameters
        ----------
        stride : int
            Trajectory stride; must be a positive integer.
        chunk_size : int
            Processing chunk size; must be a positive integer.

        Returns
        -------
        None
            Raises ValueError for invalid inputs.
        """
        if stride <= 0 and not isinstance(stride, int):
            raise ValueError("Stride must be a positive integer.")
        if chunk_size <= 0 and not isinstance(chunk_size, int):
            raise ValueError("Chunk size must be a positive integer.")

    def _init_performance_config(
        self,
        use_memmap: bool,
        use_stability_config: Optional[bool],
    ) -> None:
        """
        Build performance configuration and apply stability policy if requested.

        The configuration is stored on pipeline.config.performance. Changes
        to any of its fields automatically trigger re-application of resource
        limits through _apply_performance_config.

        Parameters
        ----------
        use_memmap : bool
            Whether this pipeline is configured to use memmaps. When
            use_stability_config is None, this determines whether stability
            settings are applied.
        use_stability_config : bool or None
            When True, overwrite the baseline (OS-default) configuration with
            stability-oriented values and apply them immediately. When False,
            leave the baseline defaults and do not apply any limits. When
            None, apply stability values only if use_memmap is True.

        Returns
        -------
        None
            Initializes self.config.performance and self.resource_limits.
        """
        self.resource_limits = {"errors": []}
        defaults = self._default_performance_config()
        perf = PerformanceConfig(defaults=defaults, on_change=self._apply_performance_config)
        self.config = SimpleNamespace(performance=perf)

        resolved_stability = use_stability_config
        if resolved_stability is None:
            resolved_stability = use_memmap

        if resolved_stability:
            stability = self._stability_performance_overrides()
            self.config.performance.update(**stability)

    def _default_performance_config(self) -> Dict[str, Any]:
        """
        Return baseline performance settings that preserve OS defaults.

        These defaults are intentionally neutral: they do not change CPU
        priority, I/O priority, affinity, or BLAS/OpenMP threading. They are
        still stored on pipeline.config.performance so users can override them.

        Returns
        -------
        dict
            Dictionary of baseline performance settings.
        """
        return {
            "auto_resource_limits": False,
            "reserve_cores": 0,
            "resource_nice": None,
            "resource_io_priority": None,
            "resource_cpu_affinity": None,
            "auto_blas_thread_limit": False,
        }

    def _stability_performance_overrides(self) -> Dict[str, Any]:
        """
        Return stability-oriented performance settings.

        These values favor responsiveness and avoid hard system stalls during
        large sequential I/O workloads. They are applied only when
        use_stability_config is enabled.

        Returns
        -------
        dict
            Dictionary of stability-focused overrides.
        """
        return {
            "auto_resource_limits": True,
            "reserve_cores": 2,
            "resource_nice": 15,
            "resource_io_priority": "low",
            "resource_cpu_affinity": None,
            "auto_blas_thread_limit": True,
        }

    def _apply_performance_config(self) -> None:
        """
        Apply process-level limits using the current performance configuration.

        This method is invoked automatically whenever pipeline.config.performance
        changes. It applies process nice value, I/O priority, CPU affinity, and
        BLAS/OpenMP thread caps. All settings are best-effort and may be ignored
        by the OS or restricted by the current scheduler allocation.

        Returns
        -------
        None
            Updates self.resource_limits with applied values and warnings.
        """
        perf = self.config.performance
        cpu_affinity = perf.resource_cpu_affinity

        if perf.auto_resource_limits and cpu_affinity is None:
            cpu_affinity = ResourceUtils.recommend_cpu_affinity(
                reserve_cores=perf.reserve_cores
            )

        should_apply = (
            perf.auto_resource_limits
            or cpu_affinity is not None
            or perf.resource_nice is not None
            or perf.resource_io_priority is not None
        )

        self.resource_limits = {"errors": []}
        if should_apply:
            self.resource_limits = ResourceUtils.apply_process_limits(
                nice=perf.resource_nice,
                io_priority=perf.resource_io_priority,
                cpu_affinity=cpu_affinity,
            )

        if perf.auto_blas_thread_limit:
            max_threads = None
            if cpu_affinity:
                max_threads = len(cpu_affinity)
            else:
                allowed = ResourceUtils.recommend_cpu_affinity(reserve_cores=0)
                if allowed:
                    max_threads = len(allowed)
            if max_threads is not None:
                self.resource_limits["blas"] = ResourceUtils.apply_blas_thread_limits(
                    max_threads
                )
        else:
            self.resource_limits["blas"] = ResourceUtils.apply_blas_thread_limits(None)

    def _init_data(
        self,
        use_memmap: bool,
        cache_dir: str,
        chunk_size: int,
        dtype: type,
        max_memory_gb: float,
    ) -> None:
        """
        Initialize the central PipelineData container.

        Parameters
        ----------
        use_memmap : bool
            Whether to store feature and decomposition data as memmaps.
        cache_dir : str
            Cache directory for memmaps and intermediate artifacts.
        chunk_size : int
            Chunk size used by downstream managers and helpers.
        dtype : type
            Data type for feature matrices.
        max_memory_gb : float
            Maximum memory budget for memory-aware sampling.

        Returns
        -------
        None
            Sets self._data.
        """
        self._data = PipelineData(
            use_memmap=use_memmap,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            dtype=dtype,
            max_memory_gb=max_memory_gb,
        )

    def _init_managers(
        self,
        stride: int,
        concat: bool,
        selection: Optional[str],
        cache_dir: str,
        use_memmap: bool,
        chunk_size: int,
    ) -> None:
        """
        Initialize manager instances with the shared pipeline configuration.

        Parameters
        ----------
        stride : int
            Trajectory stride used by the TrajectoryManager.
        concat : bool
            Concatenation flag for trajectory loading.
        selection : str, optional
            MDTraj selection string for trajectory loading.
        cache_dir : str
            Cache directory for managers that persist data.
        use_memmap : bool
            Whether managers should create memmaps for large matrices.
        chunk_size : int
            Shared chunk size for chunked processing.

        Returns
        -------
        None
            Creates manager instances on the PipelineManager.
        """
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
        self._feature_selector_manager = FeatureSelectorManager(
            use_memmap=use_memmap, chunk_size=chunk_size, cache_dir=cache_dir
        )

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

    def report_resource_limits(self) -> None:
        """
        Report applied resource limits to stdout when progress output is enabled.

        This method is public to let users re-print the current limits after
        changing settings at runtime. It is intentionally lightweight and
        avoids strong formatting guarantees; the output is meant for human
        inspection rather than machine parsing.

        Returns
        -------
        None
            Prints a one-line summary plus any warnings when applicable.
        """
        if not getattr(self, "show_progress", True):
            return
        if not getattr(self, "resource_limits", None):
            return

        parts = []
        affinity = self.resource_limits.get("cpu_affinity")
        if affinity:
            parts.append(f"cpu_affinity={len(affinity)} cores")
        if self.resource_limits.get("nice") is not None:
            parts.append(f"nice={self.resource_limits['nice']}")
        if self.resource_limits.get("io_priority") is not None:
            parts.append(f"io_priority={self.resource_limits['io_priority']}")
        blas = self.resource_limits.get("blas")
        if isinstance(blas, dict) and blas.get("max_threads") is not None:
            parts.append(f"blas_threads={blas['max_threads']}")

        if parts:
            print("Resource limits applied: " + ", ".join(parts))

        errors = []
        errors.extend(self.resource_limits.get("errors") or [])
        if isinstance(blas, dict):
            errors.extend(blas.get("errors") or [])
        if errors:
            print("Resource limit warnings: " + "; ".join(errors))

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

    def _apply_show_progress(self, show_progress: bool) -> None:
        """
        Apply progress display setting to environment and managers.

        Updates the progress controller and propagates the flag to all managers.
        Can be invoked at runtime to enable or disable progress bars without
        recreating the pipeline.

        Parameters
        ----------
        show_progress : bool
            Flag to enable (True) or disable (False) progress bars.

        Returns
        -------
        None
            Progress setting is applied to controller and managers.
        """
        self.show_progress = show_progress
        ProgressUtils.set_enabled(show_progress)

        for manager in (
            self._trajectory_manager,
            self._feature_manager,
            self._cluster_manager,
            self._decomposition_manager,
            self._feature_selector_manager,
            self._data_selector_manager,
            self._comparison_manager,
            self._feature_importance_manager,
            self._analysis_manager,
            self._plots_manager,
            self._structure_visualization_manager,
        ):
            setattr(manager, "show_progress", show_progress)

    def set_show_progress(self, enabled: bool) -> None:
        """
        Public helper to toggle progress bars at runtime.

        Parameters
        ----------
        enabled : bool
            Desired progress bar state. True enables bars, False suppresses them.

        Returns
        -------
        None
            Updates the progress controller for subsequent operations.

        Examples
        --------
        >>> pipeline = PipelineManager(show_progress=True)
        >>> pipeline.set_show_progress(False)
        """
        self._apply_show_progress(enabled)

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

    def add_custom_metadata(
        self,
        name: str,
        value: Any,
        overwrite: bool = False,
        warn_if_large: bool = True,
        max_size_gb: Optional[float] = None,
    ) -> None:
        """
        Register user-defined custom metadata in the pipeline state.

        Parameters
        ----------
        name : str
            Metadata key.
        value : Any
            Metadata payload to persist with the pipeline.
        overwrite : bool, default=False
            If False, existing keys raise ValueError.
        warn_if_large : bool, default=True
            Emit RuntimeWarning when the estimated object size exceeds the
            configured threshold.
        max_size_gb : float, optional
            Explicit warning threshold in GB.
            If None, uses ``pipeline.data.max_memory_gb``.

        Returns
        -------
        None
            Stores metadata in-place.

        Notes
        -----
        TODO: Add optional disk-backed backend (e.g. zarr/proxy) for large
        nested metadata payloads. Current implementation keeps metadata in RAM.
        """
        # TODO: Add optional disk-backed custom-metadata backends (e.g. zarr).
        self._data.add_custom_metadata(name=name, value=value, overwrite=overwrite)

        if warn_if_large:
            self._warn_if_custom_metadata_large(
                name=name,
                value=value,
                max_size_gb=max_size_gb
            )

    def get_custom_metadata(self, name: str) -> Any:
        """
        Retrieve a registered custom metadata payload by key.

        Parameters
        ----------
        name : str
            Metadata key.

        Returns
        -------
        Any
            Stored metadata payload.
        """
        return self._data.get_custom_metadata(name)

    def clear_all(self) -> None:
        """
        Clear all pipeline data.

        Resets the entire pipeline to empty state, clearing all
        trajectories, features, clustering, and decomposition results.
        """
        self._data.clear_all_data()

    def _warn_if_custom_metadata_large(
        self,
        name: str,
        value: Any,
        max_size_gb: Optional[float]
    ) -> None:
        """
        Emit a warning when custom metadata size exceeds the configured budget.

        Parameters
        ----------
        name : str
            Metadata key.
        value : Any
            Metadata payload.
        max_size_gb : Optional[float]
            Maximum allowed size in GB. If None, uses ``pipeline.data.max_memory_gb``.
            
        Returns
        -------
        None
        """
        if max_size_gb is not None:
            if isinstance(max_size_gb, bool) or not isinstance(max_size_gb, (int, float)):
                raise ValueError("max_size_gb must be a positive numeric value.")
        threshold_gb = self._data.max_memory_gb if max_size_gb is None else max_size_gb
        if threshold_gb <= 0:
            raise ValueError("max_size_gb must be > 0 when provided.")

        estimated_bytes = self._estimate_object_size_bytes(value, visited=set())
        threshold_bytes = int(threshold_gb * (1024 ** 3))
        if estimated_bytes <= threshold_bytes:
            return

        estimated_gb = estimated_bytes / (1024 ** 3)
        warnings.warn(
            (
                f"Custom metadata '{name}' is estimated at {estimated_gb:.3f} GB, "
                f"which exceeds the warning threshold of {threshold_gb:.3f} GB. "
                "This payload is currently kept in RAM and can increase memory "
                "pressure. TODO: disk-backed metadata backend support will be "
                "added in a future release."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _estimate_object_size_bytes(value: Any, visited: set) -> int:
        """
        Recursively estimate object size in bytes.

        Parameters
        ----------
        value : Any
            Object to estimate size of.
        visited : set
            Set of visited object IDs to avoid cycles.

        Returns
        -------
        int
            Estimated size in bytes.
        """
        value_id = id(value)
        if value_id in visited:
            return 0
        visited.add(value_id)

        if isinstance(value, np.ndarray):
            return int(value.nbytes)

        if isinstance(value, (str, bytes, bytearray, memoryview)):
            return sys.getsizeof(value)

        if isinstance(value, (int, float, bool, complex, np.generic, type(None))):
            return sys.getsizeof(value)

        size = sys.getsizeof(value)

        if isinstance(value, dict):
            for key, item in value.items():
                size += PipelineManager._estimate_object_size_bytes(key, visited)
                size += PipelineManager._estimate_object_size_bytes(item, visited)
            return size

        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                size += PipelineManager._estimate_object_size_bytes(item, visited)
            return size

        if hasattr(value, "__dict__"):
            size += PipelineManager._estimate_object_size_bytes(vars(value), visited)
            return size

        return size

    def close(self) -> None:
        """
        Release in-memory resources and memmap handles owned by this pipeline.

        This method closes memmap-backed arrays and trajectory runtime caches
        without deleting cache files on disk. It is safe to call multiple times.

        Returns
        -------
        None
            Frees resources and detaches in-memory references.
        """
        if getattr(self, "_closed", False):
            return

        if not hasattr(self, "_data"):
            self._closed = True
            return

        # Release trajectory runtime caches (e.g., DaskMDTrajectory caches).
        for trajectory in getattr(self._data.trajectory_data, "trajectories", []):
            cleanup = getattr(trajectory, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass

        # Release feature memmaps and analysis binding cycles.
        for feature_traj_dict in self._data.feature_data.values():
            for feature_data in feature_traj_dict.values():
                FeatureBindingHelper.release_bound_methods(feature_data)
                MemmapUtils.close_memmap_view(feature_data.data)
                MemmapUtils.close_memmap_view(feature_data.reduced_data)
                if MemmapUtils.is_memmap_view(feature_data.data):
                    feature_data.data = None
                if MemmapUtils.is_memmap_view(feature_data.reduced_data):
                    feature_data.reduced_data = None

        # Release decomposition memmaps.
        for decomposition_data in self._data.decomposition_data.values():
            MemmapUtils.close_memmap_view(decomposition_data.data)
            if MemmapUtils.is_memmap_view(decomposition_data.data):
                decomposition_data.data = None

        # Release clustering memmaps (labels and optional centers).
        for cluster_data in self._data.cluster_data.values():
            MemmapUtils.close_memmap_view(cluster_data.labels)
            if MemmapUtils.is_memmap_view(cluster_data.labels):
                cluster_data.labels = None
            if isinstance(cluster_data.metadata, dict):
                centers = cluster_data.metadata.get("centers")
                MemmapUtils.close_memmap_view(centers)
                if MemmapUtils.is_memmap_view(centers):
                    cluster_data.metadata["centers"] = None

        # Release feature-importance memmaps if present.
        for feature_importance_data in self._data.feature_importance_data.values():
            for idx, scores in enumerate(feature_importance_data.data):
                MemmapUtils.close_memmap_view(scores)
                if MemmapUtils.is_memmap_view(scores):
                    feature_importance_data.data[idx] = None

        # Clear selector matrix cache metadata (files remain on disk).
        self._data.clear_matrix_cache()

        gc.collect()
        self._closed = True

    def __del__(self) -> None:
        """
        Best-effort fallback cleanup when the pipeline object is garbage-collected.
        """
        try:
            self.close()
        except Exception:
            pass

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
        save_path = PathUtils.prepare_file_path(
            save_path,
            create_parent=True,
            purpose="save path",
        )
        self._data.save(save_path)

    @staticmethod
    def load_from_single_file(
        load_path: str,
        cache_dir: Optional[str] = "./cache",
        chunk_size: int = 1000,
        stride: int = 1,
        concat: bool = False,
        selection: Optional[str] = None,
        show_progress: bool = True,
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
        cache_dir : str, optional
            Fallback cache directory used only when saved cache metadata
            is unavailable. By default, the cache scope stored in the
            single-file payload is reused to preserve pipeline identity.
        chunk_size : int, default=1000
            Default chunk size for future operations
        stride : int, default=1
            Default stride for future trajectory loading
        concat : bool, default=False
            Default concat mode for future trajectory loading
        selection : str, optional
            Default MDTraj selection string for trajectories
        show_progress : bool, default=True
            Enable or disable progress bars globally (tqdm)

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
        load_path = PathUtils.prepare_file_path(
            load_path,
            create_parent=False,
            purpose="load path",
        )

        saved_cache_dir = LoadAndSaveHelper.peek_cache_dir(load_path)

        target_cache_dir = saved_cache_dir or cache_dir or "./cache"
        target_cache_dir = PathUtils.prepare_directory_path(
            target_cache_dir,
            create=True,
            purpose="cache directory",
        )

        pipeline = PipelineManager(
            cache_dir=target_cache_dir,
            chunk_size=chunk_size,
            stride=stride,
            concat=concat,
            selection=selection,
            show_progress=show_progress,
        )
        pipeline._data.load(load_path)

        loaded_cache_dir = PathUtils.prepare_directory_path(
            pipeline._data.cache_dir,
            create=True,
            purpose="cache directory",
        )
        pipeline._data.cache_dir = loaded_cache_dir
        pipeline._trajectory_manager.cache_dir = loaded_cache_dir
        pipeline._feature_manager.cache_dir = loaded_cache_dir
        pipeline._decomposition_manager.cache_dir = loaded_cache_dir
        pipeline._cluster_manager.cache_dir = loaded_cache_dir
        pipeline._feature_importance_manager.cache_dir = loaded_cache_dir
        pipeline._plots_manager.cache_dir = loaded_cache_dir
        pipeline._structure_visualization_manager.cache_dir = loaded_cache_dir

        scoped_name = os.path.basename(os.path.normpath(loaded_cache_dir))
        scope_match = re.fullmatch(
            r"cache_([0-9a-f]{32})_(\d{8}_\d{6})",
            scoped_name,
        )
        if scope_match is not None:
            pipeline._pipeline_uuid = scope_match.group(1)
            pipeline._pipeline_timestamp = scope_match.group(2)

        return pipeline

    def create_sharable_archive(
        self,
        archive_path: str,
        compression: str = "zst",
        exclude_visualizations: bool = True,
        include_structure_files: bool = True,
        compression_level: Optional[int] = None,
        zstd_threads: Optional[int] = None,
        zstd_reserve_cores: int = 2,
        sha: Union[bool, str] = True,
        overwrite: bool = False,
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
        compression : str, default="zst"
            Compression method: "zst", "bz2", or "gz"
        exclude_visualizations : bool, default=True
            If True, exclude PNG/PDF/SVG plot outputs
        include_structure_files : bool, default=True
            If True, include PDB/PML structure files
        compression_level : int, optional
            Compression level override (e.g. zst level 1-19).
        zstd_threads : int, optional
            Thread count for zstd compression. If None, chosen automatically.
        zstd_reserve_cores : int, default=2
            Number of CPU cores to keep free when zstd thread count is automatic.
        sha : bool or str, default=True
            If True, write ``<archive>.sha`` next to the created archive.
            When a string is provided, it is used as the explicit SHA256
            output path.
        overwrite : bool, default=False
            If True, replace existing archive outputs. When False, existing
            archive or SHA256 outputs raise ``FileExistsError``.

        Returns
        -------
        str
            Path to created archive file

        Examples
        --------
        >>> # Minimal archive (only data)
        >>> pipeline.create_sharable_archive("analysis.tar.zst")

        >>> # Full archive (with visualizations)
        >>> pipeline.create_sharable_archive(
        ...     "analysis_full.tar.zst",
        ...     exclude_visualizations=False
        ... )

        >>> # Data only (no structure files)
        >>> pipeline.create_sharable_archive(
        ...     "analysis_data.tar.zst",
        ...     include_structure_files=False
        ... )

        >>> pipeline.create_sharable_archive(
        ...     "analysis.tar.zst",
        ...     sha="checksums/analysis.sha"
        ... )

        Notes
        -----
        - zstd compression is optimized for fast runtime and low memory pressure
        - zstd compression is multithreaded by default
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
            include_structure_files,
            compression_level=compression_level,
            zstd_threads=zstd_threads if compression == "zst" else None,
            reserve_cores=zstd_reserve_cores if compression == "zst" else 2,
            sha=sha,
            overwrite=overwrite,
        )

    @staticmethod
    def load_from_archive(
        file_path: str,
        cache_dir: str = "./cache",
        verify: bool = True,
        sha: Optional[str] = None,
        download_url: Optional[str] = None,
        overwrite: bool = False,
        chunk_size: int = 1000,
        stride: int = 1,
        concat: bool = False,
        selection: Optional[str] = None,
        show_progress: bool = True,
    ) -> PipelineManager:
        """
        Load pipeline from sharable archive.

        Extracts compressed archive, moves cache files to specified
        cache directory, and loads pipeline state. Automatically
        repairs memmap file paths to point to new cache location.

        Parameters
        ----------
        file_path : str
            Local archive path, local download target, or remote archive URL.
        cache_dir : str, default="./cache"
            Target cache directory for extracted files
        verify : bool, default=True
            Whether to validate the archive via SHA256 before loading.
            When ``sha`` is provided, SHA256 verification is performed even if
            ``verify`` is False.
        sha : str, optional
            SHA256 input used for archive verification. May be provided as a
            raw SHA256 hex string, a local path to a ``.sha`` file, or a URL.
            When verification is enabled and ``sha`` is missing, loading fails.
        download_url : str, optional
            Remote source URL used when ``file_path`` should act as the local
            archive target. When omitted and ``file_path`` is itself a URL,
            downloads are stored under ``<cache_dir>/downloads/``.
        overwrite : bool, default=False
            For remote URLs, controls whether an existing local target file is
            replaced. When False, an existing downloaded file is reused and a
            warning is emitted.
        chunk_size : int, default=1000
            Default chunk size for future operations
        stride : int, default=1
            Default stride for future trajectory loading
        concat : bool, default=False
            Default concatenation setting for future trajectory loading
        selection : str, optional
            Default MDTraj selection string for trajectories
        show_progress : bool, default=True
            Enable or disable progress bars globally (tqdm)

        Returns
        -------
        PipelineManager
            Loaded pipeline instance

        Examples
        --------
        >>> # Load a remote archive with explicit verification disabled
        >>> pipeline = PipelineManager.load_from_archive(
        ...     "local_analysis.tar.zst",
        ...     cache_dir="./my_cache",
        ...     download_url="https://example.org/analysis.tar.zst",
        ...     verify=False
        ... )

        >>> # Load with SHA256 verification from a sidecar file
        >>> pipeline = PipelineManager.load_from_archive(
        ...     "analysis.tar.zst",
        ...     verify=True,
        ...     sha="analysis.tar.zst.sha"
        ... )

        Notes
        -----
        - Extracts to temporary directory
        - Moves cache files to specified cache_dir
        - Automatically repairs memmap paths for portability
        - Cache directory created if it doesn't exist
        """
        cache_dir = PathUtils.prepare_directory_path(
            cache_dir,
            create=True,
            purpose="cache directory",
        )
        archive_path = ArchiveFetchHelper.resolve_archive_path(
            file_path=file_path,
            cache_dir=cache_dir,
            verify=verify,
            sha=sha,
            download_url=download_url,
            overwrite=overwrite,
        )
        if ArchiveFetchHelper.should_verify_archive(verify=verify, sha=sha):
            ArchiveFetchHelper.verify_archive_sha256(
                file_path=archive_path,
                sha=sha,
            )

        # Create loaded pipeline first to obtain a fresh scoped runtime cache dir.
        pipeline = PipelineManager(
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            stride=stride,
            concat=concat,
            selection=selection,
            show_progress=show_progress,
        )
        runtime_cache_dir = pipeline.get_config()["cache_dir"]

        # Extract into a temporary directory to avoid clashes with existing cache
        temp_extract_dir = tempfile.mkdtemp(prefix="mdxplain_archive_")
        extract_dir = ArchiveUtils.extract_archive(
            archive_path, extract_to=temp_extract_dir
        )

        pkl_path = extract_dir / "pipeline.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"pipeline.pkl not found in extracted archive"
            )

        # Move cache files to target cache_dir
        extracted_cache = extract_dir / "cache"
        target_cache = Path(runtime_cache_dir)
        target_cache.mkdir(parents=True, exist_ok=True)

        if extracted_cache.exists():
            for item in extracted_cache.iterdir():
                target_item = target_cache / item.name
                if target_item.exists():
                    CleanupUtils.remove_path(target_item, purpose="cache path")
                shutil.move(str(item), str(target_item))

        pipeline._data.load(str(pkl_path))

        # Force update cache_dir to match extracted file location
        pipeline._data.cache_dir = runtime_cache_dir

        # Update all manager cache_dirs for consistency
        pipeline._trajectory_manager.cache_dir = runtime_cache_dir
        pipeline._feature_manager.cache_dir = runtime_cache_dir
        pipeline._decomposition_manager.cache_dir = runtime_cache_dir
        pipeline._cluster_manager.cache_dir = runtime_cache_dir
        pipeline._feature_importance_manager.cache_dir = runtime_cache_dir
        pipeline._plots_manager.cache_dir = runtime_cache_dir
        pipeline._structure_visualization_manager.cache_dir = runtime_cache_dir

        CacheRemapHelper.remap_pipeline_memmaps(
            pipeline._data,
            runtime_cache_dir=runtime_cache_dir,
        )

        # Cleanup extracted temporary directory
        CleanupUtils.remove_tree(
            extract_dir,
            ignore_errors=True,
            purpose="temporary extraction directory",
        )

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
            f"{summary['feature_importance_analyses']} feature importance analyses, "
            f"{summary['custom_metadata_entries']} custom metadata entries"
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
            cache_dir = PathUtils.create_pipeline_cache_dir(
                cache_dir,
                pipeline_uuid=self._pipeline_uuid,
                pipeline_timestamp=self._pipeline_timestamp,
                purpose="cache directory",
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

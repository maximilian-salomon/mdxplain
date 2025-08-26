# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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
FeatureManager is a manager for feature data objects.

It is used to add, reset, and reduce features to the trajectory data.
"""

import os
from tqdm import tqdm

from ..entities.feature_data import FeatureData
from ..helper.feature_validation_helper import FeatureValidationHelper
from ..helper.feature_reset_helper import FeatureResetHelper
from ..helper.feature_reduction_helper import FeatureReductionHelper
from ..helper.feature_computation_helper import FeatureComputationHelper
from ..helper.feature_binding_helper import FeatureBindingHelper
from ...utils.data_utils import DataUtils


class FeatureManager:
    """Manager for feature data objects."""

    def __init__(self, use_memmap=False, chunk_size=10000, cache_dir="./cache"):
        """
        Initialize feature manager.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for feature data
        chunk_size : int, optional
            Processing chunk size
        cache_dir : str, default="./cache"
            Cache path for feature data

        Returns:
        --------
        None
            Initializes FeatureManager instance with specified configuration
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        if chunk_size <= 0 and not isinstance(chunk_size, int):
            raise ValueError("Chunk size must be a positive integer.")

    def reset_features(self, pipeline_data, feature_type=None, strict=True):
        """
        Reset calculated features and clear feature data.

        This method removes computed features and their associated data,
        requiring features to be recalculated from scratch. Use this method
        after trajectory modifications that invalidate existing features.
        Can reset all features or specific feature types.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature.reset_features()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureManager()
        >>> manager.reset_features(pipeline_data)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_type : str, FeatureTypeBase, list, or None, default=None
            Feature type(s) to reset. If None, resets all features.
            Supports following formats:
            - "distances" (string)
            - feature_type.Distances() (instance)
            - feature_type.Distances (class)
            - ["distances", "contacts"] (list of any of above)
        strict : bool, default=False
            Whether to raise ValueError if feature_type doesn't exist.
            If False, non-existent features are silently ignored with warning.

        Returns:
        --------
        None
            Clears specified feature data from pipeline_data.feature_data in-place

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> feature_manager = FeatureManager()
        >>> feature_manager.add_feature(pipeline_data, feature_type.Distances())
        >>> feature_manager.add_feature(pipeline_data, feature_type.Contacts())

        >>> # Reset all features
        >>> feature_manager.reset_features(pipeline_data)

        >>> # Reset specific feature type
        >>> feature_manager.reset_features(pipeline_data, "distances")

        >>> # Reset multiple feature types
        >>> feature_manager.reset_features(pipeline_data, ["distances", "contacts"])

        >>> # Strict mode - raise error for non-existent features
        >>> feature_manager.reset_features(pipeline_data, "nonexistent", strict=True)  # Raises ValueError

        Notes:
        -----
        - Selected feature data is permanently deleted
        - Memory-mapped feature files remain on disk but are no longer referenced
        - Features must be recalculated after reset
        """
        if not FeatureResetHelper.check_has_features(pipeline_data):
            print("No features to reset.")
            return

        if feature_type is None:
            FeatureResetHelper.reset_all_features(pipeline_data)
        else:
            feature_types = FeatureResetHelper.normalize_feature_types(feature_type)
            FeatureResetHelper.reset_specific_features(
                pipeline_data, feature_types, strict
            )

    def add_feature(
        self, 
        pipeline_data, 
        feature_type, 
        traj_selection="all",
        force=False, 
        force_original=True,
    ):
        """
        Add and compute a feature for the loaded trajectories.

        This method creates a FeatureData instance for the specified feature type,
        handles dependency checking, and computes the feature data. Features with
        dependencies (like Contacts depending on Distances) will automatically
        use the required input data.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature.add_feature(feature_type.Distances())  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureManager()
        >>> manager.add_feature(pipeline_data, feature_type.Distances())  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_type : FeatureTypeBase
            Feature type instance(e.g., Distances(), Contacts()).
            The feature type determines what kind of analysis will be performed.
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to compute features for:
            - int: trajectory index
            - str: trajectory name, tag (prefixed with "tag:"), or "all"
            - list: list of indices/names/tags
            - "all": all trajectories (default)
        force : bool, default=False
            Whether to force recomputation of the feature even if it already exists.
        force_original : bool, default=True
            Whether to force using the original data as base for the calculation for
            features using other features as input instead of the reduced data

        Returns:
        --------
        None
            Adds computed feature to pipeline data and creates analysis methods

        Raises:
        -------
        ValueError
            If the feature already exists with computed data, if required
            dependencies are missing, or if trajectories are not loaded.

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> feature_manager = FeatureManager()
        >>> feature_manager.add_feature(pipeline_data, feature_type.Distances())
        >>> feature_manager.add_feature(pipeline_data, feature_type.Contacts())
        """
        # Get trajectory indices to process (same logic as cut_traj)
        traj_indices = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)
        
        # Check that ALL trajectories in selection have labels (res_label_data uses int keys)
        missing_labels = [traj_idx for traj_idx in traj_indices 
                         if traj_idx not in pipeline_data.trajectory_data.res_label_data]
        if missing_labels:
            raise ValueError(
                f"Trajectories {missing_labels} in selection {traj_selection} have no labels set. "
                "Use TrajectoryManager.add_labels() to set labels for selected trajectories."
            )

        feature_key = DataUtils.get_type_key(feature_type)

        FeatureComputationHelper.check_feature_existence(
            pipeline_data, feature_key, traj_indices, force
        )
        FeatureValidationHelper.validate_dependencies(
            pipeline_data, feature_type, feature_key, traj_indices
        )

        # Validate computation requirements
        FeatureValidationHelper.validate_computation_requirements(
            pipeline_data, feature_type
        )
        
        # Create FeatureData objects per trajectory
        self._create_feature_data_per_trajectory(
            pipeline_data, feature_type, feature_key, traj_indices, force_original
        )

    def reset_reduction(self, pipeline_data, feature_type):
        """
        Reset to using full original data instead of reduced dataset.

        Supports all three input variants:
        - feature_type.Distances() (instance)
        - feature_type.Distances (class with metaclass)
        - "distances" (string)

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature.reset_reduction("distances")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureManager()
        >>> manager.reset_reduction(pipeline_data, "distances")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_type : FeatureTypeBase, FeatureTypeBase class, or str
            Feature type instance, class, or string
            E.g. Distances(), Distances, "distances"

        Returns:
        --------
        None
            Clears reduced_data and prints confirmation with shape info

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> feature_manager = FeatureManager()
        >>> feature_manager.add_feature(pipeline_data, feature_type.Distances())
        >>> feature_manager.reduce_data(pipeline_data, feature_type.Distances,
                                        metric="cv", threshold_min=0.1)
        >>> feature_manager.reset_reduction(pipeline_data, "distances")  # string
        """
        feature_key = DataUtils.get_type_key(feature_type)
        FeatureReductionHelper.reset_reduction(pipeline_data, feature_key)

    def reduce_data(
        self,
        pipeline_data,
        feature_type,
        metric,
        traj_selection="all",
        threshold_min=None,
        threshold_max=None,
        transition_threshold=2.0,
        window_size=10,
        transition_mode="window",
        lag_time=1,
    ):
        """
        Filter features based on statistical criteria.

        Supports all three input variants:
        - feature_type.Distances() (instance)
        - feature_type.Distances (class with metaclass)
        - "distances" (string)

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature.reduce_data("distances", metric="cv")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureManager()
        >>> manager.reduce_data(pipeline_data, "distances", metric="cv")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_type : FeatureTypeBase, FeatureTypeBase class, or str
            Feature type instance, class, or string
            E.g. Distances(), Distances, "distances"
        metric : str or ReduceMetrics
            Statistical filtering metric ('cv', 'frequency', 'transitions', etc.)
        threshold_min : float, optional
            Minimum value to keep (features >= threshold_min)
        threshold_max : float, optional
            Maximum value to keep (features <= threshold_max)
        transition_threshold : float, default=2.0
            Distance change threshold for transition detection (Angstrom)
        window_size : int, default=10
            Number of frames for transition window analysis
        transition_mode : str, default="window"
            Mode for transition detection ('window', 'lag')
        lag_time : int, default=1
            Lag time for transition detection

        Returns:
        --------
        None
            Stores filtered data in self.reduced_data and prints reduction summary

        Raises:
        -------
        ValueError
            If the feature has no data, if the reduction has already been performed,
            or if threshold parameters are invalid (threshold_min > threshold_max)

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> feature_manager = FeatureManager()
        >>> feature_manager.add_feature(pipeline_data, feature_type.Distances())  # instance
        >>> feature_manager.reduce_data(pipeline_data, feature_type.Distances,   # class
                                        metric="cv", threshold_min=0.1)
        """
        feature_key = DataUtils.get_type_key(feature_type)
        FeatureValidationHelper.validate_reduction_state(pipeline_data, feature_key)
        FeatureValidationHelper.validate_threshold_parameters(
            threshold_min, threshold_max, metric
        )

        # Get reduction results from calculator
        results = pipeline_data.feature_data[
            feature_key
        ].feature_type.calculator.compute_dynamic_values(
            input_data=pipeline_data.feature_data[feature_key].data,
            metric=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            feature_metadata=pipeline_data.feature_data[feature_key].feature_metadata[
                "features"
            ],
            output_path=pipeline_data.feature_data[feature_key].reduced_cache_path,
            transition_mode=transition_mode,
            lag_time=lag_time,
        )

        # Process and store results
        FeatureReductionHelper.process_reduction_results(
            pipeline_data, feature_key, results, threshold_min, threshold_max, metric
        )

    def _create_feature_data_per_trajectory(
        self, pipeline_data, feature_type, feature_key, traj_indices, force_original
    ):
        """
        Create FeatureData objects for specified trajectories.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_type : FeatureTypeBase
            Feature type to compute
        feature_key : str
            Feature key for storage
        traj_indices : list
            List of trajectory indices
        force_original : bool
            Whether to force original data for dependent features
            
        Returns:
        --------
        None
            Creates and stores FeatureData objects in pipeline_data
        """
        # Initialize 2-level dict structure
        if feature_key not in pipeline_data.feature_data:
            pipeline_data.feature_data[feature_key] = {}

        # Create FeatureData per trajectory with progress bar        
        with tqdm(total=len(traj_indices), desc=f"Computing {feature_type.get_type_name()}", unit="trajectories") as pbar:
            for traj_idx in traj_indices:
                trajectory_name = pipeline_data.trajectory_data.trajectory_names[traj_idx]
                
                # Create FeatureData for this trajectory
                feature_data = FeatureData(
                    feature_type=feature_type,
                    use_memmap=self.use_memmap,
                    cache_path=self.cache_dir,
                    chunk_size=self.chunk_size,
                    trajectory_name=trajectory_name,
                )

                # Compute feature for single trajectory
                self._compute_and_store_single_trajectory(
                    pipeline_data, feature_data, feature_type, traj_idx, force_original
                )

                # Store the feature data
                pipeline_data.feature_data[feature_key][traj_idx] = feature_data
                
                # Update progress
                pbar.update(1)

    def _compute_and_store_single_trajectory(
        self, pipeline_data, feature_data, feature_type, traj_idx, force_original
    ):
        """
        Compute and store feature for single trajectory.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_data : FeatureData
            Feature data object for this trajectory
        feature_type : FeatureTypeBase
            Feature type to compute
        traj_idx : int
            Trajectory index
        force_original : bool
            Whether to force original data for dependent features
            
        Returns:
        --------
        None
            Computes and stores data in feature_data
        """
        # Compute using helper for single trajectory
        single_data, single_metadata = FeatureComputationHelper.execute_computation(
            pipeline_data, feature_data, feature_type, traj_idx, force_original
        )
        
        # Store results and bind methods
        FeatureComputationHelper.store_computation_results(
            feature_data, single_data, single_metadata
        )
        FeatureBindingHelper.bind_stats_methods(feature_data)


# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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

import functools
from typing import Any

import numpy as np

from ..entities.feature_data import FeatureData


class FeatureManager:
    """Manager for feature data objects."""

    def __init__(self, use_memmap=False, chunk_size=None, cache_dir=None):
        """
        Initialize feature manager.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for feature data
        chunk_size : int, optional
            Processing chunk size
        cache_dir : str, optional
            Cache path for feature data

        Returns:
        --------
        None
            Initializes FeatureManager instance with specified configuration
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

        self.cache_dir = cache_dir
        if use_memmap and cache_dir is None:
            self.cache_dir = "./cache"

    def reset_features(self, traj_data):
        """
        Reset all calculated features and clear feature data.

        This method removes all computed features and their associated data,
        requiring features to be recalculated from scratch. Use this method
        after trajectory modifications that invalidate existing features.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        None
            Clears all feature data from traj_data.features in-place

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> feature_manager = FeatureManager()
        >>> traj_data.add_feature(feature_type.Distances())
        >>> feature_manager.reset_features(traj_data)

        Notes:
        -----
        - All feature data is permanently deleted
        - Memory-mapped feature files remain on disk but are no longer referenced
        - Features must be recalculated after reset
        """
        if not traj_data.features:
            print("No features to reset.")
            return

        feature_list = list(traj_data.features.keys())
        traj_data.features.clear()

        print(f"Reset {len(feature_list)} feature(s): " f"{', '.join(feature_list)}")
        print("All feature data has been cleared. Features must be recalculated.")

    def add_feature(self, traj_data, feature_type, force=False):
        """
        Add and compute a feature for the loaded trajectories.

        This method creates a FeatureData instance for the specified feature type,
        handles dependency checking, and computes the feature data. Features with
        dependencies (like Contacts depending on Distances) will automatically
        use the required input data.

        Supports all three input variants:
        - feature_type.Distances() (instance)
        - feature_type.Distances (class with metaclass)
        - "distances" (string)

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_type : FeatureTypeBase, FeatureTypeBase class, or str
            Feature type instance, class, or string (e.g., Distances(), Distances, "distances").
            The feature type determines what kind of analysis will be performed.
        force : bool, default=False
            Whether to force recomputation of the feature even if it already exists.

        Returns:
        --------
        None
            Adds computed feature to trajectory data and creates analysis methods

        Raises:
        -------
        ValueError
            If the feature already exists with computed data, if required
            dependencies are missing, or if trajectories are not loaded.

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> feature_manager = FeatureManager()
        >>> feature_manager.add_feature(traj_data, feature_type.Distances())  # instance
        >>> feature_manager.add_feature(traj_data, feature_type.Contacts)     # class
        >>> feature_manager.add_feature(traj_data, "distances", force=True)   # string
        """
        if traj_data.res_label_data is None:
            raise ValueError(
                "Trajectory labels must be set before computing features. "
                "Use TrajectoryManager.set_labels() to set labels."
            )

        feature_key = self._get_feature_key(feature_type)

        self._check_feature_existence(traj_data, feature_key, force)
        self._check_dependencies(traj_data, feature_type, feature_key)

        # Create FeatureData instance
        feature_data = FeatureData(
            feature_type=feature_type,
            use_memmap=self.use_memmap,
            cache_path=self.cache_dir,
            chunk_size=self.chunk_size,
        )

        self._compute_feature(traj_data, feature_data, feature_type)
        self._bind_stats_methods(feature_data)

        # Store the feature data
        traj_data.features[feature_key] = feature_data

    def _check_feature_existence(self, traj_data, feature_key, force):
        """
        Check if feature already exists and handle accordingly.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_key : str
            Feature key
        force : bool
            Whether to force recomputation of the feature

        Returns:
        --------
        None
            Prints warning if feature already exists and force is True

        Raises:
        -------
        ValueError
            If feature already exists and force is False
        """
        feature_exists = self._feature_already_exists(traj_data, feature_key)

        if not feature_exists:
            return

        if force:
            self._handle_force_recomputation(traj_data, feature_key)
        else:
            raise ValueError(f"{feature_key.capitalize()} FeatureData already exists.")

    def _feature_already_exists(self, traj_data, feature_key):
        """
        Check if feature already exists with data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_key : str
            Feature key to check

        Returns:
        --------
        bool
            True if feature exists and has data, False otherwise
        """
        return (
            feature_key in traj_data.features
            and traj_data.features[feature_key].data is not None
        )

    def _handle_force_recomputation(self, traj_data, feature_key):
        """
        Handle forced recomputation of existing feature.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_key : str
            Feature key to remove and recompute

        Returns:
        --------
        None
            Removes existing feature and prints warning message
        """
        print(
            f"WARNING: {feature_key.capitalize()} FeatureData already "
            f"exists. Forcing recomputation."
        )
        traj_data.features.pop(feature_key)

    def _check_dependencies(self, traj_data, feature_type, feature_key):
        """
        Check that all dependencies are available.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_type : FeatureTypeBase
            Feature type object
        feature_key : str
            Feature key

        Returns:
        --------
        None
            Validates dependencies are available

        Raises:
        -------
        ValueError
            If dependency is missing
        """
        dependencies = feature_type.get_dependencies()
        for dep in dependencies:
            if (
                dep not in traj_data.features
                or traj_data.features[dep].get_data() is None
            ):
                raise ValueError(
                    f"Dependency '{dep}' must be computed before '{feature_key}'."
                )

    def _compute_feature(self, traj_data, feature_data, feature_type):
        """
        Compute the feature with appropriate input data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_data : FeatureData
            Feature data object
        feature_type : FeatureTypeBase
            Feature type object

        Returns:
        --------
        None
            Computes the feature with appropriate input data

        Raises:
        -------
        ValueError
            If the feature type has no input or if trajectories are not loaded
        """
        self._validate_computation_requirements(traj_data, feature_type)
        data, feature_metadata = self._execute_computation(
            traj_data, feature_data, feature_type
        )
        self._store_computation_results(feature_data, data, feature_metadata)

    def _validate_computation_requirements(self, traj_data, feature_type):
        """
        Validate that all requirements for computation are met.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_type : FeatureTypeBase
            Feature type object

        Returns:
        --------
        None
            Validates requirements are met

        Raises:
        -------
        ValueError
            If trajectory labels are not set or trajectories are not loaded
        """
        if traj_data.res_label_data is None:
            raise ValueError("Trajectory labels must be set before computing features.")

        if feature_type.get_input() is None and traj_data.trajectories is None:
            raise ValueError("Trajectories must be loaded before computing features.")

    def _execute_computation(self, traj_data, feature_data, feature_type) -> tuple[Any, dict]:
        """
        Execute the actual feature computation.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_data : FeatureData
            Feature data object
        feature_type : FeatureTypeBase
            Feature type object

        Returns:
        --------
        tuple[Any, dict]
            Tuple of (data, feature_metadata) from computation
        """
        if feature_type.get_input() is not None:
            input_feature = traj_data.features[feature_type.get_input()]
            return feature_data.feature_type.compute(
                input_feature.get_data(),
                input_feature.get_feature_metadata(),
            )

        return feature_data.feature_type.compute(
            traj_data.trajectories, feature_metadata=traj_data.res_label_data
        )

    def _store_computation_results(self, feature_data, data, feature_metadata):
        """
        Store computation results in the feature data object.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object to store results in
        data : numpy.ndarray
            Computed feature data
        feature_metadata : dict
            Feature metadata dictionary

        Returns:
        --------
        None
            Stores data and metadata in feature_data object
        """
        feature_data.data = data
        feature_data.feature_metadata = self._ensure_numpy_arrays(feature_metadata)

    def _ensure_numpy_arrays(self, feature_metadata):
        """
        Ensure features are always numpy arrays for consistency.

        Parameters:
        -----------
        feature_metadata : dict
            Feature metadata dictionary

        Returns:
        --------
        dict
            Feature metadata with features converted to numpy arrays
        """
        if feature_metadata and "features" in feature_metadata:
            if not isinstance(feature_metadata["features"], np.ndarray):
                feature_metadata["features"] = np.array(feature_metadata["features"])
        return feature_metadata

    def _get_feature_key(self, feature_type):
        """
        Get the feature key from the feature type.

        Supports all three input variants:
        - feature_type.Distances() (instance)
        - feature_type.Distances (class with metaclass)
        - "distances" (string)

        Parameters:
        -----------
        feature_type : FeatureTypeBase, FeatureTypeBase class, or str
            Feature type instance, class, or feature key string

        Returns:
        --------
        str
            Feature key
        """
        if isinstance(feature_type, str):
            # Direct string: "distances"
            return feature_type
        elif hasattr(feature_type, "get_type_name"):
            # Instance or class with get_type_name method
            return feature_type.get_type_name()
        else:
            # Fallback: try to convert to string (handles metaclass)
            return str(feature_type)

    def reset_reduction(self, traj_data, feature_type):
        """
        Reset to using full original data instead of reduced dataset.

        Supports all three input variants:
        - feature_type.Distances() (instance)
        - feature_type.Distances (class with metaclass)
        - "distances" (string)

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_type : FeatureTypeBase, FeatureTypeBase class, or str
            Feature type instance, class, or string

        Returns:
        --------
        None
            Clears reduced_data and prints confirmation with shape info

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> feature_manager = FeatureManager()
        >>> feature_manager.add_feature(traj_data, feature_type.Distances())
        >>> feature_manager.reduce_data(traj_data, feature_type.Distances,
                                        metric="cv", threshold_min=0.1)
        >>> feature_manager.reset_reduction(traj_data, "distances")  # string
        """
        feature_key = self._get_feature_key(feature_type)
        if traj_data.features[feature_key].reduced_data is None:
            print("No reduction to reset - already using full data.")
            return

        # Clear reduced data
        original_shape = traj_data.features[feature_key].data.shape
        reduced_shape = traj_data.features[feature_key].reduced_data.shape

        traj_data.features[feature_key].reduced_data = None
        traj_data.features[feature_key].reduced_feature_metadata = None
        old_info = traj_data.features[feature_key].reduction_info
        traj_data.features[feature_key].reduction_info = None

        print(
            f"Reset reduction: Now using full data {original_shape}. "
            f"(Data was reduced to {reduced_shape}, {old_info:.1%})"
        )

    def reduce_data(
        self,
        traj_data,
        feature_type,
        metric,
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

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        feature_type : FeatureTypeBase, FeatureTypeBase class, or str
            Feature type instance, class, or string
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
            If the feature has no data or if the reduction has already been performed

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> feature_manager = FeatureManager()
        >>> feature_manager.add_feature(traj_data, feature_type.Distances())  # instance
        >>> feature_manager.reduce_data(traj_data, feature_type.Distances,   # class
                                        metric="cv", threshold_min=0.1)
        """
        feature_key = self._get_feature_key(feature_type)
        if traj_data.features[feature_key].data is None:
            raise ValueError(
                "No data available. Add the feature to the trajectory data first."
            )

        if traj_data.features[feature_key].reduced_data is not None:
            raise ValueError(
                "Reduction already performed. Reset the reduction first using reset_reduction()."
            )

        # Get reduction results from calculator
        results = traj_data.features[
            feature_key
        ].feature_type.calculator.compute_dynamic_values(
            input_data=traj_data.features[feature_key].data,
            metric=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            feature_metadata=traj_data.features[feature_key].feature_metadata[
                "features"
            ],
            output_path=traj_data.features[feature_key].reduced_cache_path,
            transition_mode=transition_mode,
            lag_time=lag_time,
        )

        # Simply set reduced_data
        traj_data.features[feature_key].reduced_data = results["dynamic_data"]
        traj_data.features[feature_key].reduced_feature_metadata = traj_data.features[
            feature_key
        ].feature_metadata.copy()
        # Ensure features are always numpy arrays for consistency
        if isinstance(results["feature_names"], np.ndarray):
            traj_data.features[feature_key].reduced_feature_metadata["features"] = (
                results["feature_names"]
            )
        else:
            traj_data.features[feature_key].reduced_feature_metadata["features"] = (
                np.array(results["feature_names"])
            )
        traj_data.features[feature_key].reduction_info = (
            results["n_dynamic"] / results["total_pairs"]
        )

        print(
            f"Now using reduced data. "
            f"Data reduced from {traj_data.features[feature_key].data.shape} "
            f"to {traj_data.features[feature_key].reduced_data.shape}. "
            f"({traj_data.features[feature_key].reduction_info:.1%} retained)."
        )

    def _create_bound_method(
        self, feature_data, original_method, method_name, requires_full_data
    ):
        """
        Create bound method with automatic data selection.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        original_method : function
            Original method to bind
        method_name : str
            Name of the method to bind
        requires_full_data : set
            Set of method names that require full data

        Returns:
        --------
        function
            Bound method that automatically uses current data
        """

        @functools.wraps(original_method)
        def bound_method(*args, **kwargs):
            """Bound method that automatically uses current data."""
            # Check if method requires full data
            if method_name in requires_full_data:
                data = feature_data.data
            else:
                # Other methods use reduced_data if available, else data
                data = (
                    feature_data.reduced_data
                    if feature_data.reduced_data is not None
                    else feature_data.data
                )
            return original_method(data, *args, **kwargs)

        return bound_method

    def _bind_stats_methods(self, feature_data):
        """
        Bind analysis methods from calculator to feature_data.analysis with automatic data passing.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object

        Returns:
        --------
        None
            Creates feature_data.analysis object with bound methods that auto-use current data
        """
        if not hasattr(feature_data.feature_type.calculator, "analysis"):
            return

        # Create a simple object to hold bound methods
        feature_data.analysis = type("BoundStats", (), {})()

        # Get the set of methods that require full data
        requires_full_data = getattr(
            feature_data.feature_type.calculator.analysis, "REQUIRES_FULL_DATA", set()
        )

        # Bind each method from calculator.analysis
        for method_name in dir(feature_data.feature_type.calculator.analysis):
            if not method_name.startswith("_") and callable(
                getattr(feature_data.feature_type.calculator.analysis, method_name)
            ):
                original_method = getattr(
                    feature_data.feature_type.calculator.analysis, method_name
                )
                bound_method = self._create_bound_method(
                    feature_data, original_method, method_name, requires_full_data
                )
                setattr(feature_data.analysis, method_name, bound_method)

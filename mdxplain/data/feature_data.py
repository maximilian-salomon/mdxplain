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
Feature data container for computed features with analysis methods.

Container for feature data (distances, contacts) with associated calculator.
Stores feature data and provides bound analysis methods from calculators.
Supports data reduction based on statistical criteria.
"""

import functools
import os


class FeatureData:
    """
    Internal container for computed feature data with analysis methods.

    Stores feature data and provides bound analysis methods from calculators.
    Supports data reduction based on statistical criteria.
    """

    def __init__(
        self, feature_type, use_memmap=False, cache_path=None, chunk_size=None
    ):
        """
        Initialize feature data container.

        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object to compute data
        use_memmap : bool, default=False
            Whether to use memory mapping
        cache_path : str, optional
            Cache file path for memmap
        chunk_size : int, optional
            Processing chunk size
        """
        self.feature_type = feature_type
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

        # Handle cache path
        if use_memmap:
            if cache_path is None:
                os.makedirs("./cache", exist_ok=True)
                cache_path = f"./cache/{feature_type}.dat"
            self.cache_path = cache_path
            self.reduced_cache_path = (
                f"{os.path.splitext(self.cache_path)[0]}_reduced.dat"
            )
        else:
            self.cache_path = None
            self.reduced_cache_path = None

        # Initialize data as None
        self.data = None
        self.feature_names = None
        self.reduced_data = None
        self.reduced_feature_names = None
        self.reduction_info = None

        self.feature_type = feature_type
        self.feature_type.init_calculator(
            use_memmap=self.use_memmap,
            cache_path=self.cache_path,
            chunk_size=self.chunk_size,
        )

        # Bind stats methods from calculator
        self._bind_stats_methods()

    def _create_bound_method(self, original_method, method_name, requires_full_data):
        """Create bound method with automatic data selection."""
        @functools.wraps(original_method)
        def bound_method(*args, **kwargs):
            """Bound method that automatically uses current data."""
            # Check if method requires full data
            if method_name in requires_full_data:
                data = self.data
            else:
                # Other methods use reduced_data if available, else data
                data = (
                    self.reduced_data
                    if self.reduced_data is not None
                    else self.data
                )
            return original_method(data, *args, **kwargs)
        return bound_method

    def _bind_stats_methods(self):
        """
        Bind analysis methods from calculator to self.analysis with automatic data passing.

        Returns:
        --------
        None
            Creates self.analysis object with bound methods that auto-use current data
        """
        if not hasattr(self.feature_type.calculator, "analysis"):
            return

        # Create a simple object to hold bound methods
        self.analysis = type("BoundStats", (), {})()

        # Get the set of methods that require full data
        requires_full_data = getattr(
            self.feature_type.calculator.analysis, 'REQUIRES_FULL_DATA', set()
        )

        # Bind each method from calculator.analysis
        for method_name in dir(self.feature_type.calculator.analysis):
            if not method_name.startswith("_") and callable(
                getattr(self.feature_type.calculator.analysis, method_name)
            ):
                original_method = getattr(
                    self.feature_type.calculator.analysis, method_name
                )
                bound_method = self._create_bound_method(
                    original_method, method_name, requires_full_data
                )
                setattr(self.analysis, method_name, bound_method)

    def compute(self, input_data=None, feature_names=None, labels=None):
        """
        Compute feature data using the associated calculator.

        Parameters:
        -----------
        input_data : array-like or list
            Input data for computation (trajectories, distance arrays, etc.)
        feature_names : list, optional
            Existing feature names to pass through
        labels : list, optional
            Residue labels for generating feature names

        Returns:
        --------
        None
            Stores computed data in self.data and feature names in self.feature_names
        """
        # Call the compute method of the associated calculator
        self.data, self.feature_names = self.feature_type.compute(
            input_data, feature_names=feature_names, labels=labels
        )

    def reduce_data(
        self,
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

        Parameters:
        -----------
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
        """
        if self.data is None:
            raise ValueError("No data available. Call compute() first.")

        # Get reduction results from calculator
        results = self.feature_type.calculator.compute_dynamic_values(
            input_data=self.data,
            metric=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            feature_names=self.feature_names,
            output_path=self.reduced_cache_path,
            transition_mode=transition_mode,
            lag_time=lag_time,
        )

        # Simply set reduced_data
        self.reduced_data = results["dynamic_data"]
        self.reduced_feature_names = results["feature_names"]
        self.reduction_info = results["n_dynamic"] / results["total_pairs"]

        print(
            f"Now using reduced data. "
            f"Data reduced from {self.data.shape} to {self.reduced_data.shape}. "
            f"({self.reduction_info:.1%} retained)."
        )

    def get_data(self):
        """
        Get current dataset (reduced if available, else original).

        Returns:
        --------
        numpy.ndarray
            Feature data array, shape (n_frames, n_features)
        """
        if self.reduced_data is not None:
            return self.reduced_data
        return self.data

    def get_feature_names(self):
        """
        Get current feature names (reduced if available, else original).

        Returns:
        --------
        numpy.ndarray
            Feature names array, shape (n_features, 2) with residue pair indices
        """
        if self.reduced_feature_names is not None:
            return self.reduced_feature_names
        return self.feature_names

    def reset_reduction(self):
        """
        Reset to using full original data instead of reduced dataset.

        Returns:
        --------
        None
            Clears reduced_data and prints confirmation with shape info
        """
        if self.reduced_data is None:
            print("No reduction to reset - already using full data.")
            return

        # Clear reduced data
        original_shape = self.data.shape
        reduced_shape = self.reduced_data.shape

        self.reduced_data = None
        self.reduced_feature_names = None
        old_info = self.reduction_info
        self.reduction_info = None

        print(
            f"Reset reduction: Now using full data {original_shape}. "
            f"(Data was reduced to {reduced_shape}, {old_info:.1%})"
        )

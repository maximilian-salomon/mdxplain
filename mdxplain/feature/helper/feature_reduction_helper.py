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
Feature reduction helper for data reduction operations.

This module provides utilities for processing data reduction results,
handling retention warnings, and managing reduced feature metadata
in the FeatureManager.
"""

import numpy as np
from typing import Optional, Union, Dict, Any

from ...feature.entities.feature_data import FeatureData

class FeatureReductionHelper:
    """
    Helper class for feature data reduction operations.

    Provides static methods for processing reduction results, handling
    retention rate warnings, and managing reduced feature metadata.
    """

    @staticmethod
    def process_reduction_results(
        feature_data: FeatureData,
        results: Dict[str, Any],
        threshold_min: Optional[Union[int, float]],
        threshold_max: Optional[Union[int, float]],
        metric: str,
    ) -> None:
        """
        Process and store reduction results for a single trajectory FeatureData object.

        Parameters:
        -----------
        feature_data : FeatureData
            FeatureData object to update with reduction results
        results : Dict[str, Any]
            Results dictionary from compute_dynamic_values
        threshold_min : float or None
            Minimum threshold used for context in warnings
        threshold_max : float or None
            Maximum threshold used for context in warnings
        metric : str
            Metric name used for context in warnings

        Returns:
        --------
        None
            Updates feature_data with reduction results and prints warnings

        Examples:
        ---------
        >>> FeatureReductionHelper.process_reduction_results(
        ...     feature_data, results, 0.1, 0.9, "cv"
        ... )
        Now using reduced data. Data reduced from (1000, 500) to (1000, 45). (9.0% retained).
        """
        # Store reduced data
        feature_data.reduced_data = results["dynamic_data"]

        # Update reduced metadata
        FeatureReductionHelper._update_reduced_metadata(feature_data, results)

        # Calculate and store retention info
        retention_rate = results["n_dynamic"] / results["total_pairs"]
        feature_data.reduction_info = {
            "reduction_method": metric,
            "retention_rate": retention_rate,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "n_dynamic": results["n_dynamic"],
            "total_pairs": results["total_pairs"]
        }

        # Check and warn about low retention
        FeatureReductionHelper._check_retention_warnings(
            retention_rate, threshold_min, threshold_max, metric
        )

        # Print summary
        FeatureReductionHelper._print_reduction_summary(feature_data, retention_rate)

    @staticmethod
    def _update_reduced_metadata(
        feature_data: FeatureData, 
        results: Dict[str, Any]
    ) -> None:
        """
        Update reduced feature metadata from results.

        Parameters:
        -----------
        feature_data : FeatureData
            FeatureData object to update metadata for
        results : Dict[str, Any]
            Results dictionary containing feature names

        Returns:
        --------
        None
            Updates reduced_feature_metadata in feature_data
        """
        # Copy original metadata
        feature_data.reduced_feature_metadata = feature_data.feature_metadata.copy()

        # Ensure feature names are numpy arrays for consistency
        feature_names = results["feature_names"]
        if isinstance(feature_names, np.ndarray):
            reduced_features = feature_names
        else:
            reduced_features = np.array(feature_names)

        feature_data.reduced_feature_metadata["features"] = reduced_features

    @staticmethod
    def _print_reduction_summary(feature_data: Any, retention_rate: float) -> None:
        """
        Print summary of data reduction operation.

        Parameters:
        -----------
        feature_data : FeatureData
            FeatureData object with reduction results
        retention_rate : float
            Data retention rate

        Returns:
        --------
        None
            Prints reduction summary
        """
        original_shape = feature_data.data.shape
        reduced_shape = feature_data.reduced_data.shape

        print(
            f"Now using reduced data. "
            f"Data reduced from {original_shape} "
            f"to {reduced_shape}. "
            f"({retention_rate:.1%} retained)."
        )

    @staticmethod
    def _check_retention_warnings(
        retention_rate: float,
        threshold_min: Optional[Union[int, float]],
        threshold_max: Optional[Union[int, float]],
        metric: str,
    ) -> None:
        """
        Check retention rate and print warnings if necessary.

        Parameters:
        -----------
        retention_rate : float
            Data retention rate (0.0 to 1.0)
        threshold_min : float or None
            Minimum threshold for context
        threshold_max : float or None
            Maximum threshold for context
        metric : str
            Metric name for context

        Returns:
        --------
        None
            Prints warnings for very low retention rates
        """
        if retention_rate == 0.0:
            print(
                f"WARNING: Threshold parameters resulted in 0% data retention. "
                f"All features were filtered out. Consider adjusting threshold_min={threshold_min} or "
                f"threshold_max={threshold_max} for metric '{metric}'."
            )
        elif retention_rate < 0.01:  # Less than 1%
            print(
                f"WARNING: Very low data retention ({retention_rate:.1%}). "
                f"Consider adjusting threshold parameters."
            )

    @staticmethod
    def reset_reduction(pipeline_data, feature_key: str) -> None:
        """
        Reset data reduction and return to using full original data.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to reset
        feature_key : str
            Feature key to reset reduction for

        Returns:
        --------
        None
            Resets reduced data and prints summary

        Examples:
        ---------
        >>> FeatureReductionHelper.reset_reduction(pipeline_data, "distances")
        Reset reduction: Now using full data (1000, 500). (Data was reduced to (1000, 45), 9.0%)
        """
        feature_data = pipeline_data.feature_data[feature_key]

        if feature_data.reduced_data is None:
            print("No reduction to reset - already using full data.")
            return

        # Get shapes before clearing
        original_shape = feature_data.data.shape
        reduced_shape = feature_data.reduced_data.shape
        old_info = feature_data.reduction_info

        # Clear reduced data
        feature_data.reduced_data = None
        feature_data.reduced_feature_metadata = None
        feature_data.reduction_info = None

        print(
            f"Reset reduction: Now using full data {original_shape}. "
            f"(Data was reduced to {reduced_shape}, {old_info:.1%})"
        )

    @staticmethod
    def check_reduction_state(pipeline_data, feature_key: str) -> bool:
        """
        Check if feature has reduced data.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to check
        feature_key : str
            Feature key to check

        Returns:
        --------
        bool
            True if feature has reduced data, False otherwise

        Examples:
        ---------
        >>> has_reduction = FeatureReductionHelper.check_reduction_state(
        ...     pipeline_data, "distances"
        ... )
        """
        return pipeline_data.feature_data[feature_key].reduced_data is not None

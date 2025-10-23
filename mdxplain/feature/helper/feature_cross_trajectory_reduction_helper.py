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
Cross-trajectory feature reduction helper.

This module provides utilities for reducing features consistently across
multiple trajectories to ensure common denominator features for matrix
construction and comparison.
"""

import numpy as np
from typing import Dict, Any, Set, Optional

from ...feature.entities.feature_data import FeatureData
from .feature_reduction_helper import FeatureReductionHelper
from ..feature_type.helper.calculator_compute_helper import CalculatorComputeHelper


class FeatureCrossTrajectoryReductionHelper:
    """
    Helper class for cross-trajectory feature reduction operations.

    Provides static methods for finding common features across trajectories,
    performing consistent reduction, and maintaining feature alignment
    for matrix construction.
    """

    @staticmethod
    def find_common_features(
        feature_data_dict: Dict[int, FeatureData],
        metric: str,
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        **kwargs
    ) -> Set[int]:
        """
        Find feature indices that pass reduction criteria in ALL trajectories.

        Uses existing calculator logic to find features that meet the criteria
        in every trajectory, then returns the intersection of valid indices.

        Parameters
        ----------
        feature_data_dict : Dict[int, FeatureData]
            Dictionary mapping trajectory indices to their FeatureData
        metric : str
            Reduction metric to apply (e.g., 'frequency', 'cv', 'stability')
        threshold_min : float, optional
            Minimum threshold for feature retention
        threshold_max : float, optional
            Maximum threshold for feature retention
        '**'kwargs : dict
            Additional parameters for reduction metric calculation

        Returns
        -------
        Set[int]
            Set of feature indices that pass criteria in all trajectories

        Examples
        --------
        >>> common_indices = FeatureCrossTrajectoryReductionHelper.find_common_features(
        ...     feature_data_dict, "frequency", threshold_min=0.5
        ... )
        """
        all_valid_indices = []

        for _, feature_data in feature_data_dict.items():
            # Calculator handles memmap/chunking automatically
            traj_results = feature_data.feature_type.calculator.compute_dynamic_values(
                input_data=feature_data.data,
                metric=metric,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                feature_metadata=feature_data.feature_metadata["features"],
                output_path=feature_data.reduced_cache_path,
                **kwargs
            )

            # Extract indices of features that pass criteria
            retained_indices = traj_results["indices"][0]
            valid_indices = set(retained_indices)
            all_valid_indices.append(valid_indices)

        if not all_valid_indices:
            return set()

        # Find intersection of all valid indices
        common_indices = all_valid_indices[0]
        for valid_indices in all_valid_indices[1:]:
            common_indices = common_indices.intersection(valid_indices)

        return common_indices

    @staticmethod
    def apply_common_reduction(
        feature_data_dict: Dict[int, FeatureData],
        common_indices: Set[int],
        metric: str,
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Apply reduction using common feature indices across all trajectories.

        Creates results dictionary and uses FeatureReductionHelper to handle
        the reduction processing, including memmap compatibility.

        Parameters
        ----------
        feature_data_dict : Dict[int, FeatureData]
            Dictionary mapping trajectory indices to their FeatureData
        common_indices : Set[int]
            Set of common feature indices to retain
        metric : str
            Reduction metric name for metadata
        threshold_min : float, optional
            Minimum threshold used (for metadata)
        threshold_max : float, optional
            Maximum threshold used (for metadata)

        Returns
        -------
        None
            Updates reduced_data and metadata for all FeatureData objects

        Examples
        --------
        >>> FeatureCrossTrajectoryReductionHelper.apply_common_reduction(
        ...     feature_data_dict, common_indices, "frequency", 0.5, None
        ... )
        """
        common_indices_array = np.array(sorted(list(common_indices)))

        for _, feature_data in feature_data_dict.items():
            # Create boolean mask from indices for memmap-compatible processing
            mask = np.zeros(len(feature_data.feature_metadata["features"]), dtype=bool)
            if len(common_indices_array) > 0:
                mask[common_indices_array] = True

            # Use CalculatorComputeHelper for memmap-compatible data extraction
            dynamic_data = CalculatorComputeHelper._extract_dynamic_data(
                data=feature_data.data,
                mask=mask,
                use_memmap=feature_data.use_memmap,
                output_path=feature_data.reduced_cache_path,
                chunk_size=feature_data.chunk_size
            )

            # Create results dictionary compatible with FeatureReductionHelper
            if len(common_indices_array) > 0:
                feature_names = feature_data.feature_metadata["features"][common_indices_array]
            else:
                feature_names = np.array([])

            results = {
                "dynamic_data": dynamic_data,
                "feature_names": feature_names,
                "n_dynamic": len(common_indices),
                "total_pairs": len(feature_data.feature_metadata["features"]),
            }

            # Use FeatureReductionHelper for consistent processing
            FeatureReductionHelper.process_reduction_results(
                feature_data,
                results,
                threshold_min,
                threshold_max,
                f"{metric}_cross_trajectory"
            )

    @staticmethod
    def get_reduction_summary(
        feature_data_dict: Dict[int, FeatureData],
        common_indices: Set[int]
    ) -> Dict[str, Any]:
        """
        Generate summary of cross-trajectory reduction.

        Parameters
        ----------
        feature_data_dict : Dict[int, FeatureData]
            Dictionary mapping trajectory indices to their FeatureData
        common_features : Set[str]
            Set of common features retained

        Returns
        -------
        Dict[str, Any]
            Summary dictionary with reduction statistics

        Examples
        --------
        >>> summary = FeatureCrossTrajectoryReductionHelper.get_reduction_summary(
        ...     feature_data_dict, common_features
        ... )
        """
        if not feature_data_dict:
            return {"n_trajectories": 0, "common_features": 0, "retention_rates": []}

        first_feature_data = next(iter(feature_data_dict.values()))
        total_features = len(first_feature_data.feature_metadata["features"])
        n_common = len(common_indices)

        retention_rates = []
        original_shapes = []
        reduced_shapes = []

        for feature_data in feature_data_dict.values():
            retention_rate = n_common / len(feature_data.feature_metadata["features"])
            retention_rates.append(retention_rate)
            original_shapes.append(feature_data.data.shape)
            if feature_data.reduced_data is not None:
                reduced_shapes.append(feature_data.reduced_data.shape)

        return {
            "n_trajectories": len(feature_data_dict),
            "total_features": total_features,
            "common_features": n_common,
            "retention_rates": retention_rates,
            "mean_retention": np.mean(retention_rates),
            "original_shapes": original_shapes,
            "reduced_shapes": reduced_shapes
        }

    @staticmethod
    def print_cross_trajectory_summary(
        feature_data_dict: Dict[int, FeatureData],
        common_indices: Set[int]
    ) -> None:
        """
        Print summary of cross-trajectory reduction results.

        Parameters
        ----------
        feature_data_dict : Dict[int, FeatureData]
            Dictionary mapping trajectory indices to their FeatureData
        common_features : Set[str]
            Set of common features retained

        Returns
        -------
        None
            Prints reduction summary to console

        Examples
        --------
        >>> FeatureCrossTrajectoryReductionHelper.print_cross_trajectory_summary(
        ...     feature_data_dict, common_features
        ... )
        Cross-trajectory reduction: 3 trajectories, 2500 → 450 features (18.0% retained).
        """
        summary = FeatureCrossTrajectoryReductionHelper.get_reduction_summary(
            feature_data_dict, common_indices
        )

        if summary["n_trajectories"] == 0:
            print("No trajectories available for cross-trajectory reduction.")
            return

        print(
            f"Cross-trajectory reduction: {summary['n_trajectories']} trajectories, "
            f"{summary['total_features']} → {summary['common_features']} features "
            f"({summary['mean_retention']:.1%} retained)."
        )

        if summary["mean_retention"] < 0.1:
            print(
                "WARNING: Very low retention rate in cross-trajectory reduction. "
                "Consider adjusting threshold parameters."
            )
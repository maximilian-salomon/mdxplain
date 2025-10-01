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
Feature computation helper for managing feature calculation workflow.

This module provides utilities for handling feature computation processes,
including existence checks, dependency validation, and result storage
in the FeatureManager.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Any, Dict, List, TYPE_CHECKING

from ..entities.feature_data import FeatureData
from ..feature_type.interfaces.feature_type_base import FeatureTypeBase

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class FeatureComputationHelper:
    """
    Helper class for feature computation operations.

    Provides static methods for managing the feature computation workflow,
    including pre-computation checks, computation execution, and result storage.
    """

    @staticmethod
    def check_feature_existence(
        pipeline_data: PipelineData, 
        feature_key: str,
        traj_indices: List[int],
        force: bool
    ) -> None:
        """
        Check if feature already exists and handle accordingly.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to check
        feature_key : str
            Feature key to check
        force : bool
            Whether to force recomputation

        Returns
        -------
        None
            Handles existing features or raises ValueError

        Raises
        ------
        ValueError
            If feature exists and force=False

        Examples
        --------
        >>> FeatureComputationHelper.check_feature_existence(
        ...     pipeline_data, "distances", False
        ... )
        """
        feature_exists = FeatureComputationHelper._feature_already_exists(
            pipeline_data, feature_key, traj_indices
        )

        if not feature_exists:
            return

        if force:
            FeatureComputationHelper._handle_force_recomputation(
                pipeline_data, feature_key, traj_indices
            )
        else:
            raise ValueError(f"{feature_key.capitalize()} FeatureData already exists.")

    @staticmethod
    def _feature_already_exists(pipeline_data: PipelineData, feature_key: str, traj_indices: List[int]) -> bool:
        """
        Check if feature already exists with computed data.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to check
        feature_key : str
            Feature key to check

        Returns
        -------
        bool
            True if feature exists with data, False otherwise
        """
        if feature_key not in pipeline_data.feature_data:
            return False
        
        # Check if any of the specified trajectories has data for this feature
        feature_dict = pipeline_data.feature_data[feature_key]
        return any(
            traj_idx in feature_dict and feature_dict[traj_idx].data is not None
            for traj_idx in traj_indices
        )

    @staticmethod
    def _handle_force_recomputation(pipeline_data: PipelineData, feature_key: str, traj_indices: List[int]) -> None:
        """
        Handle forced recomputation of existing feature.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to modify
        feature_key : str
            Feature key to remove and recompute

        Returns
        -------
        None
            Removes existing feature and prints warning
        """
        print(
            f"WARNING: {feature_key.capitalize()} FeatureData already "
            f"exists for trajectories {traj_indices}. Forcing recomputation."
        )
        # Remove only the data for the specified trajectories
        if feature_key in pipeline_data.feature_data:
            for traj_idx in traj_indices:
                if traj_idx in pipeline_data.feature_data[feature_key]:
                    del pipeline_data.feature_data[feature_key][traj_idx]

    @staticmethod
    def execute_computation(
        pipeline_data: PipelineData, 
        feature_data: FeatureData, 
        feature_type: FeatureTypeBase,
        traj_idx: int,
        force_original: bool = True
    ) -> Tuple[Any, Dict]:
        """
        Execute feature computation for a single trajectory.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object with input data
        feature_data : FeatureData
            Feature data object for computation
        feature_type : FeatureTypeBase
            Feature type object to compute
        traj_idx : int
            Trajectory index to process
        force_original : bool, default=True
            Whether to force using original data instead of reduced data

        Returns
        -------
        Tuple[Any, Dict]
            Tuple of (computed_data, metadata) for single trajectory

        Examples
        --------
        >>> # Process single trajectory
        >>> data, metadata = FeatureComputationHelper.execute_computation(
        ...     pipeline_data, feature_data, distances_feature, traj_idx=0
        ... )
        """
        if feature_type.get_input() is not None:
            # Feature depends on other features
            input_feature_dict = pipeline_data.feature_data[feature_type.get_input()]
            
            # Get the specific trajectory's feature data
            if traj_idx not in input_feature_dict:
                raise ValueError(f"Input feature '{feature_type.get_input()}' not computed for trajectory {traj_idx}")

            input_feature = input_feature_dict[traj_idx]
            input_data = input_feature.get_data(force_original=force_original)
            input_metadata = input_feature.get_feature_metadata(force_original=force_original)
            
            return feature_data.feature_type.compute(input_data, input_metadata)
        else:
            # Base feature from trajectory
            single_traj = pipeline_data.trajectory_data.trajectories[traj_idx]
            single_labels = pipeline_data.trajectory_data.res_label_data[traj_idx]
            return feature_data.feature_type.compute(single_traj, feature_metadata=single_labels)


    @staticmethod
    def store_computation_results(
        feature_data: FeatureData, 
        data: np.ndarray, 
        feature_metadata: Dict
    ) -> None:
        """
        Store computation results in feature data object.

        Parameters
        ----------
        feature_data : FeatureData
            Feature data object to store results in
        data : np.ndarray
            Computed feature data (typically numpy array)
        feature_metadata : Dict
            Feature metadata dictionary

        Returns
        -------
        None
            Stores data and metadata in feature_data object

        Examples
        --------
        >>> FeatureComputationHelper.store_computation_results(
        ...     feature_data, distance_matrix, metadata_dict
        ... )
        """
        feature_data.data = data
        feature_data.feature_metadata = FeatureComputationHelper._ensure_numpy_arrays(
            feature_metadata
        )

    @staticmethod
    def _ensure_numpy_arrays(feature_metadata: Dict) -> Dict:
        """
        Ensure features in metadata are numpy arrays for consistency.

        Parameters
        ----------
        feature_metadata : Dict
            Feature metadata dictionary

        Returns
        -------
        Dict
            Feature metadata with features converted to numpy arrays
        """
        if feature_metadata and "features" in feature_metadata:
            if not isinstance(feature_metadata["features"], np.ndarray):
                feature_metadata["features"] = np.array(feature_metadata["features"])
        return feature_metadata


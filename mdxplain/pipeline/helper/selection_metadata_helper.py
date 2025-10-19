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
Metadata operations helper for feature selection system.

Provides metadata collection, processing, and validation operations
for the feature selection system in PipelineData.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import numpy as np

from ...feature.entities.feature_data import FeatureData

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData


class SelectionMetadataHelper:
    """
    Helper class for metadata operations in feature selection system.

    Provides static methods for collecting metadata for selections,
    processing individual feature metadata, and finalizing metadata results.
    """

    @staticmethod
    def collect_metadata_for_selection(pipeline_data: PipelineData, name: str) -> List[Dict[str, Any]]:
        """
        Collect metadata for all features in the selection using stored reference trajectory.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        name : str
            Name of the selection to collect metadata for

        Returns
        -------
        list
            List of metadata for all features in the selection

        Raises
        ------
        ValueError
            If the selection does not exist or reference trajectory not set/valid
        """
        selected_metadata = []

        # Get results from FeatureSelectorData object - new trajectory-specific structure
        selector_data = pipeline_data.selected_feature_data[name]
        all_results = selector_data.get_all_results()

        # Get reference trajectory from selector data
        reference_traj = selector_data.get_reference_trajectory()
        if reference_traj is None:
            raise ValueError(f"No reference trajectory set for selection '{name}'. Run select() first.")

        for feature_type, selection_info in all_results.items():
            # New structure: {"trajectory_indices": {traj_idx: {"indices": [...], "use_reduced": [...]}}}
            trajectory_results = selection_info.get("trajectory_indices", {})
            
            # Validate reference trajectory exists in this selection
            if reference_traj not in trajectory_results:
                available_trajs = list(trajectory_results.keys())
                raise ValueError(
                    f"Reference trajectory {reference_traj} not found in selection '{name}' "
                    f"for feature '{feature_type}'. Available trajectories: {available_trajs}"
                )
            
            # Get metadata using reference trajectory
            reference_data = trajectory_results[reference_traj]
            feature_metadata = SelectionMetadataHelper._get_metadata_for_feature(
                pipeline_data, feature_type, reference_data, reference_traj, name
            )
            selected_metadata.extend(feature_metadata)

        return selected_metadata

    @staticmethod
    def _get_metadata_for_feature(
        pipeline_data: PipelineData, feature_type: str, reference_data: Dict[str, Any], reference_traj: int, name: str
    ) -> List[Dict[str, Any]]:
        """
        Get metadata for a single feature type.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        feature_type : str
            Name of the feature type to get metadata for
        reference_data : dict
            Selection data for reference trajectory containing indices and use_reduced flags
        reference_traj : int
            Reference trajectory index
        name : str
            Name of the selection to get metadata for

        Returns
        -------
        list
            List of metadata for the feature type

        Raises
        ------
        ValueError
            If the feature type does not exist
        """
        # Get feature data for reference trajectory (trajektorien-spezifisch!)
        feature_data = pipeline_data.feature_data[feature_type][reference_traj]
        indices = reference_data["indices"]
        use_reduced_flags = reference_data["use_reduced"]

        feature_metadata = []

        for col_idx, use_reduced in zip(indices, use_reduced_flags):
            metadata_entry = SelectionMetadataHelper._get_single_metadata_entry(
                feature_data, col_idx, use_reduced, feature_type, name
            )
            if metadata_entry:
                feature_metadata.append(metadata_entry)

        return feature_metadata

    @staticmethod
    def _get_single_metadata_entry(
        feature_data: FeatureData,
        col_idx: int,
        use_reduced: bool,
        feature_type: str,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata entry for a single column with type_metadata structure.

        Parameters
        ----------
        feature_data : FeatureData
            Feature data object
        col_idx : int
            Index of the column to get metadata for
        use_reduced : bool
            Whether the column uses reduced data
        feature_type : str
            Name of the feature type to get metadata for
        name : str
            Name of the selection to get metadata for

        Returns
        -------
        Optional[dict]
            Metadata entry with structure:
            {
                "features": feature-specific metadata (single entry),
                "type": feature type string,
                "type_metadata": {
                    "computation_params": {...},
                    "is_pair": bool,
                    "algorithm": str,
                    ...
                }
            }
            Returns None if not available.

        Raises
        ------
        ValueError
            If the feature type does not exist
        """
        metadata = SelectionMetadataHelper._get_feature_metadata_by_type(
            feature_data, use_reduced, feature_type, name
        )

        if "features" in metadata:
            return {
                "features": metadata["features"][col_idx],
                "type": feature_type,
                "type_metadata": {k: v for k, v in metadata.items() if k != "features"}
            }
        return None

    @staticmethod
    def _get_feature_metadata_by_type(
        feature_data: FeatureData, use_reduced: bool, feature_type: str, name: str
    ) -> Dict[str, Any]:
        """
        Get the appropriate metadata based on use_reduced flag.

        Parameters
        ----------
        feature_data : FeatureData
            Feature data object
        use_reduced : bool
            Whether the column uses reduced data
        feature_type : str
            Name of the feature type to get metadata for
        name : str
            Name of the selection to get metadata for

        Returns
        -------
        dict
            Metadata for the feature type

        Raises
        ------
        ValueError
            If the metadata is not available
        """
        if use_reduced:
            if feature_data.reduced_feature_metadata is None:
                raise ValueError(
                    f"Reduced metadata not available for feature '{feature_type}' "
                    f"in selection '{name}'."
                )
            return feature_data.reduced_feature_metadata
        else:
            if feature_data.feature_metadata is None:
                raise ValueError(
                    f"Metadata not available for feature '{feature_type}' "
                    f"in selection '{name}'."
                )
            return feature_data.feature_metadata

    @staticmethod
    def finalize_metadata_result(selected_metadata: list, name: str) -> np.ndarray:
        """
        Finalize and validate the metadata result.

        Parameters
        ----------
        selected_metadata : list
            List of metadata for all features in the selection
        name : str
            Name of the selection to get metadata for

        Returns
        -------
        numpy.ndarray
            Array of dictionaries, one for each column in the selected matrix.

        Raises
        ------
        ValueError
            If no valid metadata is found for the selection
        """
        if not selected_metadata:
            raise ValueError(f"No valid metadata found for selection '{name}'.")

        return np.array(selected_metadata)

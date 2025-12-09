# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.5).
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
Helper for mapping reduced feature indices to original indices.

This module enables using original feature data (like distances) when the
corresponding reduced feature (like contacts) provides the feature selection,
maintaining consistency across feature types during plotting and analysis.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class ReductionIndexMapper:
    """
    Maps between reduced and original feature indices.

    When contacts are reduced but distances are not, this class enables
    using the same atom pairs from distances that correspond to the
    reduced contacts. This maintains consistency during plotting and analysis.

    Examples
    --------
    >>> # Contacts reduced from 250 to 50 features
    >>> kept_indices = np.array([5, 12, 45, 102, 203, ...])  # 50 indices
    >>>
    >>> # User wants reduced features [1, 3] from reduced contacts
    >>> reduced_indices = np.array([1, 3])
    >>>
    >>> # Map to original indices for distances
    >>> original = ReductionIndexMapper.map_reduced_to_original(
    ...     reduced_indices, kept_indices
    ... )
    >>> print(original)  # [12, 102]
    """

    @staticmethod
    def map_reduced_to_original(
        reduced_indices: np.ndarray,
        kept_indices: np.ndarray
    ) -> np.ndarray:
        """
        Map reduced feature indices to original data indices.

        Uses the kept_indices array to translate indices from the reduced
        feature space back to the original feature space.

        Parameters
        ----------
        reduced_indices : np.ndarray
            Indices in the reduced data space
            For example: [0, 5, 12] refers to features in reduced_data
        kept_indices : np.ndarray
            Original indices that were kept during reduction
            For example: [3, 45, 67, 102, 145, 203, ...]
            This maps reduced index i to original index kept_indices[i]

        Returns
        -------
        np.ndarray
            Original indices corresponding to reduced indices
            For example: If reduced_indices=[1, 3] and kept_indices=[5, 12, 45, 102, ...],
            returns [12, 102]

        Examples
        --------
        >>> # 250 original features reduced to 50
        >>> kept = np.array([5, 12, 45, 102, 203])
        >>>
        >>> # Want features 1 and 3 from reduced space
        >>> reduced = np.array([1, 3])
        >>>
        >>> # Map to original space
        >>> original = ReductionIndexMapper.map_reduced_to_original(reduced, kept)
        >>> print(original)  # [12, 102]

        Notes
        -----
        This is a simple array indexing operation: kept_indices[reduced_indices].
        The complexity is O(len(reduced_indices)).
        """
        return kept_indices[reduced_indices]

    @staticmethod
    def get_kept_indices(
        pipeline_data: PipelineData,
        feature_key: str,
        traj_idx: int
    ) -> Optional[np.ndarray]:
        """
        Get kept indices for a specific trajectory and feature type.

        Retrieves the kept_indices array from the feature's reduction_info,
        which records which original feature indices were retained during
        dimensionality reduction.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with feature data
        feature_key : str
            Feature type key such as "contacts" or "distances"
        traj_idx : int
            Trajectory index to get indices for

        Returns
        -------
        np.ndarray or None
            Array of kept indices if reduction info exists, None otherwise
            Returns None if: feature doesn't exist, trajectory doesn't exist,
            or no reduction has been performed

        Examples
        --------
        >>> # Get kept indices for contacts in trajectory 0
        >>> kept = ReductionIndexMapper.get_kept_indices(
        ...     pipeline_data, "contacts", 0
        ... )
        >>> if kept is not None:
        ...     print(f"Contacts reduced from {kept[-1]+1} to {len(kept)} features")

        Notes
        -----
        Returns None in these cases:

        - Feature type not computed: feature_key not in pipeline_data.feature_data
        - Trajectory not loaded: traj_idx not in feature_data_dict
        - No reduction performed: feature_data.reduction_info is None
        - Reduction info incomplete: 'kept_indices' key missing
        """
        # Check if feature type exists
        if feature_key not in pipeline_data.feature_data:
            return None

        # Get feature data dict for this feature type
        feature_dict = pipeline_data.feature_data[feature_key]

        # Check if trajectory exists
        if traj_idx not in feature_dict:
            return None

        # Get feature data for this trajectory
        feature_data = feature_dict[traj_idx]

        # Check if reduction info exists
        if feature_data.reduction_info is None:
            return None

        # Return kept_indices if available
        return feature_data.reduction_info.get("kept_indices")

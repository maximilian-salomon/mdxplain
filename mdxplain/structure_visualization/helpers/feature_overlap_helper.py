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
Feature overlap detection utilities for structure visualization.

This module provides utilities for detecting residues that appear in
multiple features, which require special handling in visualizations
(e.g., mixed colors for overlapping residues).
"""

from typing import Dict, List, Any


class FeatureOverlapHelper:
    """
    Helper class for detecting overlapping features.

    Provides static methods for building residue-to-feature mappings
    and identifying residues that appear in multiple features. Used
    by both NGLView and PyMOL visualizations to handle overlaps.

    Examples
    --------
    >>> features = [
    ...     {"residue_seqids": [16, 33, 50]},
    ...     {"residue_seqids": [27, 33, 60]}
    ... ]
    >>> # Residue 33 appears in both features
    >>> overlaps = FeatureOverlapHelper.detect_residue_overlaps(features)
    >>> print(overlaps)
    {33: [0, 1]}
    """

    @staticmethod
    def build_residue_feature_map(
        top_features: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """
        Build mapping from residue seqid to feature indices.

        Creates dictionary mapping each residue sequence ID to a list
        of feature indices that contain that residue. Used as
        intermediate step for overlap detection.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of feature dictionaries with 'residue_seqids' keys

        Returns
        -------
        Dict[int, List[int]]
            Mapping from residue seqid to list of feature indices

        Examples
        --------
        >>> features = [
        ...     {"residue_seqids": [16, 33]},
        ...     {"residue_seqids": [27, 33]}
        ... ]
        >>> mapping = FeatureOverlapHelper.build_residue_feature_map(
        ...     features
        ... )
        >>> print(mapping)
        {16: [0], 33: [0, 1], 27: [1]}

        Notes
        -----
        - Feature indices start at 0
        - Each residue maps to all features containing it
        - Empty 'residue_seqids' are skipped
        """
        residue_to_features = {}

        for feat_idx, feature in enumerate(top_features):
            seqids = feature.get("residue_seqids", [])
            for seqid in seqids:
                if seqid not in residue_to_features:
                    residue_to_features[seqid] = []
                residue_to_features[seqid].append(feat_idx)

        return residue_to_features

    @staticmethod
    def detect_residue_overlaps(
        top_features: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """
        Detect residues occurring in multiple features.

        Identifies residues that appear in more than one feature,
        which need special handling in visualizations (e.g., mixed
        colors for NGLView licorice or PyMOL sticks).

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of feature dictionaries with 'residue_seqids' keys

        Returns
        -------
        Dict[int, List[int]]
            Mapping from seqid to list of feature indices.
            Only includes residues appearing in 2+ features.

        Examples
        --------
        >>> features = [
        ...     {"residue_seqids": [16, 33, 50]},
        ...     {"residue_seqids": [27, 33, 60]}
        ... ]
        >>> overlaps = FeatureOverlapHelper.detect_residue_overlaps(
        ...     features
        ... )
        >>> print(overlaps)
        {33: [0, 1]}

        >>> # No overlaps
        >>> features = [
        ...     {"residue_seqids": [16]},
        ...     {"residue_seqids": [27]}
        ... ]
        >>> overlaps = FeatureOverlapHelper.detect_residue_overlaps(
        ...     features
        ... )
        >>> print(overlaps)
        {}

        Notes
        -----
        - Only residues in 2+ features are returned
        - Feature indices start at 0
        - Used by visualizers to determine which residues need
          mixed color representation
        - Non-overlapping residues are filtered out
        """
        residue_to_features = FeatureOverlapHelper.build_residue_feature_map(
            top_features
        )

        return {
            seqid: feat_indices
            for seqid, feat_indices in residue_to_features.items()
            if len(feat_indices) > 1
        }

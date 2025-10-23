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
Base numeric parser helper for feature selection.

This module provides the base class for numeric category parsers (resid, seqid),
implementing shared logic for range and single value parsing with configurable
metadata fields.
"""

import re
from typing import List, Tuple, Set


class FeatureSelectorNumericParseHelper:
    """
    Base helper class for numeric category parsing - array index and sequence ID based operations.

    Provides static methods to parse numeric categories like "123", "123-140"
    to identify matching feature indices based on trajectory metadata.
    """

    @staticmethod
    def parse_numeric_category(
        param_parts: List[str],
        features_list: List[list],
        metadata_field: str,
        category_name: str
    ) -> Tuple[List[int], Set[int]]:
        """
        Parse numeric category and return matching feature indices plus matched residue indices.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters
        ----------
        param_parts : List[str]
            List of parameter parts for numeric selection
        features_list : List[list]
            List of features from metadata
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        category_name : str
            Category name for error messages ("resid" or "seqid")

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
            - feature_indices: List of feature indices matching the criteria
            - matched_residue_indices: Set of residue.index values that matched

        Raises
        ------
        ValueError
            If the numeric specification is invalid
        """
        all_indices = []
        all_matched_residue_indices = set()

        for part in param_parts:
            if "-" in part:
                # Range like "123-140"
                indices, residue_indices = FeatureSelectorNumericParseHelper._find_by_numeric_range(
                    part, features_list, metadata_field, category_name
                )
            else:
                # Single number like "123"
                indices, residue_indices = FeatureSelectorNumericParseHelper._find_by_single_numeric(
                    part, features_list, metadata_field, category_name
                )

            all_indices.extend(indices)
            all_matched_residue_indices.update(residue_indices)

        return list(set(all_indices)), all_matched_residue_indices
    
    @staticmethod
    def _find_by_numeric_range(
        numeric_spec: str,
        features_list: List[list],
        metadata_field: str,
        category_name: str
    ) -> Tuple[List[int], Set[int]]:
        """
        Find features with numeric values in the specified range.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters
        ----------
        numeric_spec : str
            Numeric range specification (e.g., "123-140")
        features_list : List[list]
            List of features from metadata
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        category_name : str
            Category name for error messages ("resid" or "seqid")

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)

        Raises
        ------
        ValueError
            If the range specification is invalid
        """
        start_id, end_id = FeatureSelectorNumericParseHelper._parse_numeric_range(
            numeric_spec, category_name
        )

        if start_id > end_id:
            raise ValueError(
                f"Invalid {category_name} range: {numeric_spec}. Start ID cannot be greater than End ID."
            )

        indices = FeatureSelectorNumericParseHelper._find_features_in_numeric_range(
            features_list, start_id, end_id, metadata_field
        )

        # Extract matched residue indices
        if metadata_field == "index":
            # For resid: the numeric values ARE the residue indices
            matched_residue_indices = set(range(start_id, end_id + 1))
        else:
            # For seqid: extract residue.index for residues that ACTUALLY have seqid in range
            matched_residue_indices = FeatureSelectorNumericParseHelper._extract_residue_indices_in_range(
                features_list, metadata_field, start_id, end_id
            )

        return indices, matched_residue_indices
    
    @staticmethod
    def _find_by_single_numeric(
        numeric_spec: str,
        features_list: List[list],
        metadata_field: str,
        category_name: str
    ) -> Tuple[List[int], Set[int]]:
        """
        Find features containing the specified single numeric value.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters
        ----------
        numeric_spec : str
            Numeric specification (e.g., "123")
        features_list : List[list]
            List of features from metadata
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        category_name : str
            Category name for error messages ("resid" or "seqid")

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)

        Raises
        ------
        ValueError
            If the numeric specification is invalid
        """
        if not re.match(r"^\d+$", numeric_spec):
            raise ValueError(f"Invalid {category_name}: {numeric_spec}. Expected integer number.")

        target_id = int(numeric_spec)
        matching_indices = []

        for idx, feature in enumerate(features_list):
            # At least ONE partner must have the target numeric value
            for partner in feature:
                residue_info = partner.get("residue", {})
                res_value = residue_info.get(metadata_field)

                if res_value == target_id:
                    matching_indices.append(idx)
                    break  # Found match in this feature, move to next

        # Extract matched residue indices
        if metadata_field == "index":
            # For resid: the numeric value IS the residue index
            matched_residue_indices = {target_id}
        else:
            # For seqid: extract residue.index for residues that ACTUALLY have the target seqid
            matched_residue_indices = FeatureSelectorNumericParseHelper._extract_residue_indices_with_value(
                features_list, metadata_field, target_id
            )

        return matching_indices, matched_residue_indices
    
    @staticmethod
    def _parse_numeric_range(numeric_spec: str, category_name: str) -> tuple:
        """
        Parse and validate numeric range specification.

        Parameters
        ----------
        numeric_spec : str
            Numeric range specification
        category_name : str
            Category name for error messages ("resid" or "seqid")

        Returns
        -------
        tuple
            Tuple of (start_id, end_id) as integers

        Raises
        ------
        ValueError
            If the range specification is invalid
        """
        if not re.match(r"^\d+-\d+$", numeric_spec):
            raise ValueError(
                f"Invalid {category_name} range format: {numeric_spec}. "
                "Expected format: 'Start-End'"
            )

        start_str, end_str = numeric_spec.split("-")
        return int(start_str), int(end_str)
    
    @staticmethod
    def _find_features_in_numeric_range(
        features_list: List[list],
        start_id: int,
        end_id: int,
        metadata_field: str
    ) -> List[int]:
        """
        Find features with numeric values in the specified range.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters
        ----------
        features_list : List[list]
            List of features from metadata
        start_id : int
            Start of the numeric range
        end_id : int
            End of the numeric range
        metadata_field : str
            Metadata field to use ("index" or "seqid")

        Returns
        -------
        List[int]
            List of feature indices with values in the specified range
        """
        matching_indices = []

        for idx, feature in enumerate(features_list):
            if FeatureSelectorNumericParseHelper._feature_has_value_in_numeric_range(
                feature, start_id, end_id, metadata_field
            ):
                matching_indices.append(idx)

        return matching_indices
    
    @staticmethod
    def _feature_has_value_in_numeric_range(
        feature: list,
        start_id: int,
        end_id: int,
        metadata_field: str
    ) -> bool:
        """
        Check if feature has any value in the specified numeric range.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters
        ----------
        feature : list
            Feature from metadata
        start_id : int
            Start of the numeric range
        end_id : int
            End of the numeric range
        metadata_field : str
            Metadata field to use ("index" or "seqid")

        Returns
        -------
        bool
            True if feature has any value in the specified range, False otherwise
        """
        for partner in feature:
            residue_info = partner.get("residue", {})
            res_value = residue_info.get(metadata_field)

            if res_value is not None and start_id <= res_value <= end_id:
                return True

        return False

    @staticmethod
    def _extract_residue_indices_with_value(
        features_list: List[list], metadata_field: str, target_value: int
    ) -> Set[int]:
        """
        Extract residue.index values for residues that have a specific metadata value.

        Parameters
        ----------
        features_list : List[list]
            Complete list of features
        metadata_field : str
            Metadata field to check (e.g., "seqid")
        target_value : int
            Target value to match

        Returns
        -------
        Set[int]
            Set of residue.index values for residues with the target metadata value
        """
        residue_indices = set()
        for feature in features_list:
            for partner in feature:
                residue_info = partner.get("residue", {})
                if residue_info.get(metadata_field) == target_value:
                    res_index = residue_info.get("index")
                    if res_index is not None:
                        residue_indices.add(res_index)
        return residue_indices

    @staticmethod
    def _extract_residue_indices_in_range(
        features_list: List[list], metadata_field: str, start_value: int, end_value: int
    ) -> Set[int]:
        """
        Extract residue.index values for residues with metadata values in a range.

        Parameters
        ----------
        features_list : List[list]
            Complete list of features
        metadata_field : str
            Metadata field to check (e.g., "seqid")
        start_value : int
            Start of the range (inclusive)
        end_value : int
            End of the range (inclusive)

        Returns
        -------
        Set[int]
            Set of residue.index values for residues with metadata in the range
        """
        residue_indices = set()
        for feature in features_list:
            for partner in feature:
                residue_info = partner.get("residue", {})
                res_value = residue_info.get(metadata_field)
                if res_value is not None and start_value <= res_value <= end_value:
                    res_index = residue_info.get("index")
                    if res_index is not None:
                        residue_indices.add(res_index)
        return residue_indices

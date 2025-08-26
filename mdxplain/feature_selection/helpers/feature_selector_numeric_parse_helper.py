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
from typing import List


class FeatureSelectorNumericParseHelper:
    """Base helper class for numeric category parsing - array index and sequence ID based operations."""
    
    @staticmethod
    def parse_numeric_category(
        param_parts: List[str], 
        features_list: List[list], 
        metadata_field: str, 
        category_name: str,
        require_all_partners: bool = False
    ) -> List[int]:
        """
        Parse numeric category and return matching feature indices.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters:
        -----------
        param_parts : List[str]
            List of parameter parts for numeric selection
        features_list : List[list]
            List of features from metadata
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        category_name : str
            Category name for error messages ("resid" or "seqid")
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns:
        --------
        List[int]
            List of feature indices matching the numeric criteria

        Raises:
        -------
        ValueError
            If the numeric specification is invalid
        """
        all_indices = []

        for part in param_parts:
            if "-" in part:
                # Range like "123-140"
                indices = FeatureSelectorNumericParseHelper._find_by_numeric_range(
                    part, features_list, metadata_field, category_name, require_all_partners
                )
            else:
                # Single number like "123"
                indices = FeatureSelectorNumericParseHelper._find_by_single_numeric(
                    part, features_list, metadata_field, category_name, require_all_partners
                )

            all_indices.extend(indices)

        return list(set(all_indices))  # Remove duplicates
    
    @staticmethod
    def _find_by_numeric_range(
        numeric_spec: str, 
        features_list: List[list], 
        metadata_field: str, 
        category_name: str,
        require_all_partners: bool = False
    ) -> List[int]:
        """
        Find features with numeric values in the specified range.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters:
        -----------
        numeric_spec : str
            Numeric range specification (e.g., "123-140")
        features_list : List[list]
            List of features from metadata
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        category_name : str
            Category name for error messages ("resid" or "seqid")
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns:
        --------
        List[int]
            List of feature indices with values in the specified range

        Raises:
        -------
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
        
        return FeatureSelectorNumericParseHelper._find_features_in_numeric_range(
            features_list, start_id, end_id, metadata_field, require_all_partners
        )
    
    @staticmethod
    def _find_by_single_numeric(
        numeric_spec: str, 
        features_list: List[list], 
        metadata_field: str, 
        category_name: str,
        require_all_partners: bool = False
    ) -> List[int]:
        """
        Find features containing the specified single numeric value.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters:
        -----------
        numeric_spec : str
            Numeric specification (e.g., "123")
        features_list : List[list]
            List of features from metadata
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        category_name : str
            Category name for error messages ("resid" or "seqid")
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns:
        --------
        List[int]
            List of feature indices with the specified single numeric value

        Raises:
        -------
        ValueError
            If the numeric specification is invalid
        """
        if not re.match(r"^\d+$", numeric_spec):
            raise ValueError(f"Invalid {category_name}: {numeric_spec}. Expected integer number.")

        target_id = int(numeric_spec)
        matching_indices = []

        for idx, feature in enumerate(features_list):
            if require_all_partners:
                # ALL partners must have the target numeric value
                if len(feature) == 0:
                    continue  # Skip empty features
                all_match = True
                for partner in feature:
                    residue_info = partner.get("residue", {})
                    res_value = residue_info.get(metadata_field)
                    if res_value != target_id:
                        all_match = False
                        break
                if all_match:
                    matching_indices.append(idx)
            else:
                # At least ONE partner must have the target numeric value (original behavior)
                for partner in feature:
                    residue_info = partner.get("residue", {})
                    res_value = residue_info.get(metadata_field)

                    if res_value == target_id:
                        matching_indices.append(idx)
                        break  # Found match in this feature, move to next

        return matching_indices
    
    @staticmethod
    def _parse_numeric_range(numeric_spec: str, category_name: str) -> tuple:
        """
        Parse and validate numeric range specification.

        Parameters:
        -----------
        numeric_spec : str
            Numeric range specification
        category_name : str
            Category name for error messages ("resid" or "seqid")

        Returns:
        --------
        tuple
            Tuple of (start_id, end_id) as integers

        Raises:
        -------
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
        metadata_field: str,
        require_all_partners: bool = False
    ) -> List[int]:
        """
        Find features with numeric values in the specified range.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters:
        -----------
        features_list : List[list]
            List of features from metadata
        start_id : int
            Start of the numeric range
        end_id : int
            End of the numeric range
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns:
        --------
        List[int]
            List of feature indices with values in the specified range
        """
        matching_indices = []

        for idx, feature in enumerate(features_list):
            if FeatureSelectorNumericParseHelper._feature_has_value_in_numeric_range(
                feature, start_id, end_id, metadata_field, require_all_partners
            ):
                matching_indices.append(idx)

        return matching_indices
    
    @staticmethod
    def _feature_has_value_in_numeric_range(
        feature: list, 
        start_id: int, 
        end_id: int, 
        metadata_field: str,
        require_all_partners: bool = False
    ) -> bool:
        """
        Check if feature has any value in the specified numeric range.
        Uses configurable metadata field (e.g., residue.index or residue.seqid).

        Parameters:
        -----------
        feature : list
            Feature from metadata
        start_id : int
            Start of the numeric range
        end_id : int
            End of the numeric range
        metadata_field : str
            Metadata field to use ("index" or "seqid")
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns:
        --------
        bool
            True if feature has any value in the specified range, False otherwise
        """
        if require_all_partners:
            if len(feature) == 0:
                return False  # Empty feature cannot satisfy "all" condition
            for partner in feature:
                residue_info = partner.get("residue", {})
                res_value = residue_info.get(metadata_field)
                if res_value is None or not (start_id <= res_value <= end_id):
                    return False
            return True  # All partners satisfied the range condition
        
        for partner in feature:
            residue_info = partner.get("residue", {})
            res_value = residue_info.get(metadata_field)

            if res_value is not None and start_id <= res_value <= end_id:
                return True

        return False

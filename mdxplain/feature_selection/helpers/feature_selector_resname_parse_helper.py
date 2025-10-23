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
Resname parser helper for feature selection.

This module provides parsing functionality for resname (res) category selections,
which are based on amino acid names with optional sequence ID combinations.
"""

import re
from typing import List, Tuple, Set


class FeatureSelectorResnameParseHelper:
    """
    Helper class for res category - amino acid name based operations.

    Provides static methods to parse 'res' category selections like "ALA", "ALA13"
    to identify matching feature indices based on trajectory metadata.
    """

    @staticmethod
    def parse_res_category(param_tokens: List[str], features_list: List[list]) -> Tuple[List[int], Set[int]]:
        """
        Parse 'res' category and return matching feature indices plus matched residue indices.
        Uses amino acid names from metadata (residue.aaa_code, residue.a_code).

        Supports both:

        - "ALA" → all alanines (any seqid)
        - "ALA13" → specific alanine at seqid 13

        Parameters
        ----------
        param_tokens : List[str]
            List of parameter tokens for residue selection
        features_list : List[list]
            List of features from metadata

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)

        Raises
        ------
        ValueError
            If the residue specification is invalid
        """
        all_indices = []
        all_matched_residue_indices = set()

        for param in param_tokens:
            # Parse residue name with optional seqid: ALA, ALA13, A, A13
            residue_name, residue_seqid = FeatureSelectorResnameParseHelper._parse_residue_spec(param)

            if residue_seqid is not None:
                # Specific residue like ALA13
                indices, residue_indices = FeatureSelectorResnameParseHelper._find_by_name_and_seqid(
                    residue_name, residue_seqid, features_list
                )
            else:
                # Just residue type like ALA (all alanines)
                indices, residue_indices = FeatureSelectorResnameParseHelper._find_by_residue_name(
                    residue_name, features_list
                )

            all_indices.extend(indices)
            all_matched_residue_indices.update(residue_indices)

        unique_indices = list(set(all_indices))

        return unique_indices, all_matched_residue_indices
    
    @staticmethod
    def _parse_residue_spec(param: str) -> tuple:
        """
        Parse residue specification and extract name and optional seqid.
        
        Supports patterns like:
        
        - "ALA" → ("ALA", None)
        - "ALA13" → ("ALA", 13) 
        - "A" → ("A", None)
        - "A13" → ("A", 13)

        Parameters
        ----------
        param : str
            Residue specification (e.g., "ALA", "ALA13", "A", "A13")

        Returns
        -------
        tuple
            Tuple of (residue_name, residue_seqid or None)
            
        Raises
        ------
        ValueError
            If the residue specification is invalid
        """
        # Pattern: 1-3 letters optionally followed by digits
        pattern = r"^([A-Za-z]{1,3})(\d+)?$"
        match = re.match(pattern, param)
        
        if match is None:
            raise ValueError(f"Invalid residue specification: {param}")
        
        residue_name = match.group(1).upper()  # Normalize to uppercase
        residue_seqid = int(match.group(2)) if match.group(2) else None
        
        return residue_name, residue_seqid
    
    @staticmethod
    def _find_by_residue_name(residue_name: str, features_list: List[list]) -> Tuple[List[int], Set[int]]:
        """
        Find features containing the specified residue name (any seqid).

        Parameters
        ----------
        residue_name : str
            Residue name to find (e.g., "ALA", "A")
        features_list : List[list]
            List of features from metadata

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
        """
        matching_indices = []
        matched_residue_indices = set()

        for idx, feature in enumerate(features_list):
            feature_matches = False
            for partner in feature:
                residue = partner.get("residue", {})
                if FeatureSelectorResnameParseHelper._residue_name_matches(residue, residue_name):
                    feature_matches = True
                    # Track this residue index
                    residue_index = residue.get("index")
                    if residue_index is not None:
                        matched_residue_indices.add(residue_index)

            if feature_matches:
                matching_indices.append(idx)

        return matching_indices, matched_residue_indices
    
    @staticmethod
    def _find_by_name_and_seqid(
        residue_name: str, residue_seqid: int, features_list: List[list]
    ) -> Tuple[List[int], Set[int]]:
        """
        Find features matching both residue name and sequence ID.

        Parameters
        ----------
        residue_name : str
            Residue name to find (e.g., "ALA", "A")
        residue_seqid : int
            Sequence ID to find (e.g., 13)
        features_list : List[list]
            List of features from metadata

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
        """
        matching_indices = []
        matched_residue_indices = set()

        for idx, feature in enumerate(features_list):
            feature_matches = False
            for partner in feature:
                residue = partner.get("residue", {})
                if (
                    FeatureSelectorResnameParseHelper._residue_name_matches(residue, residue_name)
                    and residue.get("seqid") == residue_seqid
                ):
                    feature_matches = True
                    # Track this residue index
                    residue_index = residue.get("index")
                    if residue_index is not None:
                        matched_residue_indices.add(residue_index)

            if feature_matches:
                matching_indices.append(idx)

        return matching_indices, matched_residue_indices
    
    @staticmethod
    def _feature_has_residue_name(feature: list, residue_name: str) -> bool:
        """
        Check if feature has the specified residue name.

        Parameters
        ----------
        feature : list
            Feature from metadata
        residue_name : str
            Residue name to find

        Returns
        -------
        bool
            True if at least one partner has the residue name
        """
        for partner in feature:
            residue = partner.get("residue", {})
            if FeatureSelectorResnameParseHelper._residue_name_matches(residue, residue_name):
                return True
        return False

    @staticmethod
    def _feature_has_residue_name_and_seqid(
        feature: list, residue_name: str, residue_seqid: int
    ) -> bool:
        """
        Check if feature has the specified residue name and sequence ID.

        Parameters
        ----------
        feature : list
            Feature from metadata
        residue_name : str
            Residue name to find
        residue_seqid : int
            Sequence ID to find

        Returns
        -------
        bool
            True if at least one partner has both residue name and sequence ID
        """
        for partner in feature:
            residue = partner.get("residue", {})
            if (
                FeatureSelectorResnameParseHelper._residue_name_matches(residue, residue_name)
                and residue.get("seqid") == residue_seqid
            ):
                return True
        return False
    
    @staticmethod
    def _residue_name_matches(residue: dict, residue_name: str) -> bool:
        """
        Check if residue name matches either aaa_code or a_code.

        Parameters
        ----------
        residue : dict
            Residue from metadata
        residue_name : str
            Residue name to find

        Returns
        -------
        bool
            True if residue name matches either aaa_code or a_code, False otherwise
        """
        return (
            residue.get("aaa_code") == residue_name
            or residue.get("a_code") == residue_name
        )

    @staticmethod
    def _extract_residue_indices(features_list: List[list], feature_indices: List[int]) -> Set[int]:
        """
        Extract all unique residue.index values from specified features.

        Parameters
        ----------
        features_list : List[list]
            Complete list of features
        feature_indices : List[int]
            Indices of features to extract residue indices from

        Returns
        -------
        Set[int]
            Set of unique residue.index values
        """
        residue_indices = set()
        for idx in feature_indices:
            feature = features_list[idx]
            for partner in feature:
                residue_index = partner.get("residue", {}).get("index")
                if residue_index is not None:
                    residue_indices.add(residue_index)
        return residue_indices
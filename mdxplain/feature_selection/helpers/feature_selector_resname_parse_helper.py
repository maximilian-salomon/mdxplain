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
from typing import List


class FeatureSelectorResnameParseHelper:
    """Helper class for res category - amino acid name based operations."""
    
    @staticmethod
    def parse_res_category(
        param_tokens: List[str], features_list: List[list], require_all_partners: bool = False
    ) -> List[int]:
        """
        Parse 'res' category and return matching feature indices.
        Uses amino acid names from metadata (residue.aaa_code, residue.a_code).
        
        Supports both:
        - "ALA" → all alanines (any seqid)
        - "ALA13" → specific alanine at seqid 13

        Parameters:
        -----------
        param_tokens : List[str]
            List of parameter tokens for residue selection
        features_list : List[list]
            List of features from metadata
        require_all_partners : bool, default=False
            If True, ALL partners must contain the residue ID

        Returns:
        --------
        List[int]
            List of feature indices matching the residue criteria

        Raises:
        -------
        ValueError
            If the residue specification is invalid
        """
        all_indices = []

        for param in param_tokens:
            # Parse residue name with optional seqid: ALA, ALA13, A, A13
            residue_name, residue_seqid = FeatureSelectorResnameParseHelper._parse_residue_spec(param)
            
            if residue_seqid is not None:
                # Specific residue like ALA13
                indices = FeatureSelectorResnameParseHelper._find_by_name_and_seqid(
                    residue_name, residue_seqid, features_list, require_all_partners
                )
            else:
                # Just residue type like ALA (all alanines)
                indices = FeatureSelectorResnameParseHelper._find_by_residue_name(
                    residue_name, features_list, require_all_partners
                )

            all_indices.extend(indices)

        return list(set(all_indices))  # Remove duplicates
    
    @staticmethod
    def _parse_residue_spec(param: str) -> tuple:
        """
        Parse residue specification and extract name and optional seqid.
        
        Supports patterns like:
        - "ALA" → ("ALA", None)
        - "ALA13" → ("ALA", 13) 
        - "A" → ("A", None)
        - "A13" → ("A", 13)

        Parameters:
        -----------
        param : str
            Residue specification (e.g., "ALA", "ALA13", "A", "A13")

        Returns:
        --------
        tuple
            Tuple of (residue_name, residue_seqid or None)
            
        Raises:
        -------
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
    def _find_by_residue_name(
        residue_name: str, features_list: List[list], require_all_partners: bool = False
    ) -> List[int]:
        """
        Find features containing the specified residue name (any seqid).

        Parameters:
        -----------
        residue_name : str
            Residue name to find (e.g., "ALA", "A")
        features_list : List[list]
            List of features from metadata
        require_all_partners : bool, default=False
            If True, ALL partners must contain the residue name
            If False, at least ONE partner must contain the residue name

        Returns:
        --------
        List[int]
            List of feature indices with the specified residue name
        """
        matching_indices = []

        for idx, feature in enumerate(features_list):
            if FeatureSelectorResnameParseHelper._feature_has_residue_name(
                feature, residue_name, require_all_partners
            ):
                matching_indices.append(idx)

        return matching_indices
    
    @staticmethod
    def _find_by_name_and_seqid(
        residue_name: str, residue_seqid: int, features_list: List[list], require_all_partners: bool = False
    ) -> List[int]:
        """
        Find features matching both residue name and sequence ID.

        Parameters:
        -----------
        residue_name : str
            Residue name to find (e.g., "ALA", "A")
        residue_seqid : int
            Sequence ID to find (e.g., 13)
        features_list : List[list]
            List of features from metadata
        require_all_partners : bool, default=False
            If True, ALL partners must contain both residue name and sequence ID
            If False, at least ONE partner must contain both

        Returns:
        --------
        List[int]
            List of feature indices with the specified residue name and sequence ID
        """
        matching_indices = []

        for idx, feature in enumerate(features_list):
            if FeatureSelectorResnameParseHelper._feature_has_residue_name_and_seqid(
                feature, residue_name, residue_seqid, require_all_partners
            ):
                matching_indices.append(idx)

        return matching_indices
    
    @staticmethod
    def _feature_has_residue_name(
        feature: list, residue_name: str, require_all_partners: bool = False
    ) -> bool:
        """
        Check if feature has the specified residue name.

        Parameters:
        -----------
        feature : list
            Feature from metadata
        residue_name : str
            Residue name to find
        require_all_partners : bool, default=False
            If True, ALL partners must have the residue name
            If False, at least ONE partner must have the residue name

        Returns:
        --------
        bool
            True if residue name criteria are met
        """
        if require_all_partners:
            # ALL partners must have the residue name
            if len(feature) == 0:
                return False
            for partner in feature:
                residue = partner.get("residue", {})
                if not FeatureSelectorResnameParseHelper._residue_name_matches(residue, residue_name):
                    return False
            return True
        
        for partner in feature:
            residue = partner.get("residue", {})
            if FeatureSelectorResnameParseHelper._residue_name_matches(residue, residue_name):
                return True
        return False

    @staticmethod
    def _feature_has_residue_name_and_seqid(
        feature: list, residue_name: str, residue_seqid: int, require_all_partners: bool = False
    ) -> bool:
        """
        Check if feature has the specified residue name and sequence ID.

        Parameters:
        -----------
        feature : list
            Feature from metadata
        residue_name : str
            Residue name to find
        residue_seqid : int
            Sequence ID to find
        require_all_partners : bool, default=False
            If True, ALL partners must have both residue name and sequence ID
            If False, at least ONE partner must have both

        Returns:
        --------
        bool
            True if residue name and sequence ID criteria are met
        """
        if require_all_partners:
            # ALL partners must have both name and seqid
            if len(feature) == 0:
                return False
            for partner in feature:
                residue = partner.get("residue", {})
                if not (
                    FeatureSelectorResnameParseHelper._residue_name_matches(residue, residue_name)
                    and residue.get("seqid") == residue_seqid
                ):
                    return False
            return True
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

        Parameters:
        -----------
        residue : dict
            Residue from metadata
        residue_name : str
            Residue name to find

        Returns:
        --------
        bool
            True if residue name matches either aaa_code or a_code, False otherwise
        """
        return (
            residue.get("aaa_code") == residue_name
            or residue.get("a_code") == residue_name
        )
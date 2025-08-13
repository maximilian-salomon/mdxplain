# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
Core parser helper for feature selection.

This module provides the main coordination functionality for feature selection parsing,
including smart pattern detection and delegation to specialized category helpers.
"""

import re
import warnings
from typing import List, Union

from .feature_selector_resid_parse_helper import FeatureSelectorResidParseHelper
from .feature_selector_seqid_parse_helper import FeatureSelectorSeqidParseHelper
from .feature_selector_resname_parse_helper import FeatureSelectorResnameParseHelper


class FeatureSelectorParseCoreHelper:
    """Core parser that coordinates pattern detection and delegates to specialized helpers."""
    
    @staticmethod
    def parse_selection(selection_string: str, features_list: List[list]) -> List[int]:
        """
        Parse selection string and return matching feature indices.
        Coordinates pattern detection and delegates to specialized category helpers.

        Parameters:
        -----------
        selection_string : str
            Selection string (e.g., "res ALA HIS", "resid 123 130", "res ALA and not resid 124-130")
        features_list : List[list]
            List of features from metadata (each feature is a list of partners)

        Returns:
        --------
        List[int]
            List of feature indices that match the selection

        Raises:
        -------
        ValueError
            If the selection string is invalid
        """
        instructions = FeatureSelectorParseCoreHelper._split_instructions(selection_string)
        positive, negative = FeatureSelectorParseCoreHelper._process_instructions(
            instructions, features_list
        )
        return FeatureSelectorParseCoreHelper._combine_results(positive, negative, features_list)
    
    @staticmethod
    def _split_instructions(selection_string: str) -> List[str]:
        """
        Split selection string into individual instructions.

        Parameters:
        -----------
        selection_string : str
            Selection string to split

        Returns:
        --------
        List[str]
            List of individual instruction strings
        """
        return [s.strip() for s in selection_string.lower().split("and")]
    
    @staticmethod
    def _process_instructions(
        instructions: List[str], features_list: List[list]
    ) -> tuple:
        """
        Process all instructions and return positive and negative indices.

        Parameters:
        -----------
        instructions : List[str]
            List of instruction strings
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        tuple
            Tuple of (positive_indices, negative_indices)

        Raises:
        -------
        ValueError
            If the instruction is invalid
        """
        positive = []
        negative = []

        for instr in instructions:
            instr_positive, instr_negative = (
                FeatureSelectorParseCoreHelper._process_single_instruction(instr, features_list)
            )
            positive.extend(instr_positive)
            negative.extend(instr_negative)

        return positive, negative
    
    @staticmethod
    def _process_single_instruction(
        instruction: str, features_list: List[list]
    ) -> tuple:
        """
        Process a single instruction and return positive and negative indices.

        Parameters:
        -----------
        instruction : str
            Single instruction string
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        tuple
            Tuple of (positive_indices, negative_indices)

        Raises:
        -------
        ValueError
            If the instruction is invalid
        """
        tokens = [t for t in instruction.split() if t]
        if not tokens:
            raise ValueError(f"Invalid instruction: {instruction}")

        category, parameters, is_negative = (
            FeatureSelectorParseCoreHelper._extract_category_info(tokens)
        )
        indices = FeatureSelectorParseCoreHelper._get_indices_for_category(
            category, parameters, features_list
        )

        if is_negative:
            return [], indices
        else:
            return indices, []
    
    @staticmethod
    def _get_indices_for_category(
        category: str, parameters: str, features_list: List[list]
    ) -> List[int]:
        """
        Get indices for a specific category and parameters.
        Delegates to appropriate specialized helper.

        Parameters:
        -----------
        category : str
            Category type (e.g., 'res', 'resid', 'seqid')
        parameters : str
            Parameters string for the category
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            List of feature indices matching the category and parameters

        Raises:
        -------
        ValueError
            If the category is unknown
        """
        category_parsers = {
            "res": FeatureSelectorResnameParseHelper.parse_res_category,
            "resid": FeatureSelectorResidParseHelper.parse_resid_category,
            "seqid": FeatureSelectorSeqidParseHelper.parse_seqid_category,
        }

        normalized_params = FeatureSelectorParseCoreHelper._normalize_and_split_parameters(
            parameters
        )

        if not normalized_params:
            return []
        elif normalized_params == "ALL":
            return list(range(len(features_list)))
        else:
            # Ensure normalized_params is a list for the parser functions
            if isinstance(normalized_params, str):
                # This should only happen if normalized_params is "ALL", which is handled above
                raise ValueError(f"Unexpected string parameter: {normalized_params}")
            return category_parsers[category](normalized_params, features_list)
    
    @staticmethod
    def _combine_results(
        positive: List[int], negative: List[int], features_list: List[list]
    ) -> List[int]:
        """
        Combine positive and negative results into final selection.

        Parameters:
        -----------
        positive : List[int]
            List of feature indices
        negative : List[int]
            List of feature indices to exclude
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            Final list of selected feature indices
        """
        if not positive and negative:
            positive = list(range(len(features_list)))

        result = sorted(set(positive) - set(negative))
        
        # Warning for empty selections
        if not result:
            warnings.warn(
                "Feature selection resulted in no matches. "
                "Check your selection criteria.",
                UserWarning
            )
        
        return result
    
    @staticmethod
    def _normalize_and_split_parameters(parameters: str) -> Union[List[str], str]:
        """
        Normalize parameters to uppercase and split by separators.

        Parameters:
        -----------
        parameters : str
            Parameters string to normalize and split

        Returns:
        --------
        Union[List[str], str]
            List of parameter tokens or "ALL" string
        """
        if not parameters.strip():
            return []

        # Normalize parameters to uppercase
        parameters = parameters.strip().upper()
        if parameters == "ALL":
            return "ALL"

        # Split parameters by comma, whitespace, tab, semicolon and dot delimiters
        return [p.strip() for p in re.split(r"[,\s.;\t]+", parameters) if p.strip()]
    
    @staticmethod
    def _extract_category_info(tokens: List[str]) -> tuple:
        """
        Extract category, parameters and negative flag from tokens.
        Implements smart pattern detection for user-friendly syntax.

        Smart pattern detection:
        - '0-2' or '123' or '123-140' → 'seqid 0-2' or 'seqid 123' or 'seqid 123-140'  
        - 'ALA' or 'A' or 'LEU' → 'res ALA' or 'res A' or 'res LEU'
        - 'ALA13' or 'A13' → 'res ALA13' or 'res A13'

        Parameters:
        -----------
        tokens : List[str]
            List of tokens from instruction

        Returns:
        --------
        tuple
            Tuple of (category, parameters, is_negative)

        Raises:
        -------
        ValueError
            If the instruction is invalid
        """
        if tokens[0] == "not":
            if len(tokens) < 2:
                raise ValueError(f"Invalid 'not' instruction: {' '.join(tokens)}")
            
            # Apply smart pattern detection to ALL tokens after 'not'
            # This enables complex patterns like "not HIS" -> "not res HIS"
            remaining_tokens = tokens[1:]
            category, parameters = FeatureSelectorParseCoreHelper._apply_smart_pattern_detection_to_tokens(
                remaining_tokens
            )
            return (category, parameters, True)
        else:
            # Apply smart pattern detection to all tokens
            category, parameters = FeatureSelectorParseCoreHelper._apply_smart_pattern_detection_to_tokens(
                tokens
            )
            return (category, parameters, False)
    
    @staticmethod
    def _apply_smart_pattern_detection_to_tokens(tokens: List[str]) -> tuple:
        """
        Apply smart pattern detection to a list of tokens.

        Enables complex patterns like:
        - ['HIS'] -> ('res', 'HIS')  
        - ['13'] -> ('seqid', '13')
        - ['res', 'ALA'] -> ('res', 'ALA')
        - ['13', '14', '15'] -> ('seqid', '13 14 15')

        Parameters:
        -----------
        tokens : List[str]
            List of tokens to analyze

        Returns:
        --------
        tuple
            Tuple of (category, parameters)
        """
        if not tokens:
            raise ValueError("Empty token list provided")
        
        first_token = tokens[0]
        remaining_params = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        
        # Known explicit categories - no pattern detection needed
        known_categories = {"res", "resid", "seqid"}
        if first_token.lower() in known_categories:
            return (first_token.lower(), remaining_params)
        
        # Special case: "all" is a valid parameter for any category
        if first_token.lower() == "all":
            # Default "all" to resid category
            return ("resid", "all")
        
        # Check if ALL tokens match the same pattern type
        # This handles cases like "13 14 15" (all numbers) -> "seqid 13 14 15"
        all_numeric = all(re.match(r"^\d+$", token) or re.match(r"^\d+-\d+$", token) for token in tokens)
        all_aminoacid = all(re.match(r"^[A-Z]{1,3}\d*$", token.upper()) for token in tokens)
        
        if all_numeric:
            # All tokens are numeric -> seqid category (biological sequence ID)
            return ("seqid", " ".join(tokens))
        elif all_aminoacid:
            # All tokens are aminoacid patterns -> res category
            return ("res", " ".join(token.upper() for token in tokens))
        else:
            # Unknown or mixed patterns - require explicit syntax
            raise ValueError(
                f"Unknown pattern or mixed pattern in '{' '.join(tokens)}'. "
                "Use resid, seqid or res as category followed by the values. "
                "Combine different categories with 'and'."
            )
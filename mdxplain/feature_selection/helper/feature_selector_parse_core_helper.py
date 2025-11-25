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
Core parser helper for feature selection.

This module provides the main coordination functionality for feature selection parsing,
including smart pattern detection and delegation to specialized category helper.
"""

import re
import warnings
from typing import List, Union, Set, Tuple

from .feature_selector_resid_parse_helper import FeatureSelectorResidParseHelper
from .feature_selector_seqid_parse_helper import FeatureSelectorSeqidParseHelper
from .feature_selector_resname_parse_helper import FeatureSelectorResnameParseHelper
from .feature_selector_consensus_parse_helper import FeatureSelectorConsensusParseHelper


class FeatureSelectorParseCoreHelper:
    """
    Core parser that coordinates pattern detection and delegates to specialized helper.

    This class implements the main logic for parsing feature selection strings,
    including smart pattern detection to infer categories when not explicitly provided.
    """
    
    @staticmethod
    def parse_selection(selection_string: str, features_list: List[list], require_all_partners: bool = False) -> List[int]:
        """
        Parse selection string and return matching feature indices.
        Coordinates pattern detection and delegates to specialized category helper.

        Parameters
        ----------
        selection_string : str
            Selection string (e.g., "res ALA HIS", "resid 123 130", "res ALA and not resid 124-130")
        features_list : List[list]
            List of features from metadata (each feature is a list of partners)
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns
        -------
        List[int]
            List of feature indices that match the selection

        Raises
        ------
        ValueError
            If the selection string is invalid
        """
        instructions = FeatureSelectorParseCoreHelper._split_instructions(selection_string)

        # Process instructions and collect matched residue indices
        positive, negative, matched_residue_indices, excluded_residue_indices, had_negative_instructions = (
            FeatureSelectorParseCoreHelper._process_instructions(instructions, features_list)
        )

        # Combine matched residue indices (Union - Subtraction)
        final_matched_residue_indices = matched_residue_indices - excluded_residue_indices

        # If we had only "not" instructions without positive, use all residue indices from positive features
        if not matched_residue_indices and had_negative_instructions and positive:
            final_matched_residue_indices = FeatureSelectorParseCoreHelper._extract_all_residue_indices(
                positive, features_list
            )
            # Subtract excluded
            final_matched_residue_indices = final_matched_residue_indices - excluded_residue_indices

        # Apply require_all_partners filtering AFTER collecting all matches
        if require_all_partners and positive:
            if final_matched_residue_indices:
                positive = FeatureSelectorParseCoreHelper._apply_require_all_partners_filter(
                    positive, features_list, final_matched_residue_indices
                )
            else:
                # No matched residue indices means no features can satisfy require_all_partners
                positive = []

        result = FeatureSelectorParseCoreHelper._combine_results(positive, negative, features_list, had_negative_instructions)
        return result
    
    @staticmethod
    def _split_instructions(selection_string: str) -> List[str]:
        """
        Split selection string into individual instructions.

        Parameters
        ----------
        selection_string : str
            Selection string to split

        Returns
        -------
        List[str]
            List of individual instruction strings
        """
        return [s.strip() for s in selection_string.lower().split("and")]
    
    @staticmethod
    def _process_instructions(instructions: List[str], features_list: List[list]) -> tuple:
        """
        Process all instructions and return indices and residue indices.

        Parameters
        ----------
        instructions : List[str]
            List of instruction strings
        features_list : List[list]
            List of features from metadata

        Returns
        -------
        tuple
            Tuple of (positive_indices, negative_indices, matched_residue_indices, excluded_residue_indices, had_negative_instructions)

        Raises
        ------
        ValueError
            If the instruction is invalid
        """
        positive = []
        negative = []
        matched_residue_indices = set()
        excluded_residue_indices = set()
        had_negative_instructions = False

        for instr in instructions:
            # Check if this instruction starts with "not"
            if instr.strip().startswith("not "):
                had_negative_instructions = True

            instr_positive, instr_negative, instr_matched_residues, instr_excluded_residues = (
                FeatureSelectorParseCoreHelper._process_single_instruction(instr, features_list)
            )
            positive.extend(instr_positive)
            negative.extend(instr_negative)
            matched_residue_indices.update(instr_matched_residues)  # Union
            excluded_residue_indices.update(instr_excluded_residues)  # Union

        return positive, negative, matched_residue_indices, excluded_residue_indices, had_negative_instructions
    
    @staticmethod
    def _process_single_instruction(instruction: str, features_list: List[list]) -> tuple:
        """
        Process a single instruction and return indices and residue indices.

        Parameters
        ----------
        instruction : str
            Single instruction string
        features_list : List[list]
            List of features from metadata

        Returns
        -------
        tuple
            Tuple of (positive_indices, negative_indices, matched_residue_indices, excluded_residue_indices)

        Raises
        ------
        ValueError
            If the instruction is invalid
        """
        tokens = [t for t in instruction.split() if t]
        if not tokens:
            raise ValueError(f"Invalid instruction: {instruction}")

        category, parameters, is_negative = (
            FeatureSelectorParseCoreHelper._extract_category_info(tokens)
        )
        indices, residue_indices = FeatureSelectorParseCoreHelper._get_indices_for_category(
            category, parameters, features_list
        )

        if is_negative:
            return [], indices, set(), residue_indices
        else:
            return indices, [], residue_indices, set()
    
    @staticmethod
    def _get_indices_for_category(category: str, parameters: str, features_list: List[list]) -> Tuple[List[int], Set[int]]:
        """
        Get indices and matched residue indices for a specific category and parameters.
        Delegates to appropriate specialized helper.

        Parameters
        ----------
        category : str
            Category type (e.g., 'res', 'resid', 'seqid')
        parameters : str
            Parameters string for the category
        features_list : List[list]
            List of features from metadata

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)

        Raises
        ------
        ValueError
            If the category is unknown
        """
        category_parsers = {
            "res": FeatureSelectorResnameParseHelper.parse_res_category,
            "resid": FeatureSelectorResidParseHelper.parse_resid_category,
            "seqid": FeatureSelectorSeqidParseHelper.parse_seqid_category,
            "consensus": FeatureSelectorConsensusParseHelper.parse_consensus_category,
        }

        if category not in category_parsers:
            raise ValueError(f"Unknown category: {category}")

        # Handle parameters - don't split for consensus category
        if category == "consensus":
            normalized_params = [parameters.strip()] if parameters.strip() else []
        else:
            normalized_params = FeatureSelectorParseCoreHelper._normalize_and_split_parameters(parameters)

        if not normalized_params:
            return [], set()
        elif normalized_params == "ALL":
            # Extract all residue indices from all features
            all_indices = list(range(len(features_list)))
            all_residue_indices = FeatureSelectorParseCoreHelper._extract_all_residue_indices(
                all_indices, features_list
            )
            return all_indices, all_residue_indices
        else:
            # Ensure normalized_params is a list for the parser functions
            if isinstance(normalized_params, str):
                # This should only happen if normalized_params is "ALL", which is handled above
                raise ValueError(f"Unexpected string parameter: {normalized_params}")
            return category_parsers[category](normalized_params, features_list)
    
    @staticmethod
    def _combine_results(
        positive: List[int], negative: List[int], features_list: List[list], had_negative_instructions: bool = False
    ) -> List[int]:
        """
        Combine positive and negative results into final selection.

        Parameters
        ----------
        positive : List[int]
            List of feature indices
        negative : List[int]
            List of feature indices to exclude
        features_list : List[list]
            List of features from metadata
        had_negative_instructions : bool, default=False
            Whether the selection contained any "not" instructions

        Returns
        -------
        List[int]
            Final list of selected feature indices
        """
        # If we had negative instructions but no positive selections,
        # assume "all" as the base (e.g., "not 66" means "all except 66")
        if not positive and had_negative_instructions:
            positive = list(range(len(features_list)))

        result = sorted(set(positive) - set(negative))

        # Warning for empty selections, but not if we had negative instructions
        # (because "not nonexistent" should return all features, not trigger warning)
        if not result and not had_negative_instructions:
            warnings.warn(
                "Feature selection resulted in no matches. "
                "Check your selection criteria.",
                UserWarning
            )

        return result
    
    @staticmethod
    def _apply_require_all_partners_filter(positive_indices: List[int], features_list: List[list], matched_residue_indices: Set[int]) -> List[int]:
        """
        Filter features to only those where ALL partners match the selection criteria.

        This implements the require_all_partners logic at the top level, after all
        pattern matching is complete. Uses the matched_residue_indices to filter
        features where ALL partners are in the matched set.

        Parameters
        ----------
        positive_indices : List[int]
            Indices of features that matched the selection criteria
        features_list : List[list]
            Complete list of features
        matched_residue_indices : Set[int]
            Set of residue.index values that matched the selection criteria

        Returns
        -------
        List[int]
            Filtered indices where all partners match the criteria
        """
        filtered_indices = []
        for idx in positive_indices:
            feature = features_list[idx]
            if len(feature) == 0:
                continue
            # Check if ALL partners have their residue.index in matched_residue_indices
            all_in_set = all(
                partner.get("residue", {}).get("index") in matched_residue_indices
                for partner in feature
            )
            if all_in_set:
                filtered_indices.append(idx)
        return filtered_indices

    @staticmethod
    def _extract_all_residue_indices(feature_indices: List[int], features_list: List[list]) -> Set[int]:
        """
        Extract all unique residue.index values from specified features.

        Parameters
        ----------
        feature_indices : List[int]
            Indices of features to extract residue indices from
        features_list : List[list]
            Complete list of features

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

    @staticmethod
    def _normalize_and_split_parameters(parameters: str) -> Union[List[str], str]:
        """
        Normalize parameters to uppercase and split by separators.

        Parameters
        ----------
        parameters : str
            Parameters string to normalize and split

        Returns
        -------
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

        Parameters
        ----------
        tokens : List[str]
            List of tokens from instruction

        Returns
        -------
        tuple
            Tuple of (category, parameters, is_negative)

        Raises
        ------
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

        Parameters
        ----------
        tokens : List[str]
            List of tokens to analyze

        Returns
        -------
        tuple
            Tuple of (category, parameters)
        """
        if not tokens:
            raise ValueError("Empty token list provided")

        first_token = tokens[0]
        remaining_params = " ".join(tokens[1:]) if len(tokens) > 1 else ""

        # Check for explicit category first
        explicit_category = FeatureSelectorParseCoreHelper._check_explicit_category(
            first_token, remaining_params
        )
        if explicit_category:
            return explicit_category

        # Auto-detect category from token patterns
        return FeatureSelectorParseCoreHelper._detect_category_from_patterns(tokens)

    @staticmethod
    def _check_explicit_category(first_token: str, remaining_params: str) -> Union[tuple, None]:
        """
        Check if first token is an explicit category keyword.

        Parameters
        ----------
        first_token : str
            First token from the list
        remaining_params : str
            Remaining parameters joined

        Returns
        -------
        Union[tuple, None]
            Tuple of (category, parameters) if explicit, None otherwise
        """
        known_categories = {"res", "resid", "seqid", "consensus"}
        if first_token.lower() in known_categories:
            return (first_token.lower(), remaining_params)

        if first_token.lower() == "all":
            return ("resid", "all")

        return None

    @staticmethod
    def _detect_category_from_patterns(tokens: List[str]) -> tuple:
        """
        Auto-detect category from token patterns.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to analyze

        Returns
        -------
        tuple
            Tuple of (category, parameters)

        Raises
        ------
        ValueError
            If patterns are unknown or mixed
        """
        all_numeric = FeatureSelectorParseCoreHelper._all_tokens_numeric(tokens)
        all_aminoacid = FeatureSelectorParseCoreHelper._all_tokens_aminoacid(tokens)

        if all_numeric:
            return ("seqid", " ".join(tokens))
        elif all_aminoacid:
            return ("res", " ".join(token.upper() for token in tokens))
        else:
            raise ValueError(
                f"Unknown pattern or mixed pattern in '{' '.join(tokens)}'. "
                "Use resid, seqid, res, or consensus as category followed by the values. "
                "Combine different categories with 'and'."
            )

    @staticmethod
    def _all_tokens_numeric(tokens: List[str]) -> bool:
        """
        Check if all tokens are numeric or numeric ranges.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to check

        Returns
        -------
        bool
            True if all tokens match numeric pattern
        """
        return all(re.match(r"^\d+$", token) or re.match(r"^\d+-\d+$", token) for token in tokens)

    @staticmethod
    def _all_tokens_aminoacid(tokens: List[str]) -> bool:
        """
        Check if all tokens are amino acid patterns.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to check

        Returns
        -------
        bool
            True if all tokens match amino acid pattern
        """
        return all(re.match(r"^[A-Z]{1,3}\d*$", token.upper()) for token in tokens)
    
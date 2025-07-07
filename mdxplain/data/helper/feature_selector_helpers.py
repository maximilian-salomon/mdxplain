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
Helper classes for FeatureSelector to reduce complexity.

Contains utility classes for parsing residue labels and building lookup tables.
"""

import re
from typing import List, Union


class FeatureSelectorHelper:
    """Helper class for parsing feature selections using structured metadata."""

    @staticmethod
    def parse_selection(selection_string: str, features_list: List[list]) -> List[int]:
        """
        Parse selection string and return matching feature indices.

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
        instructions = FeatureSelectorHelper._split_instructions(selection_string)
        positive, negative = FeatureSelectorHelper._process_instructions(
            instructions, features_list
        )
        return FeatureSelectorHelper._combine_results(positive, negative, features_list)

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
                FeatureSelectorHelper._process_single_instruction(instr, features_list)
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
            FeatureSelectorHelper._extract_category_info(tokens)
        )
        indices = FeatureSelectorHelper._get_indices_for_category(
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

        Parameters:
        -----------
        category : str
            Category type (e.g., 'res', 'resid')
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
            "res": FeatureSelectorHelper._parse_res_category,
            "resid": FeatureSelectorHelper._parse_resid_category,
        }

        if category not in category_parsers:
            raise ValueError(f"Unknown category: {category}")

        normalized_params = FeatureSelectorHelper._normalize_and_split_parameters(
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
        if not positive:
            positive = list(range(len(features_list)))

        result = sorted(set(positive) - set(negative))
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

        # Split parameters by whitespace and commas
        return [p.strip() for p in re.split(r"[,\s.;\t]+", parameters) if p.strip()]

    @staticmethod
    def _extract_category_info(tokens: List[str]) -> tuple:
        """
        Extract category, parameters and negative flag from tokens.

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
            category = tokens[1]
            parameters = " ".join(tokens[2:]) if len(tokens) > 2 else ""
            return (category, parameters, True)
        else:
            category = tokens[0]
            parameters = " ".join(tokens[1:]) if len(tokens) > 1 else ""
            return (category, parameters, False)

    @staticmethod
    def _parse_res_category(
        param_tokens: List[str], features_list: List[list]
    ) -> List[int]:
        """
        Parse 'res' category and return matching feature indices.

        Parameters:
        -----------
        param_tokens : List[str]
            List of parameter tokens for residue selection
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            List of feature indices matching the residue criteria

        Raises:
        -------
        ValueError
            If the residue ID specification is invalid
        """
        all_indices = []

        for param in param_tokens:
            # Check if it's a specific residue (like HIS123) or just a type (like ALA)
            pattern = r"^([A-Za-z]{1,3})(\d+)$"
            match = re.match(pattern, param)

            if match:
                # Specific residue like HIS123
                indices = FeatureSelectorHelper._find_by_direct_residue(
                    param, features_list
                )
            else:
                # Just residue type like ALA
                indices = FeatureSelectorHelper._find_by_residue_name(
                    param, features_list
                )

            all_indices.extend(indices)

        return list(set(all_indices))  # Remove duplicates

    @staticmethod
    def _parse_resid_category(
        param_parts: List[str], features_list: List[list]
    ) -> List[int]:
        """
        Parse 'resid' category and return matching feature indices.

        Parameters:
        -----------
        param_parts : List[str]
            List of parameter parts for residue ID selection
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            List of feature indices matching the residue ID criteria

        Raises:
        -------
        ValueError
            If the residue ID specification is invalid
        """
        all_indices = []

        for part in param_parts:
            if "-" in part:
                # Range like "123-140"
                indices = FeatureSelectorHelper._find_by_resid_range(
                    part, features_list
                )
            else:
                # Single number like "123"
                indices = FeatureSelectorHelper._find_by_single_resid(
                    part, features_list
                )

            all_indices.extend(indices)

        return list(set(all_indices))  # Remove duplicates

    @staticmethod
    def _find_by_resid_range(resid_spec: str, features_list: List[list]) -> List[int]:
        """
        Find features with residue IDs in the specified range.

        Parameters:
        -----------
        resid_spec : str
            Residue ID range specification (e.g., "123-140")
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            List of feature indices with residues in the specified range

        Raises:
        -------
        ValueError
            If the residue range specification is invalid
        """
        start_id, end_id = FeatureSelectorHelper._parse_resid_range(resid_spec)
        return FeatureSelectorHelper._find_features_in_range(
            features_list, start_id, end_id
        )

    @staticmethod
    def _parse_resid_range(resid_spec: str) -> tuple:
        """
        Parse and validate residue ID range specification.

        Parameters:
        -----------
        resid_spec : str
            Residue ID range specification

        Returns:
        --------
        tuple
            Tuple of (start_id, end_id) as integers

        Raises:
        -------
        ValueError
            If the residue range specification is invalid
        """
        if not re.match(r"^\d+-\d+$", resid_spec):
            raise ValueError(
                f"Invalid resid range format: {resid_spec}. "
                "Expected format: 'Start-End'"
            )

        start_str, end_str = resid_spec.split("-")
        return int(start_str), int(end_str)

    @staticmethod
    def _find_features_in_range(
        features_list: List[list], start_id: int, end_id: int
    ) -> List[int]:
        """
        Find features with residue indices in the specified range.

        Parameters:
        -----------
        features_list : List[list]
            List of features from metadata
        start_id : int
            Start of the residue ID range
        end_id : int
            End of the residue ID range

        Returns:
        --------
        List[int]
            List of feature indices with residues in the specified range

        Raises:
        -------
        ValueError
            If the residue range specification is invalid
        """
        matching_indices = []

        for idx, feature in enumerate(features_list):
            if FeatureSelectorHelper._feature_has_residue_in_range(
                feature, start_id, end_id
            ):
                matching_indices.append(idx)

        return matching_indices

    @staticmethod
    def _feature_has_residue_in_range(
        feature: list, start_id: int, end_id: int
    ) -> bool:
        """
        Check if feature has any residue in the specified range.

        Parameters:
        -----------
        feature : list
            Feature from metadata
        start_id : int
            Start of the residue ID range
        end_id : int
            End of the residue ID range

        Returns:
        --------
        bool
            True if feature has any residue in the specified range, False otherwise
        """
        for partner in feature:
            residue_info = partner.get("residue", {})
            res_index = residue_info.get("index")

            if res_index is not None and start_id <= res_index <= end_id:
                return True

        return False

    @staticmethod
    def _find_by_single_resid(resid_spec: str, features_list: List[list]) -> List[int]:
        """
        Find features containing the specified single residue ID.

        Parameters:
        -----------
        resid_spec : str
            Residue ID specification (e.g., "123")
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            List of feature indices with the specified single residue ID

        Raises:
        -------
        ValueError
            If the residue ID specification is invalid
        """
        if not re.match(r"^\d+$", resid_spec):
            raise ValueError(f"Invalid resid: {resid_spec}. Expected format: '123'")

        target_id = int(resid_spec)
        matching_indices = []

        for idx, feature in enumerate(features_list):
            for partner in feature:
                residue_info = partner.get("residue", {})
                res_index = residue_info.get("index")

                if res_index == target_id:
                    matching_indices.append(idx)
                    break  # Found match in this feature, move to next

        return matching_indices

    @staticmethod
    def _find_by_residue_name(
        residue_name: str, features_list: List[list]
    ) -> List[int]:
        """
        Find features containing the specified residue name.

        Parameters:
        -----------
        residue_name : str
            Residue name to find
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            List of feature indices with the specified residue name
        """
        matching_indices = []

        for idx, feature in enumerate(features_list):
            # Check each partner in the feature
            for partner in feature:
                residue_info = partner.get("residue", {})
                if (
                    residue_info.get("aaa_code") == residue_name
                    or residue_info.get("a_code") == residue_name
                ):
                    matching_indices.append(idx)
                    break

        return matching_indices

    @staticmethod
    def _find_by_direct_residue(token: str, features_list: List[list]) -> List[int]:
        """
        Find features by direct residue specification like 'ALA123' or 'A123'.

        Parameters:
        -----------
        token : str
            Residue specification (e.g., "ALA123" or "A123")
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        List[int]
            List of feature indices with the specified residue name and ID
        """
        residue_name, residue_id = FeatureSelectorHelper._parse_direct_residue(token)
        return FeatureSelectorHelper._find_features_by_name_and_id(
            features_list, residue_name, residue_id
        )

    @staticmethod
    def _parse_direct_residue(token: str) -> tuple:
        """
        Parse direct residue specification and extract name and ID.

        Parameters:
        -----------
        token : str
            Residue specification (e.g., "ALA123" or "A123")
        features_list : List[list]
            List of features from metadata

        Returns:
        --------
        tuple
            Tuple of (residue_name, residue_id)
        """
        pattern = r"^([A-Za-z]{1,3})(\d+)$"
        match = re.match(pattern, token)

        if match is None:
            raise ValueError(f"Invalid residue specification: {token}")

        residue_name = match.group(1)
        residue_id = int(match.group(2))

        return residue_name, residue_id

    @staticmethod
    def _find_features_by_name_and_id(
        features_list: List[list], residue_name: str, residue_id: int
    ) -> List[int]:
        """
        Find features matching both residue name and ID.

        Parameters:
        -----------
        features_list : List[list]
            List of features from metadata
        residue_name : str
            Residue name to find
        residue_id : int
            Residue ID to find

        Returns:
        --------
        List[int]
            List of feature indices with the specified residue name and ID
        """
        result = []

        for idx, feature in enumerate(features_list):
            if FeatureSelectorHelper._feature_has_residue_name_and_id(
                feature, residue_name, residue_id
            ):
                result.append(idx)

        return result

    @staticmethod
    def _feature_has_residue_name_and_id(
        feature: list, residue_name: str, residue_id: int
    ) -> bool:
        """
        Check if feature has a residue with the specified name and ID.

        Parameters:
        -----------
        feature : list
            Feature from metadata
        residue_name : str
            Residue name to find
        residue_id : int
            Residue ID to find

        Returns:
        --------
        bool
            True if feature has a residue with the specified name and ID, False otherwise
        """
        for partner in feature:
            residue = partner.get("residue", {})
            if (
                FeatureSelectorHelper._residue_name_matches(residue, residue_name)
                and residue.get("index") == residue_id
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

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
        """
        # Category to parser mapping
        category_parsers = {
            "res": FeatureSelectorHelper._parse_res_category,
            "resid": FeatureSelectorHelper._parse_resid_category
        }
        
        # Split by 'and' for multiple instructions
        instructions = [s.strip() for s in selection_string.lower().split('and')]
        positive = []
        negative = []
        
        for instr in instructions:
            tokens = [t for t in instr.split() if t]  # Filter empty strings
            if not tokens:
                raise ValueError(f"Invalid selection string: {selection_string}")
            
            # Extract category and parameters using helper
            category, parameters, is_negative = FeatureSelectorHelper._extract_category_info(tokens)
            
            if category not in category_parsers:
                raise ValueError(f"Unknown category: {category}")
            
            # Normalize and split parameters
            normalized_params = FeatureSelectorHelper._normalize_and_split_parameters(parameters)
            
            # Get indices using appropriate parser
            if not normalized_params:
                continue;
            elif normalized_params == "ALL":
                indices = list(range(len(features_list)))
            else:
                indices = category_parsers[category](normalized_params, features_list)
            
            if is_negative:
                negative.extend(indices)
            else:
                positive.extend(indices)
        
        # Wenn keine positiven, dann alle nehmen
        if not positive:
            positive = list(range(len(features_list)))
        
        # Ergebnis: positive minus negative
        result = sorted(set(positive) - set(negative))
        return result
    
    @staticmethod
    def _normalize_and_split_parameters(parameters: str) -> Union[List[str], str]:
        """Normalize parameters to uppercase and split by separators."""
        if not parameters.strip():
            return []
            
        # Normalize parameters to uppercase
        parameters = parameters.strip().upper()
        if parameters == "ALL":
            return "ALL"
        
        # Split parameters by whitespace and commas
        return [p.strip() for p in re.split(r'[,\s.;\t]+', parameters) if p.strip()]
    
    @staticmethod
    def _extract_category_info(tokens: List[str]) -> tuple:
        """
        Extract category, parameters and negative flag from tokens.
        
        Returns:
        --------
        tuple: (category, parameters, is_negative) or None if invalid
        """
        if tokens[0] == "not":
            if len(tokens) < 2:
                raise ValueError(f"Invalid 'not' instruction: {' '.join(tokens)}")
            category = tokens[1]
            parameters = ' '.join(tokens[2:]) if len(tokens) > 2 else ""
            return (category, parameters, True)
        else:
            category = tokens[0]
            parameters = ' '.join(tokens[1:]) if len(tokens) > 1 else ""
            return (category, parameters, False)
    
    @staticmethod
    def _parse_res_category(param_tokens: List[str], features_list: List[list]) -> List[int]:
        """Parse 'res' category and return matching feature indices."""
        all_indices = []
        
        for param in param_tokens:
            # Check if it's a specific residue (like HIS123) or just a type (like ALA)
            pattern = r'^([A-Za-z]{1,3})(\d+)$'
            match = re.match(pattern, param)
            
            if match:
                # Specific residue like HIS123
                indices = FeatureSelectorHelper._find_by_direct_residue(param, features_list)
            else:
                # Just residue type like ALA
                indices = FeatureSelectorHelper._find_by_residue_name(param, features_list)
            
            all_indices.extend(indices)
        
        return list(set(all_indices))  # Remove duplicates
    
    @staticmethod
    def _parse_resid_category(param_parts: List[str], features_list: List[list]) -> List[int]:
        """Parse 'resid' category and return matching feature indices."""
        all_indices = []
        
        for part in param_parts:
            if "-" in part:
                # Range like "123-140"
                indices = FeatureSelectorHelper._find_by_resid_range(part, features_list)
            else:
                # Single number like "123"
                indices = FeatureSelectorHelper._find_by_single_resid(part, features_list)
            
            all_indices.extend(indices)
        
        return list(set(all_indices))  # Remove duplicates
    
    @staticmethod
    def _find_by_resid_range(resid_spec: str, features_list: List[list]) -> List[int]:
        """Find features with residue IDs in the specified range."""
        # Validate format first
        if not re.match(r'^\d+-\d+$', resid_spec):
            raise ValueError(
                f"Invalid resid range format: {resid_spec}. "
                "Expected format: 'Start-End'"
            )
        
        start_str, end_str = resid_spec.split("-")
        start_id = int(start_str)
        end_id = int(end_str)
        
        matching_indices = []
        
        for idx, feature in enumerate(features_list):
            for partner in feature:
                residue_info = partner.get('residue', {})
                res_index = residue_info.get('index')
                
                if res_index is not None and start_id <= res_index <= end_id:
                    matching_indices.append(idx)
                    break  # Found match in this feature, move to next
        
        return matching_indices
    
    @staticmethod
    def _find_by_single_resid(resid_spec: str, features_list: List[list]) -> List[int]:
        """Find features containing the specified single residue ID."""
        if not re.match(r'^\d+$', resid_spec):
            raise ValueError(f"Invalid resid: {resid_spec}. Expected format: '123'")
        
        target_id = int(resid_spec)
        matching_indices = []
        
        for idx, feature in enumerate(features_list):
            for partner in feature:
                residue_info = partner.get('residue', {})
                res_index = residue_info.get('index')
                
                if res_index == target_id:
                    matching_indices.append(idx)
                    break  # Found match in this feature, move to next
        
        return matching_indices
    
    @staticmethod
    def _find_by_residue_name(residue_name: str, features_list: List[list]) -> List[int]:
        """Find features containing the specified residue name."""
        matching_indices = []
        
        for idx, feature in enumerate(features_list):
            # Check each partner in the feature
            for partner in feature:
                residue_info = partner.get('residue', {})
                if (residue_info.get('aaa_code') == residue_name or 
                    residue_info.get('a_code') == residue_name):
                    matching_indices.append(idx)
                    break

        return matching_indices
    
    @staticmethod
    def _find_by_direct_residue(token: str, features_list: List[list]) -> List[int]:
        """Find features by direct residue specification like 'ALA123' or 'A123'."""
        pattern = r'^([A-Za-z]{1,3})(\d+)$'
        match = re.match(pattern, token)
        
        residue_name = match.group(1)
        residue_id = int(match.group(2))

        result = []
        
        for idx, feature in enumerate(features_list):
            for partner in feature:
                residue = partner.get('residue', {})
                if (
                    (residue.get('aaa_code') == residue_name or residue.get('a_code') == residue_name)
                    and residue.get('index') == residue_id
                ):
                    result.append(idx)
                    break 
        return result 
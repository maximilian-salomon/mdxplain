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
Seqid parser helper for feature selection.

This module provides parsing functionality for seqid category selections,
which are based on biological sequence IDs (residue.seqid) in the trajectory data.
"""

from typing import List
from .feature_selector_numeric_parse_helper import FeatureSelectorNumericParseHelper


class FeatureSelectorSeqidParseHelper(FeatureSelectorNumericParseHelper):
    """Helper class for seqid category - biological sequence ID based operations."""
    
    @staticmethod
    def parse_seqid_category(
        param_parts: List[str], features_list: List[list], require_all_partners: bool = False
    ) -> List[int]:
        """
        Parse 'seqid' category and return matching feature indices.
        Uses biological sequence ID from metadata (residue.seqid).

        Parameters:
        -----------
        param_parts : List[str]
            List of parameter parts for sequence ID selection
        features_list : List[list]
            List of features from metadata
        require_all_partners : bool, default=False
            If True, ALL partners must contain the sequence ID

        Returns:
        --------
        List[int]
            List of feature indices matching the sequence ID criteria

        Raises:
        -------
        ValueError
            If the sequence ID specification is invalid
        """
        return FeatureSelectorNumericParseHelper.parse_numeric_category(
            param_parts, features_list, "seqid", "seqid", require_all_partners
        )
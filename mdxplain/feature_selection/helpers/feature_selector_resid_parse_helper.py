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
Resid parser helper for feature selection.

This module provides parsing functionality for resid category selections,
which are based on array indices (residue.index) in the trajectory data.
"""

from typing import List
from .feature_selector_numeric_parse_helper import FeatureSelectorNumericParseHelper


class FeatureSelectorResidParseHelper(FeatureSelectorNumericParseHelper):
    """
    Helper class for resid category - array index based operations.

    Provides static methods to parse 'resid' category selections like "123", "123-140"
    to identify matching feature indices based on trajectory metadata.
    """

    @staticmethod
    def parse_resid_category(
        param_parts: List[str], features_list: List[list], require_all_partners: bool = False
    ) -> List[int]:
        """
        Parse 'resid' category and return matching feature indices.
        Uses array index from metadata (residue.index).

        Parameters
        ----------
        param_parts : List[str]
            List of parameter parts for residue ID selection
        features_list : List[list]
            List of features from metadata
        require_all_partners : bool, default=False
            If True, ALL partners must contain the residue ID

        Returns
        -------
        List[int]
            List of feature indices matching the residue ID criteria

        Raises
        ------
        ValueError
            If the residue ID specification is invalid
        """
        return FeatureSelectorNumericParseHelper.parse_numeric_category(
            param_parts, features_list, "index", "resid", require_all_partners
        )
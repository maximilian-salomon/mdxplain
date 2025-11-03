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
Sub-comparison creation helper for comparison management.

This module provides the SubComparisonCreationHelper class with static methods
for creating different types of sub-comparisons (binary, pairwise, one-vs-rest,
multiclass). Extracted from ComparisonManager to improve code organization.
"""

from typing import List
from ..entities.comparison_data import ComparisonData


class SubComparisonCreationHelper:
    """
    Static helper class for creating sub-comparisons.
    
    Provides methods for creating different types of sub-comparisons
    based on comparison modes. All methods are static and stateless.
    """

    @staticmethod
    def create_binary_sub_comparisons(
        comp_data: ComparisonData, data_selectors: List[str]
    ) -> None:
        """
        Create sub-comparisons for binary comparison mode.
        
        Creates a single sub-comparison between two data selectors with
        binary labels (0, 1). Validates that exactly 2 selectors are provided.
        
        Parameters
        ----------
        comp_data : ComparisonData
            Comparison data object to add sub-comparison to
        data_selectors : List[str]
            List of data selector names (must contain exactly 2 selectors)
            
        Returns
        -------
        None
            Adds one sub-comparison to the comp_data object
        """
        if len(data_selectors) != 2:
            raise ValueError(
                f"Binary mode requires exactly 2 data selectors, got {len(data_selectors)}"
            )

        sub_name = f"{data_selectors[0]}_vs_{data_selectors[1]}"
        comp_data.add_sub_comparison(
            sub_name, [data_selectors[0]], [data_selectors[1]], (0, 1)
        )

    @staticmethod
    def create_pairwise_sub_comparisons(
        comp_data: ComparisonData, data_selectors: List[str]
    ) -> None:
        """
        Create sub-comparisons for pairwise comparison mode.
        
        Creates all possible pairwise combinations between data selectors.
        Each pair becomes a binary comparison with labels (0, 1).
        For N selectors, creates N*(N-1)/2 sub-comparisons.
        
        Parameters
        ----------
        comp_data : ComparisonData
            Comparison data object to add sub-comparisons to
        data_selectors : List[str]
            List of data selector names (must contain at least 2 selectors)
            
        Returns
        -------
        None
            Adds multiple sub-comparisons to the comp_data object
        """
        if len(data_selectors) < 2:
            raise ValueError(
                f"Pairwise mode requires at least 2 data selectors, got {len(data_selectors)}"
            )

        # Generate all pairs
        for i, selector1 in enumerate(data_selectors):
            for selector2 in data_selectors[i + 1 :]:
                sub_name = f"{selector1}_vs_{selector2}"
                comp_data.add_sub_comparison(sub_name, [selector1], [selector2], (0, 1))

    @staticmethod
    def create_one_vs_rest_sub_comparisons(
        comp_data: ComparisonData, data_selectors: List[str]
    ) -> None:
        """
        Create sub-comparisons for one-vs-rest comparison mode.
        
        Creates N sub-comparisons where each selector is compared against
        all other selectors combined. The target selector gets label 1,
        and all others get label 0. For N selectors, creates N sub-comparisons.
        
        Parameters
        ----------
        comp_data : ComparisonData
            Comparison data object to add sub-comparisons to
        data_selectors : List[str]
            List of data selector names (must contain at least 2 selectors)
            
        Returns
        -------
        None
            Adds N sub-comparisons to the comp_data object (one per selector)
        """
        if len(data_selectors) < 2:
            raise ValueError(
                f"One-vs-rest mode requires at least 2 data selectors, got {len(data_selectors)}"
            )

        # Each selector vs all others
        for i, target_selector in enumerate(data_selectors):
            rest_selectors = data_selectors[:i] + data_selectors[i + 1 :]
            sub_name = f"{target_selector}_vs_rest"
            comp_data.add_sub_comparison(
                sub_name, [target_selector], rest_selectors, (0, 1)  # target=0, rest=1
            )

    @staticmethod
    def create_multiclass_sub_comparisons(
        comp_data: ComparisonData, data_selectors: List[str]
    ) -> None:
        """
        Create sub-comparison for multiclass comparison mode.
        
        Creates a single sub-comparison containing all selectors as separate
        classes with incremental labels (0, 1, 2, ...). This mode is handled
        specially in ComparisonData.get_sub_comparison_data().
        
        Parameters
        ----------
        comp_data : ComparisonData
            Comparison data object to add sub-comparison to
        data_selectors : List[str]
            List of data selector names (must contain at least 2 selectors)
            
        Returns
        -------
        None
            Adds one special multiclass sub-comparison to the comp_data object
        """
        if len(data_selectors) < 2:
            raise ValueError(
                f"Multiclass mode requires at least 2 data selectors, got {len(data_selectors)}"
            )

        # For multiclass, we create a single sub-comparison with multiple groups
        # This is handled differently in get_sub_comparison_data
        sub_name = "multiclass_all"
        
        # We'll store all selectors as separate groups with incremental labels
        # This requires special handling in ComparisonData.get_sub_comparison_data
        # For now, we'll create a special structure
        sub_comp = {
            "name": sub_name,
            "mode": "multiclass",
            "selectors": data_selectors,
            "labels": list(range(len(data_selectors))),  # 0, 1, 2, 3, ...
        }
        
        comp_data.sub_comparisons.append(sub_comp)

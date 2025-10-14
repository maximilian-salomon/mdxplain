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
Helper class for extracting data from comparison configurations.

Provides methods to extract data selector names and other metadata
from ComparisonData objects for use in plotting workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class ComparisonDataExtractor:
    """
    Helper class for extracting data from comparison configurations.

    Provides static methods to extract data selector names and other
    metadata from ComparisonData objects needed for plotting.

    Examples
    --------
    >>> # Get data selector for a specific sub-comparison
    >>> data_selector = ComparisonDataExtractor.get_data_selector_from_comparison(
    ...     pipeline_data, "folded_vs_unfolded", "cluster_0_vs_rest"
    ... )
    """

    @staticmethod
    def get_data_selector_from_comparison(
        pipeline_data: PipelineData,
        comparison_name: str,
        sub_comparison_name: str
    ) -> str:
        """
        Get data selector name from a sub-comparison.

        Extracts the first data selector (group1_selectors[0]) from
        a sub-comparison configuration. This represents the primary
        group being compared against others.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with comparison configurations
        comparison_name : str
            Name of the comparison
        sub_comparison_name : str
            Name of the sub-comparison within the comparison

        Returns
        -------
        str
            Name of the data selector for the primary group

        Raises
        ------
        ValueError
            If comparison or sub-comparison not found
        KeyError
            If sub-comparison structure invalid

        Examples
        --------
        >>> # Get data selector for cluster 0
        >>> selector = ComparisonDataExtractor.get_data_selector_from_comparison(
        ...     pipeline_data, "clusters_comparison", "cluster_0_vs_rest"
        ... )
        >>> print(selector)  # "cluster_0"

        Notes
        -----
        Returns group1_selectors[0] from the sub-comparison dictionary.
        This is the primary group in the comparison (e.g., in "A vs rest",
        this returns the selector for group A).
        """
        if comparison_name not in pipeline_data.comparison_data:
            available = list(pipeline_data.comparison_data.keys())
            raise ValueError(
                f"Comparison '{comparison_name}' not found. "
                f"Available: {available}"
            )

        comp_data = pipeline_data.comparison_data[comparison_name]
        sub_comp = comp_data.get_sub_comparison(sub_comparison_name)

        if sub_comp is None:
            available = comp_data.list_sub_comparisons()
            raise ValueError(
                f"Sub-comparison '{sub_comparison_name}' not found in "
                f"comparison '{comparison_name}'. Available: {available}"
            )

        if "group1_selectors" not in sub_comp or len(sub_comp["group1_selectors"]) == 0:
            raise KeyError(
                f"Sub-comparison '{sub_comparison_name}' has invalid structure: "
                "missing or empty 'group1_selectors'"
            )

        return sub_comp["group1_selectors"][0]

    @staticmethod
    def get_all_data_selectors_from_comparison(
        pipeline_data: PipelineData,
        comparison_name: str
    ) -> list:
        """
        Get all unique data selector names from a comparison.

        Returns the data selector names directly from the comparison's
        data_selectors attribute, which is set during comparison creation.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with comparison configurations
        comparison_name : str
            Name of the comparison

        Returns
        -------
        List[str]
            Sorted list of unique data selector names

        Raises
        ------
        ValueError
            If comparison not found

        Examples
        --------
        >>> # Get all data selectors from a comparison
        >>> selectors = ComparisonDataExtractor.get_all_data_selectors_from_comparison(
        ...     pipeline_data, "clusters_comparison"
        ... )
        >>> print(selectors)  # ["cluster_0", "cluster_1", "cluster_2", "folded"]

        Notes
        -----
        Returns the data_selectors attribute from ComparisonData, which
        contains all data selectors involved in the comparison. The list
        is sorted for consistent visualization ordering.
        """
        if comparison_name not in pipeline_data.comparison_data:
            available = list(pipeline_data.comparison_data.keys())
            raise ValueError(
                f"Comparison '{comparison_name}' not found. "
                f"Available: {available}"
            )

        comp_data = pipeline_data.comparison_data[comparison_name]
        return sorted(comp_data.data_selectors)

    @staticmethod
    def get_data_selector_by_name(
        pipeline_data: PipelineData,
        selector_name: str
    ):
        """
        Get DataSelector object by name.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        selector_name : str
            Name of the data selector

        Returns
        -------
        DataSelector
            DataSelector object

        Raises
        ------
        ValueError
            If selector not found

        Examples
        --------
        >>> # Get data selector object
        >>> selector = ComparisonDataExtractor.get_data_selector_by_name(
        ...     pipeline_data, "cluster_0"
        ... )
        >>> print(selector.name)  # "cluster_0"

        Notes
        -----
        Returns the DataSelector object from pipeline_data.data_selector_data
        for use in matrix building and data extraction.
        """
        if selector_name not in pipeline_data.data_selector_data:
            available = list(pipeline_data.data_selector_data.keys())
            raise ValueError(
                f"DataSelector '{selector_name}' not found. "
                f"Available: {available}"
            )

        return pipeline_data.data_selector_data[selector_name]

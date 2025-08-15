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
Comparison manager for creating data comparisons.

This module provides the ComparisonManager class that creates comparisons
between different data selections for further analysis.
It supports various comparison modes and automatically generates
appropriate sub-comparisons.
"""

from typing import List, Dict, Any

from ..entities.comparison_data import ComparisonData
from ..helpers.comparison_validation_helper import ComparisonValidationHelper
from ..helpers.sub_comparison_creation_helper import SubComparisonCreationHelper


class ComparisonManager:
    """
    Manager for creating and managing data comparisons.

    This class provides methods to create comparisons between different
    data selections (created by DataSelectorManager) for further
    analysis. It supports various comparison modes and automatically
    generates the appropriate sub-comparisons.

    Supported modes:
    - Binary: Simple A vs B comparison
    - Pairwise: All possible pairs from multiple selectors
    - One-vs-rest: Each selector vs all others combined
    - Multiclass: All selectors as separate classes

    Examples:
    ---------
    Pipeline mode (automatic injection):

    >>> pipeline = PipelineManager()
    >>> pipeline.comparison.create_comparison(
    ...     "folded_vs_unfolded", "binary", "key_features",
    ...     ["folded_frames", "unfolded_frames"]
    ... )

    Standalone mode:

    >>> pipeline_data = PipelineData()
    >>> manager = ComparisonManager()
    >>> manager.create_comparison(
    ...     pipeline_data, "folded_vs_unfolded", "binary", "key_features",
    ...     ["folded_frames", "unfolded_frames"]
    ... )
    """

    def __init__(self):
        """
        Initialize the comparison manager.

        The ComparisonManager creates and manages comparison metadata.
        Actual data processing is done via get_comparison_data() method
        which uses PipelineData for memmap-safe operations.

        Returns:
        --------
        None
            Initializes ComparisonManager instance
        """
        pass

    def create_comparison(
        self,
        pipeline_data,
        name: str,
        mode: str,
        feature_selector: str,
        data_selectors: List[str],
    ) -> None:
        """
        Create a new comparison with specified mode and selectors.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.comparison.create_comparison("folded_vs_unfolded", "binary", "key_features", ["folded_frames", "unfolded_frames"])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = ComparisonManager()
        >>> manager.create_comparison(pipeline_data, "folded_vs_unfolded", "binary", "key_features", ["folded_frames", "unfolded_frames"])  # WITH pipeline_data parameter

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to store the comparison
        name : str
            Name for the new comparison
        mode : str
            Comparison mode: "binary", "pairwise", "one_vs_rest", "multiclass"
        feature_selector : str
            Name of the feature selector to use (defines columns)
        data_selectors : List[str]
            Names of data selectors to compare

        Returns:
        --------
        None
            Creates ComparisonData in pipeline_data

        Raises:
        -------
        ValueError
            If comparison already exists, invalid mode, or selectors not found

        Examples:
        ---------
        >>> # Binary comparison
        >>> manager.create_comparison(
        ...     pipeline_data, "folded_vs_unfolded", "binary", "key_features",
        ...     ["folded_frames", "unfolded_frames"]
        ... )

        >>> # One-vs-rest comparison (creates multiple sub-comparisons)
        >>> manager.create_comparison(
        ...     pipeline_data, "conformations", "one_vs_rest", "all_features",
        ...     ["folded", "intermediate", "unfolded", "extended"]
        ... )
        """
        # Validate inputs using helper
        ComparisonValidationHelper.validate_comparison_name(pipeline_data, name)
        ComparisonValidationHelper.validate_mode(mode)
        ComparisonValidationHelper.validate_feature_selector(pipeline_data, feature_selector)
        ComparisonValidationHelper.validate_data_selectors(pipeline_data, data_selectors)

        # Create comparison data (pure metadata container)
        comp_data = ComparisonData(
            name, mode, feature_selector, data_selectors
        )

        # Generate sub-comparisons based on mode using helper
        if mode == "binary":
            SubComparisonCreationHelper.create_binary_sub_comparisons(comp_data, data_selectors)
        elif mode == "pairwise":
            SubComparisonCreationHelper.create_pairwise_sub_comparisons(comp_data, data_selectors)
        elif mode == "one_vs_rest":
            SubComparisonCreationHelper.create_one_vs_rest_sub_comparisons(comp_data, data_selectors)
        elif mode == "multiclass":
            SubComparisonCreationHelper.create_multiclass_sub_comparisons(comp_data, data_selectors)

        # Store in pipeline data
        pipeline_data.comparison_data[name] = comp_data


    def list_comparisons(self, pipeline_data) -> List[str]:
        """
        List all available comparisons.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns:
        --------
        List[str]
            List of comparison names

        Examples:
        ---------
        >>> comparisons = manager.list_comparisons(pipeline_data)
        >>> print(f"Available comparisons: {comparisons}")
        """
        return list(pipeline_data.comparison_data.keys())

    def get_comparison_info(self, pipeline_data, name: str) -> Dict[str, Any]:
        """
        Get information about a comparison.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the comparison

        Returns:
        --------
        Dict[str, Any]
            Dictionary with comparison information

        Examples:
        ---------
        >>> info = manager.get_comparison_info(pipeline_data, "conformations")
        >>> print(f"Mode: {info['mode']}")
        >>> print(f"Sub-comparisons: {info['sub_comparison_names']}")
        """
        ComparisonValidationHelper.validate_comparison_exists(pipeline_data, name)
        comp_data = pipeline_data.comparison_data[name]
        return comp_data.get_comparison_info()

    def remove_comparison(self, pipeline_data, name: str) -> None:
        """
        Remove a comparison.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the comparison to remove

        Returns:
        --------
        None
            Removes the comparison from pipeline_data

        Examples:
        ---------
        >>> manager.remove_comparison(pipeline_data, "old_comparison")
        """
        ComparisonValidationHelper.validate_comparison_exists(pipeline_data, name)
        del pipeline_data.comparison_data[name]

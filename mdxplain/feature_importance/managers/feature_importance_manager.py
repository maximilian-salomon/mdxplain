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
Feature importance manager for ML-based feature analysis.

This module provides the FeatureImportanceManager class that manages
feature importance analysis using various ML algorithms. It follows
the same pattern as DecompositionManager, working with analyzer_types
and creating FeatureImportanceData objects.
"""

from typing import List, Dict, Any, Optional

from ..analyzer_types.interfaces.analyzer_type_base import AnalyzerTypeBase
from ..helpers.analysis_runner_helper import AnalysisRunnerHelper
from ..helpers.feature_importance_validation_helper import FeatureImportanceValidationHelper
from ..helpers.top_features_helper import TopFeaturesHelper


class FeatureImportanceManager:
    """
    Manager for creating and managing feature importance analyses.

    This class provides methods to run feature importance analysis on
    comparisons created by ComparisonManager. It uses various ML algorithms
    (analyzer_types) to determine which features are most important for
    distinguishing between different data groups.

    The manager follows the same pattern as DecompositionManager:
    - Uses analyzer_type objects similar to decomposition_type
    - Creates FeatureImportanceData objects similar to DecompositionData
    - Integrates with pipeline via AutoInjectProxy

    Examples:
    ---------
    Pipeline mode (automatic injection):

    >>> pipeline = PipelineManager()
    >>> from mdxplain.feature_importance import analyzer_types
    >>> pipeline.feature_importance.add(
    ...     "my_comparison", analyzer_types.DecisionTree(max_depth=5), "tree_analysis"
    ... )

    Standalone mode:

    >>> pipeline_data = PipelineData()
    >>> manager = FeatureImportanceManager()
    >>> manager.add(
    ...     pipeline_data, "my_comparison",
    ...     analyzer_types.DecisionTree(max_depth=5), "tree_analysis"
    ... )
    """

    def __init__(self):
        """
        Initialize the feature importance manager.

        Returns:
        --------
        None
            Initializes FeatureImportanceManager instance
        """
        pass

    def add(
        self,
        pipeline_data,
        comparison_name: str,
        analyzer_type: AnalyzerTypeBase,
        analysis_name: str,
        force: bool = False,
    ) -> None:
        """
        Add feature importance analysis for a comparison.

        Runs feature importance analysis on all sub-comparisons within the
        specified comparison using the provided analyzer. Creates a single
        FeatureImportanceData object containing results for all sub-comparisons.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> from mdxplain.feature_importance import analyzer_types
        >>> pipeline.feature_importance.add("folded_vs_unfolded", analyzer_types.DecisionTree(), "tree_analysis")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.add(pipeline_data, "folded_vs_unfolded", analyzer_types.DecisionTree(), "tree_analysis")  # WITH pipeline_data parameter

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing comparisons
        comparison_name : str
            Name of the comparison to analyze
        analyzer_type : AnalyzerTypeBase
            Analyzer instance (e.g., analyzer_types.DecisionTree(max_depth=5))
        analysis_name : str
            Name to store the analysis results
        force : bool, default=False
            Whether to overwrite existing analysis with same name

        Returns:
        --------
        None
            Creates FeatureImportanceData in pipeline_data

        Raises:
        -------
        ValueError
            If analysis already exists (and force=False), comparison not found,
            or analysis computation fails

        Examples:
        ---------
        >>> from mdxplain.feature_importance import analyzer_types
        >>> manager = FeatureImportanceManager()

        >>> # Basic decision tree analysis
        >>> manager.add(
        ...     pipeline_data, "folded_vs_unfolded",
        ...     analyzer_types.DecisionTree(max_depth=5, random_state=42),
        ...     "tree_analysis"
        ... )

        >>> # Balanced tree for imbalanced data
        >>> manager.add(
        ...     pipeline_data, "conformations",
        ...     analyzer_types.DecisionTree(class_weight="balanced"),
        ...     "balanced_tree", force=True
        ... )
        """
        # Validate inputs using helper
        FeatureImportanceValidationHelper.validate_analysis_name(pipeline_data, analysis_name, force)
        FeatureImportanceValidationHelper.validate_comparison_exists(pipeline_data, comparison_name)
        FeatureImportanceValidationHelper.validate_analyzer_type(analyzer_type)

        # Get comparison data
        comp_data = pipeline_data.comparison_data[comparison_name]

        # Run analysis using helper
        fi_data = AnalysisRunnerHelper.run_comparison_analysis(
            pipeline_data, comp_data, analyzer_type, analysis_name
        )

        # Store in pipeline data
        pipeline_data.feature_importance_data[analysis_name] = fi_data

    def get_analysis_info(self, pipeline_data, analysis_name: str) -> Dict[str, Any]:
        """
        Get information about a feature importance analysis.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the analysis

        Returns:
        --------
        Dict[str, Any]
            Dictionary with analysis information

        Examples:
        ---------
        >>> info = manager.get_analysis_info(pipeline_data, "tree_analysis")
        >>> print(f"Analyzer: {info['analyzer_type']}")
        >>> print(f"Comparisons: {info['n_comparisons']}")
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(pipeline_data, analysis_name)
        fi_data = pipeline_data.feature_importance_data[analysis_name]
        return fi_data.get_analysis_info()

    def get_top_features(
        self,
        pipeline_data,
        analysis_name: str,
        comparison_identifier: Optional[str] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top N most important features from analysis.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the analysis
        comparison_identifier : str, optional
            Specific sub-comparison to get features from.
            If None, returns average across all sub-comparisons.
        n : int, default=10
            Number of top features to return

        Returns:
        --------
        List[Dict[str, Any]]
            List of dictionaries with feature information

        Examples:
        ---------
        >>> # Get top features averaged across all comparisons
        >>> top_features = manager.get_top_features(
        ...     pipeline_data, "tree_analysis", n=5
        ... )

        >>> # Get top features for specific comparison
        >>> top_features = manager.get_top_features(
        ...     pipeline_data, "tree_analysis", "folded_vs_rest", n=5
        ... )
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(pipeline_data, analysis_name)
        fi_data = pipeline_data.feature_importance_data[analysis_name]

        # Use helper for all top features processing
        return TopFeaturesHelper.get_top_features_with_names(
            pipeline_data, fi_data, comparison_identifier, n
        )

    def list_analyses(self, pipeline_data) -> List[str]:
        """
        List all available feature importance analyses.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns:
        --------
        List[str]
            List of analysis names

        Examples:
        ---------
        >>> analyses = manager.list_analyses(pipeline_data)
        >>> print(f"Available analyses: {analyses}")
        """
        return list(pipeline_data.feature_importance_data.keys())

    def remove_analysis(self, pipeline_data, analysis_name: str) -> None:
        """
        Remove a feature importance analysis.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the analysis to remove

        Returns:
        --------
        None
            Removes the analysis from pipeline_data

        Examples:
        ---------
        >>> manager.remove_analysis(pipeline_data, "old_analysis")
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(pipeline_data, analysis_name)
        del pipeline_data.feature_importance_data[analysis_name]


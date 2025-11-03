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
from __future__ import annotations

import warnings
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from ..analyzer_types.interfaces.analyzer_type_base import AnalyzerTypeBase
from ..helpers.analysis_runner_helper import AnalysisRunnerHelper
from ..helpers.feature_importance_validation_helper import FeatureImportanceValidationHelper
from ..helpers.top_features_helper import TopFeaturesHelper
from ...utils.data_utils import DataUtils
from ..services.feature_importance_add_service import FeatureImportanceAddService
from ..helpers.representative_finder_helper import RepresentativeFinderHelper


class FeatureImportanceManager:
    """
    Manager for creating and managing feature importance analyses.

    This class provides methods to run feature importance analysis on
    comparisons created by ComparisonManager. It uses various ML algorithms
    (analyzer_types) to determine which features are most important for
    distinguishing between different data groups. So basically classifiers.

    The manager follows the same pattern as DecompositionManager:
    
    - Uses analyzer_type objects similar to decomposition_type
    - Creates FeatureImportanceData objects similar to DecompositionData
    - Integrates with pipeline via AutoInjectProxy

    Examples
    --------
    Pipeline mode (automatic injection):

    >>> pipeline = PipelineManager()
    >>> from mdxplain.feature_importance import analyzer_types
    >>> pipeline.feature_importance.add_analysis(
    ...     "my_comparison", analyzer_types.DecisionTree(max_depth=5), "tree_analysis"
    ... )

    Standalone mode:

    >>> pipeline_data = PipelineData()
    >>> manager = FeatureImportanceManager()
    >>> manager.add_analysis(
    ...     pipeline_data, "my_comparison",
    ...     analyzer_types.DecisionTree(max_depth=5), "tree_analysis"
    ... )
    """

    def __init__(self, use_memmap: bool = False, chunk_size: int = 2000, cache_dir: str = "./cache") -> None:
        """
        Initialize the feature importance manager.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, default=10000
            Processing chunk size for incremental computation
        cache_dir : str, default="./cache"
            Cache directory path

        Returns
        -------
        None
            Initializes FeatureImportanceManager instance with specified configuration
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir

    def add_analysis(
        self,
        pipeline_data: PipelineData,
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

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> from mdxplain.feature_importance import analyzer_types
        >>> pipeline.feature_importance.add_analysis("folded_vs_unfolded", analyzer_types.DecisionTree(), "tree_analysis")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.add_analysis(pipeline_data, "folded_vs_unfolded", analyzer_types.DecisionTree(), "tree_analysis")  # WITH pipeline_data parameter

        Parameters
        ----------
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

        Returns
        -------
        None
            Creates FeatureImportanceData in pipeline_data

        Raises
        ------
        ValueError
            If analysis already exists (and force=False), comparison not found,
            or analysis computation fails

        Examples
        --------
        >>> from mdxplain.feature_importance import analyzer_types
        >>> manager = FeatureImportanceManager()

        >>> # Basic decision tree analysis
        >>> manager.add_analysis(
        ...     pipeline_data, "folded_vs_unfolded",
        ...     analyzer_types.DecisionTree(max_depth=5, random_state=42),
        ...     "tree_analysis"
        ... )

        >>> # Balanced tree for imbalanced data
        >>> manager.add_analysis(
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

        # Initialize calculator if needed
        if hasattr(analyzer_type, 'init_calculator'):
            analyzer_type.init_calculator(
                use_memmap=self.use_memmap,
                cache_path=f"{self.cache_dir}/{analysis_name}",
                chunk_size=self.chunk_size,
            )

        # Run analysis using helper
        fi_data = AnalysisRunnerHelper.run_comparison_analysis(
            pipeline_data, comp_data, analyzer_type, analysis_name
        )

        # Store in pipeline data
        pipeline_data.feature_importance_data[analysis_name] = fi_data

    def get_analysis_info(self, pipeline_data: PipelineData, analysis_name: str) -> Dict[str, Any]:
        """
        Get information about a feature importance analysis.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_importance.get_analysis_info("tree_analysis")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.get_analysis_info(pipeline_data, "tree_analysis")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the analysis

        Returns
        -------
        Dict[str, Any]
            Dictionary with analysis information

        Examples
        --------
        >>> info = manager.get_analysis_info(pipeline_data, "tree_analysis")
        >>> print(f"Analyzer: {info['analyzer_type']}")
        >>> print(f"Comparisons: {info['n_comparisons']}")
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(pipeline_data, analysis_name)
        fi_data = pipeline_data.feature_importance_data[analysis_name]
        return fi_data.get_analysis_info()

    def get_top_features(
        self,
        pipeline_data: PipelineData,
        analysis_name: str,
        comparison_identifier: Optional[str] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top N most important features from analysis.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the analysis
        comparison_identifier : str, optional
            Specific sub-comparison to get features from.
            If None, returns average across all sub-comparisons.
        n : int, default=10
            Number of top features to return

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with feature information

        Examples
        --------
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

    def get_all_top_features(
        self,
        pipeline_data: PipelineData,
        analysis_name: str,
        n: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top features for all sub-comparisons in an analysis.
        
        Returns a dictionary where keys are comparison identifiers
        and values are lists of top features for each comparison.
        
        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.
        
        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> all_features = pipeline.feature_importance.get_all_top_features("dt_analysis", n=5)
        
        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> all_features = manager.get_all_top_features(pipeline_data, "dt_analysis", n=5)
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the analysis
        n : int, default=10
            Number of top features per comparison
            
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary mapping comparison names to their top features
            
        Examples
        --------
        >>> all_features = manager.get_all_top_features(
        ...     pipeline_data, "dt_analysis", n=5
        ... )
        >>> # Access specific comparison
        >>> cluster_0 = all_features["cluster_0_vs_rest"]
        >>> print(f"Top feature: {cluster_0[0]['feature_name']}")
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(pipeline_data, analysis_name)
        fi_data = pipeline_data.feature_importance_data[analysis_name]
        
        result = {}
        
        # Get all comparison identifiers
        comparisons = fi_data.list_comparisons()
        
        # Get top features for each comparison
        for comp_name in comparisons:
            result[comp_name] = TopFeaturesHelper.get_top_features_with_names(
                pipeline_data, fi_data, comp_name, n
            )
        
        return result

    def print_top_n_features(
        self,
        pipeline_data: PipelineData,
        analysis_name: str,
        n: int = 3
    ) -> None:
        """
        Print top N features for all comparisons in analysis.

        Uses get_all_top_features() internally and formats output for console display.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_importance.print_top_n_features("my_analysis", n=3)

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.print_top_n_features(pipeline_data, "my_analysis", n=3)

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the feature importance analysis
        n : int, default=3
            Number of top features to print per comparison

        Returns
        -------
        None
            Prints to console

        Examples
        --------
        >>> pipeline.feature_importance.print_top_n_features(
        ...     "feature_importance", n=5
        ... )
        Top 5 features for cluster_0_vs_rest:
          1. CA-CB: 0.456
          2. CA-CG: 0.234
          ...
        """
        all_top_features = self.get_all_top_features(
            pipeline_data, analysis_name, n=n
        )

        for comparison_name, top_features in all_top_features.items():
            print(f"\nTop {n} features for {comparison_name}:")
            for j, feature_info in enumerate(top_features, 1):
                print(f"  {j}. {feature_info['feature_name']}: "
                      f"{feature_info['importance_score']:.3f}")

    def list_analyses(self, pipeline_data: PipelineData) -> List[str]:
        """
        List all available feature importance analyses.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_importance.list_analyses()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.list_analyses(pipeline_data)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns
        -------
        List[str]
            List of analysis names

        Examples
        --------
        >>> analyses = manager.list_analyses(pipeline_data)
        >>> print(f"Available analyses: {analyses}")
        """
        return list(pipeline_data.feature_importance_data.keys())

    def remove_analysis(self, pipeline_data: PipelineData, analysis_name: str) -> None:
        """
        Remove a feature importance analysis.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_importance.remove_analysis("old_analysis")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.remove_analysis(pipeline_data, "old_analysis")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of the analysis to remove

        Returns
        -------
        None
            Removes the analysis from pipeline_data

        Examples
        --------
        >>> manager.remove_analysis(pipeline_data, "old_analysis")
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(pipeline_data, analysis_name)
        del pipeline_data.feature_importance_data[analysis_name]
        
    def save(self, pipeline_data: PipelineData, save_path: str) -> None:
        """
        Save all feature importance data to single file.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_importance.save('feature_importance.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.save(pipeline_data, 'feature_importance.npy')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with feature importance data
        save_path : str
            Path where to save all feature importance data in one file

        Returns
        -------
        None
            Saves all feature importance data to the specified file
            
        Examples
        --------
        >>> manager.save(pipeline_data, 'feature_importance.npy')
        """
        DataUtils.save_object(pipeline_data.feature_importance_data, save_path)

    def load(self, pipeline_data: PipelineData, load_path: str) -> None:
        """
        Load all feature importance data from single file.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_importance.load('feature_importance.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.load(pipeline_data, 'feature_importance.npy')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container to load feature importance data into
        load_path : str
            Path to saved feature importance data file

        Returns
        -------
        None
            Loads all feature importance data from the specified file
            
        Examples
        --------
        >>> manager.load(pipeline_data, 'feature_importance.npy')
        """
        temp_dict = {}
        DataUtils.load_object(temp_dict, load_path)
        pipeline_data.feature_importance_data = temp_dict

    def print_info(self, pipeline_data: PipelineData) -> None:
        """
        Print feature importance data information.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.feature_importance.print_info()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = FeatureImportanceManager()
        >>> manager.print_info(pipeline_data)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with feature importance data
        
        Returns
        -------
        None
            Prints feature importance data information to console
        
        Examples
        --------
        >>> pipeline_data = PipelineData()
        >>> feature_importance_manager = FeatureImportanceManager()
        >>> feature_importance_manager.print_info(pipeline_data)
        """
        if len(pipeline_data.feature_importance_data) == 0:
            print("No featureimportancedata data available.")
            return

        print("=== FeatureImportanceData Information ===")
        data_names = list(pipeline_data.feature_importance_data.keys())
        print(f"FeatureImportanceData Names: {len(data_names)} ({", ".join(data_names)})")
        
        for name, data in pipeline_data.feature_importance_data.items():
            print(f"\n--- {name} ---")
            data.print_info(pipeline_data)

    @property
    def add(self):
        """
        Service for adding feature importance analyses with simplified syntax.

        Provides an intuitive interface for adding feature importance analyses without
        requiring explicit analyzer type instantiation or imports.

        Returns
        -------
        FeatureImportanceAddService
            Service instance for adding feature importance analyses with combined parameters

        Examples
        --------
        >>> # Add different analyzer types
        >>> pipeline.feature_importance.add.decision_tree("my_comparison", "tree_analysis", max_depth=5)
        >>> pipeline.feature_importance.add.decision_tree(
        ...     "folded_vs_unfolded",
        ...     "deep_tree",
        ...     max_depth=10,
        ...     criterion="entropy",
        ...     random_state=42
        ... )

        Notes
        -----
        Pipeline data is automatically injected by AutoInjectProxy.
        All analyzer type parameters are combined with add_analysis parameters.
        """
        return FeatureImportanceAddService(self, None)

    def _validate_representative_analysis(
        self,
        pipeline_data: PipelineData,
        analysis_name: str
    ) -> Tuple:
        """
        Validate analysis exists and supports representative frame finding.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of feature importance analysis

        Returns
        -------
        Tuple
            (fi_data, comp_data) for validated analysis

        Raises
        ------
        ValueError
            If analysis not found or not Decision Tree based
        """
        if analysis_name not in pipeline_data.feature_importance_data:
            raise ValueError(
                f"Analysis '{analysis_name}' not found. "
                f"Available: {list(pipeline_data.feature_importance_data.keys())}"
            )

        fi_data = pipeline_data.feature_importance_data[analysis_name]

        if fi_data.analyzer_type != "decision_tree":
            raise ValueError(
                f"get_representative_frames() currently only supports "
                f"'decision_tree' analyzer, got '{fi_data.analyzer_type}'"
            )

        comp_data = pipeline_data.comparison_data[fi_data.comparison_name]
        return fi_data, comp_data

    def _get_representatives_multiclass(
        self,
        pipeline_data: PipelineData,
        comp_data,
        fi_data
    ) -> Dict[str, List[int]]:
        """
        Find representative frames for multiclass mode using centroids.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        comp_data : ComparisonData
            Comparison configuration
        fi_data : FeatureImportanceData
            Feature importance data

        Returns
        -------
        Dict[str, List[int]]
            Mapping from data_selector_name to [traj_idx, frame_idx]
        """
        result = {}
        for ds_name in comp_data.data_selectors:
            traj_idx, frame_idx = pipeline_data.get_centroid_frame(
                fi_data.feature_selector, ds_name
            )
            result[ds_name] = [traj_idx, frame_idx]
        return result

    def _get_representatives_standard(
        self,
        pipeline_data: PipelineData,
        fi_data,
        n_top: int
    ) -> Dict[str, List[int]]:
        """
        Find representative frames for standard modes using tree-based scoring.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        fi_data : FeatureImportanceData
            Feature importance data
        n_top : int
            Number of top features to consider

        Returns
        -------
        Dict[str, List[int]]
            Mapping from sub_comparison_name to [traj_idx, frame_idx]
        """
        comparison_names = fi_data.list_comparisons()
        result = {}

        for sub_comp_name in comparison_names:
            traj_idx, frame_idx = RepresentativeFinderHelper.find_best_tree_based(
                pipeline_data, fi_data, sub_comp_name, n_top,
                use_memmap=self.use_memmap, chunk_size=self.chunk_size
            )
            result[sub_comp_name] = [traj_idx, frame_idx]

        return result

    def get_representative_frames(
        self,
        pipeline_data: PipelineData,
        analysis_name: str,
        n_top: int = 10
    ) -> Dict[str, List[int]]:
        """
        Find representative frames for each sub-comparison.

        Finds frames that most strongly exhibit the top important features
        identified by the decision tree. Uses tree split rules to determine
        optimal feature values and scores frames based on how well they
        match these criteria.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        analysis_name : str
            Name of feature importance analysis
        n_top : int, default=10
            Number of top features to consider

        Returns
        -------
        Dict[str, List[int]]
            Mapping from sub_comparison_name to [traj_idx, frame_idx]

        Examples
        --------
        >>> representatives = manager.get_representative_frames(
        ...     pipeline_data, "dt_analysis", n_top=10
        ... )
        >>> print(representatives)
        {'cluster_0_vs_rest': [1, 2341], 'cluster_1_vs_rest': [3, 156]}

        Notes
        -----
        - Uses Decision Tree split rules to find characteristic frames
        - Frames maximize expression of top important features
        - Handles periodic features (torsions) with circular distance
        - For multiclass mode, uses centroids instead

        Raises
        ------
        ValueError
            If analysis not found or not Decision Tree based
        """
        fi_data, comp_data = self._validate_representative_analysis(
            pipeline_data, analysis_name
        )

        if comp_data.mode == "multiclass":
            warnings.warn(
                "get_representative_frames() does not support multiclass mode. "
                "Finding centroids for each class instead.",
                UserWarning
            )
            return self._get_representatives_multiclass(
                pipeline_data, comp_data, fi_data
            )

        return self._get_representatives_standard(
            pipeline_data, fi_data, n_top
        )

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
    from ..entities.feature_importance_data import FeatureImportanceData
    from ...pipeline.entities.pipeline_data import PipelineData

from ..analyzer_type.interfaces.analyzer_type_base import AnalyzerTypeBase
from ..helper.analysis_runner_helper import AnalysisRunnerHelper
from ..helper.feature_importance_validation_helper import FeatureImportanceValidationHelper
from ..helper.top_features_helper import TopFeaturesHelper
from ...utils.data_utils import DataUtils
from ...utils.feature_metadata_utils import FeatureMetadataUtils
from ...utils.path_utils import PathUtils
from ..services.feature_importance_add_service import FeatureImportanceAddService
from ..helper.representative_finder_helper import RepresentativeFinderHelper
from ..helper.importance_filter_helper import ImportanceFilterHelper


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
        self.cache_dir = PathUtils.prepare_directory_path(
            cache_dir,
            create=True,
            purpose="cache directory",
        )

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
                max_memory_gb=pipeline_data.max_memory_gb,
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

    def filter_importance(
        self,
        pipeline_data: PipelineData,
        source_name: str,
        filtered_name: str,
        min_sequence_separation: int = 20,
        merge_radius: int = 5,
        force: bool = False,
    ) -> None:
        """
        Create a redundancy-filtered copy of a feature importance analysis.

        Collapses near-identical neighbour features into one representative per
        coupling: long-range pair features are kept, then a strength-ordered
        greedy pass merges near neighbours (chain-aware) into the strongest and
        counts them. The original analysis is not modified; a filtered clone is
        stored under ``filtered_name`` with merged features set to zero.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        source_name : str
            Name of the feature importance analysis to filter
        filtered_name : str
            Name to store the filtered clone under
        min_sequence_separation : int, default=20
            Minimum within-chain sequence separation to keep a pair feature
            (discards trivial short-range within-chain couplings)
        merge_radius : int, default=5
            Maximum within-chain sequence distance for two features to be
            treated as the same event
        force : bool, default=False
            Whether to overwrite an existing analysis with the same name

        Returns
        -------
        None
            Stores the filtered clone in pipeline data

        Examples
        --------
        >>> pipeline.feature_importance.filter_importance(
        ...     "feature_importance", "feature_importance_filtered"
        ... )
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(
            pipeline_data, source_name
        )
        FeatureImportanceValidationHelper.validate_analysis_name(
            pipeline_data, filtered_name, force
        )
        source = pipeline_data.feature_importance_data[source_name]
        feature_metadata = self._source_feature_metadata(
            pipeline_data, source
        )
        chain_of = self._chain_segment_map(pipeline_data)
        filtered = self._build_filtered_data(
            source,
            filtered_name,
            feature_metadata,
            chain_of,
            min_sequence_separation,
            merge_radius,
        )
        pipeline_data.feature_importance_data[filtered_name] = filtered

    @staticmethod
    def _source_feature_metadata(
        pipeline_data: PipelineData, source: "FeatureImportanceData"
    ) -> Optional[List[Any]]:
        """
        Get the feature metadata backing a source analysis.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        source : FeatureImportanceData
            Source analysis to filter

        Returns
        -------
        List[Any] or None
            Feature metadata for the source's selector, or None if it has none
        """
        if not source.feature_selector:
            return None
        return pipeline_data.get_selected_metadata(source.feature_selector)

    @staticmethod
    def _chain_segment_map(pipeline_data: PipelineData) -> Dict[int, int]:
        """
        Build a residue-index to chain-segment map from the residue labels.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns
        -------
        Dict[int, int]
            Mapping from residue index to chain-segment id (empty if no labels)
        """
        res_label_data = pipeline_data.trajectory_data.res_label_data
        if not res_label_data:
            return {}
        # res_label_data stores a per-residue label list for each trajectory.
        residue_labels = list(next(iter(res_label_data.values())))
        return ImportanceFilterHelper.build_chain_map(residue_labels)

    @staticmethod
    def _build_filtered_data(
        source: "FeatureImportanceData",
        filtered_name: str,
        feature_metadata: Optional[List[Any]],
        chain_of: Dict[int, int],
        min_sequence_separation: int,
        merge_radius: int,
    ) -> "FeatureImportanceData":
        """
        Build a filtered clone of a source analysis.

        Copies the importance arrays and metadata per sub-comparison, sharing
        the trained model reference, and applies the redundancy filter.

        Parameters
        ----------
        source : FeatureImportanceData
            Source analysis to clone and filter
        filtered_name : str
            Name for the filtered clone
        feature_metadata : List[Any] or None
            Feature metadata for residue mapping
        chain_of : Dict[int, int]
            Residue index to chain-segment map
        min_sequence_separation : int
            Minimum within-chain sequence separation for pair features
        merge_radius : int
            Maximum within-chain sequence distance to merge

        Returns
        -------
        FeatureImportanceData
            The filtered clone
        """
        filtered = type(source)(filtered_name)
        filtered.analyzer_type = source.analyzer_type
        filtered.comparison_name = source.comparison_name
        filtered.feature_selector = source.feature_selector
        filter_params = {
            "min_sequence_separation": min_sequence_separation,
            "merge_radius": merge_radius,
        }
        for importances, metadata in source.get_all_comparisons():
            new_importances, merged_counts = (
                ImportanceFilterHelper.filter_comparison(
                    importances,
                    feature_metadata,
                    chain_of,
                    min_sequence_separation,
                    merge_radius,
                )
            )
            new_metadata = dict(metadata)
            new_metadata["merged_counts"] = merged_counts
            new_metadata["filter_params"] = filter_params
            filtered.add_comparison_result(new_importances, new_metadata)
        return filtered

    def print_top_n_features(
        self,
        pipeline_data: PipelineData,
        analysis_name: str,
        n: int = 3
    ) -> None:
        """
        Print top N features for all comparisons in analysis.

        Uses get_all_top_features() internally and formats output for console
        display. If a trained Decision Tree model is available for a
        comparison, the printed label is extended with a representative split
        criterion for that feature. This keeps the output focused on the
        actual tree rule instead of generic metadata labels.

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
          1. contacts: LEU13-ARG31: Non-Contact (0.456)
          2. torsions: GLY42_phi: <= 55.20 degrees (0.234)
          ...

        Notes
        -----
        - Uses representative weighted split thresholds from the stored
          Decision Tree model when available
        - For binary discrete features, prints the left branch label
          (e.g. "Non-Contact")
        - Falls back to ``feature_type: feature_name`` if no tree rule is
          available for a feature
        """
        FeatureImportanceValidationHelper.validate_analysis_exists(
            pipeline_data, analysis_name
        )
        fi_data = pipeline_data.feature_importance_data[analysis_name]
        all_top_features = self.get_all_top_features(
            pipeline_data, analysis_name, n=n
        )
        feature_metadata = None
        if fi_data.feature_selector:
            feature_metadata = pipeline_data.get_selected_metadata(
                fi_data.feature_selector
            )

        for comparison_name, top_features in all_top_features.items():
            split_rules = self._get_split_rules_for_comparison(
                fi_data, comparison_name, top_features
            )
            merged_counts = self._get_merged_counts(fi_data, comparison_name)
            print(f"\nTop {n} features for {comparison_name}:")
            for j, feature_info in enumerate(top_features, 1):
                feature_label = self._format_top_feature_label(
                    feature_info,
                    feature_metadata,
                    split_rules.get(feature_info["feature_index"]),
                )
                feature_label = self._append_merged_count(
                    feature_label, feature_info["feature_index"], merged_counts
                )
                print(
                    f"  {j}. {feature_label} "
                    f"({feature_info['importance_score']:.3f})"
                )

    @staticmethod
    def _get_merged_counts(
        fi_data: FeatureImportanceData, comparison_name: str
    ) -> Dict[int, int]:
        """
        Get the merged-neighbour counts for a sub-comparison.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance analysis (possibly filter-produced)
        comparison_name : str
            Sub-comparison name

        Returns
        -------
        Dict[int, int]
            Mapping from representative feature index to merged count (empty
            when the analysis was not produced by filter_importance)
        """
        _, metadata = fi_data.get_comparison(comparison_name)
        return metadata.get("merged_counts", {})

    @staticmethod
    def _append_merged_count(
        label: str, feature_index: int, merged_counts: Dict[int, int]
    ) -> str:
        """
        Append the merged-neighbour count to a feature label when present.

        Parameters
        ----------
        label : str
            The formatted feature label
        feature_index : int
            Index of the feature being labelled
        merged_counts : Dict[int, int]
            Mapping from representative feature index to merged count

        Returns
        -------
        str
            The label, with " (+N)" appended when a count exists
        """
        count = merged_counts.get(feature_index)
        if count:
            return f"{label} (+{count})"
        return label

    def _get_split_rules_for_comparison(
        self,
        fi_data: FeatureImportanceData,
        comparison_name: str,
        top_features: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Extract representative split rules for the requested top features.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance analysis containing stored comparison metadata
            and trained models
        comparison_name : str
            Name of the sub-comparison to inspect
        top_features : List[Dict[str, Any]]
            Top-feature dictionaries returned by get_all_top_features()

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping from feature index to representative split-rule metadata.
            Each value contains at least ``threshold`` and ``weight``.

        Notes
        -----
        Returns an empty dictionary when:

        - no top features are provided
        - the comparison has no stored model
        - the model does not expose relevant tree splits for those features
        """
        if not top_features:
            return {}

        _, comparison_metadata = fi_data.get_comparison(comparison_name)
        model = comparison_metadata.get("model")
        if model is None or not self._is_tree_based_model(model):
            return {}

        feature_indices = [feature["feature_index"] for feature in top_features]
        target_label = comparison_metadata.get("labels", (0, 1))[0]
        return RepresentativeFinderHelper._extract_tree_rules(
            model,
            feature_indices,
            target_label,
        )

    @staticmethod
    def _is_tree_based_model(model: Any) -> bool:
        """
        Check whether a model exposes tree-based split rules.

        A single Decision Tree exposes ``tree_`` and a tree ensemble such as
        Random Forest exposes ``estimators_``. Models with neither cannot
        provide split-rule labels.

        Parameters
        ----------
        model : Any
            Trained model stored in the comparison metadata

        Returns
        -------
        bool
            True if the model exposes tree-based split rules

        Examples
        --------
        >>> FeatureImportanceManager._is_tree_based_model(decision_tree_model)
        True
        """
        return hasattr(model, "tree_") or hasattr(model, "estimators_")

    def _format_top_feature_label(
        self,
        feature_info: Dict[str, Any],
        feature_metadata: Optional[List[Any]],
        split_rule: Optional[Dict[str, float]],
    ) -> str:
        """
        Format one top feature with its representative split criterion.

        Parameters
        ----------
        feature_info : Dict[str, Any]
            Top-feature dictionary containing at least ``feature_type`` and
            ``feature_name``
        feature_metadata : list or None
            Selected feature metadata used to resolve visualization settings
            such as discrete tick labels and units
        split_rule : Dict[str, float] or None
            Representative split rule for this feature. Expected to contain a
            ``threshold`` entry. If None, only the base feature label is used.

        Returns
        -------
        str
            Formatted label such as ``contacts: LEU13-ARG31: Non-Contact`` or
            ``torsions: GLY42_phi: <= 55.20 degrees``
        """
        base_label = (
            f"{feature_info['feature_type']}: {feature_info['feature_name']}"
        )
        if split_rule is None:
            return base_label

        type_metadata = FeatureMetadataUtils.get_top_level_metadata(
            feature_info["feature_type"], feature_metadata
        )
        criterion = self._format_split_criterion(
            type_metadata,
            split_rule["threshold"],
        )
        return f"{base_label}: {criterion}"

    def _format_split_criterion(
        self,
        type_metadata: Dict[str, Any],
        threshold: float,
    ) -> str:
        """
        Convert a tree threshold into a readable split criterion label.

        Parameters
        ----------
        type_metadata : Dict[str, Any]
            Type-level feature metadata containing visualization settings and
            optional units
        threshold : float
            Representative tree split threshold for the feature

        Returns
        -------
        str
            Human-readable split criterion. For binary discrete features this
            is the left-branch class label. For continuous features this is a
            threshold string such as ``<= 4.50`` or ``<= 55.20 degrees``.

        Notes
        -----
        Binary discrete features in scikit-learn trees typically split at
        ``0.5``. In that case the left branch corresponds to the first label
        in the metadata tick-label list.
        """
        visualization = type_metadata.get("visualization", {})
        tick_labels = visualization.get("tick_labels", {})
        labels = tick_labels.get("long", tick_labels.get("short", []))
        labels = [
            str(label).replace("\n", " ").strip()
            for label in labels
            if str(label).strip()
        ]

        if (
            visualization.get("is_discrete", False)
            and len(labels) == 2
            and abs(float(threshold) - 0.5) < 1e-8
        ):
            return labels[0]

        unit = type_metadata.get("units", "")
        unit_suffix = f" {unit}" if unit else ""
        return f"<= {threshold:.2f}{unit_suffix}"

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
        print(f"FeatureImportanceData Names: {len(data_names)} ({', '.join(data_names)})")
        
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
            If analysis not found or not tree-based (decision_tree or
            random_forest)
        """
        supported_analyzers = ("decision_tree", "random_forest")

        if analysis_name not in pipeline_data.feature_importance_data:
            raise ValueError(
                f"Analysis '{analysis_name}' not found. "
                f"Available: {list(pipeline_data.feature_importance_data.keys())}"
            )

        fi_data = pipeline_data.feature_importance_data[analysis_name]

        if fi_data.analyzer_type not in supported_analyzers:
            raise ValueError(
                f"get_representative_frames() currently only supports "
                f"{supported_analyzers} analyzers, got "
                f"'{fi_data.analyzer_type}'"
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

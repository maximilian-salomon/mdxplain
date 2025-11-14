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

"""Factory for adding feature importance analyzers with simplified syntax."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager.feature_importance_manager import FeatureImportanceManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..analyzer_type import DecisionTree


class FeatureImportanceAddService:
    """
    Service for adding feature importance analyzers without explicit type instantiation.
    
    This service provides an intuitive interface for adding feature importance analyzers
    without requiring users to import and instantiate analyzer types directly.
    All analyzer type parameters are combined with manager.add_analysis parameters.
    
    Examples
    --------
    >>> pipeline.feature_importance.add.decision_tree("my_comparison", max_depth=5, analysis_name="tree_analysis")
    """
    
    def __init__(self, manager: FeatureImportanceManager, pipeline_data: PipelineData) -> None:
        """
        Initialize factory with manager and pipeline data.
        
        Parameters
        ----------
        manager : FeatureImportanceManager
            Feature importance manager instance
        pipeline_data : PipelineData
            Pipeline data container (injected by AutoInjectProxy)
            
        Returns
        -------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    def decision_tree(
        self,
        comparison_name: str,
        analysis_name: str,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[str] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Optional[str] = None,
        ccp_alpha: float = 0.0,
        max_samples: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """
        Add Decision Tree feature importance analysis.
        
        Decision Tree classifier computes feature importance scores based on how much
        each feature contributes to reducing impurity at tree splits. This provides
        interpretable feature rankings for understanding which molecular features
        are most important for distinguishing between different states.
        
        Parameters
        ----------
        comparison_name : str
            Name of the comparison to analyze
        analysis_name : str
            Name to store the analysis results
        criterion : str, default="gini"
            Function to measure quality of splits ("gini" or "entropy")
        splitter : str, default="best"
            Strategy to split at each node ("best" or "random")
        max_depth : int, optional
            Maximum depth of tree. None means unlimited depth
        min_samples_split : int, default=2
            Minimum samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum samples required to be at a leaf node
        min_weight_fraction_leaf : float, default=0.0
            Minimum weighted fraction of sum total of weights required at leaf
        max_features : str, optional
            Number of features to consider when looking for best split
        random_state : int, optional
            Random state for reproducible results
        max_leaf_nodes : int, optional
            Maximum number of leaf nodes in best-first manner
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease required to make a split
        class_weight : str, optional
            Weights associated with classes ("balanced" or None)
        ccp_alpha : float, default=0.0
            Complexity parameter for Minimal Cost-Complexity Pruning
        max_samples : int, optional
            Maximum number of samples to use for training. If None, automatically
            calculated based on max_memory_gb from pipeline. Use this to manually
            override memory-based sampling (e.g., max_samples=50000 for large datasets).
        force : bool, default=False
            Whether to overwrite existing analysis with same name

        Returns
        -------
        None
            Adds Decision Tree feature importance results to pipeline data
            
        Examples
        --------
        >>> # Basic decision tree analysis
        >>> pipeline.feature_importance.add.decision_tree(
        ...     "folded_vs_unfolded", 
        ...     "basic_tree"
        ... )
        
        >>> # Decision tree with custom parameters
        >>> pipeline.feature_importance.add.decision_tree(
        ...     "state_comparison",
        ...     "deep_tree",
        ...     max_depth=10,
        ...     min_samples_split=20,
        ...     random_state=42
        ... )
        
        >>> # Decision tree with entropy criterion
        >>> pipeline.feature_importance.add.decision_tree(
        ...     "conformational_states",
        ...     "entropy_tree",
        ...     criterion="entropy",
        ...     max_features="sqrt",
        ...     class_weight="balanced"
        ... )
        
        Notes
        -----
        Decision trees provide interpretable feature importance based on split
        criteria. Higher importance scores indicate features that contribute
        more to reducing impurity when making classification decisions.

        Uses sklearn.tree.DecisionTreeClassifier internally.
        """
        analyzer_type = DecisionTree(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        return self._manager.add_analysis(
            self._pipeline_data,
            comparison_name,
            analyzer_type,
            analysis_name,
            force=force,
        )

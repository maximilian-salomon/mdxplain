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
Decision tree analyzer type for feature importance analysis.

This module implements a Decision Tree analyzer type for feature importance
analysis, following the analyzer_type pattern similar to decomposition_types.
"""

from typing import Dict, Any, Optional
import numpy as np

from ..interfaces.analyzer_type_base import AnalyzerTypeBase
from .decision_tree_calculator import DecisionTreeCalculator


class DecisionTree(AnalyzerTypeBase):
    """
    Decision Tree analyzer type for feature importance analysis.

    Implements a Decision Tree classifier using scikit-learn to compute
    feature importance scores. The tree structure provides interpretable
    feature importance based on how much each feature contributes to
    reducing impurity at each split.

    Examples
    --------
    Basic usage via FeatureImportanceManager:

    >>> from mdxplain.feature_importance import analyzer_types
    >>> analyzer = analyzer_types.DecisionTree(max_depth=5, random_state=42)
    >>> analyzer.init_calculator()
    >>> pipeline.feature_importance.add(
    ...     "my_comparison", analyzer, "tree_analysis"
    ... )

    Direct usage:

    >>> analyzer = DecisionTree(max_depth=3)
    >>> analyzer.init_calculator()
    >>> result = analyzer.compute(X, y)
    >>> importance_scores = result['importances']
    >>> trained_model = result['model']
    """

    def __init__(
        self,
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
    ):
        """
        Initialize Decision Tree analyzer type with parameters.

        Creates a DecisionTree instance with specified parameters that will be
        used during computation via the calculator. Mainly they are from sklearn's
        DecisionTreeClassifier. For more details on parameters, see
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.

        Parameters
        ----------
        criterion : str, default="gini"
            Function to measure quality of splits ("gini" or "entropy")
        splitter : str, default="best"
            Strategy to split at each node ("best" or "random")
        max_depth : int, optional
            Maximum depth of tree. None means unlimited depth.
        min_samples_split : int, default=2
            Minimum samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum samples required to be at a leaf node
        min_weight_fraction_leaf : float, default=0.0
            Minimum weighted fraction of sum total of weights required at leaf
        max_features : str, optional
            Number of features to consider when looking for best split
        random_state : int, optional
            Controls randomness of estimator for reproducible results
        max_leaf_nodes : int, optional
            Maximum number of leaf nodes. None means unlimited nodes.
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease required for split
        class_weight : str, optional
            Weights associated with classes ("balanced" or None)
        ccp_alpha : float, default=0.0
            Complexity parameter for minimal cost-complexity pruning

        Returns
        -------
        None
            Initializes DecisionTree with specified parameters

        Examples
        --------
        >>> # Basic decision tree
        >>> dt = DecisionTree(max_depth=5, random_state=42)
        >>> print(f"Type: {dt.get_type_name()}")
        'decision_tree'
        """
        super().__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the type name for Decision Tree analyzer.

        Returns the unique string identifier for Decision Tree analyzer type
        used for storing results and type identification.

        Parameters
        ----------
        cls : type
            The DecisionTree class

        Returns
        -------
        str
            String identifier 'decision_tree'

        Examples
        --------
        >>> print(DecisionTree.get_type_name())
        'decision_tree'
        >>> # Can also be used via analyzer_types module
        >>> from mdxplain.feature_importance import analyzer_types
        >>> print(analyzer_types.DecisionTree.get_type_name())
        'decision_tree'
        """
        return "decision_tree"

    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the Decision Tree calculator with specified configuration.

        Sets up the Decision Tree calculator with options for memory mapping and
        chunk processing for large datasets.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Path for cache files (reserved for future use)
        chunk_size : int, optional
            Number of samples to process per chunk (reserved for future use)

        Returns
        -------
        None
            Sets self.calculator to initialized DecisionTreeCalculator instance

        Examples
        --------
        >>> # Basic initialization
        >>> dt = DecisionTree()
        >>> dt.init_calculator()

        >>> # With memory mapping for large datasets
        >>> dt.init_calculator(use_memmap=True, chunk_size=1000)

        >>> # With custom chunk size
        >>> dt.init_calculator(chunk_size=500)
        """
        self.calculator = DecisionTreeCalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute Decision Tree feature importance analysis.

        Performs Decision Tree classification on the input feature matrix
        using the initialized calculator and the parameters provided during
        initialization.

        Parameters
        ----------
        X : numpy.ndarray
            Input feature matrix to analyze, shape (n_samples, n_features)
        y : numpy.ndarray
            Target labels, shape (n_samples,)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            
            - importances: Feature importance scores (n_features,)
            - model: Trained DecisionTreeClassifier instance
            - metadata: Dictionary with analysis information including:
              
              * hyperparameters: Used parameters
              * train_accuracy: Training accuracy score
              * classification_report: Detailed classification metrics
              * tree_depth: Depth of the trained tree
              * tree_n_leaves: Number of leaves in the tree

        Examples
        --------
        >>> # Compute feature importance with predefined parameters
        >>> dt = DecisionTree(max_depth=5, random_state=42)
        >>> dt.init_calculator()
        >>> X = np.random.rand(1000, 50)
        >>> y = np.random.choice([0, 1], 1000)
        >>> result = dt.compute(X, y)
        >>> print(f"Importance shape: {result['importances'].shape}")

        Raises
        ------
        ValueError
            If calculator is not initialized, input data is invalid,
            or computation fails
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        # Pass all instance parameters to calculator
        return self.calculator.compute(
            X, y,
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
        )

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters used by this Decision Tree analyzer.

        Returns a dictionary of all parameters used by this analyzer instance.
        This is used for metadata storage and reproducibility.

        Parameters
        ----------
        None but self : DecisionTree
            The DecisionTree instance

        Returns
        -------
        Dict[str, Any]
            Dictionary of Decision Tree parameters

        Examples
        --------
        >>> dt = DecisionTree(max_depth=5, random_state=42)
        >>> params = dt.get_params()
        >>> print(f"Max depth: {params['max_depth']}")
        >>> print(f"Random state: {params['random_state']}")
        """
        return {
            'criterion': self.criterion,
            'splitter': self.splitter,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'class_weight': self.class_weight,
            'ccp_alpha': self.ccp_alpha,
        }
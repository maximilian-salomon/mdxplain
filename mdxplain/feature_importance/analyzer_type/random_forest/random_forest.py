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
Random Forest analyzer type for feature importance analysis.

This module implements a Random Forest analyzer type for feature importance
analysis, following the analyzer_type pattern shared with the Decision Tree
analyzer. Importance can be computed from impurity reduction (GINI) or from
SHAP values.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..interfaces.analyzer_type_base import AnalyzerTypeBase
from .random_forest_calculator import RandomForestCalculator


class RandomForest(AnalyzerTypeBase):
    """
    Random Forest analyzer type for feature importance analysis.

    Trains a scikit-learn RandomForestClassifier and computes feature
    importance scores. The importance is derived either from impurity reduction
    across the trees (``importance_method="gini"``) or from aggregated SHAP
    values (``importance_method="shap"``).

    Examples
    --------
    Basic usage via FeatureImportanceManager:

    >>> from mdxplain.feature_importance import analyzer_type
    >>> analyzer = analyzer_type.RandomForest(n_estimators=200, random_state=42)
    >>> analyzer.init_calculator()
    >>> pipeline.feature_importance.add_analysis(
    ...     "my_comparison", analyzer, "forest_analysis"
    ... )

    Direct usage:

    >>> analyzer = RandomForest(n_estimators=100, importance_method="shap")
    >>> analyzer.init_calculator()
    >>> result = analyzer.compute(X, y)
    >>> importance_scores = result['importances']
    >>> trained_model = result['model']
    """

    def __init__(
        self,
        n_estimators: int = 3000,
        importance_method: str = "shap",
        criterion: str = "gini",
        max_depth: Optional[int] = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[str] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Optional[str] = "balanced",
        ccp_alpha: float = 0.0,
        max_samples: Optional[int] = None,
        shap_sample_size: Optional[int] = None,
        n_jobs: Optional[int] = -1,
        max_blas_threads: Optional[int] = 1,
        auto_limit_blas: bool = True,
    ):
        """
        Initialize Random Forest analyzer type with parameters.

        Creates a RandomForest instance with parameters used during computation
        via the calculator. Most parameters map to sklearn's
        RandomForestClassifier. For details, see
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.

        Parameters
        ----------
        n_estimators : int, default=3000
            Number of trees in the forest
        importance_method : str, default="shap"
            Feature importance method, "shap" or "gini" (impurity reduction)
        criterion : str, default="gini"
            Function to measure the quality of a split ("gini" or "entropy")
        max_depth : int, optional, default=6
            Maximum depth of each tree. None means unlimited depth. The shallow
            default keeps SHAP affordable and spreads importance across
            correlated features instead of concentrating it on one.
        min_samples_split : int, default=2
            Minimum samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum samples required to be at a leaf node
        min_weight_fraction_leaf : float, default=0.0
            Minimum weighted fraction of the sum total of weights at a leaf
        max_features : str, optional, default="sqrt"
            Number of features to consider when looking for the best split
        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees
        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate generalization score
        random_state : int, optional
            Controls randomness of the estimator for reproducible results
        max_leaf_nodes : int, optional
            Maximum number of leaf nodes. None means unlimited nodes.
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease required for a split
        class_weight : str, optional, default="balanced"
            Weights associated with classes ("balanced", "balanced_subsample"
            or None)
        ccp_alpha : float, default=0.0
            Complexity parameter for minimal cost-complexity pruning
        max_samples : int, optional
            Maximum number of samples to use for training. If None, it is
            calculated from max_memory_gb. This is a memory-based row cap
            applied before training; it is not sklearn's bootstrap max_samples.
        shap_sample_size : int, optional
            Maximum number of rows SHAP is evaluated on. If None, SHAP uses all
            training rows (already capped by max_samples / max_memory_gb).
        n_jobs : int, optional, default=-1
            Number of parallel jobs for training the forest
        max_blas_threads : int, optional, default=1
            Preferred BLAS/OpenMP thread limit during training
        auto_limit_blas : bool, default=True
            Apply BLAS=1 when n_jobs != 1 to avoid thread oversubscription

        Returns
        -------
        None
            Initializes RandomForest with the specified parameters

        Examples
        --------
        >>> rf = RandomForest(n_estimators=200, random_state=42)
        >>> print(f"Type: {rf.get_type_name()}")
        'random_forest'
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.importance_method = importance_method
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.shap_sample_size = shap_sample_size
        self.n_jobs = n_jobs
        self.max_blas_threads = max_blas_threads
        self.auto_limit_blas = auto_limit_blas

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the type name for the Random Forest analyzer.

        Returns the unique string identifier for the Random Forest analyzer
        type used for storing results and type identification.

        Parameters
        ----------
        cls : type
            The RandomForest class

        Returns
        -------
        str
            String identifier 'random_forest'

        Examples
        --------
        >>> print(RandomForest.get_type_name())
        'random_forest'
        """
        return "random_forest"

    def init_calculator(
        self,
        use_memmap: bool = False,
        cache_path: str = "./cache",
        chunk_size: int = 2000,
        max_memory_gb: float = 6.0,
    ) -> None:
        """
        Initialize the Random Forest calculator with the given configuration.

        Sets up the Random Forest calculator with memory and thread options.
        The BLAS thread settings are taken from this analyzer instance.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, default="./cache"
            Path for cache files (reserved for future use)
        chunk_size : int, default=2000
            Number of samples processed per chunk during SHAP computation
        max_memory_gb : float, default=6.0
            Maximum memory in GB for dataset processing. Datasets exceeding
            this limit are stratified sampled before training.

        Returns
        -------
        None
            Sets self.calculator to an initialized RandomForestCalculator

        Examples
        --------
        >>> rf = RandomForest()
        >>> rf.init_calculator()

        >>> rf.init_calculator(max_memory_gb=8.0, chunk_size=5000)
        """
        self.calculator = RandomForestCalculator(
            use_memmap=use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size,
            max_memory_gb=max_memory_gb,
            max_blas_threads=self.max_blas_threads,
            auto_limit_blas=self.auto_limit_blas,
        )

    def compute(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute Random Forest feature importance analysis.

        Performs Random Forest classification on the input feature matrix using
        the initialized calculator and the parameters provided during
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
            - model: Trained RandomForestClassifier instance
            - metadata: Dictionary with analysis information including
              hyperparameters, train_accuracy, classification_report, and
              either importance_std (GINI) or shap_sample_size (SHAP)

        Raises
        ------
        ValueError
            If the calculator is not initialized, the input data is invalid,
            or the computation fails

        Examples
        --------
        >>> rf = RandomForest(n_estimators=100, random_state=42)
        >>> rf.init_calculator()
        >>> X = np.random.rand(1000, 50)
        >>> y = np.random.choice([0, 1], 1000)
        >>> result = rf.compute(X, y)
        >>> print(f"Importance shape: {result['importances'].shape}")
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        return self.calculator.compute(
            X,
            y,
            n_estimators=self.n_estimators,
            importance_method=self.importance_method,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            shap_sample_size=self.shap_sample_size,
            n_jobs=self.n_jobs,
        )

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters used by this Random Forest analyzer.

        Returns a dictionary of the parameters used by this analyzer instance.
        This is used for metadata storage and reproducibility.

        Parameters
        ----------
        None but self : RandomForest
            The RandomForest instance

        Returns
        -------
        Dict[str, Any]
            Dictionary of Random Forest parameters

        Examples
        --------
        >>> rf = RandomForest(n_estimators=200, random_state=42)
        >>> params = rf.get_params()
        >>> print(f"Trees: {params['n_estimators']}")
        """
        return {
            "n_estimators": self.n_estimators,
            "importance_method": self.importance_method,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "random_state": self.random_state,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "class_weight": self.class_weight,
            "ccp_alpha": self.ccp_alpha,
        }

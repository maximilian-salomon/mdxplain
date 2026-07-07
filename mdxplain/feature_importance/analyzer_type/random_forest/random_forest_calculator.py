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
Random Forest calculator for feature importance analysis.

This module implements the Random Forest classifier computation using
scikit-learn, following the calculator pattern shared with the Decision Tree
analyzer. Feature importance is derived either from impurity reduction (GINI)
or from SHAP values. Memory-based subsampling and BLAS thread limiting bound
the memory and thread usage during training.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ..interfaces.calculator_base import CalculatorBase


class RandomForestCalculator(CalculatorBase):
    """
    Calculator for Random Forest feature importance analysis.

    Trains a scikit-learn RandomForestClassifier and extracts feature
    importance scores either from impurity reduction (``importance_method
    ="gini"``) or from aggregated SHAP values (``importance_method="shap"``).
    Reuses the memory-based stratified subsampling and BLAS thread limiting
    from :class:`CalculatorBase`; subsampling limits the number of training
    rows to the configured memory budget.

    Examples
    --------
    >>> calculator = RandomForestCalculator()
    >>> X = np.random.rand(1000, 50)
    >>> y = np.random.choice([0, 1], 1000)
    >>> result = calculator.compute(X, y, n_estimators=100, random_state=42)
    >>> importance_scores = result['importances']
    >>> trained_model = result['model']
    """

    def compute(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute feature importance using a Random Forest classifier.

        Runs validation, memory-based subsampling, forest training and
        importance extraction. Each step is delegated to a helper method.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        y : np.ndarray
            Target labels with shape (n_samples,)
        kwargs : dict
            Random Forest parameters (n_estimators, importance_method, n_jobs,
            shap_sample_size, max_samples, random_state, and sklearn
            RandomForestClassifier hyperparameters).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:

            - 'importances': np.ndarray of feature importance scores
            - 'model': Trained RandomForestClassifier instance
            - 'metadata': Dict with training metrics and parameters

        Raises
        ------
        ValueError
            If input data has invalid shape or contains NaN values

        Examples
        --------
        >>> result = calculator.compute(X, y, n_estimators=200, random_state=1)
        >>> importance_scores = result['importances']
        >>> accuracy = result['metadata']['train_accuracy']
        """
        self._validate_input_data(X, y)

        importance_method = kwargs.get("importance_method", "shap")
        n_jobs = kwargs.get("n_jobs", -1)
        shap_sample_size = kwargs.get("shap_sample_size", None)
        random_state = kwargs.get("random_state", None)

        max_samples = self._resolve_max_samples(
            X, kwargs.get("max_samples", None)
        )
        X_train, y_train = self._apply_stratified_sampling(
            X, y, max_samples, random_state
        )

        rf_params = self._build_rf_params(kwargs)
        rf = self._train_forest(X_train, y_train, rf_params, n_jobs)

        importances, importance_std = self._resolve_importances(
            rf,
            X_train,
            y_train,
            importance_method,
            shap_sample_size,
            random_state,
        )
        extras = self._importance_metadata(
            importance_method, importance_std, shap_sample_size
        )
        metadata = self._build_training_metadata(
            rf, X_train, y_train, rf_params, X, max_samples, extras
        )

        return {
            "importances": importances,
            "model": rf,
            "metadata": metadata,
        }

    def _resolve_max_samples(
        self, X: np.ndarray, user_max_samples: Optional[int]
    ) -> int:
        """
        Resolve the effective sample cap for training.

        Uses the user-provided override when given, otherwise derives the cap
        from the configured memory budget.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        user_max_samples : int or None
            Explicit sample cap provided by the user, or None for auto

        Returns
        -------
        int
            Maximum number of training samples to use

        Examples
        --------
        >>> calc = RandomForestCalculator(max_memory_gb=6.0)
        >>> calc._resolve_max_samples(np.zeros((10, 4)), None)
        10
        """
        if user_max_samples is not None:
            return user_max_samples
        return self._calculate_max_samples(X, self.max_memory_gb)

    @staticmethod
    def _build_rf_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assemble the scikit-learn RandomForestClassifier parameter dict.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments forwarded from the analyzer, holding the Random
            Forest hyperparameters.

        Returns
        -------
        Dict[str, Any]
            Parameter dictionary for RandomForestClassifier (excluding n_jobs,
            which is applied separately during training).

        Examples
        --------
        >>> params = RandomForestCalculator._build_rf_params(
        ...     {"n_estimators": 50}
        ... )
        >>> params["n_estimators"]
        50
        """
        return {
            "n_estimators": kwargs.get("n_estimators", 3000),
            "criterion": kwargs.get("criterion", "gini"),
            "max_depth": kwargs.get("max_depth", 6),
            "min_samples_split": kwargs.get("min_samples_split", 2),
            "min_samples_leaf": kwargs.get("min_samples_leaf", 1),
            "min_weight_fraction_leaf": kwargs.get(
                "min_weight_fraction_leaf", 0.0
            ),
            "max_features": kwargs.get("max_features", "sqrt"),
            "bootstrap": kwargs.get("bootstrap", True),
            "oob_score": kwargs.get("oob_score", False),
            "random_state": kwargs.get("random_state", None),
            "max_leaf_nodes": kwargs.get("max_leaf_nodes", None),
            "min_impurity_decrease": kwargs.get("min_impurity_decrease", 0.0),
            "class_weight": kwargs.get("class_weight", "balanced"),
            "ccp_alpha": kwargs.get("ccp_alpha", 0.0),
        }

    def _train_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        rf_params: Dict[str, Any],
        n_jobs: Optional[int],
    ) -> RandomForestClassifier:
        """
        Create and fit a RandomForestClassifier within a BLAS thread limit.

        The fit runs inside the shared thread-limiting context to avoid thread
        oversubscription (parallel trees x BLAS threads).

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix (potentially subsampled)
        y_train : np.ndarray
            Training labels (potentially subsampled)
        rf_params : Dict[str, Any]
            RandomForestClassifier hyperparameters
        n_jobs : int or None
            Number of parallel jobs for the forest

        Returns
        -------
        RandomForestClassifier
            The trained forest

        Examples
        --------
        >>> calc = RandomForestCalculator()
        >>> params = RandomForestCalculator._build_rf_params({"n_estimators": 10})
        >>> rf = calc._train_forest(X, y, params, n_jobs=-1)
        """
        rf = RandomForestClassifier(n_jobs=n_jobs, **rf_params)
        with self._limit_threadpools(n_jobs):
            rf.fit(X_train, y_train)
        return rf

    def _resolve_importances(
        self,
        rf: RandomForestClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        importance_method: str,
        shap_sample_size: Optional[int],
        random_state: Optional[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Dispatch to the requested feature importance method.

        Parameters
        ----------
        rf : RandomForestClassifier
            Trained forest
        X_train : np.ndarray
            Training feature matrix used as the SHAP evaluation pool
        y_train : np.ndarray
            Training labels (used to stratify optional SHAP subsampling)
        importance_method : str
            Either "gini" (impurity reduction) or "shap"
        shap_sample_size : int or None
            Optional cap on the number of rows used for SHAP
        random_state : int, optional
            Random state for reproducible SHAP subsampling

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (importances, importance_std). importance_std is the per-tree
            standard deviation for GINI and the per-frame standard deviation
            for SHAP.

        Examples
        --------
        >>> imp, std = calc._resolve_importances(rf, X, y, "gini", None, 42)
        >>> imp.shape[0] == X.shape[1]
        True
        """
        if importance_method == "shap":
            X_eval = self._prepare_shap_eval_set(
                X_train, y_train, shap_sample_size, random_state
            )
            return self._compute_shap_importance(rf, X_eval)
        return self._compute_gini_importance(rf)

    @staticmethod
    def _compute_gini_importance(
        rf: RandomForestClassifier,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute impurity-based (GINI) importance and its per-tree spread.

        Parameters
        ----------
        rf : RandomForestClassifier
            Trained forest

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (importances, importance_std), where importance_std is the standard
            deviation of feature importances across the individual trees.

        Examples
        --------
        >>> imp, std = RandomForestCalculator._compute_gini_importance(rf)
        >>> imp.shape == std.shape
        True
        """
        importances = np.asarray(rf.feature_importances_)
        per_tree = np.array(
            [tree.feature_importances_ for tree in rf.estimators_]
        )
        importance_std = np.asarray(per_tree.std(axis=0))
        return importances, importance_std

    @staticmethod
    def _prepare_shap_eval_set(
        X_train: np.ndarray,
        y_train: np.ndarray,
        shap_sample_size: Optional[int],
        random_state: Optional[int],
    ) -> np.ndarray:
        """
        Select the evaluation rows for SHAP value computation.

        Uses the full training set by default; when ``shap_sample_size`` is set
        and smaller than the training set, a stratified subsample is drawn to
        bound SHAP compute time.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix
        y_train : np.ndarray
            Training labels (used to stratify the subsample)
        shap_sample_size : int or None
            Optional cap on the number of evaluation rows
        random_state : int, optional
            Random state for reproducible subsampling

        Returns
        -------
        np.ndarray
            The feature matrix SHAP will be evaluated on

        Examples
        --------
        >>> X_eval = RandomForestCalculator._prepare_shap_eval_set(
        ...     X, y, 2000, 42
        ... )
        """
        if shap_sample_size is None or X_train.shape[0] <= shap_sample_size:
            return X_train
        X_eval, _, _, _ = train_test_split(
            X_train,
            y_train,
            train_size=shap_sample_size,
            stratify=y_train,
            random_state=random_state,
        )
        return np.asarray(X_eval)

    def _compute_shap_importance(
        self, rf: RandomForestClassifier, X_eval: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute global SHAP importance and its per-frame spread, chunk by chunk.

        Uses ``shap.TreeExplainer`` and accumulates the sum and the sum of
        squares of absolute SHAP values in chunks of ``self.chunk_size`` rows,
        so the full (samples x features x classes) SHAP matrix is never
        materialised. The result is the mean absolute SHAP value per feature
        and its standard deviation across the evaluation frames.

        Parameters
        ----------
        rf : RandomForestClassifier
            Trained forest
        X_eval : np.ndarray
            Feature matrix to evaluate SHAP on

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (importances, importance_std) per feature, shape (n_features,).
            importance_std is the standard deviation across the frames.

        Examples
        --------
        >>> imp, std = calc._compute_shap_importance(rf, X)
        >>> imp.shape[0] == X.shape[1]
        True
        """
        explainer = shap.TreeExplainer(
            rf, feature_perturbation="tree_path_dependent"
        )
        n_samples = X_eval.shape[0]
        abs_sum = np.zeros(X_eval.shape[1], dtype=np.float64)
        sq_sum = np.zeros(X_eval.shape[1], dtype=np.float64)
        count = 0
        for start in range(0, n_samples, self.chunk_size):
            chunk = np.asarray(X_eval[start : start + self.chunk_size])
            per_sample = self._aggregate_abs_shap(explainer.shap_values(chunk))
            abs_sum += per_sample.sum(axis=0)
            sq_sum += np.square(per_sample).sum(axis=0)
            count += per_sample.shape[0]
        mean = abs_sum / count
        variance = np.maximum(sq_sum / count - np.square(mean), 0.0)
        return mean, np.sqrt(variance)

    @staticmethod
    def _aggregate_abs_shap(shap_values: np.ndarray) -> np.ndarray:
        """
        Reduce SHAP values to per-sample, per-feature absolute values.

        The classifier SHAP output has shape
        (n_samples, n_features, n_classes); the absolute values are averaged
        over the class axis.

        Parameters
        ----------
        shap_values : np.ndarray
            The array returned by ``TreeExplainer.shap_values``

        Returns
        -------
        np.ndarray
            Absolute SHAP values with shape (n_samples, n_features)

        Examples
        --------
        >>> arr = np.random.rand(8, 5, 3)
        >>> RandomForestCalculator._aggregate_abs_shap(arr).shape
        (8, 5)
        """
        return np.abs(np.asarray(shap_values)).mean(axis=-1)

    def _build_training_metadata(
        self,
        rf: RandomForestClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        rf_params: Dict[str, Any],
        X_original: np.ndarray,
        max_samples: int,
        extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the analysis metadata dictionary.

        Parameters
        ----------
        rf : RandomForestClassifier
            Trained forest
        X_train : np.ndarray
            Training feature matrix (potentially subsampled)
        y_train : np.ndarray
            Training labels (potentially subsampled)
        rf_params : Dict[str, Any]
            RandomForestClassifier hyperparameters used
        X_original : np.ndarray
            Original feature matrix before subsampling
        max_samples : int
            Sample cap that was applied
        extras : Dict[str, Any]
            Importance-method-specific metadata (see _importance_metadata)

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with training metrics and model info

        Examples
        --------
        >>> meta = calc._build_training_metadata(
        ...     rf, X, y, params, X, 50000, {"importance_method": "gini"}
        ... )
        >>> meta["algorithm"]
        'random_forest'
        """
        train_accuracy, class_report = self._train_metrics(rf, X_train, y_train)
        metadata = {
            "algorithm": "random_forest",
            "hyperparameters": rf_params,
            "train_accuracy": train_accuracy,
            "classification_report": class_report,
            "n_samples": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_classes": len(np.unique(y_train)),
            "oob_score": getattr(rf, "oob_score_", None),
            "sampling": self._sampling_metadata(
                X_original, X_train, max_samples
            ),
        }
        metadata.update(extras)
        return metadata

    @staticmethod
    def _train_metrics(
        rf: RandomForestClassifier, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[float, Any]:
        """
        Compute training accuracy and classification report.

        Parameters
        ----------
        rf : RandomForestClassifier
            Trained forest
        X_train : np.ndarray
            Training feature matrix
        y_train : np.ndarray
            Training labels

        Returns
        -------
        Tuple[float, Any]
            (train_accuracy, classification_report_dict)

        Examples
        --------
        >>> acc, report = RandomForestCalculator._train_metrics(rf, X, y)
        """
        y_pred = rf.predict(X_train)
        train_accuracy = float(accuracy_score(y_train, y_pred))
        class_report = classification_report(y_train, y_pred, output_dict=True)
        return train_accuracy, class_report

    def _sampling_metadata(
        self, X_original: np.ndarray, X_train: np.ndarray, max_samples: int
    ) -> Dict[str, Any]:
        """
        Build the subsampling section of the metadata.

        Parameters
        ----------
        X_original : np.ndarray
            Original feature matrix before subsampling
        X_train : np.ndarray
            Feature matrix used for training (potentially subsampled)
        max_samples : int
            Sample cap that was applied

        Returns
        -------
        Dict[str, Any]
            Dictionary describing the applied subsampling

        Examples
        --------
        >>> info = calc._sampling_metadata(X, X, 50000)
        >>> info["sampled"]
        False
        """
        return {
            "original_samples": int(X_original.shape[0]),
            "used_samples": int(X_train.shape[0]),
            "sampled": bool(X_original.shape[0] > max_samples),
            "max_memory_gb": self.max_memory_gb,
        }

    @staticmethod
    def _importance_metadata(
        importance_method: str,
        importance_std: Optional[np.ndarray],
        shap_sample_size: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build importance-method-specific metadata.

        Parameters
        ----------
        importance_method : str
            Either "gini" or "shap"
        importance_std : np.ndarray or None
            Importance standard deviation (per-tree for GINI, per-frame for
            SHAP)
        shap_sample_size : int or None
            SHAP evaluation cap (SHAP only)

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the chosen importance method

        Examples
        --------
        >>> RandomForestCalculator._importance_metadata(
        ...     "shap", None, 2000
        ... )["importance_method"]
        'shap'
        """
        std_list = importance_std.tolist() if importance_std is not None else []
        if importance_method == "shap":
            return {
                "importance_method": "shap",
                "shap_sample_size": shap_sample_size,
                "importance_std": std_list,
            }
        return {
            "importance_method": "gini",
            "importance_std": std_list,
        }

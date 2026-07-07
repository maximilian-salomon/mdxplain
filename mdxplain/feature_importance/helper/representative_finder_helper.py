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
Representative frame finder for structure visualization.

This module provides utilities for finding representative frames from
DataSelectors, supporting both "best" (feature-based) and "centroid"
(distance-based) selection modes. Includes memmap-safe implementations.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ...feature_importance.entities.feature_importance_data import (
        FeatureImportanceData,
    )



class RepresentativeFinderHelper:
    """
    Helper for finding representative frames from DataSelectors.

    Provides methods to find frames that best represent a DataSelector,
    either by maximizing alignment with top important features ("best")
    or by finding the centroid frame ("centroid").

    Examples
    --------
    >>> # Find best representative for a comparison
    >>> traj_idx, frame_idx = RepresentativeFinderHelper.find_best_representative(
    ...     pipeline_data, fi_data, "cluster_0_vs_rest", n_top=10
    ... )

    >>> # Find centroid frame
    >>> traj_idx, frame_idx = RepresentativeFinderHelper.find_centroid_frame(
    ...     pipeline_data, "cluster_0", "my_features"
    ... )
    """

    @staticmethod
    def find_best_tree_based(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        comparison_identifier: str,
        n_top: int = 10,
        use_memmap: bool = False,
        chunk_size: int = 1000
    ) -> Tuple[int, int]:
        """
        Find frame using tree-based scoring from Decision Tree splits.

        Analyzes Decision Tree split rules to find frames that most strongly
        exhibit the top important features. Uses actual tree thresholds and
        split directions rather than median values.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        fi_data : FeatureImportanceData
            Feature importance data with Decision Tree model
        comparison_identifier : str
            Sub-comparison identifier
        n_top : int, default=10
            Number of top features to consider
        use_memmap : bool, default=False
            Whether to use memmap-safe processing
        chunk_size : int, default=1000
            Chunk size for memmap processing

        Returns
        -------
        Tuple[int, int]
            (trajectory_index, frame_index) of best representative

        Examples
        --------
        >>> traj_idx, frame_idx = RepresentativeFinderHelper.find_best_tree_based(
        ...     pipeline_data, fi_data, "cluster_0_vs_rest", n_top=10
        ... )

        Notes
        -----
        - Uses sklearn DecisionTree split thresholds
        - Handles periodic features with circular distance
        - Scores frames by alignment with tree rules
        """
        comp_data = pipeline_data.comparison_data[fi_data.comparison_name]
        sub_comp = comp_data.get_sub_comparison(comparison_identifier)
        ds_name = sub_comp["group1_selectors"][0]
        target_label = sub_comp.get("labels", (0, 1))[0]

        if not use_memmap:
            return RepresentativeFinderHelper._find_best_tree_fast(
                pipeline_data, fi_data, comparison_identifier,
                ds_name, target_label, n_top
            )
        else:
            return RepresentativeFinderHelper._find_best_tree_chunked(
                pipeline_data, fi_data, comparison_identifier,
                ds_name, target_label, n_top, chunk_size
            )

    @staticmethod
    def _find_best_tree_fast(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        comparison_identifier: str,
        ds_name: str,
        target_label: int,
        n_top: int
    ) -> Tuple[int, int]:
        """
        Fast tree-based scoring without memmap constraints.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        fi_data : FeatureImportanceData
            Feature importance data
        comparison_identifier : str
            Sub-comparison identifier
        ds_name : str
            DataSelector name
        target_label : int
            Label corresponding to group1 in the comparison
        n_top : int
            Number of top features

        Returns
        -------
        Tuple[int, int]
            (trajectory_index, frame_index) of best frame
        """
        top_features = fi_data.get_top_features(comparison_identifier, n_top)
        feature_indices = [f[0] for f in top_features]
        feature_importances = [f[1] for f in top_features]

        _, metadata = fi_data.get_comparison(comparison_identifier)
        model = metadata.get("model")

        if model is None:
            raise ValueError("No Decision Tree model found in metadata")

        tree_rules = RepresentativeFinderHelper._extract_tree_rules(
            model, feature_indices, target_label
        )

        selected_data, frame_mapping = pipeline_data.get_selected_data(
            fi_data.feature_selector, ds_name, return_frame_mapping=True
        )
        feature_scales = RepresentativeFinderHelper._compute_feature_scales(
            selected_data, feature_indices
        )

        scores = RepresentativeFinderHelper._score_frames_tree_based(
            selected_data, feature_indices, feature_importances,
            tree_rules, feature_scales
        )

        best_local_idx = np.argmax(scores)

        return frame_mapping[best_local_idx]

    @staticmethod
    def _find_best_tree_chunked(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        comparison_identifier: str,
        ds_name: str,
        target_label: int,
        n_top: int,
        chunk_size: int
    ) -> Tuple[int, int]:
        """
        Memmap-safe tree-based scoring with chunked processing.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        fi_data : FeatureImportanceData
            Feature importance data
        comparison_identifier : str
            Sub-comparison identifier
        ds_name : str
            DataSelector name
        target_label : int
            Label corresponding to group1 in the comparison
        n_top : int
            Number of top features
        chunk_size : int
            Chunk size for processing

        Returns
        -------
        Tuple[int, int]
            (trajectory_index, frame_index) of best frame
        """
        top_features = fi_data.get_top_features(comparison_identifier, n_top)
        feature_indices = [f[0] for f in top_features]
        feature_importances = [f[1] for f in top_features]

        _, metadata = fi_data.get_comparison(comparison_identifier)
        model = metadata.get("model")

        if model is None:
            raise ValueError("No Decision Tree model found in metadata")

        tree_rules = RepresentativeFinderHelper._extract_tree_rules(
            model, feature_indices, target_label
        )

        selected_data, frame_mapping = pipeline_data.get_selected_data(
            fi_data.feature_selector, ds_name, return_frame_mapping=True
        )
        feature_scales = RepresentativeFinderHelper._compute_feature_scales(
            selected_data, feature_indices, chunk_size
        )

        best_idx = RepresentativeFinderHelper._find_best_in_chunks(
            selected_data, feature_indices, feature_importances,
            tree_rules, feature_scales, chunk_size
        )

        if hasattr(selected_data, '_mmap') and selected_data._mmap is not None:
            selected_data._mmap.close()

        return frame_mapping[best_idx]

    @staticmethod
    def _find_best_in_chunks(
        selected_data: np.ndarray,
        feature_indices: List[int],
        feature_importances: List[float],
        tree_rules: Dict[int, Dict[str, float]],
        feature_scales: Dict[int, float],
        chunk_size: int
    ) -> int:
        """
        Find frame with best tree-based score across chunks.

        Parameters
        ----------
        selected_data : np.ndarray
            Pre-loaded feature data
        feature_indices : List[int]
            Feature indices to score
        feature_importances : List[float]
            Importance weights
        tree_rules : Dict[int, Dict[str, float]]
            Tree split rules
        feature_scales : Dict[int, float]
            Mapping from feature index to scale for margin normalization
        chunk_size : int
            Chunk size for processing

        Returns
        -------
        int
            Local index of best frame
        """
        best_score = -np.inf
        best_idx = 0
        current_offset = 0
        n_frames = selected_data.shape[0]

        for start_idx in range(0, n_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, n_frames)
            chunk = selected_data[start_idx:end_idx]

            scores = RepresentativeFinderHelper._score_frames_tree_based(
                chunk, feature_indices, feature_importances,
                tree_rules, feature_scales
            )

            chunk_max_idx = np.argmax(scores)
            chunk_max_score = scores[chunk_max_idx]

            if chunk_max_score > best_score:
                best_score = chunk_max_score
                best_idx = current_offset + chunk_max_idx

            current_offset += chunk.shape[0]

        return best_idx

    @staticmethod
    def _extract_tree_rules(
        model,
        feature_indices: List[int],
        target_label: int,
    ) -> Dict[int, Dict[str, float]]:
        """
        Extract one representative split rule per feature from a model.

        Supports single Decision Trees (via ``model.tree_``) and tree
        ensembles such as Random Forest (via ``model.estimators_``). For an
        ensemble the per-tree rules are aggregated into one rule per feature.

        Parameters
        ----------
        model : sklearn tree or ensemble classifier
            Trained DecisionTreeClassifier or RandomForestClassifier
        feature_indices : List[int]
            Indices of features to extract rules for
        target_label : int
            Label corresponding to group1 in the comparison

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping from feature_idx to dict with threshold, direction,
            and weight
        """
        target_class_index = RepresentativeFinderHelper._get_target_class_index(
            model, target_label
        )
        if hasattr(model, "estimators_"):
            return RepresentativeFinderHelper._extract_ensemble_tree_rules(
                model, feature_indices, target_class_index
            )
        return RepresentativeFinderHelper._collect_single_tree_rules(
            model.tree_, feature_indices, target_class_index
        )

    @staticmethod
    def _collect_single_tree_rules(
        tree,
        feature_indices: List[int],
        target_class_index: int,
    ) -> Dict[int, Dict[str, float]]:
        """
        Collect the best split rule per feature from a single tree.

        Parameters
        ----------
        tree : sklearn tree object
            Tree structure from a fitted DecisionTreeClassifier
        feature_indices : List[int]
            Indices of features to extract rules for
        target_class_index : int
            Target class index in tree.value

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping from feature_idx to its best split rule
        """
        rules = {}
        for feat_idx in feature_indices:
            split_rule = RepresentativeFinderHelper._find_best_feature_split(
                tree, feat_idx, target_class_index
            )
            if split_rule is not None:
                rules[feat_idx] = split_rule
        return rules

    @staticmethod
    def _extract_ensemble_tree_rules(
        model,
        feature_indices: List[int],
        target_class_index: int,
    ) -> Dict[int, Dict[str, float]]:
        """
        Aggregate per-feature split rules across all trees of an ensemble.

        Parameters
        ----------
        model : sklearn ensemble classifier
            Trained classifier exposing ``estimators_``
        feature_indices : List[int]
            Indices of features to extract rules for
        target_class_index : int
            Target class index in tree.value

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping from feature_idx to the aggregated split rule
        """
        rules_per_feature: Dict[int, List[Dict[str, float]]] = {
            feat_idx: [] for feat_idx in feature_indices
        }
        for estimator in model.estimators_:
            tree_rules = RepresentativeFinderHelper._collect_single_tree_rules(
                estimator.tree_, feature_indices, target_class_index
            )
            for feat_idx, rule in tree_rules.items():
                rules_per_feature[feat_idx].append(rule)
        return RepresentativeFinderHelper._aggregate_all_feature_rules(
            rules_per_feature
        )

    @staticmethod
    def _aggregate_all_feature_rules(
        rules_per_feature: Dict[int, List[Dict[str, float]]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Aggregate the collected per-tree rules for every feature.

        Parameters
        ----------
        rules_per_feature : Dict[int, List[Dict[str, float]]]
            Mapping from feature_idx to the list of per-tree split rules

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping from feature_idx to the single aggregated rule
        """
        aggregated = {}
        for feat_idx, rules in rules_per_feature.items():
            combined = RepresentativeFinderHelper._aggregate_feature_rules(rules)
            if combined is not None:
                aggregated[feat_idx] = combined
        return aggregated

    @staticmethod
    def _aggregate_feature_rules(
        rules: List[Dict[str, float]],
    ) -> Dict[str, float] | None:
        """
        Combine several per-tree rules for one feature into a single rule.

        The threshold is the weight-weighted mean of the per-tree thresholds,
        the weight is the mean per-tree weight, and the direction is the
        weight-based majority direction.

        Parameters
        ----------
        rules : List[Dict[str, float]]
            Per-tree split rules for one feature

        Returns
        -------
        Dict[str, float] or None
            Aggregated split rule, or None if no usable rule exists
        """
        if not rules:
            return None
        weights = np.array([rule["weight"] for rule in rules], dtype=float)
        thresholds = np.array(
            [rule["threshold"] for rule in rules], dtype=float
        )
        total_weight = float(weights.sum())
        if total_weight <= 0:
            return None
        return {
            "threshold": float(np.average(thresholds, weights=weights)),
            "weight": total_weight / len(rules),
            "direction": RepresentativeFinderHelper._majority_direction(
                rules, weights
            ),
        }

    @staticmethod
    def _majority_direction(
        rules: List[Dict[str, float]], weights: np.ndarray
    ) -> str:
        """
        Determine the weight-based majority split direction.

        Parameters
        ----------
        rules : List[Dict[str, float]]
            Per-tree split rules for one feature
        weights : np.ndarray
            Per-tree weights aligned with rules

        Returns
        -------
        str
            "le" if the weighted "le" votes dominate, otherwise "gt"
        """
        le_weight = sum(
            weight
            for rule, weight in zip(rules, weights)
            if rule["direction"] == "le"
        )
        gt_weight = float(weights.sum()) - le_weight
        return "le" if le_weight >= gt_weight else "gt"

    @staticmethod
    def _get_target_class_index(model, target_label: int) -> int:
        """
        Resolve target label to class index used by the sklearn tree.

        Parameters
        ----------
        model : sklearn DecisionTreeClassifier
            Trained Decision Tree model
        target_label : int
            Label corresponding to group1 in the comparison

        Returns
        -------
        int
            Index of target label in model.classes_
        """
        classes = np.asarray(model.classes_)
        matches = np.where(classes == target_label)[0]
        if len(matches) == 0:
            raise ValueError(
                f"Target label {target_label} not found in model classes "
                f"{classes.tolist()}"
            )
        return int(matches[0])

    @staticmethod
    def _find_best_feature_split(
        tree,
        feat_idx: int,
        target_class_index: int,
    ) -> Dict[str, float] | None:
        """
        Find the single most informative split for one feature.

        Parameters
        ----------
        tree : sklearn tree object
            Tree structure from DecisionTreeClassifier
        feat_idx : int
            Feature index to inspect
        target_class_index : int
            Target class index in tree.value

        Returns
        -------
        Dict[str, float] or None
            Best split rule or None if feature is not used meaningfully
        """
        best_rule = None
        best_score = -np.inf

        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] != feat_idx:
                continue

            left_idx = tree.children_left[node_idx]
            right_idx = tree.children_right[node_idx]
            if left_idx < 0 or right_idx < 0:
                continue

            gain = RepresentativeFinderHelper._compute_split_gain(tree, node_idx)
            if gain <= 0:
                continue

            left_fraction = RepresentativeFinderHelper._get_target_fraction(
                tree.value[left_idx], target_class_index
            )
            right_fraction = RepresentativeFinderHelper._get_target_fraction(
                tree.value[right_idx], target_class_index
            )
            separation = abs(left_fraction - right_fraction)
            if separation <= 0:
                continue

            direction = "le" if left_fraction > right_fraction else "gt"
            split_score = gain * separation

            if split_score > best_score:
                best_score = split_score
                best_rule = {
                    "threshold": float(tree.threshold[node_idx]),
                    "weight": float(split_score),
                    "direction": direction,
                }

        return best_rule

    @staticmethod
    def _compute_split_gain(tree, node_idx: int) -> float:
        """
        Compute weighted impurity gain for one split node.

        Parameters
        ----------
        tree : sklearn tree object
            Tree structure from DecisionTreeClassifier
        node_idx : int
            Node index of the split

        Returns
        -------
        float
            Weighted impurity reduction
        """
        left_idx = tree.children_left[node_idx]
        right_idx = tree.children_right[node_idx]
        parent_weight = float(tree.weighted_n_node_samples[node_idx])
        if parent_weight <= 0:
            return 0.0

        left_weight = float(tree.weighted_n_node_samples[left_idx])
        right_weight = float(tree.weighted_n_node_samples[right_idx])

        impurity_reduction = (
            float(tree.impurity[node_idx])
            - (left_weight / parent_weight) * float(tree.impurity[left_idx])
            - (right_weight / parent_weight) * float(tree.impurity[right_idx])
        )
        return max(parent_weight * impurity_reduction, 0.0)

    @staticmethod
    def _get_target_fraction(node_value: np.ndarray, target_class_index: int) -> float:
        """
        Compute target-class fraction at a tree node.

        Parameters
        ----------
        node_value : np.ndarray
            Tree value array for one node
        target_class_index : int
            Target class index in node counts

        Returns
        -------
        float
            Fraction of target class at the node
        """
        counts = np.asarray(node_value).reshape(-1)
        total = float(np.sum(counts))
        if total <= 0:
            return 0.0
        return float(counts[target_class_index] / total)

    @staticmethod
    def _compute_feature_scales(
        selected_data: np.ndarray | None,
        feature_indices: List[int],
        chunk_size: int | None = None,
        feature_data_getter=None,
    ) -> Dict[int, float]:
        """
        Compute per-feature scales for margin normalization.

        Parameters
        ----------
        selected_data : np.ndarray or None
            Feature matrix or memmap-backed array
        feature_indices : List[int]
            Indices of features to normalize
        chunk_size : int, optional
            Chunk size for memmap-safe processing
        feature_data_getter : callable, optional
            Lazy getter returning selected_data when needed

        Returns
        -------
        Dict[int, float]
            Mapping from feature index to positive scale
        """
        if selected_data is None:
            if feature_data_getter is None:
                raise ValueError("Either selected_data or feature_data_getter must be provided")
            selected_data = feature_data_getter()

        if chunk_size is None or selected_data.shape[0] <= chunk_size:
            scales = {}
            for feat_idx in feature_indices:
                scale = float(np.std(selected_data[:, feat_idx]))
                scales[feat_idx] = scale if scale > 1e-12 else 1.0
            return scales

        sums = np.zeros(len(feature_indices), dtype=float)
        sumsq = np.zeros(len(feature_indices), dtype=float)
        n_frames = selected_data.shape[0]

        for start_idx in range(0, n_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, n_frames)
            chunk = selected_data[start_idx:end_idx][:, feature_indices]
            sums += np.sum(chunk, axis=0)
            sumsq += np.sum(np.square(chunk), axis=0)

        means = sums / n_frames
        variances = np.maximum((sumsq / n_frames) - np.square(means), 0.0)

        return {
            feat_idx: float(np.sqrt(var)) if var > 1e-12 else 1.0
            for feat_idx, var in zip(feature_indices, variances)
        }

    @staticmethod
    def _score_frames_tree_based(
        feature_data: np.ndarray,
        feature_indices: List[int],
        feature_importances: List[float],
        tree_rules: Dict[int, Dict[str, float]],
        feature_scales: Dict[int, float],
    ) -> np.ndarray:
        """
        Score frames by positive margin beyond the best split threshold.

        A feature contributes only when the frame lies on the side of the
        threshold that is more characteristic for group1. Larger margins
        yield higher scores.

        Parameters
        ----------
        feature_data : np.ndarray
            Feature matrix (n_frames, n_features)
        feature_indices : List[int]
            Indices of features to score
        feature_importances : List[float]
            Importance weights for each feature
        tree_rules : Dict[int, Dict[str, float]]
            Best split rule per feature
        feature_scales : Dict[int, float]
            Scale per feature for margin normalization

        Returns
        -------
        np.ndarray
            Score for each frame (n_frames,)
        """
        n_frames = feature_data.shape[0]
        scores = np.zeros(n_frames)

        for feat_idx, importance in zip(feature_indices, feature_importances):
            if feat_idx not in tree_rules:
                continue

            rule = tree_rules[feat_idx]
            threshold = rule['threshold']
            direction = rule['direction']
            split_weight = rule['weight']
            scale = feature_scales.get(feat_idx, 1.0)

            feature_values = feature_data[:, feat_idx]
            if direction == "gt":
                margins = feature_values - threshold
            else:
                margins = threshold - feature_values

            positive_margins = np.maximum(margins, 0.0)
            feature_scores = importance * split_weight * np.log1p(
                positive_margins / scale
            )
            scores += feature_scores

        return scores

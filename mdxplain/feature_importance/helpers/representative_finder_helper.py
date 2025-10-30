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

        if not use_memmap:
            return RepresentativeFinderHelper._find_best_tree_fast(
                pipeline_data, fi_data, comparison_identifier,
                ds_name, n_top
            )
        else:
            return RepresentativeFinderHelper._find_best_tree_chunked(
                pipeline_data, fi_data, comparison_identifier,
                ds_name, n_top, chunk_size
            )

    @staticmethod
    def _find_best_tree_fast(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        comparison_identifier: str,
        ds_name: str,
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
            model, feature_indices
        )

        is_periodic_mapping = RepresentativeFinderHelper._get_is_periodic_mapping(
            pipeline_data, fi_data.feature_selector
        )

        selected_data, frame_mapping = pipeline_data.get_selected_data(
            fi_data.feature_selector, ds_name, return_frame_mapping=True
        )

        scores = RepresentativeFinderHelper._score_frames_tree_based(
            selected_data, feature_indices, feature_importances,
            tree_rules, is_periodic_mapping
        )

        best_local_idx = np.argmax(scores)

        return frame_mapping[best_local_idx]

    @staticmethod
    def _find_best_tree_chunked(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        comparison_identifier: str,
        ds_name: str,
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
            model, feature_indices
        )

        is_periodic_mapping = RepresentativeFinderHelper._get_is_periodic_mapping(
            pipeline_data, fi_data.feature_selector
        )

        selected_data, frame_mapping = pipeline_data.get_selected_data(
            fi_data.feature_selector, ds_name, return_frame_mapping=True
        )

        best_idx = RepresentativeFinderHelper._find_best_in_chunks(
            selected_data, feature_indices, feature_importances,
            tree_rules, is_periodic_mapping, chunk_size
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
        is_periodic_mapping: Dict[int, bool],
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
        is_periodic_mapping : Dict[int, bool]
            Mapping from feature index to is_periodic flag
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
                tree_rules, is_periodic_mapping
            )

            chunk_max_idx = np.argmax(scores)
            chunk_max_score = scores[chunk_max_idx]

            if chunk_max_score > best_score:
                best_score = chunk_max_score
                best_idx = current_offset + chunk_max_idx

            current_offset += chunk.shape[0]

        return best_idx

    @staticmethod
    def _extract_tree_rules(model, feature_indices: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Extract split rules from Decision Tree for specified features.

        Analyzes the tree structure to find representative thresholds
        for the most important features.

        Parameters
        ----------
        model : sklearn DecisionTreeClassifier
            Trained Decision Tree model
        feature_indices : List[int]
            Indices of features to extract rules for

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping from feature_idx to dict with 'threshold' and 'weight'
        """
        tree = model.tree_
        rules = {}

        for feat_idx in feature_indices:
            thresholds, weights = RepresentativeFinderHelper._collect_feature_splits(
                tree, feat_idx
            )

            if thresholds:
                weighted_threshold = RepresentativeFinderHelper._compute_weighted_threshold(
                    thresholds, weights
                )

                rules[feat_idx] = {
                    'threshold': weighted_threshold,
                    'weight': sum(weights)
                }

        return rules

    @staticmethod
    def _collect_feature_splits(tree, feat_idx: int) -> Tuple[List[float], List[float]]:
        """
        Collect all splits for a specific feature from tree.

        Parameters
        ----------
        tree : sklearn tree object
            Tree structure from DecisionTreeClassifier
        feat_idx : int
            Feature index to collect splits for

        Returns
        -------
        Tuple[List[float], List[float]]
            Lists of (thresholds, weights) for all splits on this feature
        """
        thresholds = []
        weights = []

        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] == feat_idx:
                thresholds.append(tree.threshold[node_idx])
                weights.append(tree.impurity[node_idx])

        return thresholds, weights

    @staticmethod
    def _compute_weighted_threshold(thresholds: List[float], weights: List[float]) -> float:
        """
        Compute weighted average threshold.

        Parameters
        ----------
        thresholds : List[float]
            Threshold values
        weights : List[float]
            Weight values

        Returns
        -------
        float
            Weighted threshold value
        """
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(t * w for t, w in zip(thresholds, weights)) / total_weight
        else:
            return np.mean(thresholds)

    @staticmethod
    def _score_frames_tree_based(
        feature_data: np.ndarray,
        feature_indices: List[int],
        feature_importances: List[float],
        tree_rules: Dict[int, Dict[str, float]],
        is_periodic_mapping: Dict[int, bool] = None
    ) -> np.ndarray:
        """
        Score frames based on alignment with tree split rules.

        Parameters
        ----------
        feature_data : np.ndarray
            Feature matrix (n_frames, n_features)
        feature_indices : List[int]
            Indices of features to score
        feature_importances : List[float]
            Importance weights for each feature
        tree_rules : Dict[int, Dict[str, float]]
            Tree rules extracted from model
        is_periodic_mapping : Dict[int, bool], optional
            Mapping from feature index to is_periodic flag

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

            feature_values = feature_data[:, feat_idx]

            raw_distances = feature_values - threshold
            is_periodic = is_periodic_mapping.get(feat_idx, False) if is_periodic_mapping else False
            distances = RepresentativeFinderHelper._circular_distance(
                raw_distances, is_periodic
            )

            feature_scores = importance / (1.0 + distances)
            scores += feature_scores

        return scores

    @staticmethod
    def _get_is_periodic_mapping(
        pipeline_data: PipelineData,
        feature_selector_name: str
    ) -> Dict[int, bool]:
        """
        Create mapping from feature index to is_periodic flag.

        Extracts is_periodic information from feature metadata for each
        feature type in the selector. Used to determine which features
        require circular distance calculation (e.g., torsion angles).

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_selector_name : str
            Feature selector name

        Returns
        -------
        Dict[int, bool]
            Mapping from global feature index to is_periodic flag

        Examples
        --------
        >>> is_periodic = RepresentativeFinderHelper._get_is_periodic_mapping(
        ...     pipeline_data, "my_features"
        ... )
        >>> is_periodic[42]  # True for torsion, False for distance
        """
        selector_data = pipeline_data.selected_feature_data[feature_selector_name]
        is_periodic_mapping = {}
        current_offset = 0

        for feature_key in selector_data.selection_results.keys():
            feature_data_dict = pipeline_data.feature_data[feature_key]
            ref_traj = selector_data.reference_trajectory
            if ref_traj is None:
                ref_traj = 0

            feature_data_obj = feature_data_dict[ref_traj]
            is_periodic = feature_data_obj.feature_metadata.get('is_periodic', False)

            result = selector_data.selection_results[feature_key]
            traj_results = result['trajectory_indices'][ref_traj]
            n_features = len(traj_results['indices'])

            for i in range(n_features):
                is_periodic_mapping[current_offset + i] = is_periodic

            current_offset += n_features

        return is_periodic_mapping

    @staticmethod
    def _circular_distance(
        distances: np.ndarray,
        is_periodic: bool
    ) -> np.ndarray:
        """
        Compute circular distance for periodic features.

        For periodic features (torsion angles in degrees), computes the
        shortest angular distance accounting for periodicity. For non-periodic
        features, returns absolute distance unchanged.

        Parameters
        ----------
        distances : np.ndarray
            Raw distances (feature_value - threshold)
        is_periodic : bool
            Whether feature is periodic (True) or not (False)

        Returns
        -------
        np.ndarray
            Corrected distances accounting for periodicity

        Examples
        --------
        >>> # Torsion angle: 350° vs 10° → distance is 20°, not 340°
        >>> distances = np.array([340.0])  # 350 - 10
        >>> circular = RepresentativeFinderHelper._circular_distance(
        ...     distances, is_periodic=True
        ... )
        >>> print(circular)  # [20.0]

        >>> # Non-periodic feature: standard absolute distance
        >>> distances = np.array([5.0, -3.0])
        >>> result = RepresentativeFinderHelper._circular_distance(
        ...     distances, is_periodic=False
        ... )
        >>> print(result)  # [5.0, 3.0]
        """
        if is_periodic:
            return np.abs((distances + 180) % 360 - 180)
        else:
            return np.abs(distances)

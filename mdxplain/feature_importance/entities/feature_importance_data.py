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
Feature importance data entity for storing ML analysis results.

This module contains the FeatureImportanceData class that stores feature
importance analysis results from various ML algorithms. It provides
separated data and metadata storage with flexible access methods.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Union, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from ...utils.data_utils import DataUtils
from ..helpers.representative_finder_helper import RepresentativeFinderHelper


class FeatureImportanceData:
    """
    Data entity for storing feature importance analysis results.

    Stores feature importance results from ML algorithms, 
    in this case mainly classifiers with separated
    data and metadata storage. Each FeatureImportanceData contains results
    for all sub-comparisons from a single analysis run.

    Attributes
    ----------
    name : str
        Name identifier for this analysis
    analyzer_type : str
        Type of analyzer used (e.g., "decision_tree")
    comparison_name : str
        Name of the comparison this analysis was run on
    data : List[np.ndarray]
        List of feature importance arrays (one per sub-comparison)
    metadata : List[Dict[str, Any]]
        List of metadata dictionaries (parallel to data list)

    Examples
    --------
    >>> fi_data = FeatureImportanceData("tree_analysis")
    >>> fi_data.analyzer_type = "decision_tree"
    >>> fi_data.comparison_name = "folded_vs_unfolded"

    >>> # Access by index
    >>> importance_0, meta_0 = fi_data.get_comparison(0)

    >>> # Access by name
    >>> importance, meta = fi_data.get_comparison("folded_vs_rest")
    """

    def __init__(self, name: str):
        """
        Initialize feature importance data with given name.

        Parameters
        ----------
        name : str
            Name identifier for this analysis

        Returns
        -------
        None
            Initializes FeatureImportanceData with given name

        Examples
        --------
        >>> fi_data = FeatureImportanceData("my_analysis")
        >>> print(fi_data.name)
        'my_analysis'
        """
        self.name = name
        self.analyzer_type: Optional[str] = None
        self.comparison_name: Optional[str] = None
        self.feature_selector: Optional[str] = None

        # Separated data and metadata lists (parallel indexed)
        self.data: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_comparison_result(
        self, importance_scores: np.ndarray, metadata: Dict[str, Any]
    ) -> None:
        """
        Add results for a sub-comparison.

        Parameters
        ----------
        importance_scores : np.ndarray
            Feature importance scores from ML algorithm
        metadata : Dict[str, Any]
            Metadata for this sub-comparison

        Returns
        -------
        None
            Adds results to data and metadata lists

        Examples
        --------
        >>> importance = np.array([0.3, 0.2, 0.1, 0.4])
        >>> meta = {
        ...     "comparison": "folded_vs_unfolded",
        ...     "n_samples": 1000,
        ...     "accuracy": 0.85
        ... }
        >>> fi_data.add_comparison_result(importance, meta)
        """
        self.data.append(importance_scores.copy())
        self.metadata.append(metadata.copy())

    def get_comparison(
        self, identifier: Union[int, str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get comparison results by index or name.

        Parameters
        ----------
        identifier : int or str
        
            - int: Index of the comparison (0-based)
            - str: Name of the comparison from metadata

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Tuple of (importance_scores, metadata)

        Raises
        ------
        ValueError
            If identifier not found
        TypeError
            If identifier is neither int nor str

        Examples
        --------
        >>> # Access by index
        >>> scores, meta = fi_data.get_comparison(0)

        >>> # Access by name
        >>> scores, meta = fi_data.get_comparison("folded_vs_rest")
        """
        # Case 1: Integer index
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.data):
                return self.data[identifier], self.metadata[identifier]
            
            raise ValueError(
                f"Index {identifier} out of range. "
                f"Available indices: 0-{len(self.data)-1}"
            )

        # Case 2: String name
        if isinstance(identifier, str):
            for i, meta in enumerate(self.metadata):
                if meta.get("comparison") == identifier:
                    return self.data[i], self.metadata[i]

            # Name not found - show available names
            available_names = [
                m.get("comparison", f"unnamed_{i}") for i, m in enumerate(self.metadata)
            ]
            raise ValueError(
                f"Comparison '{identifier}' not found. "
                f"Available comparisons: {available_names}"
            )

        raise TypeError(
            f"Identifier must be int or str, got {type(identifier).__name__}"
        )

    def get_all_comparisons(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get all comparison results.

        Parameters
        ----------
        None

        Returns
        -------
        List[Tuple[np.ndarray, Dict[str, Any]]]
            List of (importance_scores, metadata) tuples

        Examples
        --------
        >>> all_results = fi_data.get_all_comparisons()
        >>> for scores, meta in all_results:
        ...     print(f"{meta['comparison']}: {scores[:3]}")
        """
        return list(zip(self.data, self.metadata))

    def list_comparisons(self) -> List[str]:
        """
        List all available comparison names.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            List of comparison names from metadata

        Examples
        --------
        >>> names = fi_data.list_comparisons()
        >>> print(f"Available comparisons: {names}")
        """
        return [
            meta.get("comparison", f"unnamed_{i}")
            for i, meta in enumerate(self.metadata)
        ]

    def get_top_features(
        self, identifier: Union[int, str], n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top N most important features for a comparison.

        Parameters
        ----------
        identifier : int or str
            Comparison identifier (index or name)
        n : int, default=10
            Number of top features to return

        Returns
        -------
        List[Tuple[int, float]]
            List of (feature_index, importance_score) tuples, sorted by importance

        Examples
        --------
        >>> top_features = fi_data.get_top_features("folded_vs_rest", n=5)
        >>> for feat_idx, score in top_features:
        ...     print(f"Feature {feat_idx}: {score:.3f}")
        """
        scores, _ = self.get_comparison(identifier)
        
        # Get indices sorted by importance (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Return top N as (index, score) tuples
        top_n = min(n, len(scores))
        return [(int(idx), float(scores[idx])) for idx in sorted_indices[:top_n]]

    def get_average_importance(self) -> np.ndarray:
        """
        Get average feature importance across all comparisons.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Average importance scores across all sub-comparisons

        Examples
        --------
        >>> avg_importance = fi_data.get_average_importance()
        >>> top_overall = np.argmax(avg_importance)
        >>> print(f"Most important feature overall: {top_overall}")
        """
        if not self.data:
            return np.array([])

        # Stack all importance arrays and compute mean
        return np.mean(self.data, axis=0)

    def get_analysis_info(self) -> Dict[str, Any]:
        """
        Get summary information about this analysis.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Dictionary with analysis summary information

        Examples
        --------
        >>> info = fi_data.get_analysis_info()
        >>> print(f"Analyzer: {info['analyzer_type']}")
        >>> print(f"Comparisons: {info['n_comparisons']}")
        """
        return {
            "name": self.name,
            "analyzer_type": self.analyzer_type,
            "comparison_name": self.comparison_name,
            "n_comparisons": len(self.data),
            "n_features": len(self.data[0]) if self.data else 0,
            "comparison_names": self.list_comparisons(),
        }

    def __len__(self) -> int:
        """
        Get number of stored comparisons.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of comparisons

        Examples
        --------
        >>> print(f"Number of comparisons: {len(fi_data)}")
        """
        return len(self.data)

    def __contains__(self, comparison_name: str) -> bool:
        """
        Check if a comparison exists.

        Parameters
        ----------
        comparison_name : str
            Name of the comparison to check

        Returns
        -------
        bool
            True if comparison exists, False otherwise

        Examples
        --------
        >>> if "folded_vs_rest" in fi_data:
        ...     print("Comparison exists")
        """
        return comparison_name in self.list_comparisons()

    def __repr__(self) -> str:
        """
        String representation of the FeatureImportanceData.

        Parameters
        ----------
        None

        Returns
        -------
        str
            String representation

        Examples
        --------
        >>> print(repr(fi_data))
        FeatureImportanceData(name='tree_analysis', analyzer='decision_tree', n_comp=4)
        """
        return (
            f"FeatureImportanceData(name='{self.name}', "
            f"analyzer='{self.analyzer_type}', n_comp={len(self.data)})"
        )

    def save(self, save_path: str) -> None:
        """
        Save FeatureImportanceData object to disk.

        Parameters
        ----------
        save_path : str
            Path where to save the FeatureImportanceData object

        Returns
        -------
        None
            Saves the FeatureImportanceData object to the specified path

        Examples
        --------
        >>> feature_importance_data.save('analysis_results/tree_importance.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load FeatureImportanceData object from disk.

        Parameters
        ----------
        load_path : str
            Path to the saved FeatureImportanceData file

        Returns
        -------
        None
            Loads the FeatureImportanceData object from the specified path

        Examples
        --------
        >>> feature_importance_data.load('analysis_results/tree_importance.pkl')
        """
        DataUtils.load_object(self, load_path)

    def print_info(self) -> None:
        """
        Print comprehensive feature importance information.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints feature importance information to console

        Examples
        --------
        >>> feature_importance_data.print_info()
        === FeatureImportanceData ===
        Name: tree_analysis
        Analyzer Type: RandomForest
        Comparison: folded_vs_unfolded
        Sub-Comparisons: 3 (folded_vs_rest, intermediate_vs_rest, unfolded_vs_rest)
        Features Analyzed: 150
        """
        if self._has_no_data():
            print("No feature importance data available.")
            return

        self._print_importance_header()
        self._print_importance_details()
        self._print_top_features_summary()

    def _has_no_data(self) -> bool:
        """
        Check if no feature importance data is available.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if no data is available, False otherwise
        """
        return len(self.data) == 0

    def _print_importance_header(self) -> None:
        """
        Print header with analysis name.

        Returns
        -------
        None
        """
        print("=== FeatureImportanceData ===")
        print(f"Name: {self.name}")

    def _print_importance_details(self) -> None:
        """
        Print detailed importance analysis information.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints detailed importance analysis information
        """
        print(f"Analyzer Type: {self.analyzer_type}")
        print(f"Comparison: {self.comparison_name}")
        
        comparison_names = self.list_comparisons()
        if comparison_names:
            comparison_str = ", ".join(comparison_names)
            print(f"Sub-Comparisons: {len(comparison_names)} ({comparison_str})")
        
        info = self.get_analysis_info()
        if info['n_features'] > 0:
            print(f"Features Analyzed: {info['n_features']}")

    def _print_top_features_summary(self) -> None:
        """
        Print summary of top features across comparisons.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if len(self.data) == 0:
            return

        # Show average importance summary
        avg_importance = self.get_average_importance()
        if len(avg_importance) > 0:
            top_feature_idx = int(np.argmax(avg_importance))
            top_importance = float(avg_importance[top_feature_idx])
            print(f"Top Feature Overall: Feature {top_feature_idx} (avg importance: {top_importance:.4f})")
            
            # Show top 3 features on average
            top_indices = np.argsort(avg_importance)[::-1][:3]
            top_3_str = ", ".join([
                f"Feature {int(idx)} ({float(avg_importance[idx]):.3f})"
                for idx in top_indices
            ])
            print(f"Top 3 Features: {top_3_str}")

    def get_representative_frame(
        self,
        pipeline_data: PipelineData,
        comparison_identifier: str,
        representative_mode: str = "best",
        n_top: int = 10,
        use_memmap: bool = False,
        chunk_size: int = 2000
    ) -> Tuple[int, int]:
        """
        Get representative frame for a sub-comparison.

        Finds the most representative frame for a given sub-comparison
        based on feature importance. Supports two modes: "best" (frame
        maximizing top feature values) and "centroid" (frame closest to
        cluster center).

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing trajectories and features
        comparison_identifier : str
            Sub-comparison identifier
        representative_mode : str, default="best"
            Mode for frame selection:
            - "best": Frame maximizing top important features
            - "centroid": Frame closest to cluster centroid
        n_top : int, default=10
            Number of top features to consider (for "best" mode)
        use_memmap : bool, default=False
            Whether to use memory-mapped processing
        chunk_size : int, default=2000
            Chunk size for memory-mapped processing

        Returns
        -------
        Tuple[int, int]
            Trajectory index and frame index of representative frame

        Examples
        --------
        >>> # Get best representative frame
        >>> traj_idx, frame_idx = fi_data.get_representative_frame(
        ...     pipeline_data, "cluster_0_vs_rest", representative_mode="best"
        ... )

        >>> # Get centroid frame
        >>> traj_idx, frame_idx = fi_data.get_representative_frame(
        ...     pipeline_data, "cluster_0_vs_rest", representative_mode="centroid"
        ... )

        Notes
        -----
        - "best" mode uses Decision Tree split rules for scoring
        - "centroid" mode finds frame minimizing distance to mean
        - Use memmap mode for large trajectories to save memory
        """
        comp_data = pipeline_data.comparison_data[self.comparison_name]
        sub_comp = comp_data.get_sub_comparison(comparison_identifier)
        ds_name = sub_comp["group1_selectors"][0]

        if representative_mode == "best":
            return RepresentativeFinderHelper.find_best_tree_based(
                pipeline_data, self, comparison_identifier,
                n_top, use_memmap, chunk_size
            )
        else:
            return pipeline_data.get_centroid_frame(
                self.feature_selector, ds_name
            )

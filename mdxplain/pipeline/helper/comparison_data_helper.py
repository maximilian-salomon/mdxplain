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
Helper class for comparison data processing.

This module provides the ComparisonDataHelper class with static methods
for processing comparison data without creating circular dependencies.
Used by PipelineData to provide comparison data access functionality.
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData

from ...comparison.entities.comparison_data import ComparisonData
from ..helper.selection_memmap_helper import SelectionMemmapHelper


class ComparisonDataHelper:
    """
    Static helper class for comparison data processing.
    
    Provides static methods to process comparison data by combining
    ComparisonData metadata with PipelineData's get_selected_data() method.
    Avoids circular dependencies by being part of the pipeline module.
    
    Examples
    --------
    >>> # Used internally by PipelineData
    >>> X, y = ComparisonDataHelper.get_sub_comparison_data(
    ...     pipeline_data, comparison_data, "folded_vs_rest"
    ... )
    """

    @staticmethod
    def get_sub_comparison_data(
        pipeline_data: PipelineData, comparison_data: ComparisonData, sub_comparison_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get X (features) and y (labels) for a specific sub-comparison.
        
        This method combines ComparisonData metadata with PipelineData's
        data processing capabilities to create ML-ready datasets.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing feature and data selections
        comparison_data : ComparisonData
            Comparison metadata container
        sub_comparison_name : str
            Name of the sub-comparison to process
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (X, y) where X is feature matrix and y is label array
            
        Raises
        ------
        ValueError
            If sub-comparison not found or required data missing
            
        Examples
        --------
        >>> X, y = ComparisonDataHelper.get_sub_comparison_data(
        ...     pipeline_data, comp_data, "folded_vs_rest"
        ... )
        >>> print(f"Features shape: {X.shape}")
        >>> print(f"Labels shape: {y.shape}")
        """
        # Get sub-comparison metadata
        sub_comp = comparison_data.get_sub_comparison(sub_comparison_name)
        if sub_comp is None:
            available = comparison_data.list_sub_comparisons()
            raise ValueError(
                f"Sub-comparison '{sub_comparison_name}' not found. "
                f"Available: {available}"
            )

        # Get feature matrix AND frame_mapping (single call for performance)
        full_matrix, frame_mapping = pipeline_data.get_selected_data(
            comparison_data.feature_selector, return_frame_mapping=True
        )
        
        # Handle multiclass mode differently
        if sub_comp.get("mode") == "multiclass":
            return ComparisonDataHelper._get_multiclass_data(
                pipeline_data, comparison_data, sub_comp, full_matrix, frame_mapping
            )
        
        # Standard binary comparison
        return ComparisonDataHelper._get_binary_comparison_data(
            pipeline_data, comparison_data, sub_comp, full_matrix, frame_mapping
        )

    @staticmethod
    def _get_binary_comparison_data(
        pipeline_data: PipelineData, comparison_data: ComparisonData, sub_comp: Dict[str, Any], 
        full_matrix: np.ndarray, frame_mapping: Dict[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for binary comparison mode with trajectory-specific support.
        
        Uses pre-computed feature matrix and frame_mapping for trajectory-to-global
        index conversion. Supports memmap-safe processing for large datasets.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        comparison_data : ComparisonData
            Comparison metadata container
        sub_comp : Dict[str, Any]
            Sub-comparison configuration
        full_matrix : np.ndarray
            Pre-computed feature matrix from get_selected_data()
        frame_mapping : Dict[int, Tuple[int, int]]
            Mapping from global frame index to (traj_idx, local_frame_idx)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (X, y) where X is feature matrix and y is label array
        """
        # Collect frame indices and labels for both groups
        all_frame_indices = []
        all_labels = []

        # Group 1 (pass frame_mapping for trajectory conversion)
        group1_indices = ComparisonDataHelper._collect_frame_indices(
            pipeline_data, sub_comp["group1_selectors"], frame_mapping
        )
        all_frame_indices.extend(group1_indices)
        all_labels.extend([sub_comp["labels"][0]] * len(group1_indices))

        # Group 2 (pass frame_mapping for trajectory conversion)
        group2_indices = ComparisonDataHelper._collect_frame_indices(
            pipeline_data, sub_comp["group2_selectors"], frame_mapping
        )
        all_frame_indices.extend(group2_indices)
        all_labels.extend([sub_comp["labels"][1]] * len(group2_indices))

        # Extract comparison matrix using pre-computed matrix
        return ComparisonDataHelper._extract_comparison_matrix(
            pipeline_data, comparison_data, all_frame_indices, all_labels, 
            sub_comp["name"], full_matrix
        )

    @staticmethod
    def _get_multiclass_data(
        pipeline_data: PipelineData, comparison_data: ComparisonData, sub_comp: Dict[str, Any],
        full_matrix: np.ndarray, frame_mapping: Dict[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for multiclass comparison mode with trajectory-specific support.
        
        Uses pre-computed feature matrix and frame_mapping for trajectory-to-global
        index conversion. Supports memmap-safe processing for large datasets.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        comparison_data : ComparisonData
            Comparison metadata container
        sub_comp : Dict[str, Any]
            Multiclass sub-comparison configuration
        full_matrix : np.ndarray
            Pre-computed feature matrix from get_selected_data()
        frame_mapping : Dict[int, Tuple[int, int]]
            Mapping from global frame index to (traj_idx, local_frame_idx)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (X, y) where X is feature matrix and y contains class labels
        """
        all_frame_indices = []
        all_labels = []
        
        # Iterate through all selectors with their corresponding labels
        for selector_name, label in zip(sub_comp["selectors"], sub_comp["labels"]):
            selector_indices = ComparisonDataHelper._collect_frame_indices(
                pipeline_data, [selector_name], frame_mapping
            )
            all_frame_indices.extend(selector_indices)
            all_labels.extend([label] * len(selector_indices))
                
        # Extract comparison matrix using pre-computed matrix
        return ComparisonDataHelper._extract_comparison_matrix(
            pipeline_data, comparison_data, all_frame_indices, all_labels, 
            "multiclass", full_matrix
        )

    @staticmethod
    def _collect_frame_indices(
        pipeline_data: PipelineData, selector_names: List[str], frame_mapping: Dict[int, Tuple[int, int]]
    ) -> List[int]:
        """
        Collect global frame indices from trajectory-specific data selectors.
        
        Converts trajectory_frames to global indices using frame_mapping.
        Combines frame indices from multiple data selectors into a single
        sorted list with duplicates removed.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing data selector data
        selector_names : List[str]
            List of data selector names to combine indices from
        frame_mapping : Dict[int, Tuple[int, int]]
            Mapping from global frame index to (traj_idx, local_frame_idx)

        Returns
        -------
        List[int]
            Combined list of unique global frame indices sorted in ascending order
        """
        all_indices = set()
        
        # Create inverse mapping: (traj_idx, local_frame_idx) -> global_idx
        inverse_mapping = {v: k for k, v in frame_mapping.items()}

        for selector_name in selector_names:
            if selector_name not in pipeline_data.data_selector_data:
                available = list(pipeline_data.data_selector_data.keys())
                raise ValueError(
                    f"Data selector '{selector_name}' not found. "
                    f"Available selectors: {available}"
                )

            selector_data = pipeline_data.data_selector_data[selector_name]
            
            # Convert trajectory_frames to global indices
            for traj_idx, frame_indices in selector_data.trajectory_frames.items():
                for local_frame_idx in frame_indices:
                    key = (traj_idx, local_frame_idx)
                    if key in inverse_mapping:
                        global_idx = inverse_mapping[key]
                        all_indices.add(global_idx)

        return sorted(list(all_indices))

    @staticmethod
    def _extract_comparison_matrix(
        pipeline_data: PipelineData, comparison_data: ComparisonData, all_frame_indices: List[int], 
        all_labels: List[int], sub_comparison_name: str, full_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract comparison matrix with memmap-safe processing for overlapping frames.
        
        This method handles the common logic for both binary and multiclass comparisons,
        supporting overlapping frames that cannot be handled by DataSelector (which works as a set).
        Uses pre-computed feature matrix to avoid duplicate get_selected_data() calls.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        comparison_data : ComparisonData
            Comparison metadata container
        all_frame_indices : List[int]
            Frame indices including potential duplicates
        all_labels : List[int]
            Labels corresponding to frame indices
        sub_comparison_name : str
            Name of the sub-comparison for error messages
        full_matrix : np.ndarray
            Pre-computed feature matrix from get_selected_data()
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (X, y) where X is feature matrix and y is label array
            
        Raises
        ------
        ValueError
            If no frames found for comparison
        """
        y = np.array(all_labels)
        
        if not all_frame_indices:
            raise ValueError(f"No frames found for sub-comparison '{sub_comparison_name}'")

        # Use memmap-safe approach for large datasets
        if pipeline_data.use_memmap and len(all_frame_indices) > pipeline_data.chunk_size:
            X = ComparisonDataHelper._get_comparison_data_memmap(
                pipeline_data, comparison_data, all_frame_indices, full_matrix
            )
            return X, y
        
        # For small datasets, use direct indexing with pre-computed matrix
        return full_matrix[all_frame_indices], y

    @staticmethod
    def _get_comparison_data_memmap(
        pipeline_data: PipelineData, comparison_data: ComparisonData, frame_indices: List[int], full_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Get comparison data in a memmap-safe way for large datasets.
        
        Uses SelectionMemmapHelper to efficiently process large frame selections
        without loading entire datasets into RAM. Uses pre-computed feature matrix
        to avoid duplicate get_selected_data() calls.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        comparison_data : ComparisonData
            Comparison metadata container
        frame_indices : List[int]
            Frame indices to select
        full_matrix : np.ndarray
            Pre-computed feature matrix from get_selected_data()
            
        Returns
        -------
        np.ndarray
            Feature matrix for the specified frames
        """
        # Use memmap-safe frame selection with pre-computed matrix
        return SelectionMemmapHelper.create_memmap_frame_selection(
            full_matrix, frame_indices, 
            f"comparison_{comparison_data.name}_{len(frame_indices)}frames",
            pipeline_data.cache_dir, pipeline_data.chunk_size
        )

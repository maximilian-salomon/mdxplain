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
Helper for applying post-selection reduction.

This module provides the PostSelectionReductionHelper class that applies
statistical reduction to feature selections. The reduction is applied ONLY
to the specific selection where it's defined, not to all selections of that
feature type.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np
import warnings
import os
import tempfile

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ...feature.feature_type.interfaces.calculator_base import CalculatorBase


class PostSelectionReductionHelper:
    """
    Helper for applying post-selection reduction.

    WICHTIG: Reduction wird NUR auf die spezifische Selection angewendet!

    This helper applies statistical reduction to features after initial selection.
    It uses the appropriate calculator for each feature type to compute reduction
    metrics and filters features based on threshold criteria.

    The reduction process:
    1. Get feature data for each trajectory
    2. Calculate which columns to remove using calculator
    3. Apply cross-trajectory common denominator
    4. Update trajectory_results with reduced indices
    """

    @staticmethod
    def apply_reduction(
        pipeline_data: PipelineData,
        feature_key: str,
        selection_dict: Dict[str, Any],
        trajectory_results: Dict[int, Dict],
        selected_traj_indices: List[int],
        use_memmap: bool = False,
        chunk_size: int = 2000,
        cache_dir: str = "./cache"
    ) -> Dict[int, Dict]:
        """
        Apply reduction to a specific selection.

        Process:
        1. Get feature data for each trajectory
        2. Calculate which columns to remove using calculator
        3. Apply cross-trajectory common denominator
        4. Update trajectory_results

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key (e.g., "distances", "contacts")
        selection_dict : dict
            Selection configuration with reduction config
        trajectory_results : dict
            Current selection results per trajectory
        selected_traj_indices : list
            Trajectory indices for this selection
        use_memmap : bool, default=False
            Whether to use memory-mapped files for large data processing
        chunk_size : int, default=2000
            Size of data chunks for memory-efficient processing
        cache_dir : str, default="./cache"
            Directory for temporary cache files

        Returns:
        --------
        dict
            Updated trajectory_results with reduced indices

        Examples:
        ---------
        >>> results = PostSelectionReductionHelper.apply_reduction(
        ...     pipeline_data, "distances", selection_dict, results, [0, 1])
        """
        reduction_config = selection_dict.get("reduction")
        if not reduction_config:
            return trajectory_results

        cross_trajectory = reduction_config.get("cross_trajectory", True)

        # Process each trajectory and collect temp paths
        reduced_indices_per_traj = {}
        temp_paths = []

        for traj_idx in selected_traj_indices:
            if traj_idx not in trajectory_results:
                continue

            kept_indices, temp_path = PostSelectionReductionHelper._process_trajectory(
                pipeline_data, feature_key, traj_idx, trajectory_results[traj_idx],
                selection_dict, reduction_config, use_memmap, chunk_size, cache_dir
            )
            reduced_indices_per_traj[traj_idx] = kept_indices
            if temp_path:
                temp_paths.append(temp_path)

        # Apply cross-trajectory logic
        PostSelectionReductionHelper._update_trajectory_results(
            trajectory_results, reduced_indices_per_traj,
            selected_traj_indices, selection_dict, cross_trajectory
        )

        # Clean up temporary files
        PostSelectionReductionHelper._cleanup_temp_files(temp_paths)

        return trajectory_results

    @staticmethod
    def _process_trajectory(
        pipeline_data: PipelineData,
        feature_key: str,
        traj_idx: int,
        traj_results: Dict,
        selection_dict: Dict,
        reduction_config: Dict,
        use_memmap: bool,
        chunk_size: int,
        cache_dir: str
    ) -> Tuple[List[int], Optional[str]]:
        """
        Process reduction for single trajectory.

        Extracts selected columns, applies calculator reduction, and maps
        back to original indices for one trajectory.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key
        traj_idx : int
            Trajectory index
        traj_results : dict
            Results for this trajectory
        selection_dict : dict
            Selection configuration
        reduction_config : dict
            Reduction configuration
        use_memmap : bool
            Use memmap processing
        chunk_size : int
            Chunk size for processing
        cache_dir : str
            Cache directory

        Returns:
        --------
        tuple
            (kept_original_indices, temp_path or None)
        """
        feature_data = pipeline_data.feature_data[feature_key][traj_idx]
        selection_indices = traj_results["indices"]
        use_reduced = selection_dict["use_reduced"]

        # Choose data matrix
        data_matrix = feature_data.reduced_data if use_reduced else feature_data.data

        # Extract selected columns
        if use_memmap:
            selected_data, temp_path = PostSelectionReductionHelper._extract_memmap_columns(
                data_matrix, selection_indices, chunk_size, cache_dir
            )
        else:
            selected_data = data_matrix[:, selection_indices]
            temp_path = None

        # Apply calculator
        calculator = feature_data.feature_type.calculator
        result = PostSelectionReductionHelper._apply_calculator(
            calculator, selected_data, reduction_config,
            feature_data.feature_metadata, feature_data.cache_path
        )

        # Map indices back to original
        kept_local_indices = result["indices"]
        if isinstance(kept_local_indices, np.ndarray):
            kept_local_indices = kept_local_indices.tolist()

        selection_array = np.array(selection_indices)
        kept_original = selection_array[kept_local_indices].tolist()

        return kept_original, temp_path

    @staticmethod
    def _update_trajectory_results(
        trajectory_results: Dict,
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
        selection_dict: Dict,
        cross_trajectory: bool
    ) -> None:
        """
        Update trajectory results based on cross_trajectory setting.

        Handles common denominator (cross_trajectory=True) or individual
        trajectory reduction (cross_trajectory=False).

        Parameters:
        -----------
        trajectory_results : dict
            Results to update in-place
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Selected trajectory indices
        selection_dict : dict
            Selection configuration
        cross_trajectory : bool
            Apply common denominator

        Returns:
        --------
        None
            Updates trajectory_results in-place
        """
        if len(reduced_indices_per_traj) <= 1:
            # Single trajectory - simple update
            for traj_idx, indices in reduced_indices_per_traj.items():
                trajectory_results[traj_idx]["indices"] = indices
                trajectory_results[traj_idx]["use_reduced"] = [
                    selection_dict["use_reduced"]
                ] * len(indices)
            return

        if cross_trajectory:
            PostSelectionReductionHelper._apply_common_denominator(
                trajectory_results, reduced_indices_per_traj,
                selected_traj_indices, selection_dict
            )
        else:
            warnings.warn(
                "cross_trajectory=False: Features may differ between trajectories!"
            )
            for traj_idx, indices in reduced_indices_per_traj.items():
                trajectory_results[traj_idx]["indices"] = indices
                trajectory_results[traj_idx]["use_reduced"] = [
                    selection_dict["use_reduced"]
                ] * len(indices)

    @staticmethod
    def _apply_common_denominator(
        trajectory_results: Dict,
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
        selection_dict: Dict
    ) -> None:
        """
        Apply common denominator to all trajectories.

        Finds intersection of reduced indices across trajectories and
        updates all trajectories with common features only.

        Parameters:
        -----------
        trajectory_results : dict
            Results to update
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Selected trajectory indices
        selection_dict : dict
            Selection configuration

        Returns:
        --------
        None
            Updates trajectory_results in-place
        """
        # Find intersection
        common = set(reduced_indices_per_traj[selected_traj_indices[0]])
        for traj_idx in selected_traj_indices[1:]:
            if traj_idx in reduced_indices_per_traj:
                common &= set(reduced_indices_per_traj[traj_idx])

        # Update all with common indices
        for traj_idx in trajectory_results:
            old = trajectory_results[traj_idx]["indices"]
            new = [idx for idx in old if idx in common]
            trajectory_results[traj_idx]["indices"] = new
            trajectory_results[traj_idx]["use_reduced"] = [
                selection_dict["use_reduced"]
            ] * len(new)

    @staticmethod
    def _extract_memmap_columns(
        data_matrix: np.ndarray,
        column_indices: List[int],
        chunk_size: int,
        cache_dir: str
    ) -> Tuple[np.ndarray, str]:
        """
        Extract selected columns using memmap for memory efficiency.

        Creates temporary memmap file and copies selected columns
        chunk-wise to minimize memory usage.

        Parameters:
        -----------
        data_matrix : np.ndarray
            Full data matrix (can be memmap)
        column_indices : list
            Indices of columns to extract
        chunk_size : int
            Frame chunk size for processing
        cache_dir : str
            Directory for temporary cache files

        Returns:
        --------
        tuple
            (selected_data, temp_path)

        Examples:
        ---------
        >>> selected, path = PostSelectionReductionHelper._extract_memmap_columns(
        ...     data_matrix, [0, 5, 10], chunk_size=500, cache_dir="./cache")
        >>> print(selected.shape)
        (n_frames, 3)
        """
        n_frames = data_matrix.shape[0]
        n_selected = len(column_indices)

        # Create temporary memmap file
        os.makedirs(cache_dir, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.dat', dir=cache_dir)
        os.close(temp_fd)

        # Create memmap for output
        selected_data = np.memmap(
            temp_path, dtype=data_matrix.dtype, mode='w+',
            shape=(n_frames, n_selected)
        )

        # Copy chunk-wise
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk = data_matrix[start:end, :][:, column_indices]
            selected_data[start:end, :] = chunk

        selected_data.flush()
        return selected_data, temp_path

    @staticmethod
    def _apply_calculator(
        calculator: 'CalculatorBase',
        selected_data: np.ndarray,
        reduction_config: Dict,
        metadata: Dict,
        output_path: str
    ) -> Dict:
        """
        Apply calculator with reduction parameters.

        Calls calculator's compute_dynamic_values with all necessary
        parameters for reduction calculation.

        Parameters:
        -----------
        calculator : CalculatorBase
            Feature type specific calculator instance
        selected_data : np.ndarray
            Data matrix with only selected features
        reduction_config : dict
            Full reduction configuration
        metadata : dict
            Feature metadata
        output_path : str
            Output path for calculator operations

        Returns:
        --------
        dict
            Calculator result with indices of features that meet criteria

        Examples:
        ---------
        >>> result = PostSelectionReductionHelper._apply_calculator(
        ...     calculator, data, config, metadata, path)
        >>> kept_indices = result["indices"]
        """
        # Base parameters
        params = {
            "metric": reduction_config["metric"],
            "threshold_min": reduction_config.get("threshold_min"),
            "threshold_max": reduction_config.get("threshold_max"),
            "feature_metadata": metadata,
            "output_path": output_path
        }

        # Optional transition parameters
        for key in ["transition_threshold", "window_size", "transition_mode", "lag_time"]:
            if key in reduction_config:
                params[key] = reduction_config[key]

        return calculator.compute_dynamic_values(selected_data, **params)

    @staticmethod
    def _cleanup_temp_files(temp_paths: List[str]) -> None:
        """
        Clean up temporary files.

        Parameters:
        -----------
        temp_paths : list
            List of temporary file paths to remove

        Returns:
        --------
        None
        """
        for path in temp_paths:
            if path and os.path.exists(path):
                os.unlink(path)
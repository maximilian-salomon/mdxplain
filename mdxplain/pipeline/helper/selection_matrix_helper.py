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
Matrix operations helper for feature selection system.

Provides efficient matrix construction directly with proper shape calculation, 
memory management, and frame mapping instead of collecting and merging matrices.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData


class SelectionMatrixHelper:
    """
    Helper class for efficient matrix construction from selection data.
    
    Builds matrices directly with correct shape. Supports both regular arrays
    and memory-mapped files for large datasets.
    """

    @staticmethod
    def build_selection_matrix(
        pipeline_data: PipelineData, feature_selector_name: str, data_selector_name: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
        """
        Build selection matrix directly with efficient memory usage.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing all data
        feature_selector_name : str
            Name of the feature selection
        data_selector_name : str, optional
            Name of data selector for frame filtering
            
        Returns:
        --------
        Tuple[np.ndarray, Dict[int, Tuple[int, int]]]
            Complete matrix and frame mapping
        """
        # Calculate matrix shape
        n_rows, n_cols = SelectionMatrixHelper._calculate_matrix_shape(
            pipeline_data, feature_selector_name, data_selector_name
        )
        
        # Create matrix with correct shape
        matrix = SelectionMatrixHelper._create_matrix(
            (n_rows, n_cols), pipeline_data.use_memmap, 
            pipeline_data.cache_dir, feature_selector_name
        )
        
        # Fill matrix and create frame mapping
        frame_mapping = SelectionMatrixHelper._fill_matrix(
            matrix, pipeline_data, feature_selector_name, data_selector_name
        )
        
        return matrix, frame_mapping
    
    @staticmethod
    def _calculate_matrix_shape(
        pipeline_data: PipelineData, feature_selector_name: str, data_selector_name: Optional[str]
    ) -> Tuple[int, int]:
        """
        Calculate final matrix shape efficiently.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_selector_name : str
            Selection name
        data_selector_name : str, optional
            Data selector name
            
        Returns:
        --------
        Tuple[int, int]
            (n_rows, n_columns) for final matrix
        """
        # Get number of columns from FeatureSelectorData
        selector_data = pipeline_data.selected_feature_data[feature_selector_name]
        n_cols = selector_data.get_n_columns()
        
        if n_cols is None:
            raise ValueError(f"Selection '{feature_selector_name}' not processed yet. Run select() first.")
        
        # Calculate number of rows
        if data_selector_name is None:
            # Extract relevant trajectories from selector data
            all_results = selector_data.get_all_results()
            relevant_trajectories = set()
            for feature_type, selection_info in all_results.items():
                if "trajectory_indices" in selection_info:
                    relevant_trajectories.update(selection_info["trajectory_indices"].keys())
            
            # Sum frames only from selected trajectories
            n_rows = sum(
                pipeline_data.trajectory_data.trajectories[idx].n_frames 
                for idx in relevant_trajectories
                if idx < len(pipeline_data.trajectory_data.trajectories)
            )
        else:
            # Filtered frames - get from data selector
            data_selector = pipeline_data.data_selector_data[data_selector_name]
            n_rows = data_selector.n_selected_frames
        
        return n_rows, n_cols
    
    @staticmethod
    def _create_matrix(
        shape: Tuple[int, int], use_memmap: bool, cache_dir: str, name: str
    ) -> np.ndarray:
        """
        Create matrix with optimal memory management.
        
        Parameters:
        -----------
        shape : Tuple[int, int]
            Matrix shape (rows, columns)
        use_memmap : bool
            Whether to use memory mapping
        cache_dir : str
            Cache directory for memmap files
        name : str
            Matrix name for memmap filename
            
        Returns:
        --------
        np.ndarray
            Empty matrix ready for filling
        """
        if use_memmap:
            # Use DataUtils for consistent path handling
            from ...utils.data_utils import DataUtils
            memmap_path = DataUtils.get_cache_file_path(
                cache_path=cache_dir,
                cache_name=f"selection_matrix_{name}.dat"
            )
            return np.memmap(
                memmap_path, dtype=np.float64, mode='w+', shape=shape
            )
        return np.zeros(shape, dtype=np.float64)
    
    @staticmethod
    def _fill_matrix(
        matrix: np.ndarray, pipeline_data: PipelineData, name: str, 
        data_selector_name: Optional[str]
    ) -> Dict[int, Tuple[int, int]]:
        """
        Fill matrix with data and create frame mapping.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Pre-allocated matrix to fill
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Selection name
        data_selector_name : str, optional
            Data selector name
            
        Returns:
        --------
        Dict[int, Tuple[int, int]]
            Frame mapping {global_idx: (traj_idx, local_idx)}
        """
        frame_mapping = {}
        current_row = 0
        
        selector_data = pipeline_data.selected_feature_data[name]
        all_results = selector_data.get_all_results()
        
        # Get frame selection (all frames or filtered)
        if data_selector_name is None:
            frame_selection = None  # All frames
        else:
            frame_selection = pipeline_data.data_selector_data[data_selector_name]
        
        # Extract relevant trajectories from feature selections
        relevant_trajectories = set()
        for feature_type, selection_info in all_results.items():
            if "trajectory_indices" in selection_info:
                relevant_trajectories.update(selection_info["trajectory_indices"].keys())
        
        # Fill matrix trajectory by trajectory (only selected trajectories)
        for traj_idx in sorted(relevant_trajectories):
            current_row = SelectionMatrixHelper._fill_trajectory_data(
                matrix, pipeline_data, all_results, traj_idx, 
                current_row, frame_selection, frame_mapping
            )
        
        return frame_mapping
    
    @staticmethod
    def _fill_trajectory_data(
        matrix: np.ndarray, pipeline_data: PipelineData, all_results: Dict[str, Any], traj_idx: int,
        start_row: int, frame_selection: Optional[Any], frame_mapping: Dict[int, Tuple[int, int]]
    ) -> int:
        """
        Fill matrix with data from one trajectory.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Matrix to fill
        pipeline_data : PipelineData
            Pipeline data object
        all_results : dict
            All feature selection results
        traj_idx : int
            Trajectory index to process
        start_row : int
            Starting row index for this trajectory
        frame_selection : DataSelectorData or None
            Frame selection (None = all frames)
        frame_mapping : dict
            Frame mapping to update
            
        Returns:
        --------
        int
            Next available row index
        """
        # Get frame indices for this trajectory
        if frame_selection is None:
            # All frames
            traj_data = pipeline_data.trajectory_data.trajectories[traj_idx]
            frame_indices = list(range(len(traj_data)))
        else:
            # Filtered frames
            traj_frames = frame_selection.get_trajectory_frames()
            frame_indices = traj_frames.get(traj_idx, [])
        
        if not frame_indices:
            return start_row  # No frames for this trajectory
        
        # Fill matrix with data from each feature
        current_col = 0
        for feature_type, selection_info in all_results.items():
            current_col = SelectionMatrixHelper._fill_feature_data(
                matrix, pipeline_data, feature_type, selection_info, 
                traj_idx, frame_indices, start_row, current_col
            )
        
        # Update frame mapping
        for i, frame_idx in enumerate(frame_indices):
            frame_mapping[start_row + i] = (traj_idx, frame_idx)
        
        return start_row + len(frame_indices)
    
    @staticmethod
    def _fill_feature_data(
        matrix: np.ndarray, pipeline_data: PipelineData, feature_type: str, 
        selection_info: Dict[str, Any], traj_idx: int, frame_indices: List[int],
        start_row: int, start_col: int
    ) -> int:
        """
        Fill matrix with data from one feature type.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Matrix to fill
        pipeline_data : PipelineData
            Pipeline data object
        feature_type : str
            Feature type name
        selection_info : dict
            Selection info for this feature
        traj_idx : int
            Trajectory index
        frame_indices : list
            Frame indices to extract
        start_row : int
            Starting row in matrix
        start_col : int
            Starting column in matrix
            
        Returns:
        --------
        int
            Next available column index
        """
        # Get trajectory-specific feature data
        feature_data_dict = pipeline_data.feature_data[feature_type]
        if traj_idx not in feature_data_dict:
            return start_col  # No feature data for this trajectory
        
        feature_data = feature_data_dict[traj_idx]
        
        # Get trajectory-specific selection indices
        trajectory_indices_data = selection_info.get("trajectory_indices", {})
        if traj_idx not in trajectory_indices_data:
            return start_col  # No selection for this trajectory
        
        traj_selection = trajectory_indices_data[traj_idx]
        indices = traj_selection.get("indices", [])
        use_reduced_flags = traj_selection.get("use_reduced", [])
        
        if not indices:
            return start_col
        
        # Fill matrix column by column
        for i, (col_idx, use_reduced) in enumerate(zip(indices, use_reduced_flags)):
            # Get source data
            if use_reduced and feature_data.reduced_data is not None:
                source_data = feature_data.reduced_data
            else:
                source_data = feature_data.data
            
            if source_data is not None:
                # Copy data for selected frames
                matrix[start_row:start_row+len(frame_indices), start_col+i] = (
                    source_data[frame_indices, col_idx]
                )
        
        return start_col + len(indices)

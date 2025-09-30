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

"""Label operation utilities for trajectory data consistency."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...entities.trajectory_data import TrajectoryData


class LabelOperationHelper:
    """Utility class for maintaining label consistency during trajectory operations."""

    @staticmethod
    def apply_atom_selection_to_labels(
        traj_data: TrajectoryData,
        traj_indices: List[int],
        original_residue_indices: Dict[int, np.ndarray]
    ) -> None:
        """
        Apply atom selection operations to trajectory labels.
        
        When atoms are selected, the residue structure changes. Labels need to
        be filtered to only include residues that are still present after
        atom selection.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object containing labels
        traj_indices : List[int]
            Indices of trajectories that were modified
        original_residue_indices : Dict[int, np.ndarray]
            Mapping of trajectory index to original residue indices that were kept
            
        Returns
        -------
        None
            Modifies res_label_data in-place to match new residue structure
        """
        for traj_idx in traj_indices:
            if traj_idx not in traj_data.res_label_data:
                continue
                
            if traj_idx not in original_residue_indices:
                continue
            
            # Get original residue indices that were kept (in original numbering)
            kept_residue_ids = original_residue_indices[traj_idx]
            
            # Filter and renumber labels
            old_labels = traj_data.res_label_data[traj_idx]

            # Filter: keep only residues that are in kept_residue_ids
            new_labels = []
            for original_residue_id in kept_residue_ids:
                if original_residue_id < len(old_labels):
                    # Copy the residue dict and update the index to new numbering
                    residue_dict = old_labels[original_residue_id].copy()
                    residue_dict['index'] = len(new_labels)  # New sequential index
                    new_labels.append(residue_dict)
                    
            # Update labels in trajectory data
            traj_data.res_label_data[traj_idx] = new_labels

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
import mdtraj as md
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

    @staticmethod
    def combine_stack_labels(
        traj_data: TrajectoryData,
        target_idx: int,
        source_idx: int,
        target_n_residues: int,
    ) -> None:
        """
        Combine and renumber labels from stacked trajectories.

        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data containing labels
        target_idx : int
            Index of target trajectory
        source_idx : int
            Index of source trajectory
        target_n_residues : int
            Number of residues in target trajectory before stacking

        Returns
        -------
        None
            Updates labels in traj_data in-place
        """
        if (target_idx not in traj_data.res_label_data or
            source_idx not in traj_data.res_label_data):
            return

        target_labels = traj_data.res_label_data[target_idx]
        source_labels = traj_data.res_label_data[source_idx]

        # Renumber source labels to append after target
        renumbered_source_labels = []
        for residue_dict in source_labels:
            new_dict = residue_dict.copy()
            new_dict['index'] = residue_dict['index'] + target_n_residues
            renumbered_source_labels.append(new_dict)

        # Combine labels
        combined_labels = target_labels + renumbered_source_labels
        traj_data.res_label_data[target_idx] = combined_labels

    @staticmethod
    def map_residues_to_original_indices(
        original_residue_info: List[tuple],
        current_topology: md.Topology
    ) -> np.ndarray:
        """
        Map residues by sequential structure matching.

        After operations like remove_solvent(), MDTraj may renumber residues,
        making resSeq-based matching unreliable. We match residues sequentially
        by name and atom composition, preserving order.

        This method uses lightweight residue info (just names and atoms) instead
        of full trajectory data to avoid memory overhead.

        Parameters
        ----------
        original_residue_info : List[tuple]
            Original residue structure as list of (name, atoms) tuples where
            atoms is a tuple of (atom_name, element_symbol) tuples
        current_topology : md.Topology
            Topology after modification (e.g., after solvent removal)

        Returns
        -------
        np.ndarray
            Original residue indices of residues that were kept

        Examples
        --------
        >>> # Prepare lightweight residue info
        >>> original_info = [
        ...     ('ALA', (('CA', 'C'), ('C', 'C'))),
        ...     ('HOH', (('O', 'O'),)),
        ...     ('GLY', (('CA', 'C'), ('C', 'C'))),
        ...     ('HOH', (('O', 'O'),)),
        ...     ('ALA', (('CA', 'C'), ('C', 'C')))
        ... ]
        >>> # After removal: 3 residues (ALA, GLY, ALA)
        >>> # Returns: [0, 2, 4] - original indices of kept residues
        >>> kept_indices = LabelOperationHelper.map_residues_to_original_indices(
        ...     original_info, modified_topology
        ... )
        array([0, 2, 4])
        """
        kept_original_indices = []

        original_idx = 0
        for current_res in current_topology.residues:
            current_atoms = tuple((a.name, a.element.symbol) for a in current_res.atoms)

            while original_idx < len(original_residue_info):
                orig_name, orig_atoms = original_residue_info[original_idx]

                if orig_name == current_res.name and orig_atoms == current_atoms:
                    kept_original_indices.append(original_idx)
                    original_idx += 1
                    break

                original_idx += 1

        return np.array(kept_original_indices)

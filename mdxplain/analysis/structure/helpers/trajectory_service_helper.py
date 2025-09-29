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

"""Helper for common trajectory service operations."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import mdtraj as md
import numpy as np

from mdxplain.pipeline.entities.pipeline_data import PipelineData


class TrajectoryServiceHelper:
    """Helper class for common operations in trajectory services.

    Eliminates code duplication between RMSF services by providing
    common trajectory resolution, atom selection, and result mapping.
    """

    def __init__(self, pipeline_data: PipelineData) -> None:
        """Initialize helper with pipeline data.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline context with trajectory data and settings.
        """
        self.pipeline_data = pipeline_data

    def resolve_trajectories_and_atoms(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"],
        atom_selection: str,
    ) -> Tuple[List[md.Trajectory], List[str], np.ndarray]:
        """Resolve trajectory and atom selections to concrete objects.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all']
            Selection describing which trajectories to analyze.
        atom_selection : str
            MDTraj atom selection string.

        Returns
        -------
        tuple
            Trajectories, trajectory names, and atom indices.
        """
        # Resolve trajectory indices
        traj_indices = self._resolve_trajectory_indices(traj_selection)

        # Get trajectory objects and names
        trajectories = self._get_trajectories(traj_indices)
        trajectory_names = self._get_trajectory_names(traj_indices)

        # Resolve atom indices using first trajectory
        atom_indices = self._resolve_atom_indices(trajectories[0], atom_selection)

        return trajectories, trajectory_names, atom_indices

    def build_result_map(
        self,
        trajectory_names: List[str],
        arrays: List[np.ndarray],
        cross_trajectory: bool,
    ) -> Dict[str, np.ndarray]:
        """Create mapping from trajectory names to result arrays.

        Parameters
        ----------
        trajectory_names : list[str]
            Names of processed trajectories.
        arrays : list[np.ndarray]
            Result arrays for each trajectory.
        cross_trajectory : bool
            Whether calculation was cross-trajectory.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of names to arrays.
        """
        if cross_trajectory:
            return {"combined": arrays[0]}

        result: Dict[str, np.ndarray] = {}
        for name, values in zip(trajectory_names, arrays):
            result[name] = values
        return result

    def _resolve_trajectory_indices(
        self,
        selection: Union[int, str, List[Union[int, str]], "all"],
    ) -> List[int]:
        """Resolve trajectory selection to indices.

        Parameters
        ----------
        selection : Union[int, str, list[Union[int, str]], 'all']
            Trajectory selection specification.

        Returns
        -------
        list[int]
            Resolved trajectory indices.
        """
        indices = self.pipeline_data.trajectory_data.get_trajectory_indices(selection)
        if not indices:
            raise ValueError("No trajectories found for the requested selection.")
        return indices

    def _get_trajectories(self, indices: List[int]) -> List[md.Trajectory]:
        """Get trajectory objects for indices.

        Parameters
        ----------
        indices : list[int]
            Trajectory indices to retrieve.

        Returns
        -------
        list[md.Trajectory]
            Trajectory objects.
        """
        trajectories = []
        for idx in indices:
            traj = self.pipeline_data.trajectory_data.trajectories[idx]
            trajectories.append(traj)
        return trajectories

    def _get_trajectory_names(self, indices: List[int]) -> List[str]:
        """Get trajectory names for indices.

        Parameters
        ----------
        indices : list[int]
            Trajectory indices.

        Returns
        -------
        list[str]
            Trajectory names.
        """
        names = self.pipeline_data.trajectory_data.trajectory_names
        return [
            names[idx] if idx < len(names) else f"trajectory_{idx}"
            for idx in indices
        ]

    def _resolve_atom_indices(
        self, trajectory: md.Trajectory, selection: str
    ) -> np.ndarray | None:
        """Resolve atom selection to indices.

        Parameters
        ----------
        trajectory : md.Trajectory
            Trajectory for selection context.
        selection : str
            MDTraj atom selection string.

        Returns
        -------
        np.ndarray or None
            Atom indices or None for all atoms.
        """
        if selection == "all":
            return None

        indices = trajectory.topology.select(selection)
        if indices.size == 0:
            raise ValueError(f"Atom selection '{selection}' produced no atoms.")
        return indices

    def map_reference_to_local_index(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"],
        reference_traj_selection: Union[int, str, List[Union[int, str]], "all"],
    ) -> int:
        """Map global reference trajectory index to local index within selection.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all']
            Selection describing which trajectories are selected.
        reference_traj_selection : Union[int, str, list[Union[int, str]], 'all']
            Selection describing the reference trajectory.

        Returns
        -------
        int
            Local index of reference trajectory within selected trajectories.

        Raises
        ------
        ValueError
            If reference trajectory is not in selected trajectories.

        Examples
        --------
        >>> helper.map_reference_to_local_index([1, 2, 3], 2)
        1
        """
        # Get global indices for both selections
        selected_indices = self._resolve_trajectory_indices(traj_selection)
        ref_global_indices = self._resolve_trajectory_indices(reference_traj_selection)
        ref_global_idx = ref_global_indices[0]

        # Find position of reference in selected trajectories
        if ref_global_idx in selected_indices:
            return selected_indices.index(ref_global_idx)
        else:
            raise ValueError(
                f"Reference trajectory {ref_global_idx} not in selected "
                f"trajectories {selected_indices}"
            )
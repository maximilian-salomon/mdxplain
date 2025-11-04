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
PDB beta factor helper for structure visualization.

This module provides utilities for extracting trajectory frames and
creating PDB files with custom beta factors for visualization purposes.
"""

from __future__ import annotations

import numpy as np
import mdtraj as md
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class PdbBetaFactorHelper:
    """
    Helper for creating PDB files with custom beta factors.

    Provides methods for extracting frames from trajectories and
    saving them as PDB files with custom beta factors for
    visualization in PyMOL.

    Examples
    --------
    >>> # Extract frame as PDB with beta factors
    >>> beta_factors = np.array([0.0, 50.0, 100.0, ...])  # Per-atom
    >>> pdb_path = PdbBetaFactorHelper.create_pdb_with_beta_factors(
    ...     pipeline_data, traj_idx=0, frame_idx=100,
    ...     beta_factors=beta_factors, output_path="output.pdb"
    ... )
    """

    @staticmethod
    def create_pdb_with_beta_factors(
        pipeline_data: PipelineData,
        traj_idx: int,
        frame_idx: int,
        beta_factors: np.ndarray,
        output_path: str
    ) -> str:
        """
        Create PDB with custom beta factors.

        Extracts a frame and saves it with custom beta factors
        for each atom. Beta factors are used for coloring in PyMOL.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        traj_idx : int
            Trajectory index
        frame_idx : int
            Frame index within trajectory
        beta_factors : np.ndarray
            Beta factors for all atoms, shape (n_atoms,)
        output_path : str
            Path where to save the PDB file

        Returns
        -------
        str
            Path to created PDB file

        Raises
        ------
        ValueError
            If beta_factors shape doesn't match number of atoms

        Examples
        --------
        >>> # Create PDB with importance-based beta factors
        >>> topology = pipeline.data.trajectory_data.trajectories[0].topology
        >>> beta_factors = np.zeros(topology.n_atoms)
        >>> # Set beta factors based on residue importance...
        >>> pdb_path = PdbBetaFactorHelper.create_pdb_with_beta_factors(
        ...     pipeline_data, 0, 100, beta_factors, "output.pdb"
        ... )

        Notes
        -----
        - Beta factors should be in range 0-100 for optimal PyMOL visualization
        - All atoms in the frame receive the provided beta factors
        - Memmap-safe: only loads one frame
        """
        trajectory = pipeline_data.trajectory_data.trajectories[traj_idx]

        # Extract single frame (memmap-safe!)
        frame = trajectory[frame_idx]

        # Validate beta factors
        n_atoms = frame.topology.n_atoms
        if len(beta_factors) != n_atoms:
            raise ValueError(
                f"Beta factors shape {len(beta_factors)} doesn't match "
                f"number of atoms {n_atoms}"
            )

        # MDTraj doesn't directly support setting beta factors during save
        # We need to save and then modify the PDB file
        frame.save_pdb(output_path)

        # Update beta factors in the PDB file
        PdbBetaFactorHelper._update_pdb_beta_factors(
            output_path, beta_factors
        )

        return output_path

    @staticmethod
    def _update_pdb_beta_factors(pdb_path: str, beta_factors: np.ndarray) -> None:
        """
        Update beta factors in existing PDB file.

        Reads PDB file, updates beta factor column, and writes back.

        Parameters
        ----------
        pdb_path : str
            Path to PDB file to modify
        beta_factors : np.ndarray
            Beta factors for all atoms

        Returns
        -------
        None
            Modifies PDB file in place

        Notes
        -----
        PDB format: Beta factor is in columns 61-66 (0-indexed: 60-66)
        Format: %6.2f
        """
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        modified_lines = PdbBetaFactorHelper._process_pdb_lines(
            lines, beta_factors
        )

        with open(pdb_path, 'w') as f:
            f.writelines(modified_lines)

    @staticmethod
    def _process_pdb_lines(
        lines: List[str],
        beta_factors: np.ndarray
    ) -> List[str]:
        """
        Process PDB lines and update beta factors.

        Parameters
        ----------
        lines : List[str]
            PDB file lines
        beta_factors : np.ndarray
            Beta factors for atoms

        Returns
        -------
        List[str]
            Modified PDB lines
        """
        modified_lines = []
        atom_idx = 0

        for line in lines:
            if PdbBetaFactorHelper._is_atom_line(line):
                modified_line = PdbBetaFactorHelper._update_atom_line(
                    line, beta_factors, atom_idx
                )
                modified_lines.append(modified_line)
                atom_idx += 1
            else:
                modified_lines.append(line)

        return modified_lines

    @staticmethod
    def _is_atom_line(line: str) -> bool:
        """Check if line is an ATOM or HETATM record."""
        return line.startswith('ATOM') or line.startswith('HETATM')

    @staticmethod
    def _update_atom_line(
        line: str,
        beta_factors: np.ndarray,
        atom_idx: int
    ) -> str:
        """
        Update beta factor in atom line.

        Parameters
        ----------
        line : str
            Original PDB line
        beta_factors : np.ndarray
            Beta factors array (0-1 range)
        atom_idx : int
            Current atom index

        Returns
        -------
        str
            Modified PDB line

        Notes
        -----
        Beta factors are expected to be in 0-1 range for nglview visualization.
        """
        if atom_idx >= len(beta_factors):
            return line

        beta_value = float(beta_factors[atom_idx])
        prefix = line[:60]
        beta_str = f"{beta_value:6.2f}"
        suffix = line[66:] if len(line) > 66 else "\n"

        return prefix + beta_str + suffix

    @staticmethod
    def get_topology(pipeline_data: PipelineData, traj_idx: int = 0) -> md.Topology:
        """
        Get topology from a trajectory.

        Convenience method to access topology for residue information.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        traj_idx : int, default=0
            Trajectory index to get topology from

        Returns
        -------
        mdtraj.Topology
            Topology object

        Examples
        --------
        >>> topology = PdbBetaFactorHelper.get_topology(pipeline_data, 0)
        >>> print(f"Number of residues: {topology.n_residues}")
        """
        trajectory = pipeline_data.trajectory_data.trajectories[traj_idx]
        return trajectory.topology

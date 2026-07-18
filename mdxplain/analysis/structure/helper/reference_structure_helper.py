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

"""Helpers for building reference structures in a memory-efficient way."""

from __future__ import annotations

from typing import Iterable
import numpy as np


class ReferenceStructureHelper:
    """Collection of reference structure utilities for structure analysis.

    Helper methods compute mean, median, and MAD-based reference structures in a
    streaming fashion so large trajectories can be processed without exhausting
    memory resources.
    """

    @staticmethod
    def get_mean_coordinates(
        trajectories: Iterable,
        atom_chunk_size: int,
        use_memmap: bool,
        cross_trajectory: bool = True,
        frame_chunk_size: int = 2000,
    ) -> np.ndarray | dict[int, np.ndarray]:
        """Calculate mean coordinates across trajectories using atom-batching.

        Parameters
        ----------
        trajectories : Iterable
            Collection of trajectory-like objects supporting slicing.
        atom_chunk_size : int
            Number of atoms processed per batch.
        use_memmap : bool
            Whether to use memory-mapped processing with frame chunks.
        cross_trajectory : bool, optional
            If True, compute reference across all trajectories combined.
            If False, compute separate reference per trajectory.
            Defaults to True.
        frame_chunk_size : int, optional
            Number of frames to process per chunk when use_memmap=True.
            Defaults to 2000.

        Returns
        -------
        numpy.ndarray or dict[int, numpy.ndarray]
            If cross_trajectory=True: Mean coordinates with shape ``(n_atoms, 3)``.
            If cross_trajectory=False: Dict mapping trajectory index to mean coordinates.

        Raises
        ------
        ValueError
            If no frames are available across all trajectories.

        Examples
        --------
        >>> import mdtraj as md
        >>> import numpy as np
        >>> topology = md.Topology()
        >>> chain = topology.add_chain()
        >>> residue = topology.add_residue("ALA", chain)
        >>> topology.add_atom("CA", md.element.carbon, residue)
        >>> coords = np.random.rand(100, 1, 3).astype(np.float32)
        >>> traj = md.Trajectory(coords, topology)
        >>> mean_ref = ReferenceStructureHelper.get_mean_coordinates(
        ...     [traj], atom_chunk_size=50, use_memmap=False
        ... )
        >>> mean_ref.shape
        (1, 3)

        Notes
        -----
        All frames of each atom batch are collected and concatenated before the
        reduction, so peak memory scales with ``atom_chunk_size`` times the total
        frame count; lower ``atom_chunk_size`` to reduce it. Under
        ``use_memmap=True`` ``frame_chunk_size`` only sets how many frames are
        read per step and does not change the peak, since the batch is
        concatenated in full.
        """
        trajectories = list(trajectories)

        if not trajectories:
            raise ValueError("Cannot compute mean coordinates without trajectories.")

        return ReferenceStructureHelper._compute_reference_coordinates(
            trajectories, atom_chunk_size, use_memmap,
            cross_trajectory, frame_chunk_size, np.mean
        )

    @staticmethod
    def get_median_coordinates(
        trajectories: Iterable,
        atom_chunk_size: int,
        use_memmap: bool,
        cross_trajectory: bool = True,
        frame_chunk_size: int = 2000,
    ) -> np.ndarray | dict[int, np.ndarray]:
        """Calculate median coordinates across trajectories using atom-batching.

        Parameters
        ----------
        trajectories : Iterable
            Collection of trajectory-like objects supporting slicing.
        atom_chunk_size : int
            Number of atoms processed per batch.
        use_memmap : bool
            Whether to use memory-mapped processing with frame chunks.
        cross_trajectory : bool, optional
            If True, compute reference across all trajectories combined.
            If False, compute separate reference per trajectory.
            Defaults to True.
        frame_chunk_size : int, optional
            Number of frames to process per chunk when use_memmap=True.
            Defaults to 2000.

        Returns
        -------
        numpy.ndarray or dict[int, numpy.ndarray]
            If cross_trajectory=True: Median coordinates with shape ``(n_atoms, 3)``.
            If cross_trajectory=False: Dict mapping trajectory index to median coordinates.

        Raises
        ------
        ValueError
            If no frames are available across all trajectories.

        Examples
        --------
        >>> import mdtraj as md
        >>> import numpy as np
        >>> topology = md.Topology()
        >>> chain = topology.add_chain()
        >>> residue = topology.add_residue("ALA", chain)
        >>> topology.add_atom("CA", md.element.carbon, residue)
        >>> coords = np.random.rand(100, 1, 3).astype(np.float32)
        >>> traj = md.Trajectory(coords, topology)
        >>> median_ref = ReferenceStructureHelper.get_median_coordinates(
        ...     [traj], atom_chunk_size=50, use_memmap=False
        ... )
        >>> median_ref.shape
        (1, 3)

        Notes
        -----
        The median collects all frames of each atom batch before reducing, the
        same as the mean here, so both have the same peak memory. Lower
        ``atom_chunk_size`` to reduce it; ``frame_chunk_size`` only sets the read
        granularity under ``use_memmap`` and does not change the peak.
        """
        trajectories = list(trajectories)

        if not trajectories:
            raise ValueError("Cannot compute median coordinates without trajectories.")

        return ReferenceStructureHelper._compute_reference_coordinates(
            trajectories, atom_chunk_size, use_memmap,
            cross_trajectory, frame_chunk_size, np.median
        )

    @staticmethod
    def _compute_reference_coordinates(
        trajectories, atom_chunk_size, use_memmap, cross_trajectory, frame_chunk_size, reduction_func
    ):
        """Compute reference coordinates with given reduction function.

        Parameters
        ----------
        trajectories : list
            List of trajectories.
        atom_chunk_size : int
            Number of atoms per batch.
        use_memmap : bool
            Whether to use memory-mapped processing.
        cross_trajectory : bool
            If True, compute across all trajectories.
        frame_chunk_size : int
            Frames per chunk when using memmap.
        reduction_func : callable
            Function to reduce coordinates (np.mean or np.median).

        Returns
        -------
        np.ndarray or dict
            Reference coordinates.
        """
        if cross_trajectory:
            return ReferenceStructureHelper._compute_cross_trajectory(
                trajectories, atom_chunk_size, use_memmap, frame_chunk_size, reduction_func
            )
        else:
            return ReferenceStructureHelper._compute_per_trajectory(
                trajectories, atom_chunk_size, use_memmap, frame_chunk_size, reduction_func
            )

    @staticmethod
    def _compute_cross_trajectory(trajectories, atom_chunk_size, use_memmap, frame_chunk_size, reduction_func):
        """Compute reference across all trajectories.

        Parameters
        ----------
        trajectories : list
            List of trajectories.
        atom_chunk_size : int
            Number of atoms per batch.
        use_memmap : bool
            Whether to use memory-mapped processing.
        frame_chunk_size : int
            Frames per chunk when using memmap.
        reduction_func : callable
            Function to reduce coordinates.

        Returns
        -------
        np.ndarray
            Reference coordinates.
        """
        n_atoms = trajectories[0].n_atoms
        result = np.empty((n_atoms, 3))

        for atom_start in range(0, n_atoms, atom_chunk_size):
            atom_end = min(n_atoms, atom_start + atom_chunk_size)

            all_coords = []
            for trajectory in trajectories:
                if trajectory.n_atoms != n_atoms:
                    raise ValueError("All trajectories must have the same number of atoms.")
                
                coords = ReferenceStructureHelper._collect_trajectory_coords(
                    trajectory, atom_start, atom_end, use_memmap, frame_chunk_size
                )
                all_coords.extend(coords)

            if not all_coords:
                raise ValueError("Cannot compute reference coordinates without frames.")

            stacked = np.concatenate(all_coords, axis=0)
            result[atom_start:atom_end] = reduction_func(stacked, axis=0)

        return result

    @staticmethod
    def _compute_per_trajectory(trajectories, atom_chunk_size, use_memmap, frame_chunk_size, reduction_func):
        """Compute reference separately for each trajectory.

        Parameters
        ----------
        trajectories : list
            List of trajectories.
        atom_chunk_size : int
            Number of atoms per batch.
        use_memmap : bool
            Whether to use memory-mapped processing.
        frame_chunk_size : int
            Frames per chunk when using memmap.
        reduction_func : callable
            Function to reduce coordinates.

        Returns
        -------
        dict
            Dictionary mapping trajectory index to reference coordinates.
        """
        results = {}

        for traj_idx, trajectory in enumerate(trajectories):
            traj_n_atoms = trajectory.n_atoms
            result = np.empty((traj_n_atoms, 3))

            for atom_start in range(0, traj_n_atoms, atom_chunk_size):
                atom_end = min(traj_n_atoms, atom_start + atom_chunk_size)

                coords = ReferenceStructureHelper._collect_trajectory_coords(
                    trajectory, atom_start, atom_end, use_memmap, frame_chunk_size
                )

                if not coords:
                    raise ValueError(f"Cannot compute reference coordinates for trajectory {traj_idx} without frames.")

                stacked = np.concatenate(coords, axis=0)
                result[atom_start:atom_end] = reduction_func(stacked, axis=0)

            results[traj_idx] = result

        return results

    @staticmethod
    def _collect_trajectory_coords(trajectory, atom_start, atom_end, use_memmap, frame_chunk_size):
        """Collect coordinates for atom range from trajectory.

        Parameters
        ----------
        trajectory : md.Trajectory
            Trajectory to collect from.
        atom_start : int
            Start atom index.
        atom_end : int
            End atom index.
        use_memmap : bool
            Whether to use memory-mapped processing.
        frame_chunk_size : int
            Frames per chunk.

        Returns
        -------
        list
            List of coordinate arrays.
        """
        coords = []

        if use_memmap and trajectory.n_frames > frame_chunk_size:
            for frame_chunk in ReferenceStructureHelper._iterate_frame_chunks(trajectory, frame_chunk_size):
                coords.append(frame_chunk.xyz[:, atom_start:atom_end, :])
        else:
            coords.append(trajectory.xyz[:, atom_start:atom_end, :])

        return coords

    @staticmethod
    def _iterate_frame_chunks(trajectory, frame_chunk_size: int = 2000):
        """Iterate trajectory in frame chunks for memory-mapped processing.

        Parameters
        ----------
        trajectory : md.Trajectory
            Trajectory to iterate in frame chunks.
        frame_chunk_size : int, optional
            Number of frames per chunk. Defaults to 1000.

        Yields
        ------
        md.Trajectory
            Trajectory chunks with frame_chunk_size frames each.
        """
        for start in range(0, trajectory.n_frames, frame_chunk_size):
            end = min(start + frame_chunk_size, trajectory.n_frames)
            yield trajectory[start:end]

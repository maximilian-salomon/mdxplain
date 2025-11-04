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

"""Shared helper functions for structure calculations.

These helpers provide chunk iteration and stacking utilities that allow
structure analyses to process large trajectories without loading all frames
into memory simultaneously.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Tuple, TYPE_CHECKING

import mdtraj as md
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory


class StructureCalculationHelper:
    """Bundle of shared low-level operations for structure analysis.

    The helper centralises chunk iteration and frame stacking logic so higher
    level services/calculators can operate on homogeneous APIs regardless of
    whether trajectories are processed in-memory or via memmap streaming.
    """

    @staticmethod
    def iterate_chunks(
        trajectory: md.Trajectory | DaskMDTrajectory,
        chunk_size: int,
    ) -> Iterator[Tuple[md.Trajectory, int, int]]:
        """Yield slices of a trajectory with the requested chunk size.

        Parameters
        ----------
        trajectory : md.Trajectory or DaskMDTrajectory
            Trajectory object supporting slicing via ``__getitem__``.
        chunk_size : int
            Number of frames processed per chunk. Values less than one default to
            the full trajectory length.

        Returns
        -------
        Iterator[Tuple[md.Trajectory, int, int]]
            Iterator yielding tuples containing ``(chunk, start_index, end_index)``.

        Notes
        -----
        When chunking is disabled the helper yields a single tuple containing the
        original trajectory and the full frame range.

        Examples
        --------
        >>> topology = md.Topology()
        >>> chain = topology.add_chain()
        >>> residue = topology.add_residue("ALA", chain)
        >>> topology.add_atom("CA", md.element.carbon, residue)
        >>> coords = np.zeros((5, 1, 3), dtype=np.float32)
        >>> traj = md.Trajectory(coords, topology)
        >>> iterator = StructureCalculationHelper.iterate_chunks(traj, chunk_size=2)
        >>> chunk, start, end = next(iterator)
        >>> (chunk.n_frames, start, end)
        (2, 0, 2)
        """
        if chunk_size is not None and chunk_size < 0:
            raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")

        total_frames = int(trajectory.n_frames)
        if total_frames == 0:
            return

        effective_chunk = chunk_size if chunk_size and chunk_size > 0 else total_frames
        effective_chunk = min(effective_chunk, total_frames)

        for start in range(0, total_frames, effective_chunk):
            end = min(start + effective_chunk, total_frames)
            yield trajectory[start:end], start, end

    @staticmethod
    def stack_frames(frames: Iterable[md.Trajectory]) -> md.Trajectory:
        """Stack individual frames into a single trajectory.

        Parameters
        ----------
        frames : Iterable[md.Trajectory]
            Iterable of trajectories with exactly one frame each.

        Returns
        -------
        md.Trajectory
            Combined trajectory containing the stacked frames.

        Raises
        ------
        ValueError
            If the iterable is empty.

        Examples
        --------
        >>> topology = md.Topology()
        >>> chain = topology.add_chain()
        >>> residue = topology.add_residue("ALA", chain)
        >>> topology.add_atom("CA", md.element.carbon, residue)
        >>> frame = md.Trajectory(np.zeros((1, 1, 3), dtype=np.float32), topology)
        >>> stacked = StructureCalculationHelper.stack_frames([frame, frame])
        >>> stacked.n_frames
        2
        """

        frames = list(frames)
        if not frames:
            raise ValueError("Expected at least one frame to stack.")

        coords = np.concatenate([frame.xyz for frame in frames], axis=0)
        time = np.concatenate([frame.time for frame in frames], axis=0)
        topology = frames[0].topology
        return md.Trajectory(coords, topology, time=time)

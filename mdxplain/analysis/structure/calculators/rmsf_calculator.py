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

"""Calculator utilities for RMSF computations."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Literal, Sequence, Tuple, Union

import mdtraj as md
import numpy as np

from ..helpers.reference_structure_helper import ReferenceStructureHelper
from ..helpers.residue_aggregation_helper import AGGREGATOR_CODES, aggregate_residues_jit
from ..helpers.structure_calculation_helper import StructureCalculationHelper


class RMSFCalculator:
    """Perform RMSF calculations with optional chunk-aware processing.

    The calculator mirrors :class:`RMSDCalculator` but focuses on atom-wise and
    residue-wise fluctuations relative to configurable reference structures.
    Reference coordinates can be derived from mean, median frames
    and deviations support the same robust metrics.
    """

    def __init__(
        self,
        trajectories: Iterable[md.Trajectory],
        chunk_size: int,
        use_memmap: bool,
        trajectory_names: List[str] = None,
    ) -> None:
        """Initialise the RMSF calculator with trajectory data.

        Parameters
        ----------
        trajectories : Iterable[md.Trajectory]
            Iterable of trajectory-like objects.
        chunk_size : int
            Chunk size used for streaming calculations.
        use_memmap : bool
            Flag indicating whether the trajectories are memory mapped.
        trajectory_names : List[str], optional
            Names for result mapping. Defaults to "trajectory_0", "trajectory_1", etc.

        Raises
        ------
        ValueError
            If no trajectories are supplied.
        """
        self.trajectories = list(trajectories)
        if not self.trajectories:
            raise ValueError("At least one trajectory must be provided.")

        self.chunk_size = chunk_size
        self.use_memmap = use_memmap
        self.trajectory_names = trajectory_names or [
            f"trajectory_{i}" for i in range(len(self.trajectories))
        ]

    def calculate_per_atom(
        self,
        reference_mode: Literal["mean", "median"],
        metric: Literal["mean", "median", "mad"],
        atom_indices: np.ndarray = None,
        cross_trajectory: bool = False,
    ) -> List[np.ndarray]:
        """Return per-atom RMSF values for the trajectories.

        Builds the requested reference structure, streams trajectory coordinates
        in chunks and applies the chosen robust metric to the atom-wise squared
        deviations.

        Parameters
        ----------
        reference_mode : {'mean', 'median'}
            Strategy used to construct the reference structure.
        metric : {'mean', 'median', 'mad'}
            Robust metric applied to the squared deviations.
        atom_indices : np.ndarray, optional
            Atom indices used for the RMSF calculation. None selects all atoms.
        cross_trajectory : bool, optional
            When ``True``, all trajectories are combined into a single RMSF profile.

        Returns
        -------
        List[np.ndarray]
            List of per-atom RMSF arrays. If cross_trajectory=True, returns single element.
        """

        trajectories = self.trajectories
        if atom_indices is not None:
            trajectories = [t.atom_slice(atom_indices) for t in trajectories]

        atom_batch_size = self._determine_atom_batch_size(trajectories)

        if cross_trajectory:
            reference = self._compute_reference_coordinates(
                trajectories,
                reference_mode,
                cross_trajectory=True,
            )
            combined = self._compute_cross_trajectory_rmsf(
                trajectories, reference, metric, atom_batch_size
            )
            return [combined]

        # Per-trajectory RMSF calculation
        references = self._compute_reference_coordinates(
            trajectories,
            reference_mode,
            cross_trajectory=False,
        )
        per_atom_arrays = []

        for idx, trajectory in enumerate(trajectories):
            reference = references[idx]
            traj_result = self._compute_single_trajectory_rmsf(
                trajectory, reference, metric, atom_batch_size
            )
            per_atom_arrays.append(traj_result)

        return per_atom_arrays

    def calculate_per_residue(
        self,
        reference_mode: Literal["mean", "median"],
        metric: Literal["mean", "median", "mad"],
        residue_aggregator: Literal["mean", "median", "rms", "rms_median"],
        atom_indices: np.ndarray = None,
        cross_trajectory: bool = False,
        reference_trajectory_index: int = 0,
    ) -> List[np.ndarray]:
        """Return per-residue RMSF values for the trajectories.

        Computes per-atom RMSF values with the requested reference and metric and
        aggregates them residue-wise using the selected aggregator.

        Parameters
        ----------
        reference_mode : {'mean', 'median'}
            Strategy used to construct the reference structure.
        metric : {'mean', 'median', 'mad'}
            Robust metric applied to the atom-wise squared deviations.
        residue_aggregator : {'mean', 'median', 'rms', 'rms_median'}
            Aggregator applied to per-atom RMSF values within each residue.
        atom_indices : np.ndarray, optional
            Atom indices used for the RMSF calculation. None selects all atoms.
        cross_trajectory : bool, optional
            When ``True``, all trajectories are combined into a single RMSF profile before
            residue aggregation.
        reference_trajectory_index : int, optional
            Index of trajectory providing the topology for residue aggregation.

        Returns
        -------
        List[np.ndarray]
            List of per-residue RMSF arrays. If cross_trajectory=True, returns single element.
        """
        # Get per-atom RMSF first
        per_atom_arrays = self.calculate_per_atom(
            reference_mode, metric, atom_indices, cross_trajectory
        )

        trajectories = self.trajectories
        if atom_indices is not None:
            trajectories = [t.atom_slice(atom_indices) for t in trajectories]

        # Use reference trajectory for residue grouping
        reference_trajectory = trajectories[reference_trajectory_index]
        groups = self._residue_groups(reference_trajectory)

        if cross_trajectory:
            residues = self._aggregate_residues(per_atom_arrays[0], groups, residue_aggregator)
            return [residues]

        # Per-trajectory residue aggregation
        per_residue_arrays = []
        for atom_values in per_atom_arrays:
            residues = self._aggregate_residues(atom_values, groups, residue_aggregator)
            per_residue_arrays.append(residues)

        return per_residue_arrays


    def _compute_reference_coordinates(
        self,
        trajectories: Sequence[md.Trajectory],
        mode: Literal["mean", "median"],
        cross_trajectory: bool = True,
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Return reference coordinates matching the requested mode.
        
        Parameters
        ----------
        trajectories : Sequence[md.Trajectory]
            List of trajectories to derive the reference structure from.
        mode : {'mean', 'median'}
            Strategy used to construct the reference structure.
        cross_trajectory : bool, optional
            If True, compute reference across all trajectories combined.
            Defaults to True.

        Returns
        -------
        Union[np.ndarray, Dict[int, np.ndarray]]
            If cross_trajectory=True: Array of shape (n_atoms, 3) containing the reference coordinates.
            If cross_trajectory=False: Dict mapping trajectory index to reference coordinates.
        """

        if mode == "mean":
            result = ReferenceStructureHelper.get_mean_coordinates(
                trajectories,
                atom_chunk_size=self._determine_atom_batch_size(trajectories),
                use_memmap=self.use_memmap,
                cross_trajectory=cross_trajectory,
                frame_chunk_size=self.chunk_size,
            )
        elif mode == "median":
            result = ReferenceStructureHelper.get_median_coordinates(
                trajectories,
                atom_chunk_size=self._determine_atom_batch_size(trajectories),
                use_memmap=self.use_memmap,
                cross_trajectory=cross_trajectory,
                frame_chunk_size=self.chunk_size,
            )
        else:
            raise ValueError("Unsupported reference mode, must be 'mean' or 'median'.")
        
        return result

    def _compute_cross_trajectory_rmsf(
        self,
        trajectories: List[md.Trajectory],
        reference: np.ndarray,
        metric: Literal["mean", "median", "mad"],
        atom_batch_size: int,
    ) -> np.ndarray:
        """Compute RMSF across all trajectories using a single reference.

        Parameters
        ----------
        trajectories : List[md.Trajectory]
            List of trajectories to compute RMSF for.
        reference : np.ndarray
            Single reference structure for all trajectories.
        metric : {'mean', 'median', 'mad'}
            Metric to apply to squared deviations.
        atom_batch_size : int
            Number of atoms to process per batch.

        Returns
        -------
        np.ndarray
            Combined RMSF values across all trajectories.
        """
        combined = np.empty(reference.shape[0], dtype=np.float32)

        for start in range(0, reference.shape[0], atom_batch_size):
            end = min(reference.shape[0], start + atom_batch_size)
            all_timeseries = []
            for trajectory in trajectories:
                timeseries = self._squared_chunks(trajectory, reference, start, end)
                all_timeseries.append(timeseries)
            combined_timeseries = np.concatenate(all_timeseries, axis=0)
            combined[start:end] = self._reduce_metric(metric, combined_timeseries)

        return combined

    def _compute_single_trajectory_rmsf(
        self,
        trajectory: md.Trajectory,
        reference: np.ndarray,
        metric: Literal["mean", "median", "mad"],
        atom_batch_size: int,
    ) -> np.ndarray:
        """Compute RMSF for a single trajectory using its specific reference.

        Parameters
        ----------
        trajectory : md.Trajectory
            Trajectory to compute RMSF for.
        reference : np.ndarray
            Reference structure specific to this trajectory.
        metric : {'mean', 'median', 'mad'}
            Metric to apply to squared deviations.
        atom_batch_size : int
            Number of atoms to process per batch.

        Returns
        -------
        np.ndarray
            RMSF values for the trajectory.
        """
        result = np.empty(reference.shape[0], dtype=np.float32)

        for start in range(0, reference.shape[0], atom_batch_size):
            end = min(reference.shape[0], start + atom_batch_size)
            timeseries = self._squared_chunks(trajectory, reference, start, end)
            result[start:end] = self._reduce_metric(metric, timeseries)

        return result

    def _squared_chunks(
        self,
        trajectory: md.Trajectory,
        reference_coords: np.ndarray,
        atom_start: int,
        atom_end: int,
    ) -> np.ndarray:
        """
        Return complete time series of squared deviations for atom subset.
        
        Parameters
        ----------
        trajectory : md.Trajectory
            Trajectory to process in chunks.
        reference_coords : np.ndarray
            Reference coordinates with shape (n_atoms, 3).
        atom_start : int
            Starting atom index for the subset.
        atom_end : int
            Ending atom index for the subset.

        Returns
        -------
        np.ndarray
            Array of shape (n_frames, n_atoms_in_subset) containing squared deviations.
        """

        if atom_start >= atom_end:
            raise ValueError("Invalid atom range for RMSF calculation. " \
            "This should never happen. Please report a bug.")

        reference_slice = reference_coords[atom_start:atom_end]
        all_squared = []

        for chunk, _, _ in self._chunk_iterator(trajectory):
            coords = chunk.xyz[:, atom_start:atom_end, :]
            diff = coords - reference_slice
            squared_chunk = np.sum(diff * diff, axis=-1)
            all_squared.append(squared_chunk)

        return np.concatenate(all_squared, axis=0).astype(np.float32)

    def _reduce_metric(
        self,
        metric: Literal["mean", "median", "mad"],
        squared_timeseries: np.ndarray,
    ) -> np.ndarray:
        """
        Reduce squared deviations according to the requested metric.

        Parameters
        ----------
        metric : {'mean', 'median', 'mad'}
            Metric to use for reduction.
        squared_timeseries : np.ndarray
            Array of squared deviations to reduce.

        Returns
        -------
        np.ndarray
            Reduced RMSF values.
        """
        if metric not in ("mean", "median", "mad"):
            raise ValueError(f"Unsupported metric: {metric}")

        if squared_timeseries.size == 0:
            raise ValueError("No frames processed for RMSF calculation.")

        if metric == "mean":
            mean_values = np.mean(squared_timeseries, axis=0)
            return np.sqrt(np.clip(mean_values, a_min=0.0, a_max=None))

        if metric == "median":
            values = np.median(squared_timeseries, axis=0)
            return np.sqrt(np.clip(values, a_min=0.0, a_max=None))

        distances = np.sqrt(np.clip(squared_timeseries, a_min=0.0, a_max=None))
        median = np.median(distances, axis=0)
        mad = np.median(np.abs(distances - median), axis=0)
        return mad



    def _residue_groups(
        self,
        reference_trajectory: md.Trajectory,
    ) -> List[List[int]]:
        """
        Return residue-to-atom index groups for aggregation.

        Parameters
        ----------
        reference_trajectory : md.Trajectory
            Trajectory providing the topology for residue grouping.

        Returns
        -------
        List[List[int]]
            List of residue groups, each containing the indices of atoms
            belonging to that residue.
        """
        atom_array = np.arange(reference_trajectory.n_atoms, dtype=int)

        index_map = {int(atom_idx): position for position, atom_idx in enumerate(atom_array)}
        included_atoms = set(int(idx) for idx in atom_array)

        groups: List[List[int]] = []
        for residue in reference_trajectory.topology.residues:
            group: List[int] = []
            for atom in residue.atoms:
                if atom.index in included_atoms:
                    group.append(index_map[atom.index])
            if group:
                groups.append(group)

        if not groups:
            raise ValueError("Atom selection produced no residues for aggregation.")
        return groups

    def _aggregate_residues(
        self,
        atom_values: np.ndarray,
        groups: List[List[int]],
        aggregator: Literal["mean", "median", "rms", "rms_median"],
    ) -> np.ndarray:
        """
        Aggregate per-atom RMSF values to residue-level RMSF.

        Parameters
        ----------
        atom_values : np.ndarray
            Array of per-atom RMSF values.
        groups : List[List[int]]
            List of residue groups, each containing the indices of atoms
            belonging to that residue.
        aggregator : {'mean', 'median', 'rms', 'rms_median'}
            Aggregator applied to per-atom RMSF values within each residue.

        Returns
        -------
        np.ndarray
            Aggregated RMSF values per residue.
        """
        if not groups:
            raise ValueError("No residue groups provided for aggregation. This is very likely a bug.")

        # Convert groups to flat arrays for JIT function
        group_indices = np.concatenate(groups)
        group_sizes = [len(group) for group in groups]
        group_boundaries = np.cumsum([0] + group_sizes)  # Include end marker

        # Get aggregator code from imported mapping
        if aggregator not in AGGREGATOR_CODES:
            raise ValueError(f"Unsupported residue aggregator: {aggregator}")

        aggregator_code = AGGREGATOR_CODES[aggregator]

        # Call JIT-compiled function
        return aggregate_residues_jit(
            atom_values,
            group_indices,
            group_boundaries,
            aggregator_code,
        )

    def _chunk_iterator(
        self,
        trajectory: md.Trajectory,
    ) -> Iterator[tuple[md.Trajectory, int, int]]:
        """
        Yield trajectory chunks respecting memmap configuration.

        Parameters
        ----------
        trajectory : md.Trajectory
            Trajectory to iterate in chunks.
        
        Yields
        ------
        Iterator[tuple[md.Trajectory, int, int]]
            Tuples of (chunk, start_frame, end_frame).
        """

        if self.use_memmap and self.chunk_size and self.chunk_size > 0:
            yield from StructureCalculationHelper.iterate_chunks(trajectory, self.chunk_size)
        else:
            yield trajectory, 0, trajectory.n_frames

    def _determine_atom_batch_size(
        self,
        trajectories: Sequence[md.Trajectory],
    ) -> int:
        """Return the maximum number of atoms that fits into one batch.

        The calculation respects the user-configured frame chunk size and the size of
        the first trajectory. It assumes that one floating point value consumes
        four bytes (``numpy.float32``).

        Parameters
        ----------
        trajectories : Sequence[md.Trajectory]
            Trajectories that have already been filtered by atom selection.

        Returns
        -------
        int
            Recommended atom batch size, bounded to at least one atom.
        """

        if not trajectories:
            return 1

        first = trajectories[0]
        atom_count = max(1, first.n_atoms)
        frame_chunk = self.chunk_size if self.chunk_size and self.chunk_size > 0 else first.n_frames
        frame_chunk = max(1, frame_chunk)

        # Bytes available per batch equals the size currently needed for a full
        # frame chunk across the entire atom selection.
        available_bytes = atom_count * frame_chunk * 4

        total_frames = sum(max(0, traj.n_frames) for traj in trajectories)
        if total_frames <= 0:
            return atom_count

        max_atoms = available_bytes // (total_frames * 4)
        max_atoms = max(1, max_atoms)
        return min(max_atoms, atom_count)

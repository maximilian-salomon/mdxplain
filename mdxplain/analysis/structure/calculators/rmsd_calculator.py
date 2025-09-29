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

"""Calculator utilities for RMSD computations."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Literal, Sequence, Tuple
import warnings

import mdtraj as md
import numpy as np

from ..helpers.structure_calculation_helper import StructureCalculationHelper


class RMSDCalculator:
    """Perform RMSD calculations with optional chunk-aware processing.

    The calculator accepts a collection of trajectory-like objects and exposes
    three computation modes:

    * RMSD to an externally supplied reference frame.
    * Frame-to-frame RMSD with configurable lag.
    * Window-based RMSD in ``frame_to_start``/``frame_to_frame`` modes.

    All calculations are vectorised and can operate on streamed chunks when the
    caller indicates ``use_memmap=True``. The class itself performs only the
    numerical work; selection of trajectories, atom filters, and reference
    frames happens in :class:`mdxplain.analysis.structure.services.rmsd_variant_service.RMSDVariantService`.
    """

    def __init__(self, trajectories: Iterable, chunk_size: int, use_memmap: bool) -> None:
        """Initialise the calculator with shared runtime settings.

        The calculator stores the trajectory collection alongside the streaming
        configuration so all subsequent computations can reuse the same setup.

        Parameters
        ----------
        trajectories : Iterable
            Iterable of trajectory-like objects.
        chunk_size : int
            Chunk size used for streaming calculations.
        use_memmap : bool
            Flag indicating whether the trajectories are memory mapped.

        Returns
        -------
        None
            The initializer does not return anything.

        Raises
        ------
        ValueError
            If no trajectories are supplied.

        Examples
        --------
        >>> topology = md.Topology()
        >>> _ = topology.add_chain()
        >>> trajectory = md.Trajectory(np.zeros((5, 1, 3), dtype=np.float32), topology)
        >>> calculator = RMSDCalculator([trajectory], chunk_size=100, use_memmap=False)
        >>> calculator.chunk_size
        100
        """

        self.trajectories = list(trajectories)
        if not self.trajectories:
            raise ValueError("RMSDCalculator requires at least one trajectory.")
        self.chunk_size = chunk_size
        self.use_memmap = use_memmap

    def rmsd_to_reference(
        self,
        reference_frame: md.Trajectory,
        atom_indices: Sequence[int] | None,
        metric: Literal["mean", "median", "mad"],
    ) -> List[np.ndarray]:
        """Compute RMSD of each frame against a reference frame.

        Each trajectory is streamed in chunks, squared deviations towards the
        reference frame are accumulated and the requested metric is applied to
        obtain frame-wise RMSD values.

        Parameters
        ----------
        reference_frame : md.Trajectory
            Single-frame trajectory used as reference.
        atom_indices : Sequence[int] or None
            Atom indices used for the RMSD calculation. ``None`` selects all atoms.
        metric : {'mean', 'median', 'mad'}
            Robust aggregation metric applied to atom-wise squared deviations.

        Returns
        -------
        list of numpy.ndarray
            List containing RMSD values for each trajectory.

        Examples
        --------
        >>> topology = md.Topology()
        >>> _ = topology.add_chain()
        >>> trajectory = md.Trajectory(np.zeros((5, 1, 3), dtype=np.float32), topology)
        >>> calculator = RMSDCalculator([trajectory], chunk_size=100, use_memmap=False)
        >>> values = calculator.rmsd_to_reference(trajectory[0], atom_indices=None, metric="mean")
        >>> isinstance(values[0], np.ndarray)
        True
        """

        results: List[np.ndarray] = []
        reference_coords = reference_frame.xyz[0]
        atom_indices_array = None if atom_indices is None else np.asarray(atom_indices, dtype=int)

        for trajectory in self.trajectories:
            chunk_arrays: List[np.ndarray] = []
            for chunk, _, _ in self._chunk_iterator(trajectory):
                squared = self._compute_squared_differences(
                    coords=chunk.xyz,
                    reference=reference_coords,
                    atom_indices=atom_indices_array,
                )
                chunk_arrays.append(self._apply_metric(squared, metric))
            results.append(np.concatenate(chunk_arrays) if chunk_arrays else np.array([], dtype=np.float32))

        return results

    def frame_to_frame(
        self,
        atom_indices: Sequence[int] | None,
        lag: int,
        metric: Literal["mean", "median", "mad"],
    ) -> List[np.ndarray]:
        """Compute frame-to-frame RMSD with a configurable lag.

        Coordinates are paired according to the requested lag and the robust
        metric is applied per frame to obtain RMSD trajectories.

        Parameters
        ----------
        atom_indices : Sequence[int] or None
            Atom indices used for the RMSD calculation. ``None`` selects all atoms.
        lag : int
            Distance between frames when calculating the RMSD.
        metric : {'mean', 'median', 'mad'}
            Robust aggregation metric applied to atom-wise squared deviations.

        Returns
        -------
        list of numpy.ndarray
            List containing RMSD values for each trajectory.

        Examples
        --------
        >>> topology = md.Topology()
        >>> _ = topology.add_chain()
        >>> trajectory = md.Trajectory(np.zeros((5, 1, 3), dtype=np.float32), topology)
        >>> calculator = RMSDCalculator([trajectory], chunk_size=100, use_memmap=False)
        >>> values = calculator.frame_to_frame(atom_indices=None, lag=1, metric="median")
        >>> isinstance(values[0], np.ndarray)
        True
        """

        if lag <= 0:
            raise ValueError("lag must be positive.")

        results: List[np.ndarray] = []
        atom_indices_array = None if atom_indices is None else np.asarray(atom_indices, dtype=int)

        for trajectory in self.trajectories:
            if trajectory.n_frames <= lag:
                warnings.warn(
                    "Trajectory shorter than the requested lag; returning empty RMSD array.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                results.append(np.array([], dtype=np.float32))
                continue

            pair_values: List[np.ndarray] = []
            for ref_chunk, tgt_chunk in self._iter_frame_pairs(trajectory, lag):
                squared = self._compute_squared_differences(
                    coords=tgt_chunk,
                    reference=None,
                    atom_indices=atom_indices_array,
                    reference_batch=ref_chunk,
                )
                pair_values.append(self._apply_metric(squared, metric))

            results.append(np.concatenate(pair_values) if pair_values else np.array([], dtype=np.float32))

        return results

    def window(
        self,
        atom_indices: Sequence[int] | None,
        window_size: int,
        stride: int,
        metric: Literal["mean", "median", "mad"],
        mode: str = "frame_to_start",
        lag: int | None = None,
    ) -> List[np.ndarray]:
        """Compute window-based RMSD according to the requested mode.

        Depending on ``mode`` the method either compares all frames in a window
        with the first frame or evaluates lag-separated frame pairs. The final
        window value is derived by applying the metric across frames.

        Parameters
        ----------
        atom_indices : Sequence[int] or None
            Atom indices used for the RMSD calculation. ``None`` selects all atoms.
        window_size : int
            Number of frames per window.
        stride : int
            Step size when advancing the window.
        metric : {'mean', 'median', 'mad'}
            Robust aggregation metric applied to atom-wise squared deviations and
            to the frame summary inside each window.
        mode : str, optional
            Window mode: ``"frame_to_start"`` (default) or ``"frame_to_frame"``.
        lag : int, optional
            Lag between frames when ``mode`` is ``"frame_to_frame"``. Must be positive.

        Returns
        -------
        list of numpy.ndarray
            Window-wise RMSD arrays for each trajectory.

        Examples
        --------
        >>> topology = md.Topology()
        >>> _ = topology.add_chain()
        >>> trajectory = md.Trajectory(np.zeros((5, 1, 3), dtype=np.float32), topology)
        >>> calculator = RMSDCalculator([trajectory], chunk_size=100, use_memmap=False)
        >>> windows = calculator.window(
        ...     atom_indices=None,
        ...     window_size=3,
        ...     stride=1,
        ...     metric="mad",
        ... )
        >>> windows[0].shape[0] > 0
        True
        """

        if mode not in {"frame_to_start", "frame_to_frame"}:
            raise ValueError("mode must be 'frame_to_start' or 'frame_to_frame'")

        if mode == "frame_to_frame" and lag is None:
            raise ValueError("lag must be provided for frame_to_frame mode")

        if mode == "frame_to_frame":
            return self._window_frame_to_frame(atom_indices, window_size, stride, lag, metric)
        return self._window_frame_to_start(atom_indices, window_size, stride, metric)

    def _window_frame_to_start(
        self,
        atom_indices: Sequence[int] | None,
        window_size: int,
        stride: int,
        metric: Literal["mean", "median", "mad"],
    ) -> List[np.ndarray]:
        """
        Return RMSD windows using the first frame of each window as reference.

        Parameters
        ----------
        atom_indices : Sequence[int] or None
            Atom indices used for the RMSD calculation.
        window_size : int
            Number of frames per window.
        stride : int
            Step size when advancing the window.
        metric : {'mean', 'median', 'mad'}
            Robust aggregation metric applied to atom-wise squared deviations and
            to the frame summary inside each window.

        Returns
        -------
        list of numpy.ndarray
            Window-wise RMSD arrays for each trajectory.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        results: List[np.ndarray] = []
        atom_indices_array = None if atom_indices is None else np.asarray(atom_indices, dtype=int)

        for trajectory in self.trajectories:
            if trajectory.n_frames < window_size:
                raise ValueError("window_size exceeds trajectory length.")

            window_values: List[float] = []
            for start in range(0, trajectory.n_frames - window_size + 1, stride):
                value = self._process_window_frame_to_start(
                    trajectory, start, window_size, atom_indices_array, metric
                )
                window_values.append(value)

            results.append(np.asarray(window_values, dtype=np.float32))

        return results

    def _window_frame_to_frame(
        self,
        atom_indices: Sequence[int] | None,
        window_size: int,
        stride: int,
        lag: int,
        metric: Literal["mean", "median", "mad"],
    ) -> List[np.ndarray]:
        """
        Return RMSD windows measured between lag-separated frames.

        Parameters
        ----------
        atom_indices : Sequence[int] or None
            Atom indices used for the RMSD calculation.
        window_size : int
            Number of frames per window.
        stride : int
            Step size when advancing the window.
        lag : int
            Lag between frames inside the window.
        metric : {'mean', 'median', 'mad'}
            Robust aggregation metric applied to atom-wise squared deviations and
            to the frame summary inside each window.

        Returns
        -------
        list of numpy.ndarray
            Window-wise RMSD arrays for each trajectory.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if lag <= 0:
            raise ValueError("lag must be positive.")

        results: List[np.ndarray] = []
        atom_indices_array = None if atom_indices is None else np.asarray(atom_indices, dtype=int)

        for trajectory in self.trajectories:
            if trajectory.n_frames < window_size:
                raise ValueError("window_size exceeds trajectory length.")
            if window_size <= lag:
                raise ValueError("window_size must be greater than lag.")

            window_values: List[float] = []
            for start in range(0, trajectory.n_frames - window_size + 1, stride):
                value = self._process_window_frame_to_frame(
                    trajectory, start, window_size, lag, atom_indices_array, metric
                )
                window_values.append(value)

            results.append(np.asarray(window_values, dtype=np.float32))

        return results

    def _should_chunk_window(self, window_size: int) -> bool:
        """Determine if window should be processed in chunks.

        Parameters
        ----------
        window_size : int
            Size of the window to be processed.

        Returns
        -------
        bool
            True if window should be chunked, False otherwise.
        """
        return (
            self.use_memmap
            and self.chunk_size
            and self.chunk_size > 0
            and window_size > self.chunk_size
        )

    def _process_window_frame_to_start(
        self,
        trajectory: md.Trajectory,
        window_start: int,
        window_size: int,
        atom_indices_array: np.ndarray | None,
        metric: Literal["mean", "median", "mad"],
    ) -> float:
        """Process single window comparing all frames to first frame.

        Parameters
        ----------
        trajectory : md.Trajectory
            Source trajectory.
        window_start : int
            Starting frame index for the window.
        window_size : int
            Number of frames in the window.
        atom_indices_array : np.ndarray or None
            Atom indices for RMSD calculation.
        metric : {'mean', 'median', 'mad'}
            Aggregation metric for squared deviations.

        Returns
        -------
        float
            RMSD value for the window.
        """
        if window_size <= 1:
            return 0.0

        if self._should_chunk_window(window_size):
            reference_coords = trajectory[window_start : window_start + 1].xyz[0]
            squared = self._compute_window_rmsd_chunked(
                trajectory,
                window_start + 1,
                window_size - 1,
                reference_coords,
                atom_indices_array,
            )
        else:
            window = trajectory[window_start : window_start + window_size]
            squared = self._compute_squared_differences(
                coords=window.xyz[1:],
                reference=window.xyz[0],
                atom_indices=atom_indices_array,
            )

        frame_rmsd = self._apply_metric(squared, metric)
        return self._aggregate_series(frame_rmsd, metric)

    def _process_window_frame_to_frame(
        self,
        trajectory: md.Trajectory,
        window_start: int,
        window_size: int,
        lag: int,
        atom_indices_array: np.ndarray | None,
        metric: Literal["mean", "median", "mad"],
    ) -> float:
        """Process single window with lag-separated frame pairs.

        Parameters
        ----------
        trajectory : md.Trajectory
            Source trajectory.
        window_start : int
            Starting frame index for the window.
        window_size : int
            Number of frames in the window.
        lag : int
            Distance between reference and target frames.
        atom_indices_array : np.ndarray or None
            Atom indices for RMSD calculation.
        metric : {'mean', 'median', 'mad'}
            Aggregation metric for squared deviations.

        Returns
        -------
        float
            RMSD value for the window.
        """
        if self._should_chunk_window(window_size):
            squared = self._compute_window_frame_to_frame_chunked(
                trajectory, window_start, window_size, lag, atom_indices_array
            )
        else:
            window = trajectory[window_start : window_start + window_size]
            target = window.xyz[lag:]
            reference = window.xyz[:-lag]
            squared = self._compute_squared_differences(
                coords=target,
                reference=None,
                atom_indices=atom_indices_array,
                reference_batch=reference,
            )

        frame_rmsd = self._apply_metric(squared, metric)
        return self._aggregate_series(frame_rmsd, metric)

    def _compute_window_rmsd_chunked(
        self,
        trajectory: md.Trajectory,
        window_start: int,
        window_size: int,
        reference_coords: np.ndarray,
        atom_indices_array: np.ndarray | None,
    ) -> np.ndarray:
        """Compute RMSD for large window using chunks against single reference.

        Parameters
        ----------
        trajectory : md.Trajectory
            Source trajectory.
        window_start : int
            Starting frame index for the window.
        window_size : int
            Number of frames to process.
        reference_coords : np.ndarray
            Reference coordinates for comparison.
        atom_indices_array : np.ndarray or None
            Atom indices for RMSD calculation.

        Returns
        -------
        np.ndarray
            Squared deviations for all frames in the window.
        """
        chunk_arrays: List[np.ndarray] = []

        for start in range(window_start, window_start + window_size, self.chunk_size):
            end = min(start + self.chunk_size, window_start + window_size)
            chunk = trajectory[start:end]
            squared = self._compute_squared_differences(
                coords=chunk.xyz,
                reference=reference_coords,
                atom_indices=atom_indices_array,
            )
            chunk_arrays.append(squared)

        return np.concatenate(chunk_arrays, axis=0)

    def _compute_window_frame_to_frame_chunked(
        self,
        trajectory: md.Trajectory,
        window_start: int,
        window_size: int,
        lag: int,
        atom_indices_array: np.ndarray | None,
    ) -> np.ndarray:
        """Compute RMSD for large window using chunks with lag-separated pairs.

        Parameters
        ----------
        trajectory : md.Trajectory
            Source trajectory.
        window_start : int
            Starting frame index for the window.
        window_size : int
            Number of frames in the window.
        lag : int
            Distance between reference and target frames.
        atom_indices_array : np.ndarray or None
            Atom indices for RMSD calculation.

        Returns
        -------
        np.ndarray
            Squared deviations for all lag-separated pairs in the window.
        """
        chunk_arrays: List[np.ndarray] = []

        total_pairs = window_size - lag
        for start in range(0, total_pairs, self.chunk_size):
            end = min(start + self.chunk_size, total_pairs)
            ref_start = window_start + start
            ref_end = window_start + end
            tgt_start = window_start + start + lag
            tgt_end = window_start + end + lag

            ref_chunk = trajectory[ref_start:ref_end]
            tgt_chunk = trajectory[tgt_start:tgt_end]

            squared = self._compute_squared_differences(
                coords=tgt_chunk.xyz,
                reference=None,
                atom_indices=atom_indices_array,
                reference_batch=ref_chunk.xyz,
            )
            chunk_arrays.append(squared)

        return np.concatenate(chunk_arrays, axis=0)

    @staticmethod
    def _compute_squared_differences(
        coords: np.ndarray,
        reference: np.ndarray | None,
        atom_indices: np.ndarray | None,
        reference_batch: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return squared coordinate differences for a batch of frames.

        Computes squared deviations between each frame in ``coords`` and the
        provided reference frame or batch. If ``atom_indices`` is given, only the
        selected atoms are considered.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape (n_frames, n_atoms, 3) containing frame coordinates.
        reference : np.ndarray or None
            Array of shape (n_atoms, 3) containing the reference coordinates.
            Must be provided if ``reference_batch`` is ``None``.
        atom_indices : np.ndarray or None
            Array of atom indices to select from ``coords`` and ``reference``.
            ``None`` selects all atoms.
        reference_batch : np.ndarray or None, optional
            Array of shape (n_frames, n_atoms, 3) containing reference coordinates
            for each frame in ``coords``. If provided, ``reference`` is ignored.

        Returns
        -------
        np.ndarray
            Array of shape (n_frames, n_selected_atoms) containing squared
            deviations for each frame and atom.
        """

        if reference_batch is not None:
            ref_coords = reference_batch
        else:
            if reference is None:
                raise ValueError("reference must be provided when reference_batch is None")
            ref_coords = np.broadcast_to(reference, coords.shape)

        if atom_indices is not None:
            coords = coords[:, atom_indices, :]
            ref_coords = ref_coords[:, atom_indices, :]

        diff = coords - ref_coords
        return np.sum(diff * diff, axis=-1)

    @staticmethod
    def _apply_metric(squared: np.ndarray, metric: Literal["mean", "median", "mad"]) -> np.ndarray:
        """
        Apply the requested metric to atom-wise squared deviations.
        
        Parameters
        ----------
        squared : np.ndarray
            Array of shape (n_frames, n_atoms) containing squared deviations.
        metric : {'mean', 'median', 'mad'}
            Robust aggregation metric to apply. 'mad' returns the median
            absolute deviation of atom-wise distances.
            mean and median return the respective statistic.

        Returns
        -------
        np.ndarray
            Array of shape (n_frames,) containing RMSD values.
        """

        if metric == "mad":
            distances = np.sqrt(np.clip(squared, a_min=0.0, a_max=None), dtype=np.float64)
            med = np.median(distances, axis=-1, keepdims=True)
            mad = np.median(np.abs(distances - med), axis=-1)
            return mad.astype(np.float32)

        if metric == "mean":
            values = np.mean(squared, axis=-1, dtype=np.float64)
        elif metric == "median":
            values = np.median(squared, axis=-1)
        else:
            raise ValueError(f"Unknown metric '{metric}'")

        return np.sqrt(np.clip(values, a_min=0.0, a_max=None), dtype=np.float64).astype(np.float32)

    @staticmethod
    def _aggregate_series(values: np.ndarray, metric: Literal["mean", "median", "mad"]) -> float:
        """
        Aggregate a one-dimensional RMSD series according to the metric.

        Parameters
        ----------
        values : np.ndarray
            Array of shape (n_frames,) containing RMSD values.
        metric : {'mean', 'median', 'mad'}
            Robust aggregation metric to apply.

        Returns
        -------
        float
            Aggregated RMSD value.
        """

        if values.size == 0:
            return 0.0
        if metric == "mean":
            return float(np.mean(values, dtype=np.float64))
        if metric == "median":
            return float(np.median(values))
        if metric == "mad":
            return float(np.median(np.abs(values - np.median(values))))
        
        raise ValueError(f"Unknown metric '{metric}'")

    def _chunk_iterator(
        self,
        trajectory: md.Trajectory,
    ) -> Iterator[Tuple[md.Trajectory, int, int]]:
        """
        Yield trajectory chunks respecting memmap configuration.
        
        Parameters
        ----------
        trajectory : md.Trajectory
            Source trajectory to be chunked.

        Returns
        -------
        Iterator[Tuple[md.Trajectory, int, int]]
            Iterator of tuples ``(chunk, start_frame, end_frame)``.
        """

        if self.use_memmap and self.chunk_size and self.chunk_size > 0:
            yield from StructureCalculationHelper.iterate_chunks(trajectory, self.chunk_size)
        else:
            yield trajectory, 0, trajectory.n_frames

    def _iter_frame_pairs(
        self,
        trajectory: md.Trajectory,
        lag: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield reference/target frame coordinate batches for a given lag.

        Parameters
        ----------
        trajectory : md.Trajectory
            Source trajectory providing coordinate arrays.
        lag : int
            Distance between reference and target frames.

        Returns
        -------
        Iterator[Tuple[numpy.ndarray, numpy.ndarray]]
            Iterator of coordinate arrays ``(reference, target)`` aligned by lag.
        """

        total_frames = trajectory.n_frames
        if not (self.use_memmap and self.chunk_size and self.chunk_size > 0):
            yield trajectory.xyz[:-lag], trajectory.xyz[lag:]
            return

        step = self.chunk_size
        for start in range(0, total_frames - lag, step):
            end = min(start + step, total_frames - lag)
            ref_chunk = trajectory[start:end]
            tgt_chunk = trajectory[start + lag : start + lag + (end - start)]
            yield ref_chunk.xyz, tgt_chunk.xyz

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

"""Metric-specific RMSD service implementation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Sequence, Union

import mdtraj as md
import numpy as np

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from ..calculators.rmsd_calculator import RMSDCalculator

RMSDMetric = Literal["mean", "median", "mad"]


class RMSDVariantService:
    """Expose RMSD workflows for a concrete robust metric.

    The service mediates between the pipeline-facing API and the numerical
    :class:`RMSDCalculator`. It resolves trajectory selections and atom filters,
    ensures consistent reference handling, and delegates the computation while
    the chosen robust metric stays fixed for every helper method.

    Examples
    --------
    >>> service = RMSDVariantService(pipeline_data, metric="mean")  # doctest: +SKIP
    >>> service.metric  # doctest: +SKIP
    'mean'
    """

    def __init__(self, pipeline_data: PipelineData | None, metric: RMSDMetric) -> None:
        """Store pipeline context and the requested metric.

        Ensures that valid pipeline data and a supported robust metric are
        supplied before storing both configuration values for later use.

        Parameters
        ----------
        pipeline_data : PipelineData | None
            Pipeline configuration controlling chunk size and memmap usage. Must
            not be ``None``.
        metric : RMSDMetric
            Robust aggregation metric (``"mean"``, ``"median"`` or ``"mad"``).

        Returns
        -------
        None
            This initializer returns ``None``.

        Examples
        --------
        >>> service = RMSDVariantService(pipeline_data, metric="median")  # doctest: +SKIP
        >>> service.metric  # doctest: +SKIP
        'median'
        """

        if pipeline_data is None:
            raise ValueError("RMSDVariantService requires pipeline_data")
        if metric not in {"mean", "median", "mad"}:
            raise ValueError("metric must be 'mean', 'median' or 'mad'")

        self._pipeline_data = pipeline_data
        self._metric: RMSDMetric = metric
        self._chunk_size = getattr(pipeline_data, "chunk_size", None)
        self._use_memmap = getattr(pipeline_data, "use_memmap", False)

    @property
    def metric(self) -> RMSDMetric:
        """Return the configured RMSD aggregation metric.

        Provides direct access to the robust metric configured for this service
        instance.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSDMetric
            The robust aggregation metric associated with the service instance.

        Examples
        --------
        >>> service = RMSDVariantService(pipeline_data, metric="mad")  # doctest: +SKIP
        >>> service.metric  # doctest: +SKIP
        'mad'
        """

        return self._metric

    def to_reference(
        self,
        reference_traj: int = 0,
        reference_frame: int = 0,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
    ) -> Dict[str, np.ndarray]:
        """Calculate RMSD against a fixed reference frame.

        Selects the requested trajectories, extracts the reference frame and
        delegates the numerical work to the calculator using the configured
        robust metric.

        Parameters
        ----------
        reference_traj : int, optional
            Index of the trajectory containing the reference frame. Defaults to
            ``0``.
        reference_frame : int, optional
            Frame index within the reference trajectory. Defaults to ``0``.
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing the trajectories to process. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string forwarded to the calculation. Defaults
            to ``"all"``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to RMSD arrays.

        Examples
        --------
        >>> service = RMSDVariantService(pipeline_data, metric="mean")  # doctest: +SKIP
        >>> service.to_reference()  # doctest: +SKIP
        {'traj_0': array([...])}
        """

        all_trajectories = self._pipeline_data.trajectory_data.trajectories
        if not all_trajectories:
            raise ValueError("No trajectories available for RMSD calculation.")
        if reference_traj < 0 or reference_traj >= len(all_trajectories):
            raise ValueError("Reference trajectory index is out of range.")

        traj_indices = self._resolve_trajectory_indices(traj_selection)
        if reference_traj not in traj_indices:
            raise ValueError(
                f"Reference trajectory {reference_traj} must be included in trajectory selection. "
                f"Selected trajectories: {traj_indices}"
            )

        trajectories = [all_trajectories[idx] for idx in traj_indices]
        reference_trajectory = all_trajectories[reference_traj]
        if reference_frame < 0 or reference_frame >= reference_trajectory.n_frames:
            raise ValueError("Reference frame index is out of range.")

        atom_indices = self._resolve_atom_indices(reference_trajectory, atom_selection)
        self._validate_atom_indices_for_all(trajectories, atom_selection, atom_indices)

        reference_frame_traj = reference_trajectory[reference_frame]
        calculator = self._build_calculator(trajectories)
        rmsd_arrays = calculator.rmsd_to_reference(reference_frame_traj, atom_indices, self.metric)
        return self._build_result_map(traj_indices, rmsd_arrays)

    def frame_to_frame(
        self,
        lag: int = 1,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
    ) -> Dict[str, np.ndarray]:
        """Calculate RMSD between lag-separated frames.

        Resolves the requested trajectories and atom selection, then evaluates
        RMSD values between frames separated by ``lag`` using the configured
        metric.

        Parameters
        ----------
        lag : int, optional
            Distance between frames. Must be positive. Defaults to ``1``.
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing the trajectories to process. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to RMSD arrays computed for the
            requested lag.

        Examples
        --------
        >>> service = RMSDVariantService(pipeline_data, metric="median")  # doctest: +SKIP
        >>> service.frame_to_frame(lag=1)  # doctest: +SKIP
        {'traj_0': array([...])}
        """
        # Validate parameters
        if lag <= 0:
            raise ValueError("lag must be positive")

        traj_indices = self._resolve_trajectory_indices(traj_selection)
        trajectories = [self._pipeline_data.trajectory_data.trajectories[idx] for idx in traj_indices]
        atom_indices = self._resolve_atom_indices(trajectories[0], atom_selection)
        self._validate_atom_indices_for_all(trajectories, atom_selection, atom_indices)

        # Validate lag against trajectory lengths
        min_trajectory_length = min(traj.n_frames for traj in trajectories)
        if lag >= min_trajectory_length:
            raise ValueError(
                f"lag ({lag}) exceeds trajectory length ({min_trajectory_length})"
            )

        calculator = self._build_calculator(trajectories)
        rmsd_arrays = calculator.frame_to_frame(atom_indices, lag, self.metric)
        return self._build_result_map(traj_indices, rmsd_arrays)

    def window_frame_to_start(
        self,
        window_size: int,
        stride: int | None = None,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
    ) -> Dict[str, np.ndarray]:
        """Calculate window-wise RMSD using the first frame as reference.

        Sliding windows are extracted per trajectory, aligned to the window's
        first frame and condensed via the variant's metric into a single RMSD
        value per window.

        Parameters
        ----------
        window_size : int
            Number of frames within each window.
        stride : int, optional
            Step size between windows. Defaults to ``window_size`` when omitted.
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing the trajectories to process. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to window-wise RMSD arrays.

        Examples
        --------
        >>> service = RMSDVariantService(pipeline_data, metric="mad")  # doctest: +SKIP
        >>> service.window_frame_to_start(window_size=3)  # doctest: +SKIP
        {'traj_0': array([...])}
        """
        # Validate parameters
        self._validate_window_parameters(window_size=window_size, stride=stride)

        effective_stride = stride or window_size
        traj_indices = self._resolve_trajectory_indices(traj_selection)
        trajectories = [self._pipeline_data.trajectory_data.trajectories[idx] for idx in traj_indices]
        atom_indices = self._resolve_atom_indices(trajectories[0], atom_selection)
        self._validate_atom_indices_for_all(trajectories, atom_selection, atom_indices)

        # Validate window size against trajectory lengths
        min_trajectory_length = min(traj.n_frames for traj in trajectories)
        if window_size > min_trajectory_length:
            raise ValueError(
                f"window_size ({window_size}) exceeds trajectory length ({min_trajectory_length})"
            )

        calculator = self._build_calculator(trajectories)
        rmsd_arrays = calculator.window(
            atom_indices,
            window_size,
            effective_stride,
            self.metric,
            mode="frame_to_start",
        )
        return self._build_result_map(traj_indices, rmsd_arrays)

    def window_frame_to_frame(
        self,
        window_size: int,
        stride: int | None = None,
        lag: int = 1,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
    ) -> Dict[str, np.ndarray]:
        """Calculate window-wise RMSD between lag-separated frames.

        Sliding windows are processed with lag-separated frame pairs and
        condensed using the variant's robust metric.

        Parameters
        ----------
        window_size : int
            Number of frames within each window.
        stride : int, optional
            Step size between windows. Defaults to ``window_size`` when omitted.
        lag : int, optional
            Lag between frames inside each window. Defaults to ``1``.
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing the trajectories to process. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to window-wise RMSD arrays.

        Examples
        --------
        >>> service = RMSDVariantService(pipeline_data, metric="mean")  # doctest: +SKIP
        >>> service.window_frame_to_frame(window_size=4, lag=1)  # doctest: +SKIP
        {'traj_0': array([...])}
        """
        # Validate parameters
        self._validate_window_parameters(window_size=window_size, stride=stride, lag=lag)

        effective_stride = stride or window_size
        traj_indices = self._resolve_trajectory_indices(traj_selection)
        trajectories = [self._pipeline_data.trajectory_data.trajectories[idx] for idx in traj_indices]
        atom_indices = self._resolve_atom_indices(trajectories[0], atom_selection)
        self._validate_atom_indices_for_all(trajectories, atom_selection, atom_indices)

        # Validate window size and lag against trajectory lengths
        min_trajectory_length = min(traj.n_frames for traj in trajectories)
        if window_size > min_trajectory_length:
            raise ValueError(
                f"window_size ({window_size}) exceeds trajectory length ({min_trajectory_length})"
            )
        if lag >= min_trajectory_length:
            raise ValueError(
                f"lag ({lag}) exceeds trajectory length ({min_trajectory_length})"
            )

        calculator = self._build_calculator(trajectories)
        rmsd_arrays = calculator.window(
            atom_indices,
            window_size,
            effective_stride,
            self.metric,
            mode="frame_to_frame",
            lag=lag,
        )
        return self._build_result_map(traj_indices, rmsd_arrays)

    def _build_calculator(self, trajectories: Iterable) -> RMSDCalculator:
        """Instantiate an :class:`RMSDCalculator` for the given trajectories.

        Creates the calculator shared across helper methods so that streaming
        configuration is reused for every computation.

        Parameters
        ----------
        trajectories : Iterable
            Collection of trajectory-like objects to be processed.

        Returns
        -------
        RMSDCalculator
            Calculator configured with the shared runtime settings.
        """

        return RMSDCalculator(list(trajectories), self._chunk_size, self._use_memmap)

    def _resolve_trajectory_indices(
        self,
        selection: Union[int, str, List[Union[int, str]], "all"],
    ) -> List[int]:
        """Resolve user selections into trajectory indices.

        Converts user-provided selection specifications into concrete trajectory
        indices understood by the calculator.

        Parameters
        ----------
        selection : Union[int, str, list[Union[int, str]], 'all']
            Selection describing which trajectories to resolve.

        Returns
        -------
        list[int]
            Trajectory indices matching the selection.
        """

        indices = self._pipeline_data.trajectory_data.get_trajectory_indices(selection)
        if not indices:
            raise ValueError("No trajectories found for the requested selection.")
        return indices

    def _resolve_atom_indices(self, trajectory: md.Trajectory, selection: str) -> np.ndarray | None:
        """Return MDTraj atom indices for the requested selection.

        Translates an MDTraj atom selection string into a concrete array of
        indices. When ``selection`` is ``"all"`` the method returns ``None`` to
        signal that the full atom set should be used.

        Parameters
        ----------
        trajectory : md.Trajectory
            Trajectory used to interpret the selection.
        selection : str
            MDTraj atom selection string.

        Returns
        -------
        numpy.ndarray or None
            Array of atom indices or ``None`` when all atoms are requested.
        """

        if selection == "all":
            return None

        indices = trajectory.topology.select(selection)
        if indices.size == 0:
            raise ValueError(f"Atom selection '{selection}' produced no atoms.")
        return indices

    def _validate_atom_indices_for_all(
        self,
        trajectories: Sequence[md.Trajectory],
        selection: str,
        reference_indices: np.ndarray | None,
    ) -> None:
        """Ensure the atom selection resolves consistently across trajectories.

        Checks that the atom selection yields the same number of atoms for every
        trajectory after the initial reference selection.

        Parameters
        ----------
        trajectories : Sequence[md.Trajectory]
            Trajectories that must share the same atom selection cardinality.
        selection : str
            MDTraj selection string applied to every trajectory.
        reference_indices : numpy.ndarray or None
            Atom indices obtained from the reference trajectory.

        Returns
        -------
        None
            This validator returns ``None``.
        """

        if selection == "all":
            return

        expected = int(reference_indices.size if reference_indices is not None else 0)
        for trajectory in trajectories[1:]:
            indices = trajectory.topology.select(selection)
            if indices.size != expected:
                raise ValueError(
                    "Atom selection results differ between trajectories: "
                    f"expected {expected}, got {indices.size}."
                )

    def _validate_window_parameters(
        self,
        window_size: int,
        stride: int | None = None,
        lag: int | None = None,
    ) -> None:
        """Validate window analysis parameters.

        Ensures that window_size, stride, and lag are positive integers
        as required by the calculator.

        Parameters
        ----------
        window_size : int
            Size of the sliding window. Must be positive.
        stride : int, optional
            Step size for window advancement. Must be positive if provided.
        lag : int, optional
            Frame offset for frame-to-frame comparisons. Must be positive if provided.

        Returns
        -------
        None
            This validator returns ``None``.

        Raises
        ------
        ValueError
            If any parameter is zero or negative.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")

        if lag is not None and lag <= 0:
            raise ValueError("lag must be positive")

    def _build_result_map(
        self,
        indices: Sequence[int],
        arrays: Sequence[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Create a mapping from trajectory names to result arrays.

        Couples the computed RMSD arrays with the corresponding trajectory names
        for convenient downstream consumption.

        Parameters
        ----------
        indices : Sequence[int]
            Trajectory indices processed by the calculator.
        arrays : Sequence[numpy.ndarray]
            RMSD arrays produced by the calculator for each trajectory.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to RMSD arrays.
        """

        names = self._pipeline_data.trajectory_data.trajectory_names
        result: Dict[str, np.ndarray] = {}
        for idx, values in zip(indices, arrays):
            name = names[idx] if idx < len(names) else f"trajectory_{idx}"
            result[name] = values
        return result

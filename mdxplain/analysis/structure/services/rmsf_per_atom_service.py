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

"""Per-atom RMSF service variant."""

from __future__ import annotations

from typing import Dict, List, Literal, Union

import numpy as np
import mdtraj as md

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from ..calculators.rmsf_calculator import RMSFCalculator
from ..helper.trajectory_service_helper import TrajectoryServiceHelper


class RMSFPerAtomService:
    """Provide per-atom RMSF computations for a fixed metric variant.

    Resolves trajectory selections and atom filters before delegating the
    numerical calculations to :class:`RMSFCalculator`. The deviation metric is
    fixed during construction while the reference strategy remains selectable
    per method call.

    Returns
    -------
    RMSFPerAtomService
        Service exposing helper to compute per-atom RMSF values for the stored
        metric.

    Examples
    --------
    >>> service = RMSFPerAtomService(pipeline_data, metric="mean")
    >>> isinstance(service, RMSFPerAtomService)
    True
    """

    def __init__(
        self,
        pipeline_data: PipelineData | None,
        metric: Literal["mean", "median", "mad"],
    ) -> None:
        """Store pipeline settings and the robust deviation metric.

        Validates presence of pipeline data and records the chosen metric so all
        subsequent computations use consistent aggregation rules.

        Parameters
        ----------
        pipeline_data : PipelineData | None
            Pipeline context carrying chunking and memmap configuration.
        metric : {'mean', 'median', 'mad'}
            Robust deviation metric (``"mean"``, ``"median"``, or ``"mad"``).

        Returns
        -------
        None
            The initializer does not return anything.

        Examples
        --------
        >>> service = RMSFPerAtomService(pipeline_data, metric="mean")
        >>> service.metric
        'mean'
        """

        if pipeline_data is None:
            raise ValueError("RMSFPerAtomService requires pipeline_data")

        # Validate metric
        valid_metrics = ["mean", "median", "mad"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        self._pipeline_data = pipeline_data
        self._metric = metric
        self._helper = TrajectoryServiceHelper(pipeline_data)

    def __call__(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Shortcut for :meth:`to_mean_reference` on this service.

        Calls :meth:`to_mean_reference` with the provided parameters for a
        concise mean RMSF workflow.

        So its the basic RMSF for atoms.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.
        cross_trajectory : bool, optional
            Combine all selected trajectories into a single RMSF profile when
            ``True``. Defaults to ``False``.

        Returns
        -------
        dict
            Mapping of trajectory names to per-atom RMSF arrays.

        Examples
        --------
        >>> service = RMSFPerAtomService(pipeline_data, metric="mean")
        >>> service(traj_selection="all", atom_selection="all")  # doctest: +ELLIPSIS
        {...}
        """

        return self.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
        )

    def _compute_per_atom_rmsf(
        self,
        reference_mode: Literal["mean", "median"],
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute per-atom RMSF with specified reference mode.

        Common implementation for both mean and median reference structures.

        Parameters
        ----------
        reference_mode : {'mean', 'median'}
            Strategy to construct reference structure.
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse.
        atom_selection : str, optional
            MDTraj atom selection string.
        cross_trajectory : bool, optional
            Combine trajectories into single RMSF profile.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to RMSF arrays.
        """
        # Resolve trajectories and atoms using helper
        trajectories, trajectory_names, atom_indices = (
            self._helper.resolve_trajectories_and_atoms(traj_selection, atom_selection)
        )

        # Validate resolved trajectories and parameters
        if not trajectories:
            raise ValueError("No trajectories found for the given selection")

        # Validate all trajectories have frames
        for i, traj in enumerate(trajectories):
            if traj.n_frames == 0:
                raise ValueError(f"Trajectory '{trajectory_names[i]}' contains no frames")

        # Validate atom selection consistency across trajectories (only for cross_trajectory=True)
        if cross_trajectory:
            self._validate_atom_indices_for_all(trajectories, atom_selection, atom_indices)

        # Create calculator with resolved trajectories
        calculator = RMSFCalculator(
            trajectories=trajectories,
            chunk_size=self._pipeline_data.chunk_size,
            use_memmap=self._pipeline_data.use_memmap,
            trajectory_names=trajectory_names,
        )

        # Calculate RMSF
        rmsf_arrays = calculator.calculate_per_atom(
            reference_mode=reference_mode,
            metric=self._metric,
            atom_indices=atom_indices,
            cross_trajectory=cross_trajectory,
        )

        # Build result mapping using helper
        return self._helper.build_result_map(
            trajectory_names, rmsf_arrays, cross_trajectory
        )

    def to_mean_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute per-atom RMSF relative to the mean structure.

        Derives a mean reference structure across the selected trajectories and
        returns per-atom RMSF values using the variant metric.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.
        cross_trajectory : bool, optional
            Combine all selected trajectories into a single RMSF profile when
            ``True``. Defaults to ``False``.

        Returns
        -------
        dict
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-atom RMSF arrays.

        Examples
        --------
        >>> service = RMSFPerAtomService(pipeline_data, metric="median")
        >>> result = service.to_mean_reference()  # doctest: +ELLIPSIS
        {...}
        """
        return self._compute_per_atom_rmsf(
            "mean", traj_selection, atom_selection, cross_trajectory
        )

    def to_median_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute per-atom RMSF relative to the median structure.

        Uses a median reference structure across the selected trajectories and
        returns per-atom RMSF values using the configured metric.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.
        cross_trajectory : bool, optional
            Combine all selected trajectories into a single RMSF profile when
            ``True``. Defaults to ``False``.

        Returns
        -------
        dict
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-atom RMSF arrays.

        Examples
        --------
        >>> service = RMSFPerAtomService(pipeline_data, metric="mad")
        >>> result = service.to_median_reference()  # doctest: +ELLIPSIS
        {...}
        """
        return self._compute_per_atom_rmsf(
            "median", traj_selection, atom_selection, cross_trajectory
        )

    def _validate_atom_indices_for_all(
        self,
        trajectories: List[md.Trajectory],
        selection: str,
        reference_indices: np.ndarray | None,
    ) -> None:
        """Ensure the atom selection resolves consistently across trajectories.

        Checks that the atom selection yields the same number of atoms for every
        trajectory after the initial reference selection.

        Parameters
        ----------
        trajectories : list[md.Trajectory]
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

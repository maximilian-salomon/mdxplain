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

"""Per-residue RMSF service variant."""

from __future__ import annotations

from typing import Dict, List, Literal, Union

import numpy as np
import mdtraj as md

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from ..calculators.rmsf_calculator import RMSFCalculator
from ..helper.trajectory_service_helper import TrajectoryServiceHelper


class RMSFPerResidueService:
    """Provide residue-wise RMSF computations for a fixed metric and aggregator.

    The service resolves trajectory selections, derives atom subsets and
    delegates residue-level RMSF calculations to :class:`RMSFCalculator` using a
    fixed deviation metric and residue aggregator.
    """

    def __init__(
        self,
        pipeline_data: PipelineData | None,
        metric: Literal["mean", "median", "mad"],
        aggregator: Literal["mean", "median", "rms", "rms_median"],
    ) -> None:
        """Store pipeline settings, deviation metric and residue aggregator.

        Ensures pipeline data is available and records both the deviation metric
        and residue aggregator for subsequent computations.

        Parameters
        ----------
        pipeline_data : PipelineData, optional
            Pipeline context carrying chunking and memmap configuration.
        metric : {'mean', 'median', 'mad'}
            Robust deviation metric applied to squared deviations.
        aggregator : {'mean', 'median', 'rms', 'rms_median'}
            Aggregator used when condensing per-atom values to residue level.

        Returns
        -------
        None
            The initializer does not return anything.

        Examples
        --------
        >>> service = RMSFPerResidueService(pipeline_data, metric="mean", aggregator="median")
        >>> service._aggregator
        'median'
        """

        if pipeline_data is None:
            raise ValueError("RMSFPerResidueService requires pipeline_data")

        # Validate metric
        valid_metrics = ["mean", "median", "mad"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        # Validate aggregator
        valid_aggregators = ["mean", "median", "rms", "rms_median"]
        if aggregator not in valid_aggregators:
            raise ValueError(f"aggregator must be one of {valid_aggregators}, got '{aggregator}'")

        self._pipeline_data = pipeline_data
        self._metric = metric
        self._aggregator = aggregator
        self._helper = TrajectoryServiceHelper(pipeline_data)

    def to_mean_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: (
            Union[int, str, List[Union[int, str]], "all"] | None
        ) = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-residue RMSF relative to the mean structure.

        Derives a mean reference structure across the selected trajectories and
        condenses per-atom RMSF values to residue level using the configured
        aggregator. Optional parameters allow combining trajectories into a
        single RMSF profile and choosing which trajectory supplies the topology
        for residue grouping.

        Parameters
        ----------
        traj_selection : Union[int, str, List[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.

        cross_trajectory : bool, optional
            When ``True``, all selected trajectories are combined into a single
            RMSF profile before residue aggregation.
        reference_traj_selection : optional
            Selection describing which trajectory provides the topology for
            residue aggregation. ``None`` defaults to the first trajectory from
            ``traj_selection``.

        Returns
        -------
        dict
            Mapping of trajectory names – or ``'combined'`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> service = RMSFPerResidueService(pipeline_data, metric="mean", aggregator="mean")
        >>> result = service.to_mean_reference()
        >>> isinstance(result, dict)
        True
        """
        return self._compute_per_residue_rmsf(
            "mean", traj_selection, atom_selection, cross_trajectory, reference_traj_selection
        )

    def _compute_per_residue_rmsf(
        self,
        reference_mode: Literal["mean", "median"],
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: (
            Union[int, str, List[Union[int, str]], "all"] | None
        ) = None,
    ) -> Dict[str, np.ndarray]:
        """Compute per-residue RMSF with specified reference mode.

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
        reference_traj_selection : optional
            Selection describing which trajectory provides the topology for
            residue aggregation.

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

        # Resolve reference trajectory index
        reference_idx = 0
        if reference_traj_selection is not None:
            reference_idx = self._helper.map_reference_to_local_index(
                traj_selection, reference_traj_selection
            )

        # Create calculator with resolved trajectories
        calculator = RMSFCalculator(
            trajectories=trajectories,
            chunk_size=self._pipeline_data.chunk_size,
            use_memmap=self._pipeline_data.use_memmap,
            trajectory_names=trajectory_names,
        )

        # Calculate RMSF
        rmsf_arrays = calculator.calculate_per_residue(
            reference_mode=reference_mode,
            metric=self._metric,
            residue_aggregator=self._aggregator,
            atom_indices=atom_indices,
            cross_trajectory=cross_trajectory,
            reference_trajectory_index=reference_idx,
        )

        # Build result mapping using helper
        return self._helper.build_result_map(
            trajectory_names, rmsf_arrays, cross_trajectory
        )

    def to_median_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: (
            Union[int, str, List[Union[int, str]], "all"] | None
        ) = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-residue RMSF relative to the median structure.

        Uses the median reference structure for the selected trajectories and
        aggregates per-atom RMSF values to residue level. Optional parameters
        mirror :meth:`to_mean_reference` for cross-trajectory computation and
        reference topology selection.

        Parameters
        ----------
        traj_selection : Union[int, str, List[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse.
        atom_selection : str, optional
            MDTraj atom selection string. Defaults to ``"all"``.

        cross_trajectory : bool, optional
            When ``True``, all selected trajectories are combined into a single
            RMSF profile before residue aggregation.
        reference_traj_selection : optional
            Selection describing which trajectory provides the topology for
            residue aggregation. ``None`` defaults to the first trajectory from
            ``traj_selection``.

        Returns
        -------
        dict
            Mapping of trajectory names – or ``'combined'`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> service = RMSFPerResidueService(pipeline_data, metric="median", aggregator="rms")
        >>> result = service.to_median_reference()
        >>> isinstance(result, dict)
        True
        """
        return self._compute_per_residue_rmsf(
            "median", traj_selection, atom_selection, cross_trajectory, reference_traj_selection
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

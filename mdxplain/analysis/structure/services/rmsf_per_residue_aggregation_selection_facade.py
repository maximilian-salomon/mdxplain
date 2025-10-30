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

"""Facade exposing per-residue RMSF aggregation selection."""

from __future__ import annotations

from typing import Dict, List, Literal, Union

import numpy as np

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsf_per_residue_mean_agg_facade import RMSFPerResidueMeanAggFacade
from .rmsf_per_residue_median_agg_facade import RMSFPerResidueMedianAggFacade
from .rmsf_per_residue_rms_agg_facade import RMSFPerResidueRmsAggFacade
from .rmsf_per_residue_rms_median_agg_facade import RMSFPerResidueRmsMedianAggFacade


class RMSFPerResidueAggregationSelectionFacade:
    """Expose per-residue RMSF aggregator selection for a fixed metric.

    Provides access to all residue-level aggregation strategies (mean, median,
    RMS, RMS-median) while preserving the metric context from the parent
    variant facade. The metric is fixed during construction to enable proper
    type inference for autocomplete.

    Returns
    -------
    RMSFPerResidueAggregationSelectionFacade
        Facade exposing all aggregation options for the configured metric.

    Examples
    --------
    >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
    >>> isinstance(facade, RMSFPerResidueAggregationSelectionFacade)
    True
    """

    def __init__(
        self, pipeline_data: PipelineData | None, metric: Literal["mean", "median", "mad"]
    ) -> None:
        """Store pipeline context and metric for aggregator facades.

        Validates the pipeline context and captures the deviation metric that
        will be passed to all aggregator facades.

        Parameters
        ----------
        pipeline_data : PipelineData | None
            Pipeline context injected by the analysis manager. Must not be ``None``.
        metric : {'mean', 'median', 'mad'}
            Robust deviation metric passed to all aggregator facades.

        Returns
        -------
        None
            The initializer does not return anything.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
        >>> facade._metric
        'mean'
        """

        if pipeline_data is None:
            raise ValueError(
                "RMSFPerResidueAggregationSelectionFacade requires pipeline_data"
            )

        self._pipeline_data = pipeline_data
        self._metric = metric

    @property
    def with_mean_aggregation(self) -> RMSFPerResidueMeanAggFacade:
        """Access residue RMSF facade with mean aggregation.

        Returns the facade providing residue-level RMSF calculations using
        arithmetic mean aggregation for the configured metric.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFPerResidueMeanAggFacade
            Facade exposing mean aggregation with the configured metric.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
        >>> agg_facade = facade.with_mean_aggregation
        >>> isinstance(agg_facade, RMSFPerResidueMeanAggFacade)
        True
        """
        return RMSFPerResidueMeanAggFacade(self._pipeline_data, self._metric)

    @property
    def with_median_aggregation(self) -> RMSFPerResidueMedianAggFacade:
        """Access residue RMSF facade with median aggregation.

        Returns the facade providing residue-level RMSF calculations using
        median aggregation for the configured metric.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFPerResidueMedianAggFacade
            Facade exposing median aggregation with the configured metric.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
        >>> agg_facade = facade.with_median_aggregation
        >>> isinstance(agg_facade, RMSFPerResidueMedianAggFacade)
        True
        """
        return RMSFPerResidueMedianAggFacade(self._pipeline_data, self._metric)

    @property
    def with_rms_aggregation(self) -> RMSFPerResidueRmsAggFacade:
        """Access residue RMSF facade with RMS aggregation.

        Returns the facade providing residue-level RMSF calculations using
        root-mean-square aggregation for the configured metric.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFPerResidueRmsAggFacade
            Facade exposing RMS aggregation with the configured metric.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
        >>> agg_facade = facade.with_rms_aggregation
        >>> isinstance(agg_facade, RMSFPerResidueRmsAggFacade)
        True
        """
        return RMSFPerResidueRmsAggFacade(self._pipeline_data, self._metric)

    @property
    def with_rms_median_aggregation(self) -> RMSFPerResidueRmsMedianAggFacade:
        """Access residue RMSF facade with RMS-median aggregation.

        Returns the facade providing residue-level RMSF calculations using
        RMS-median aggregation for the configured metric.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFPerResidueRmsMedianAggFacade
            Facade exposing RMS-median aggregation with the configured metric.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
        >>> agg_facade = facade.with_rms_median_aggregation
        >>> isinstance(agg_facade, RMSFPerResidueRmsMedianAggFacade)
        True
        """
        return RMSFPerResidueRmsMedianAggFacade(self._pipeline_data, self._metric)

    def to_mean_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: Union[int, str, List[Union[int, str]], "all"] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate residue RMSF values relative to the mean structure.

        Delegates to :meth:`RMSFPerResidueMeanAggFacade.to_mean_reference` using
        the mean aggregation facade while forwarding optional cross-trajectory and
        reference-topology parameters.

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
        reference_traj_selection : Union[int, str, list[Union[int, str]], 'all'] | None, optional
            Selection describing which trajectory provides the topology for
            residue aggregation. ``None`` defaults to the first trajectory from
            ``traj_selection``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
        >>> facade.to_mean_reference(cross_trajectory=True)  # doctest: +ELLIPSIS
        {...}
        """
        return self.with_mean_aggregation.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
            reference_traj_selection=reference_traj_selection,
        )

    def __call__(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: Union[int, str, List[Union[int, str]], "all"] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Shortcut for to_mean_reference.

        Calls :meth:`to_mean_reference` with the provided parameters to execute
        the mean aggregation workflow via call syntax. Calculates the per-residue
        RMSF relative to the mean structure using mean aggregation.

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
        reference_traj_selection : Union[int, str, list[Union[int, str]], 'all'] | None, optional
            Selection describing which trajectory provides the topology for
            residue aggregation. ``None`` defaults to the first trajectory from
            ``traj_selection``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationSelectionFacade(pipeline_data, "mean")
        >>> facade(traj_selection="all", atom_selection="all")  # doctest: +ELLIPSIS
        {...}
        """
        return self.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
            reference_traj_selection=reference_traj_selection,
        )

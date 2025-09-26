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

"""Facade exposing residue-level RMSF aggregations."""

from __future__ import annotations

from typing import Dict, List, Union
import numpy as np
from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsf_per_residue_service import RMSFPerResidueService


class RMSFPerResidueAggregationFacade:
    """Expose residue-level RMSF services for different aggregators.

    Lazily constructs :class:`RMSFPerResidueService` instances for all supported
    residue aggregators while reusing the shared pipeline context and deviation
    metric supplied by the parent variant facade.
    """

    def __init__(self, pipeline_data: PipelineData | None, metric: str) -> None:
        """Store pipeline context and deviation metric for residue services.

        Validates the pipeline context and captures the deviation metric so all
        residue aggregations share consistent settings.

        Parameters
        ----------
        pipeline_data : PipelineData, optional
            Pipeline context injected by the analysis manager. Must not be ``None``.
        metric : str
            Robust deviation metric (``"mean"``, ``"median"`` or ``"mad"``).

        Returns
        -------
        None
            The initializer does not return anything.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="mean")
        >>> isinstance(facade.with_mean_aggregation, RMSFPerResidueService)
        True
        """

        if pipeline_data is None:
            raise ValueError("RMSFPerResidueAggregationFacade requires pipeline_data")

        self._pipeline_data = pipeline_data
        self._metric = metric

    @property
    def with_mean_aggregation(self) -> RMSFPerResidueService:
        """Return the residue service using arithmetic mean aggregation.

        Lazily instantiates and caches the residue service that averages
        atom-level RMSF values via the arithmetic mean.

        Parameters
        ----------
        None
            The property accepts no parameters.

        Returns
        -------
        RMSFPerResidueService
            Service bound to the ``mean`` residue aggregator.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="median")
        >>> service = facade.with_mean_aggregation
        >>> service.aggregator
        'mean'
        """

        return self._get_service("mean")

    @property
    def with_median_aggregation(self) -> RMSFPerResidueService:
        """Return the residue service using median aggregation.

        Exposes the residue service that aggregates atom-level RMSF values via
        the statistical median.

        Parameters
        ----------
        None
            The property accepts no parameters.

        Returns
        -------
        RMSFPerResidueService
            Service bound to the ``median`` residue aggregator.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="mean")
        >>> service = facade.with_median_aggregation
        >>> service.aggregator
        'median'
        """

        return self._get_service("median")

    @property
    def with_rms_aggregation(self) -> RMSFPerResidueService:
        """Return the residue service using RMS aggregation.

        Provides the residue service that applies a root-mean-square
        aggregation to atom-level RMSF values.

        Parameters
        ----------
        None
            The property accepts no parameters.

        Returns
        -------
        RMSFPerResidueService
            Service bound to the ``rms`` residue aggregator.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="mad")
        >>> service = facade.with_rms_aggregation
        >>> service.aggregator
        'rms'
        """

        return self._get_service("rms")

    @property
    def with_rms_median_aggregation(self) -> RMSFPerResidueService:
        """Return the residue service using RMS-median aggregation.

        Provides the residue service that applies the square root of the median
        squared deviations to atom-level RMSF values.

        Parameters
        ----------
        None
            The property accepts no parameters.

        Returns
        -------
        RMSFPerResidueService
            Service bound to the ``rms_median`` residue aggregator.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="mean")
        >>> service = facade.with_rms_median_aggregation
        >>> service.aggregator
        'rms_median'
        """

        return self._get_service("rms_median")

    def _get_service(self, aggregator: str) -> RMSFPerResidueService:
        """
        Return cached residue service for the given aggregator.

        Retrieves an existing residue service from the cache or instantiates a
        new one when the aggregator is requested for the first time.

        Parameters
        ----------
        aggregator : str
            Name of the residue-level aggregation strategy
            (``"mean"``, ``"median"``, ``"rms"``, ``"rms_median"``).

        Returns
        -------
        RMSFPerResidueService
            Residue-level RMSF service configured for the requested aggregator.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="mean")
        >>> isinstance(facade._get_service("mean"), RMSFPerResidueService)
        True
        """

        return RMSFPerResidueService(
            self._pipeline_data,
            metric=self._metric,
            aggregator=aggregator,
        )

    def to_mean_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: Union[int, str, List[Union[int, str]], "all"] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate residue RMSF values relative to the mean structure.

        Delegates to :meth:`RMSFPerResidueService.to_mean_reference` using the
        mean aggregation service while forwarding optional cross-trajectory and
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
            ``True``.
        reference_traj_selection : optional
            Selection describing which trajectory provides the topology for
            residue aggregation. ``None`` defaults to the first trajectory from
            ``traj_selection``.

        Returns
        -------
        dict
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="mean")
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
        """
        Shortcut for :meth:`to_mean_reference`.

        Calls :meth:`to_mean_reference` with the provided parameters to execute
        the mean aggregation workflow via call syntax.

        So it is basically calculates the per-residue RMSF relative to the mean
        structure.

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
        reference_traj_selection : optional
            Selection describing which trajectory provides the topology for
            residue aggregation. ``None`` defaults to the first trajectory from
            ``traj_selection``.

        Returns
        -------
        dict
            Mapping of trajectory names to per-residue RMSF arrays.

        Examples
        --------
        >>> facade = RMSFPerResidueAggregationFacade(pipeline_data, metric="mean")
        >>> facade(traj_selection="all", atom_selection="all")  # doctest: +ELLIPSIS
        {...}
        """

        return self.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
            reference_traj_selection=reference_traj_selection,
        )

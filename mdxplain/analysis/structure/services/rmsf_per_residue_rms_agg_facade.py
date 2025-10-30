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

"""Facade exposing per-residue RMSF service with RMS aggregation."""

from __future__ import annotations

from typing import Literal

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsf_per_residue_base_agg_facade import BaseRMSFPerResidueAggFacade
from .rmsf_per_residue_service import RMSFPerResidueService


class RMSFPerResidueRmsAggFacade(BaseRMSFPerResidueAggFacade):
    """Expose per-residue RMSF service with RMS aggregation.

    Provides access to :class:`RMSFPerResidueService` configured with RMS
    aggregation for residue-level RMSF calculations. The aggregation type is
    fixed while the metric is provided during construction.

    Returns
    -------
    RMSFPerResidueRmsAggFacade
        Facade exposing RMS aggregation service.

    Examples
    --------
    >>> facade = RMSFPerResidueRmsAggFacade(pipeline_data, "mean")
    >>> isinstance(facade, RMSFPerResidueRmsAggFacade)
    True
    """

    def __init__(
        self, pipeline_data: PipelineData | None, metric: Literal["mean", "median", "mad"]
    ) -> None:
        """Store pipeline context and metric for residue service.

        Validates the pipeline context and captures the deviation metric. The
        aggregation type is hard-coded to rms for this facade.

        Parameters
        ----------
        pipeline_data : PipelineData | None
            Pipeline context injected by the analysis manager. Must not be ``None``.
        metric : {'mean', 'median', 'mad'}
            Robust deviation metric passed to the residue service.

        Returns
        -------
        None
            The initializer does not return anything.

        Examples
        --------
        >>> facade = RMSFPerResidueRmsAggFacade(pipeline_data, "mean")
        >>> facade._aggregator
        'rms'
        """

        if pipeline_data is None:
            raise ValueError("RMSFPerResidueRmsAggFacade requires pipeline_data")

        self._pipeline_data = pipeline_data
        self._metric = metric
        self._aggregator = "rms"

    @property
    def service(self) -> RMSFPerResidueService:
        """Return the residue service using RMS aggregation.

        Provides the residue service that applies a root-mean-square aggregation
        to atom-level RMSF values with the configured deviation metric.

        Parameters
        ----------
        None
            The property accepts no parameters.

        Returns
        -------
        RMSFPerResidueService
            Service configured with RMS aggregation and the supplied metric.

        Examples
        --------
        >>> facade = RMSFPerResidueRmsAggFacade(pipeline_data, "mad")
        >>> isinstance(facade.service, RMSFPerResidueService)
        True
        """

        return RMSFPerResidueService(
            self._pipeline_data,
            metric=self._metric,
            aggregator=self._aggregator,
        )

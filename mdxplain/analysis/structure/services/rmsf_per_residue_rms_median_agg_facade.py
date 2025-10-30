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

"""Facade exposing per-residue RMSF service with RMS-median aggregation."""

from __future__ import annotations

from typing import Literal

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsf_per_residue_base_agg_facade import BaseRMSFPerResidueAggFacade
from .rmsf_per_residue_service import RMSFPerResidueService


class RMSFPerResidueRmsMedianAggFacade(BaseRMSFPerResidueAggFacade):
    """Expose per-residue RMSF service with RMS-median aggregation.

    Provides access to :class:`RMSFPerResidueService` configured with RMS-median
    aggregation for residue-level RMSF calculations. The aggregation type is
    fixed while the metric is provided during construction.

    Returns
    -------
    RMSFPerResidueRmsMedianAggFacade
        Facade exposing RMS-median aggregation service.

    Examples
    --------
    >>> facade = RMSFPerResidueRmsMedianAggFacade(pipeline_data, "mean")
    >>> isinstance(facade, RMSFPerResidueRmsMedianAggFacade)
    True
    """

    def __init__(
        self, pipeline_data: PipelineData | None, metric: Literal["mean", "median", "mad"]
    ) -> None:
        """Store pipeline context and metric for residue service.

        Validates the pipeline context and captures the deviation metric. The
        aggregation type is hard-coded to rms_median for this facade.

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
        >>> facade = RMSFPerResidueRmsMedianAggFacade(pipeline_data, "mean")
        >>> facade._aggregator
        'rms_median'
        """

        if pipeline_data is None:
            raise ValueError("RMSFPerResidueRmsMedianAggFacade requires pipeline_data")

        self._pipeline_data = pipeline_data
        self._metric = metric
        self._aggregator = "rms_median"

    @property
    def service(self) -> RMSFPerResidueService:
        """Return the residue service using RMS-median aggregation.

        Provides the residue service that applies the square root of the median
        squared deviations to atom-level RMSF values with the configured deviation
        metric.

        Parameters
        ----------
        None
            The property accepts no parameters.

        Returns
        -------
        RMSFPerResidueService
            Service configured with RMS-median aggregation and the supplied metric.

        Examples
        --------
        >>> facade = RMSFPerResidueRmsMedianAggFacade(pipeline_data, "mean")
        >>> isinstance(facade.service, RMSFPerResidueService)
        True
        """

        return RMSFPerResidueService(
            self._pipeline_data,
            metric=self._metric,
            aggregator=self._aggregator,
        )

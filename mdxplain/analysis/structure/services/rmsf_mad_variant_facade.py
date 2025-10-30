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

"""Facade exposing per-atom and per-residue RMSF services using MAD metric."""

from __future__ import annotations

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsf_per_atom_service import RMSFPerAtomService
from .rmsf_per_residue_aggregation_selection_facade import (
    RMSFPerResidueAggregationSelectionFacade,
)


class RMSFMadVariantFacade:
    """Expose per-atom and per-residue RMSF services using MAD metric.

    Lazily initialises the per-atom service and the residue aggregation
    facade, ensuring both reuse the same pipeline context and median absolute
    deviation (MAD) metric for maximum outlier resistance.

    Returns
    -------
    RMSFMadVariantFacade
        Helper object that provides access to per-atom and per-residue RMSF
        services using the MAD metric.

    Examples
    --------
    >>> facade = RMSFMadVariantFacade(pipeline_data)
    >>> isinstance(facade.per_atom, RMSFPerAtomService)
    True
    """

    def __init__(self, pipeline_data: PipelineData | None) -> None:
        """Store the pipeline context with MAD metric for child services.

        Ensures that pipeline data was injected prior to accessing RMSF
        services and records the MAD metric shared by per-atom and per-residue
        computations.

        Parameters
        ----------
        pipeline_data : PipelineData | None
            Pipeline context injected by the analysis manager.

        Returns
        -------
        None
            The initializer does not return anything.

        Examples
        --------
        >>> facade = RMSFMadVariantFacade(pipeline_data)
        >>> isinstance(facade, RMSFMadVariantFacade)
        True
        """

        if pipeline_data is None:
            raise ValueError("RMSFMadVariantFacade requires pipeline_data")

        self._pipeline_data = pipeline_data
        self._metric = "mad"

    @property
    def per_atom(self) -> RMSFPerAtomService:
        """Access the per-atom RMSF service using the MAD metric.

        Lazily instantiates the per-atom RMSF service using the stored
        pipeline data and MAD deviation metric.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFPerAtomService
            MAD-specific per-atom RMSF service exposing ``to_*_reference``
            helpers.

        Examples
        --------
        >>> facade = RMSFMadVariantFacade(pipeline_data)
        >>> service = facade.per_atom
        >>> service.metric
        'mad'
        """
        return RMSFPerAtomService(self._pipeline_data, self._metric)

    @property
    def per_residue(self) -> RMSFPerResidueAggregationSelectionFacade:
        """Access the per-residue RMSF helper using the MAD metric.

        Lazily instantiates the residue aggregation selection facade using the
        stored pipeline data and MAD deviation metric.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFPerResidueAggregationSelectionFacade
            Facade exposing residue-level RMSF aggregation selection for the
            MAD metric.

        Examples
        --------
        >>> facade = RMSFMadVariantFacade(pipeline_data)
        >>> per_residue = facade.per_residue
        >>> isinstance(per_residue, RMSFPerResidueAggregationSelectionFacade)
        True
        """

        return RMSFPerResidueAggregationSelectionFacade(self._pipeline_data, self._metric)

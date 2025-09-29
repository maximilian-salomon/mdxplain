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

"""Structure analysis entry point service.

This module provides the :class:`StructureAnalysisService` facade that exposes
structure-related analysis services such as RMSD and RMSF. The individual
services are instantiated lazily to avoid importing heavy dependencies until a
calculation is requested.
"""

from __future__ import annotations

from typing import Optional

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsd_facade import RMSDFacade
from .rmsf_facade import RMSFFacade


class StructureAnalysisService:
    """Facade exposing the concrete structure analysis services.

    Provides lazily initialised access to RMSD and RMSF services bound to the
    current pipeline context. This keeps subsystem creation cheap while still
    guaranteeing consistent pipeline data injection.
    """

    def __init__(self, pipeline_data: Optional[PipelineData] = None) -> None:
        """Create the structure analysis facade.

        Parameters
        ----------
        pipeline_data : PipelineData, optional
            Pipeline context injected by :class:`AutoInjectProxy`.

        Returns
        -------
        None
            The initializer does not return anything.
        """

        self._pipeline_data = pipeline_data
        self._rmsd_facade: Optional[RMSDFacade] = None
        self._rmsf_facade: Optional[RMSFFacade] = None

    @property
    def pipeline_data(self) -> PipelineData:
        """Return the bound pipeline data.

        Parameters
        ----------
        None
            The property does not accept parameters.

        Returns
        -------
        PipelineData
            Pipeline context required for analysis services.

        Raises
        ------
        ValueError
            If the service has not received pipeline data yet. The AutoInject
            infrastructure must set this before use.

        Examples
        --------
        >>> service = StructureAnalysisService(pipeline_data)
        >>> data = service.pipeline_data
        """

        if self._pipeline_data is None:
            raise ValueError("StructureAnalysisService requires pipeline_data injection")
        return self._pipeline_data

    @property
    def rmsd(self) -> RMSDFacade:
        """
        Return the RMSD helper exposing mean/median/mad variants.
        
        Provides access to the RMSD facade, which lazily constructs and caches
        :class:`RMSDVariantService` instances for the mean, median, and mad
        metrics.
        """

        if self._rmsd_facade is None:
            self._rmsd_facade = RMSDFacade(self.pipeline_data)
        return self._rmsd_facade

    @property
    def rmsf(self) -> RMSFFacade:
        """
        Return the RMSF helper exposing per-atom/per-residue variants.
        
        Provides access to the RMSF facade, which lazily constructs and caches
        :class:`RMSFVariantFacade` instances for per-atom and per-residue
        calculations using the mean, median, and mad metrics.
        """

        if self._rmsf_facade is None:
            self._rmsf_facade = RMSFFacade(self.pipeline_data)
        return self._rmsf_facade

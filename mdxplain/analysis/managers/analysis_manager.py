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

"""Analysis manager for coordinating different types of analysis."""

from __future__ import annotations

from ...feature.services.feature_analysis_service import FeatureAnalysisService
from ..structure import StructureAnalysisService


class AnalysisManager:
    """
    Main manager for all analysis operations.

    Provides access to different analysis types through properties.
    Currently supports feature analysis, with plans to extend to
    structure analysis, importance analysis, and more.

    Examples
    --------
    >>> # Feature analysis (services)
    >>> pipeline.analysis.features.distances.mean()
    >>> pipeline.analysis.features.contacts.std()

    >>> # Future extensions
    >>> pipeline.analysis.structure.rmsd(reference_traj=0, reference_frame=0)
    >>> pipeline.analysis.structure.rmsd.median.to_reference()
    """

    def __init__(self) -> None:
        """Initialize analysis manager.

        Parameters
        ----------
        None
            The initializer does not accept parameters.

        Returns
        -------
        None
            The initializer does not return anything.

        Notes
        -----
        Pipeline data is injected automatically by :class:`AutoInjectProxy`.
        """
        pass

    @property
    def features(self) -> FeatureAnalysisService:
        """Get feature analysis service.

        Parameters
        ----------
        None
            The property does not accept parameters.

        Returns
        -------
        FeatureAnalysisService
            Service providing access to feature analysis operations.
        """
        return FeatureAnalysisService(None)

    @property
    def structure(self) -> StructureAnalysisService:
        """Get structure analysis service.

        Parameters
        ----------
        None
            The property does not accept parameters.

        Returns
        -------
        StructureAnalysisService
            Service providing access to structure analysis operations.
        """

        return StructureAnalysisService(None)

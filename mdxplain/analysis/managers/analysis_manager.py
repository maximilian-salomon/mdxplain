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


class AnalysisManager:
    """
    Main manager for all analysis operations.

    Provides access to different analysis types through properties.
    Currently supports feature analysis, with plans to extend to
    structure analysis, importance analysis, and more.

    Examples:
    ---------
    >>> # Feature analysis (services)
    >>> pipeline.analysis.features.distances.mean()
    >>> pipeline.analysis.features.contacts.std()

    >>> # Future manager methods (AutoInject)
    >>> pipeline.analysis.compare_features("distances", [0, 1], [2, 3])

    >>> # Future extensions
    >>> pipeline.analysis.structure.rmsd()
    >>> pipeline.analysis.importance.feature_ranking()
    """

    def __init__(self) -> None:
        """
        Initialize analysis manager.

        Uses minimal initialization consistent with other managers.
        PipelineData is injected automatically by AutoInjectProxy.
        """
        pass

    @property
    def features(self) -> FeatureAnalysisService:
        """
        Get feature analysis service.

        Provides direct access to all feature analysis operations
        with full VS Code autocompletion support. Uses AutoInjectProxy
        pattern for consistent pipeline_data access.

        Returns:
        --------
        FeatureAnalysisService
            Service providing access to feature analysis operations
        """
        return FeatureAnalysisService(None)

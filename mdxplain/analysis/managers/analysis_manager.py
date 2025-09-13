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
from typing import TYPE_CHECKING

from ...feature.services.feature_analysis_service import FeatureAnalysisService

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class AnalysisManager:
    """
    Main manager for all analysis operations.
    
    Provides access to different analysis types through properties.
    Currently supports feature analysis, with plans to extend to
    structure analysis, importance analysis, and more.
    
    The AnalysisManager uses a special pattern where services (like features)
    receive PipelineData directly, while manager methods use AutoInject.
    This maintains the clean API while supporting both patterns.
    
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
        PipelineData is set via _set_pipeline_data() by PipelineManager.
        """
        self._pipeline_data = None
        self._services_cache = {}
    
    def _set_pipeline_data(self, pipeline_data: 'PipelineData') -> None:
        """
        Set pipeline data for services.
        
        Called by PipelineManager to provide current pipeline data.
        Invalidates service cache when pipeline data changes.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Current pipeline data container
        """
        if self._pipeline_data != pipeline_data:
            self._pipeline_data = pipeline_data
            self._services_cache.clear()  # Invalidate cache
    
    @property
    def features(self) -> FeatureAnalysisService:
        """
        Get feature analysis service.
        
        Provides direct access to all feature analysis operations
        with full VS Code autocompletion support. Uses lazy initialization
        and caching for performance.
        
        Returns:
        --------
        FeatureAnalysisService
            Service providing access to feature analysis operations
        
        Raises:
        -------
        RuntimeError
            If pipeline data has not been set
        """
        if 'features' not in self._services_cache:
            if self._pipeline_data is None:
                raise RuntimeError(
                    "Pipeline data not set. Make sure to use AnalysisManager "
                    "through PipelineManager or call _set_pipeline_data() first."
                )
            self._services_cache['features'] = FeatureAnalysisService(pipeline_data=self._pipeline_data)
        return self._services_cache['features']

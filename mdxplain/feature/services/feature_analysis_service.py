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

"""Main factory for feature analysis operations."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

# Feature type service imports
from ..feature_type.distances.distances_analysis_service import DistancesAnalysisService
from ..feature_type.contacts.contacts_analysis_service import ContactsAnalysisService
from ..feature_type.torsions.torsions_analysis_service import TorsionsAnalysisService
from ..feature_type.dssp.dssp_analysis_service import DSSPAnalysisService
from ..feature_type.sasa.sasa_analysis_service import SASAAnalysisService
from ..feature_type.coordinates.coordinates_analysis_service import CoordinatesAnalysisService


class FeatureAnalysisService:
    """Main service for accessing analysis operations on feature types."""
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """
        Initialize feature analysis service.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container
        """
        self._pipeline_data = pipeline_data
    
    @property
    def distances(self) -> DistancesAnalysisService:
        """Access distance-specific analysis operations."""
        return DistancesAnalysisService(self._pipeline_data)
    
    @property
    def contacts(self) -> ContactsAnalysisService:
        """Access contact-specific analysis operations."""
        return ContactsAnalysisService(self._pipeline_data)
    
    @property
    def torsions(self) -> TorsionsAnalysisService:
        """Access torsion-specific analysis operations."""
        return TorsionsAnalysisService(self._pipeline_data)
    
    @property
    def dssp(self) -> DSSPAnalysisService:
        """Access DSSP-specific analysis operations."""
        return DSSPAnalysisService(self._pipeline_data)
    
    @property
    def sasa(self) -> SASAAnalysisService:
        """Access SASA-specific analysis operations."""
        return SASAAnalysisService(self._pipeline_data)
    
    @property
    def coordinates(self) -> CoordinatesAnalysisService:
        """Access coordinate-specific analysis operations."""
        return CoordinatesAnalysisService(self._pipeline_data)

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

"""Main factory for feature reduce operations."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.feature_manager import FeatureManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..feature_type.distances.services.distances_reduce_service import DistancesReduceService
from ..feature_type.contacts.services.contacts_reduce_service import ContactsReduceService
from ..feature_type.torsions.services.torsions_reduce_service import TorsionsReduceService
from ..feature_type.dssp.services.dssp_reduce_service import DSSPReduceService
from ..feature_type.sasa.services.sasa_reduce_service import SASAReduceService
from ..feature_type.coordinates.services.coordinates_reduce_service import CoordinatesReduceService


class FeatureReduceService:
    """Main service for accessing reduce operations on feature types."""
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """Initialize feature reduce service."""
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    @property
    def distances(self) -> DistancesReduceService:
        """Access distance-specific reduce metrics."""
        return DistancesReduceService(self._manager, self._pipeline_data)

    @property
    def contacts(self) -> ContactsReduceService:
        """Access contact-specific reduce metrics."""
        return ContactsReduceService(self._manager, self._pipeline_data)

    @property
    def torsions(self) -> TorsionsReduceService:
        """Access torsion-specific reduce metrics."""
        return TorsionsReduceService(self._manager, self._pipeline_data)

    @property
    def dssp(self) -> DSSPReduceService:
        """Access DSSP-specific reduce metrics."""
        return DSSPReduceService(self._manager, self._pipeline_data)

    @property
    def sasa(self) -> SASAReduceService:
        """Access SASA-specific reduce metrics."""
        return SASAReduceService(self._manager, self._pipeline_data)

    @property
    def coordinates(self) -> CoordinatesReduceService:
        """Access coordinates-specific reduce metrics."""
        return CoordinatesReduceService(self._manager, self._pipeline_data)

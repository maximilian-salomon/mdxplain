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

"""
Service providing feature-type-specific properties for adding selections.

This module provides the FeatureSelectorAddService class that offers
feature-type-specific properties for adding selections with reduction options.
Each property returns a type-specific service with reduction methods.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from ..feature_type.distances.services.distances_selection_service import DistancesSelectionService
from ..feature_type.contacts.services.contacts_selection_service import ContactsSelectionService
from ..feature_type.coordinates.services.coordinates_selection_service import CoordinatesSelectionService
from ..feature_type.torsions.services.torsions_selection_service import TorsionsSelectionService
from ..feature_type.dssp.services.dssp_selection_service import DSSPSelectionService
from ..feature_type.sasa.services.sasa_selection_service import SasaSelectionService

if TYPE_CHECKING:
    from ...feature_selection.managers.feature_selector_manager import FeatureSelectorManager
    from ...pipeline.entities.pipeline_data import PipelineData


class FeatureSelectorAddService:
    """
    Service providing feature-type-specific properties for adding selections.

    Module separation:

    - Knows feature types (distances, contacts, etc.)
    - Does NOT know reduction metrics
    - Each property returns a type-specific service

    Each feature type service provides:

    - Basic add method via __call__()
    - Reduction methods like with_cv_reduction(), with_std_reduction(), etc.

    Examples
    --------
    Basic usage:
    >>> pipeline.feature_selector.add.distances("test", "res ALA")
    >>> pipeline.feature_selector.add.contacts("test", "resid 120-140")

    With reduction:
    >>> pipeline.feature_selector.add.distances.with_cv_reduction("test", "res ALA", threshold_min=0.1)
    >>> pipeline.feature_selector.add.contacts.with_frequency_reduction("test", "resid 120-140", threshold_max=0.8)
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData):
        """
        Initialize feature selector add service.

        Parameters
        ----------
        manager : FeatureSelectorManager
            Manager instance for executing add operations
        pipeline_data : PipelineData
            Pipeline data container with trajectory and feature data

        Returns
        -------
        None
            Initializes service with manager and pipeline_data references
        """
        self._manager = manager
        self._pipeline_data = pipeline_data

    @property
    def distances(self) -> DistancesSelectionService:
        """
        Get distances add service with distances-specific reduction methods.

        Returns a service that provides methods for adding distance feature
        selections with optional post-selection reduction based on statistical
        metrics like CV, standard deviation, variance, etc.

        Parameters
        ----------
        None

        Returns
        -------
        DistancesSelectionService
            Service with distances-specific reduction methods

        Available reduction methods:

        - with_cv_reduction(): Coefficient of variation filtering
        - with_std_reduction(): Standard deviation filtering
        - with_variance_reduction(): Variance filtering
        - with_range_reduction(): Range filtering
        - with_transitions_reduction(): Transition-based filtering
        - with_min_reduction(): Minimum value filtering
        - with_mad_reduction(): Median absolute deviation filtering
        - with_mean_reduction(): Mean value filtering
        - with_max_reduction(): Maximum value filtering

        Examples
        --------
        >>> pipeline.feature_selector.add.distances("test", "res ALA")
        >>> pipeline.feature_selector.add.distances.with_cv_reduction("test", "res ALA", threshold_min=0.1)
        """
        return DistancesSelectionService(self._manager, self._pipeline_data)

    @property
    def contacts(self) -> ContactsSelectionService:
        """
        Get contacts add service with contacts-specific reduction methods.

        Returns a service that provides methods for adding contact feature
        selections with optional post-selection reduction based on contact
        statistics like frequency, stability, and transitions.

        Parameters
        ----------
        None

        Returns
        -------
        ContactsSelectionService
            Service with contacts-specific reduction methods

        Available reduction methods:

        - with_frequency_reduction(): Contact frequency filtering
        - with_stability_reduction(): Contact stability filtering
        - with_transitions_reduction(): Contact transition filtering

        Examples
        --------
        >>> pipeline.feature_selector.add.contacts("test", "resid 120-140")
        >>> pipeline.feature_selector.add.contacts.with_frequency_reduction("test", "resid 120-140", threshold_min=0.3)
        """
        return ContactsSelectionService(self._manager, self._pipeline_data)

    @property
    def coordinates(self) -> CoordinatesSelectionService:
        """
        Get coordinates add service with coordinates-specific reduction methods.

        Returns a service that provides methods for adding coordinate feature
        selections with optional post-selection reduction based on structural
        flexibility metrics like RMSF, standard deviation, etc.

        Parameters
        ----------
        None

        Returns
        -------
        CoordinatesSelectionService
            Service with coordinates-specific reduction methods

        Available reduction methods:

        - with_std_reduction(): Standard deviation filtering
        - with_rmsf_reduction(): Root mean square fluctuation filtering
        - with_cv_reduction(): Coefficient of variation filtering
        - with_variance_reduction(): Variance filtering
        - with_range_reduction(): Range filtering
        - with_mad_reduction(): Median absolute deviation filtering
        - with_mean_reduction(): Mean value filtering
        - with_min_reduction(): Minimum value filtering
        - with_max_reduction(): Maximum value filtering

        Examples
        --------
        >>> pipeline.feature_selector.add.coordinates("test", "backbone")
        >>> pipeline.feature_selector.add.coordinates.with_rmsf_reduction("test", "backbone", threshold_max=2.0)
        """
        return CoordinatesSelectionService(self._manager, self._pipeline_data)

    @property
    def torsions(self) -> TorsionsSelectionService:
        """
        Get torsions add service with torsions-specific reduction methods.

        Returns a service that provides methods for adding torsion angle feature
        selections with optional post-selection reduction based on angular
        flexibility and transition metrics.

        Parameters
        ----------
        None

        Returns
        -------
        TorsionsSelectionService
            Service with torsions-specific reduction methods

        Available reduction methods:

        - with_transitions_reduction(): Angular transition filtering
        - with_std_reduction(): Standard deviation filtering
        - with_mad_reduction(): Median absolute deviation filtering
        - with_mean_reduction(): Mean value filtering
        - with_range_reduction(): Range filtering
        - with_min_reduction(): Minimum value filtering
        - with_max_reduction(): Maximum value filtering
        - with_cv_reduction(): Coefficient of variation filtering
        - with_variance_reduction(): Variance filtering

        Examples
        --------
        >>> pipeline.feature_selector.add.torsions("test", "phi psi")
        >>> pipeline.feature_selector.add.torsions.with_transitions_reduction("test", "phi psi", threshold_min=5)
        """
        return TorsionsSelectionService(self._manager, self._pipeline_data)

    @property
    def dssp(self) -> DSSPSelectionService:
        """
        Get DSSP add service with DSSP-specific reduction methods.

        Returns a service that provides methods for adding secondary structure
        feature selections with optional post-selection reduction based on
        structural stability and transition frequencies.

        Parameters
        ----------
        None

        Returns
        -------
        DsspSelectionService
            Service with DSSP-specific reduction methods

        Available reduction methods:

        - with_transitions_reduction(): Secondary structure transition filtering
        - with_transition_frequency_reduction(): Transition frequency filtering
        - with_stability_reduction(): Structural stability filtering
        - with_class_frequencies_reduction(): Structure class frequency filtering

        Examples
        --------
        >>> pipeline.feature_selector.add.dssp("test", "resid 50-100")
        >>> pipeline.feature_selector.add.dssp.with_stability_reduction("test", "resid 50-100", threshold_min=0.7)
        """
        return DSSPSelectionService(self._manager, self._pipeline_data)

    @property
    def sasa(self) -> SasaSelectionService:
        """
        Get SASA add service with SASA-specific reduction methods.

        Returns a service that provides methods for adding solvent accessible
        surface area feature selections with optional post-selection reduction
        based on exposure variability and burial statistics.

        Parameters
        ----------
        None

        Returns
        -------
        SasaSelectionService
            Service with SASA-specific reduction methods

        Available reduction methods:
        
        - with_cv_reduction(): Coefficient of variation filtering
        - with_range_reduction(): Range filtering
        - with_std_reduction(): Standard deviation filtering
        - with_variance_reduction(): Variance filtering
        - with_mad_reduction(): Median absolute deviation filtering
        - with_mean_reduction(): Mean value filtering
        - with_min_reduction(): Minimum value filtering
        - with_max_reduction(): Maximum value filtering
        - with_burial_fraction_reduction(): Burial fraction filtering

        Examples
        --------
        >>> pipeline.feature_selector.add.sasa("test", "resid 1-50")
        >>> pipeline.feature_selector.add.sasa.with_cv_reduction("test", "resid 1-50", threshold_max=0.5)
        """
        return SasaSelectionService(self._manager, self._pipeline_data)

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

"""Factory for coordinates-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....managers.feature_manager import FeatureManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ..coordinates import Coordinates
from ...interfaces.reduce_service_base import ReduceServiceBase


class CoordinatesReduceService(ReduceServiceBase):
    """
    Service for coordinates-specific reduce metrics.
    
    Provides methods to reduce coordinate features based on various statistical metrics.
    Each method applies a different reduction criterion, such as standard deviation,
    root mean square fluctuation, coefficient of variation, variance, range, median absolute deviation,
    mean, min, max, and transitions.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize coordinates reduce factory.
        
        Parameters
        ----------
        manager : FeatureManager
            The feature manager instance
        pipeline_data : PipelineData
            The pipeline data instance

        Returns
        -------
        None
            Initializes the coordinates reduce service
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "coordinates"
    
    def std(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by standard deviation.

        Standard deviation measures the amount of variation or dispersion in a set of values.

        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum standard deviation threshold (Angstrom)
        threshold_max : float, optional
            Maximum standard deviation threshold (Angstrom)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep coordinates with significant variation
        >>> pipeline.feature.reduce.coordinates.std(threshold_min=0.5)

        >>> # Remove extremely variable coordinates
        >>> pipeline.feature.reduce.coordinates.std(threshold_max=3.0)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="std",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def rmsf(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by root mean square fluctuation.

        RMSF measures average positional deviation from mean position.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum RMSF threshold (Angstrom)
        threshold_max : float, optional
            Maximum RMSF threshold (Angstrom)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep highly flexible coordinates
        >>> pipeline.feature.reduce.coordinates.rmsf(threshold_min=1.0)

        >>> # Keep stable coordinates only
        >>> pipeline.feature.reduce.coordinates.rmsf(threshold_max=0.5)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="rmsf",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def cv(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by coefficient of variation.

        Filters coordinate features based on their relative variability.
        CV = std/mean, indicating positional flexibility.

        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum CV threshold
        threshold_max : float, optional
            Maximum CV threshold
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep highly flexible coordinates
        >>> pipeline.feature.reduce.coordinates.cv(threshold_min=0.3)

        >>> # Remove extremely variable coordinates
        >>> pipeline.feature.reduce.coordinates.cv(threshold_max=1.0)

        >>> # Focus on moderate flexibility
        >>> pipeline.feature.reduce.coordinates.cv(
        ...     threshold_min=0.1,
        ...     threshold_max=0.8
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="cv",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def variance(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by variance.

        Filters coordinates based on positional variance (squared fluctuations).
        Higher variance indicates larger positional deviations.

        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum variance threshold (Ų)
        threshold_max : float, optional
            Maximum variance threshold (Ų)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep highly variable coordinates
        >>> pipeline.feature.reduce.coordinates.variance(threshold_min=1.0)

        >>> # Remove extremely variable coordinates
        >>> pipeline.feature.reduce.coordinates.variance(threshold_max=5.0)

        >>> # Focus on moderate positional variance
        >>> pipeline.feature.reduce.coordinates.variance(
        ...     threshold_min=0.5,
        ...     threshold_max=3.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="variance",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def range(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by range (max - min).

        Filters coordinates based on their positional range of motion.
        Larger range indicates greater conformational sampling.

        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum range threshold (Angstrom)
        threshold_max : float, optional
            Maximum range threshold (Angstrom)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep coordinates with large positional range
        >>> pipeline.feature.reduce.coordinates.range(threshold_min=2.0)

        >>> # Keep relatively stable coordinates
        >>> pipeline.feature.reduce.coordinates.range(threshold_max=1.5)

        >>> # Focus on moderate motion range
        >>> pipeline.feature.reduce.coordinates.range(
        ...     threshold_min=0.5,
        ...     threshold_max=3.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="range",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def mad(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by median absolute deviation.

        Robust measure of positional variability less sensitive to outliers.
        MAD provides stable estimate of coordinate fluctuations.

        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum MAD threshold (Ų)
        threshold_max : float, optional
            Maximum MAD threshold (Ų)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep robustly variable coordinates
        >>> pipeline.feature.reduce.coordinates.mad(threshold_min=0.4)

        >>> # Remove extreme outlier coordinates
        >>> pipeline.feature.reduce.coordinates.mad(threshold_max=2.0)

        >>> # MAD-based selection for stable analysis
        >>> pipeline.feature.reduce.coordinates.mad(
        ...     threshold_min=0.2,
        ...     threshold_max=1.5
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="mad",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def mean(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by mean position.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum mean position threshold (Angstrom)
        threshold_max : float, optional
            Maximum mean position threshold (Angstrom)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep coordinates near protein core
        >>> pipeline.feature.reduce.coordinates.mean(threshold_max=10.0)

        >>> # Select peripheral coordinates only
        >>> pipeline.feature.reduce.coordinates.mean(threshold_min=15.0)

        >>> # Focus on intermediate regions
        >>> pipeline.feature.reduce.coordinates.mean(
        ...     threshold_min=5.0,
        ...     threshold_max=20.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="mean",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def min(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by minimum position.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum position minimum threshold (Angstrom)
        threshold_max : float, optional
            Maximum position minimum threshold (Angstrom)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep coordinates that get close to origin
        >>> pipeline.feature.reduce.coordinates.min(threshold_max=5.0)

        >>> # Exclude coordinates that get too close
        >>> pipeline.feature.reduce.coordinates.min(threshold_min=2.0)

        >>> # Focus on specific minimum distance range
        >>> pipeline.feature.reduce.coordinates.min(
        ...     threshold_min=3.0,
        ...     threshold_max=8.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="min",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def max(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by maximum position.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum position maximum threshold (Angstrom)
        threshold_max : float, optional
            Maximum position maximum threshold (Angstrom)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep coordinates with large excursions
        >>> pipeline.feature.reduce.coordinates.max(threshold_min=20.0)

        >>> # Keep relatively constrained coordinates
        >>> pipeline.feature.reduce.coordinates.max(threshold_max=15.0)

        >>> # Focus on specific maximum distance range
        >>> pipeline.feature.reduce.coordinates.max(
        ...     threshold_min=10.0,
        ...     threshold_max=25.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="max",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 1.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce coordinates by transition detection.
        
        Filters coordinate features based on number of positional transitions.
        Detects significant changes in atomic positions.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        transition_threshold : float, default=1.0
            Position threshold for detecting transitions (Angstrom)
        window_size : int, default=10
            Window size for transition analysis
        transition_mode : str, default="window"
            Transition analysis mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for transition detection (if mode='lag')
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep highly dynamic coordinates
        >>> pipeline.feature.reduce.coordinates.transitions(
        ...     threshold_min=5,
        ...     transition_threshold=1.5
        ... )

        >>> # Keep stable coordinates with few transitions
        >>> pipeline.feature.reduce.coordinates.transitions(
        ...     threshold_max=3,
        ...     window_size=20
        ... )

        >>> # Focus on moderate transition activity
        >>> pipeline.feature.reduce.coordinates.transitions(
        ...     threshold_min=2,
        ...     threshold_max=8,
        ...     transition_threshold=1.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            transition_mode=transition_mode,
            lag_time=lag_time,
            cross_trajectory=cross_trajectory,
        )

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

"""Factory for torsion-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....managers.feature_manager import FeatureManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ..torsions import Torsions
from ...interfaces.reduce_service_base import ReduceServiceBase


class TorsionsReduceService(ReduceServiceBase):
    """
    Service for torsion-specific reduce metrics.
    
    Provides metrics specifically useful for torsion angle features,
    using circular statistics appropriate for angular data.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize torsion reduce factory.

        Parameters
        ----------
        manager : FeatureManager
            Feature manager instance
        pipeline_data : PipelineData
            Pipeline data container

        Returns
        -------
        None
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "torsions"

    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 45.0,
        window_size: int = 10,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce torsions by transition detection.
        
        Detects transitions in torsion angles using circular statistics.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        transition_threshold : float, default=45.0
            Angle threshold in degrees for transition detection
        window_size : int, default=10
            Window size for transition detection
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep dynamic torsions with many angle changes
        >>> pipeline.feature.reduce.torsions.transitions(
        ...     threshold_min=8,
        ...     transition_threshold=30.0
        ... )

        >>> # Keep stable torsions with few transitions
        >>> pipeline.feature.reduce.torsions.transitions(
        ...     threshold_max=5,
        ...     window_size=15
        ... )

        >>> # Focus on moderate torsional dynamics
        >>> pipeline.feature.reduce.torsions.transitions(
        ...     threshold_min=3,
        ...     threshold_max=12,
        ...     transition_threshold=45.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            cross_trajectory=cross_trajectory,
        )
    
    def std(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce torsions by standard deviation (circular).
        
        Filters torsion features based on their circular standard deviation.
        Higher values indicate more flexible/dynamic torsional angles.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum standard deviation threshold (degrees)
        threshold_max : float, optional
            Maximum standard deviation threshold (degrees)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep highly flexible torsions
        >>> pipeline.feature.reduce.torsions.std(threshold_min=30.0)

        >>> # Remove extremely variable torsions
        >>> pipeline.feature.reduce.torsions.std(threshold_max=60.0)

        >>> # Focus on moderately flexible angles
        >>> pipeline.feature.reduce.torsions.std(
        ...     threshold_min=15.0,
        ...     threshold_max=45.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="std",
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
        Reduce torsions by median absolute deviation (circular).
        
        Filters torsion features based on their circular median absolute deviation.
        More robust to outliers than standard deviation.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum MAD threshold (degrees)
        threshold_max : float, optional
            Maximum MAD threshold (degrees)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep robustly variable torsions (outlier-resistant)
        >>> pipeline.feature.reduce.torsions.mad(threshold_min=20.0)

        >>> # Remove extreme outlier torsions
        >>> pipeline.feature.reduce.torsions.mad(threshold_max=40.0)

        >>> # MAD-based selection for stable analysis
        >>> pipeline.feature.reduce.torsions.mad(
        ...     threshold_min=10.0,
        ...     threshold_max=35.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
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
        Reduce torsions by circular mean angle.
        
        Filters torsion features based on their mean angle values.
        Useful for selecting angles in specific conformational ranges.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum mean angle threshold (degrees, -180 to 180)
        threshold_max : float, optional
            Maximum mean angle threshold (degrees, -180 to 180)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep torsions in alpha-helix range
        >>> pipeline.feature.reduce.torsions.mean(
        ...     threshold_min=-90.0,
        ...     threshold_max=-30.0
        ... )

        >>> # Select beta-sheet conformations
        >>> pipeline.feature.reduce.torsions.mean(
        ...     threshold_min=90.0,
        ...     threshold_max=150.0
        ... )

        >>> # Exclude unfavorable conformations
        >>> pipeline.feature.reduce.torsions.mean(threshold_max=0.0)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="mean",
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
        Reduce torsions by angular range (circular).
        
        Filters torsion features based on their angular range accounting for circularity.
        Higher values indicate more flexible torsional motion.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum angular range threshold (degrees, 0-180)
        threshold_max : float, optional
            Maximum angular range threshold (degrees, 0-180)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep torsions with large conformational sampling
        >>> pipeline.feature.reduce.torsions.range(threshold_min=60.0)

        >>> # Keep relatively constrained torsions
        >>> pipeline.feature.reduce.torsions.range(threshold_max=45.0)

        >>> # Focus on moderate angular flexibility
        >>> pipeline.feature.reduce.torsions.range(
        ...     threshold_min=30.0,
        ...     threshold_max=90.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="range",
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
        Reduce torsions by minimum angle value.

        Filters torsion features based on their minimum angle values.

        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum angle threshold (degrees, -180 to 180)
        threshold_max : float, optional
            Maximum angle threshold (degrees, -180 to 180)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep torsions that reach negative angles
        >>> pipeline.feature.reduce.torsions.min(threshold_max=-30.0)

        >>> # Exclude torsions that get too negative
        >>> pipeline.feature.reduce.torsions.min(threshold_min=-120.0)

        >>> # Focus on specific minimum angle range
        >>> pipeline.feature.reduce.torsions.min(
        ...     threshold_min=-90.0,
        ...     threshold_max=-10.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
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
        Reduce torsions by maximum angle value.

        Filters torsion features based on their maximum angle values.

        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum angle threshold (degrees, -180 to 180)
        threshold_max : float, optional
            Maximum angle threshold (degrees, -180 to 180)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep torsions that reach high positive angles
        >>> pipeline.feature.reduce.torsions.max(threshold_min=120.0)

        >>> # Keep constrained torsions
        >>> pipeline.feature.reduce.torsions.max(threshold_max=90.0)

        >>> # Focus on specific maximum angle range
        >>> pipeline.feature.reduce.torsions.max(
        ...     threshold_min=60.0,
        ...     threshold_max=150.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="max",
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
        Reduce torsions by coefficient of variation (circular).
        
        Filters torsion features based on their relative variability.
        CV = circular_std / abs(circular_mean), indicating flexibility.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum CV threshold for flexibility
        threshold_max : float, optional
            Maximum CV threshold for flexibility
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep highly flexible torsions (high relative variability)
        >>> pipeline.feature.reduce.torsions.cv(threshold_min=0.5)

        >>> # Remove extremely variable torsions
        >>> pipeline.feature.reduce.torsions.cv(threshold_max=1.5)

        >>> # Focus on moderate torsional flexibility
        >>> pipeline.feature.reduce.torsions.cv(
        ...     threshold_min=0.2,
        ...     threshold_max=1.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
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
        Reduce torsions by circular variance.
        
        Filters torsion features based on their circular variance.
        Values 0-1 where higher values indicate more flexible angles.
        
        Parameters
        ----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum variance threshold (0-1)
        threshold_max : float, optional
            Maximum variance threshold (0-1)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns
        -------
        None
            Updates reduced data in pipeline

        Examples
        --------
        >>> # Keep torsions with high circular variance
        >>> pipeline.feature.reduce.torsions.variance(threshold_min=0.7)

        >>> # Keep relatively stable torsions
        >>> pipeline.feature.reduce.torsions.variance(threshold_max=0.3)

        >>> # Focus on moderate angular variance
        >>> pipeline.feature.reduce.torsions.variance(
        ...     threshold_min=0.2,
        ...     threshold_max=0.8
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="variance",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )

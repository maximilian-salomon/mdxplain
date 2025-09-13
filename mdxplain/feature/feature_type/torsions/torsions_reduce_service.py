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
    from ...managers.feature_manager import FeatureManager
    from ....pipeline.entities.pipeline_data import PipelineData

from . import Torsions


class TorsionsReduceService:
    """
    Service for torsion-specific reduce metrics.
    
    Provides metrics specifically useful for torsion angle features,
    using circular statistics appropriate for angular data.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """Initialize torsion reduce factory."""
        self._manager = manager
        self._pipeline_data = pipeline_data

    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 45.0,
        window_size: int = 10,
    ) -> None:
        """
        Reduce torsions by transition detection.
        
        Detects transitions in torsion angles using circular statistics.
        
        Parameters:
        -----------
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
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
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
        )
    
    def std(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by standard deviation (circular).
        
        Filters torsion features based on their circular standard deviation.
        Higher values indicate more flexible/dynamic torsional angles.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum standard deviation threshold (degrees)
        threshold_max : float, optional
            Maximum standard deviation threshold (degrees)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="std",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def mad(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by median absolute deviation (circular).
        
        Filters torsion features based on their circular median absolute deviation.
        More robust to outliers than standard deviation.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum MAD threshold (degrees)
        threshold_max : float, optional
            Maximum MAD threshold (degrees)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="mad",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def mean(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by circular mean angle.
        
        Filters torsion features based on their mean angle values.
        Useful for selecting angles in specific conformational ranges.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum mean angle threshold (degrees, -180 to 180)
        threshold_max : float, optional
            Maximum mean angle threshold (degrees, -180 to 180)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="mean",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def range(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by angular range (circular).
        
        Filters torsion features based on their angular range accounting for circularity.
        Higher values indicate more flexible torsional motion.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum angular range threshold (degrees, 0-180)
        threshold_max : float, optional
            Maximum angular range threshold (degrees, 0-180)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="range",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def min(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by minimum angle value.
        
        Filters torsion features based on their minimum angle values.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum angle threshold (degrees, -180 to 180)
        threshold_max : float, optional
            Maximum angle threshold (degrees, -180 to 180)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="min",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def max(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by maximum angle value.
        
        Filters torsion features based on their maximum angle values.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum angle threshold (degrees, -180 to 180)
        threshold_max : float, optional
            Maximum angle threshold (degrees, -180 to 180)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="max",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def cv(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by coefficient of variation (circular).
        
        Filters torsion features based on their relative variability.
        CV = circular_std / abs(circular_mean), indicating flexibility.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum CV threshold for flexibility
        threshold_max : float, optional
            Maximum CV threshold for flexibility
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="cv",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def variance(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce torsions by circular variance.
        
        Filters torsion features based on their circular variance.
        Values 0-1 where higher values indicate more flexible angles.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum variance threshold (0-1)
        threshold_max : float, optional
            Maximum variance threshold (0-1)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Torsions,
            metric="variance",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )

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

"""Factory for DSSP-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....managers.feature_manager import FeatureManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ..dssp import DSSP
from ...interfaces.reduce_service_base import ReduceServiceBase


class DSSPReduceService(ReduceServiceBase):
    """
    Service for DSSP-specific reduce metrics.
    
    Provides metrics specifically useful for DSSP features,
    focusing on secondary structure transitions and stability.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize DSSP reduce factory.
        
        Parameters:
        -----------
        manager : FeatureManager
            Feature manager instance
        pipeline_data : PipelineData
            Pipeline data container

        Returns:
        --------
        None
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "dssp"
    
    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 1.0,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce DSSP by secondary structure transitions.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        transition_threshold : float, default=1.0
            Threshold for transition detection
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Find residues with many secondary structure transitions
        >>> pipeline.feature.reduce.dssp.transitions(threshold_min=5)

        >>> # Keep residues with moderate transition activity
        >>> pipeline.feature.reduce.dssp.transitions(
        ...     threshold_min=2,
        ...     threshold_max=10
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            cross_trajectory=cross_trajectory,
        )
    
    def transition_frequency(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce DSSP by transition frequency.
        
        Filters residues based on frequency of secondary structure transitions.
        Same as transitions but normalized by trajectory length.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum transition frequency
        threshold_max : float, optional
            Maximum transition frequency
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues with high transition frequency
        >>> pipeline.feature.reduce.dssp.transition_frequency(threshold_min=0.1)

        >>> # Keep residues with moderate transition rates
        >>> pipeline.feature.reduce.dssp.transition_frequency(
        ...     threshold_min=0.05,
        ...     threshold_max=0.2
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="transition_frequency",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def stability(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce DSSP by structural stability.
        
        Filters residues based on secondary structure stability.
        Stability = 1 - transition_frequency, higher values = more stable.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum stability threshold (0-1)
        threshold_max : float, optional
            Maximum stability threshold (0-1)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep highly stable residues (low transitions)
        >>> pipeline.feature.reduce.dssp.stability(threshold_min=0.8)

        >>> # Keep moderately stable residues
        >>> pipeline.feature.reduce.dssp.stability(
        ...     threshold_min=0.5,
        ...     threshold_max=0.9
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="stability",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def class_frequencies(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce DSSP by secondary structure class frequencies.
        
        Filters residues based on frequency distribution of structural classes.
        Analyzes how often each residue adopts different secondary structures.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum class frequency threshold
        threshold_max : float, optional
            Maximum class frequency threshold
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues with specific class frequency patterns
        >>> pipeline.feature.reduce.dssp.class_frequencies(threshold_min=0.3)

        >>> # Keep residues with balanced class distributions
        >>> pipeline.feature.reduce.dssp.class_frequencies(
        ...     threshold_min=0.1,
        ...     threshold_max=0.7
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="class_frequencies",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )

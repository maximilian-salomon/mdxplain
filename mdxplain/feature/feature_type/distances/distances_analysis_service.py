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

"""Service for distances-specific analysis operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING, Callable
import numpy as np

from ...services.helpers.analysis_data_helper import AnalysisDataHelper
from .distance_calculator_analysis import DistanceCalculatorAnalysis
from ..interfaces.analysis_service_base import AnalysisServiceBase

if TYPE_CHECKING:
    from ....pipeline.entities.pipeline_data import PipelineData


class DistancesAnalysisService(AnalysisServiceBase):
    """
    Service for distances-specific analysis operations.
    
    Provides a simplified interface for distance analysis methods by automatically
    handling data selection and forwarding method calls to the underlying
    calculator_analysis class. All methods from DistanceCalculatorAnalysis
    are automatically available with additional feature_selector and 
    traj_selection parameters.
    
    Examples:
    ---------
    >>> # All methods from DistanceCalculatorAnalysis are available
    >>> mean = pipeline.feature.analysis.distances.compute_mean()
    >>> std = pipeline.feature.analysis.distances.compute_std(
    ...     feature_selector="my_selector",
    ...     traj_selection=[0, 1, 2]
    ... )
    """
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """
        Initialize distances analysis service.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container with all necessary data
            
        Returns:
        --------
        None
        """
        super().__init__(pipeline_data)
        self._feature_type = "distances"
        self._calculator = DistanceCalculatorAnalysis(
            use_memmap=pipeline_data.use_memmap,
            chunk_size=pipeline_data.chunk_size
        )
    
    # === BASIC STATISTICS ===
    
    def mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean distances per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Mean distance for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Compute mean for all distance pairs
        >>> pipeline.analysis.features.distances.mean()
        
        >>> # Compute mean for selected features
        >>> pipeline.analysis.features.distances.mean(feature_selector="my_selection")
        
        >>> # Compute mean for specific trajectories
        >>> pipeline.analysis.features.distances.mean(traj_selection=[0, 1, 2])
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mean(data)
    
    def std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation of distances per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Standard deviation for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Compute std for all distance pairs
        >>> pipeline.analysis.features.distances.std()
        
        >>> # Compute std for selected features
        >>> pipeline.analysis.features.distances.std(feature_selector="variable_distances")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_std(data)
    
    def min(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum distances per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Minimum distance for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Find closest approach for all pairs
        >>> pipeline.analysis.features.distances.min()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_min(data)
    
    def max(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum distances per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Maximum distance for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Find maximum separation for all pairs
        >>> pipeline.analysis.features.distances.max()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_max(data)
    
    def median(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median distances per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Median distance for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Find median distance for all pairs
        >>> pipeline.analysis.features.distances.median()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_median(data)
    
    def variance(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute variance of distances per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Variance for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Compute variance for all distance pairs
        >>> pipeline.analysis.features.distances.variance()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_variance(data)
    
    def range(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute range (max - min) for each distance pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Range for each distance pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Find pairs with largest distance variations
        >>> pipeline.analysis.features.distances.range()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_range(data)
    
    def q25(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute 25th percentile for each distance pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            25th percentile for each distance pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Get first quartile distances
        >>> pipeline.analysis.features.distances.q25()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_q25(data)
    
    def q75(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute 75th percentile for each distance pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            75th percentile for each distance pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Get third quartile distances
        >>> pipeline.analysis.features.distances.q75()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_q75(data)
    
    def iqr(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute interquartile range (Q75 - Q25) for each distance pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Interquartile range for each distance pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Robust measure of distance spread
        >>> pipeline.analysis.features.distances.iqr()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_iqr(data)
    
    def mad(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median absolute deviation for each distance pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Median absolute deviation for each distance pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Robust measure of distance variability
        >>> pipeline.analysis.features.distances.mad()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mad(data)
    
    def cv(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coefficient of variation (CV = std/mean) for distances per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Coefficient of variation for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Find most variable distances (high CV)
        >>> pipeline.analysis.features.distances.cv()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_cv(data)
    
    # === PER-FRAME ANALYSIS METHODS ===
    
    def distances_per_frame_mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean distances per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Mean distance across all pairs for each frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Track average distance over trajectory
        >>> pipeline.analysis.features.distances.distances_per_frame_mean()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.distances_per_frame_mean(data)
    
    def distances_per_frame_std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation of distances per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Standard deviation across all pairs for each frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Track distance variability over trajectory
        >>> pipeline.analysis.features.distances.distances_per_frame_std()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.distances_per_frame_std(data)
    
    def distances_per_frame_min(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum distances per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Minimum distance across all pairs for each frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Track closest approach over trajectory
        >>> pipeline.analysis.features.distances.distances_per_frame_min()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.distances_per_frame_min(data)
    
    def distances_per_frame_max(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum distances per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Maximum distance across all pairs for each frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Track maximum separation over trajectory
        >>> pipeline.analysis.features.distances.distances_per_frame_max()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.distances_per_frame_max(data)
    
    def distances_per_frame_median(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median distances per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Median distance across all pairs for each frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Track robust average distance over trajectory
        >>> pipeline.analysis.features.distances.distances_per_frame_median()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.distances_per_frame_median(data)
    
    def distances_per_frame_range(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute range of distances per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Range (max - min) across all pairs for each frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Track distance spread over trajectory
        >>> pipeline.analysis.features.distances.distances_per_frame_range()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.distances_per_frame_range(data)
    
    def distances_per_frame_sum(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute sum of distances per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Sum of distances across all pairs for each frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Track total distance involvement over trajectory
        >>> pipeline.analysis.features.distances.distances_per_frame_sum()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.distances_per_frame_sum(data)
    
    # === PER-RESIDUE ANALYSIS METHODS ===
    
    def per_residue_mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean distance per residue (auto-converts condensed to squareform).
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Mean distance per residue
            
        Examples:
        ---------
        >>> # Find most connected residues
        >>> pipeline.analysis.features.distances.per_residue_mean()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_mean(data)
    
    def per_residue_std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation of distances per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Standard deviation per residue
            
        Examples:
        ---------
        >>> # Find most variable residues
        >>> pipeline.analysis.features.distances.per_residue_std()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_std(data)
    
    def per_residue_min(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum distance per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Minimum distance per residue
            
        Examples:
        ---------
        >>> # Find closest approach distances per residue
        >>> pipeline.analysis.features.distances.per_residue_min()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_min(data)
    
    def per_residue_max(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum distance per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Maximum distance per residue
            
        Examples:
        ---------
        >>> # Find maximum separation distances per residue
        >>> pipeline.analysis.features.distances.per_residue_max()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_max(data)
    
    def per_residue_median(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median distance per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Median distance per residue
            
        Examples:
        ---------
        >>> # Robust average distance per residue
        >>> pipeline.analysis.features.distances.per_residue_median()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_median(data)
    
    def per_residue_sum(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute sum of distances per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Sum of distances per residue
            
        Examples:
        ---------
        >>> # Total distance involvement per residue
        >>> pipeline.analysis.features.distances.per_residue_sum()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_sum(data)
    
    def per_residue_variance(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute variance of distances per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Variance per residue
            
        Examples:
        ---------
        >>> # Distance variance per residue
        >>> pipeline.analysis.features.distances.per_residue_variance()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_variance(data)
    
    def per_residue_range(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute range of distances per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Distance range per residue
            
        Examples:
        ---------
        >>> # Distance range per residue
        >>> pipeline.analysis.features.distances.per_residue_range()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_range(data)
    
    # === TRANSITIONS & STABILITY ===
    
    def transitions_lagtime(self, threshold: float = 2.0, lag_time: int = 10,
                           feature_selector: Optional[str] = None, 
                           traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute distance transitions using lag time analysis.
        
        Parameters:
        -----------
        threshold : float, default=2.0
            Distance threshold for transition detection in Angstroms
        lag_time : int, default=10
            Number of frames to look ahead for transitions
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Transition counts for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Detect transitions with 2Å threshold
        >>> pipeline.analysis.features.distances.transitions_lagtime(threshold=2.0)
        
        >>> # Use longer lag time for slower dynamics
        >>> pipeline.analysis.features.distances.transitions_lagtime(
        ...     threshold=3.0, lag_time=50
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_lagtime(data, threshold, lag_time)
    
    def transitions_window(self, threshold: float = 2.0, window_size: int = 10,
                          feature_selector: Optional[str] = None, 
                          traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute distance transitions within sliding window.
        
        Parameters:
        -----------
        threshold : float, default=2.0
            Distance threshold for transition detection in Angstroms
        window_size : int, default=10
            Size of sliding window for transition detection
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Transition counts for each pair with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Window-based transition detection
        >>> pipeline.analysis.features.distances.transitions_window(
        ...     threshold=3.0, window_size=20
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_window(data, threshold, window_size)
    
    def stability(self, threshold: float = 2.0, window_size: int = 10, mode: str = "lagtime",
                 feature_selector: Optional[str] = None, 
                 traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute distance stability (inverse of transition rate).
        
        Parameters:
        -----------
        threshold : float, default=2.0
            Distance threshold for stability detection in Angstroms
        window_size : int, default=10
            Window size for calculation
        mode : str, default='lagtime'
            Calculation mode ('lagtime' or 'window')
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Stability values per pair (0=unstable, 1=stable)
            
        Examples:
        ---------
        >>> # Find most stable distance pairs
        >>> pipeline.analysis.features.distances.stability(
        ...     threshold=2.5, window_size=15, mode='window'
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_stability(data, threshold, window_size, mode)
    
    # === COMPARISON ===
    
    def differences(self, feature_selector2: Optional[str] = None, 
                   traj_selection2: Optional[Union[str, int, List]] = None,
                   preprocessing_func: Optional[Callable] = None,
                   feature_selector: Optional[str] = None, 
                   traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute differences between two distance data selections.
        
        Parameters:
        -----------
        feature_selector2 : str, optional
            Second feature selector for comparison
        traj_selection2 : str, int, list, optional
            Second trajectory selection for comparison
        preprocessing_func : callable, optional
            Function to preprocess data before comparison
        feature_selector : str, optional
            First feature selector for column selection
        traj_selection : str, int, list, optional
            First trajectory selection for row selection
            
        Returns:
        --------
        np.ndarray
            Differences between the two selections
            
        Examples:
        ---------
        >>> # Compare same features across different trajectories
        >>> pipeline.analysis.features.distances.differences(
        ...     traj_selection=[0, 1],      # Wild type
        ...     traj_selection2=[2, 3]      # Mutant
        ... )
        
        >>> # Compare different feature selections
        >>> pipeline.analysis.features.distances.differences(
        ...     feature_selector="active_site",
        ...     feature_selector2="allosteric_site"
        ... )
        """
        data1 = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        data2 = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector2, traj_selection2
        )
        return self._calculator.compute_differences(data1, data2, preprocessing_func)
    
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

"""Service for coordinates-specific analysis operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING, Callable
import numpy as np

from ....services.helpers.analysis_data_helper import AnalysisDataHelper
from ..coordinates_calculator_analysis import CoordinatesCalculatorAnalysis
from ...interfaces.analysis_service_base import AnalysisServiceBase

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData


class CoordinatesAnalysisService(AnalysisServiceBase):
    """
    Service for coordinates-specific analysis operations.
    
    Provides a simplified interface for coordinate analysis methods by automatically
    handling data selection and forwarding method calls to the underlying
    calculator_analysis class. All methods from CoordinatesCalculatorAnalysis
    are automatically available with additional feature_selector and 
    traj_selection parameters.
    """
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """
        Initialize coordinates analysis service.

        Parameters
        ----------
        pipeline_data : PipelineData
            The pipeline data containing trajectory and feature information.

        Returns
        -------
        None
        """
        super().__init__(pipeline_data)
        self._feature_type = "coordinates"
        self._calculator = CoordinatesCalculatorAnalysis(
            use_memmap=pipeline_data.use_memmap,
            chunk_size=pipeline_data.chunk_size
        )
    
    # === PER-FEATURE METHODS (per atom coordinate) ===
    
    def mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean coordinates for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Mean coordinates for each atom (average structure)
            
        Examples
        --------
        >>> # Compute average structure coordinates
        >>> pipeline.analysis.features.coordinates.mean()
        
        >>> # Average structure for selected atoms
        >>> pipeline.analysis.features.coordinates.mean(feature_selector="ca_atoms")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mean(data)
    
    def std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation of coordinates for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Standard deviation for each coordinate dimension
            
        Examples
        --------
        >>> # Find most flexible atoms (high std)
        >>> pipeline.analysis.features.coordinates.std()
        
        >>> # Flexibility of backbone atoms
        >>> pipeline.analysis.features.coordinates.std(feature_selector="backbone")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_std(data)
    
    def min(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum coordinates for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Minimum coordinate values for each dimension
            
        Examples
        --------
        >>> # Find coordinate bounds (minimum)
        >>> pipeline.analysis.features.coordinates.min()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_min(data)
    
    def max(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum coordinates for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Maximum coordinate values for each dimension
            
        Examples
        --------
        >>> # Find coordinate bounds (maximum)
        >>> pipeline.analysis.features.coordinates.max()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_max(data)
    
    def median(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median coordinates for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Median coordinates for each atom
            
        Examples
        --------
        >>> # Robust average structure (median)
        >>> pipeline.analysis.features.coordinates.median()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_median(data)
    
    def variance(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute variance of coordinates for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Variance for each coordinate dimension
            
        Examples
        --------
        >>> # Coordinate variance per atom
        >>> pipeline.analysis.features.coordinates.variance()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_variance(data)
    
    def range(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coordinate range for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Coordinate range (max - min) for each dimension
            
        Examples
        --------
        >>> # Find atoms with largest coordinate changes
        >>> pipeline.analysis.features.coordinates.range()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_range(data)
    
    def mad(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median absolute deviation of coordinates for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Median absolute deviation for each coordinate dimension
            
        Examples
        --------
        >>> # Robust measure of coordinate variability
        >>> pipeline.analysis.features.coordinates.mad()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mad(data)
    
    def cv(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coefficient of variation for each coordinate dimension.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Coefficient of variation for each coordinate dimension
            
        Examples
        --------
        >>> # Find most variable coordinates (high CV)
        >>> pipeline.analysis.features.coordinates.cv()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_cv(data)
    
    # === PER-FRAME ANALYSIS METHODS ===
    
    def coordinates_per_frame_mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean coordinates per frame.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Mean coordinate across all coordinates for each frame with shape (n_frames,)
            
        Examples
        --------
        >>> # Track average coordinate values over trajectory
        >>> pipeline.analysis.features.coordinates.coordinates_per_frame_mean()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.coordinates_per_frame_mean(data)
    
    def coordinates_per_frame_std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation of coordinates per frame.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Standard deviation across all coordinates for each frame with shape (n_frames,)
            
        Examples
        --------
        >>> # Track coordinate variability over trajectory
        >>> pipeline.analysis.features.coordinates.coordinates_per_frame_std()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.coordinates_per_frame_std(data)
    
    def coordinates_per_frame_min(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum coordinates per frame.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Minimum coordinate across all coordinates for each frame with shape (n_frames,)
            
        Examples
        --------
        >>> # Track minimum coordinate values over trajectory
        >>> pipeline.analysis.features.coordinates.coordinates_per_frame_min()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.coordinates_per_frame_min(data)
    
    def coordinates_per_frame_max(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum coordinates per frame.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Maximum coordinate across all coordinates for each frame with shape (n_frames,)
            
        Examples
        --------
        >>> # Track maximum coordinate values over trajectory
        >>> pipeline.analysis.features.coordinates.coordinates_per_frame_max()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.coordinates_per_frame_max(data)
    
    def coordinates_per_frame_range(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute range of coordinates per frame.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Range (max - min) across all coordinates for each frame with shape (n_frames,)
            
        Examples
        --------
        >>> # Track coordinate spread over trajectory
        >>> pipeline.analysis.features.coordinates.coordinates_per_frame_range()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.coordinates_per_frame_range(data)
    
    def rmsf(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute root mean square fluctuation (RMSF) for each atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            RMSF values for each atom
            
        Examples
        --------
        >>> # Compute atomic flexibility (RMSF)
        >>> pipeline.analysis.features.coordinates.rmsf()
        
        >>> # RMSF for CA atoms only
        >>> pipeline.analysis.features.coordinates.rmsf(feature_selector="ca_atoms")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_rmsf(data)
    
    # === TRANSITIONS/DYNAMICS METHODS ===
    
    def transitions_lagtime(self, threshold: float = 1.0, lag_time: int = 10,
                           feature_selector: Optional[str] = None, 
                           traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coordinate transitions with lag time for each atom.
        
        Parameters
        ----------
        threshold : float, default=1.0
            Distance threshold for detecting transitions (in Ångstroms)
        lag_time : int, default=10
            Number of frames to look ahead
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per atom
            
        Examples
        --------
        >>> # Find dynamic atoms with 1.0 Å threshold
        >>> pipeline.analysis.features.coordinates.transitions_lagtime(threshold=1.0)
        
        >>> # Slower dynamics with longer lag time
        >>> pipeline.analysis.features.coordinates.transitions_lagtime(
        ...     threshold=2.0, lag_time=50
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_lagtime(data, threshold, lag_time)
    
    def transitions_window(self, threshold: float = 1.0, window_size: int = 10,
                          feature_selector: Optional[str] = None, 
                          traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coordinate transitions within sliding window for each atom.
        
        Parameters
        ----------
        threshold : float, default=1.0
            Distance threshold for detecting transitions (in Ångstroms)
        window_size : int, default=10
            Size of sliding window
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per atom
            
        Examples
        --------
        >>> # Window-based transition detection
        >>> pipeline.analysis.features.coordinates.transitions_window(
        ...     threshold=1.0, window_size=10
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_window(data, threshold, window_size)
    
    def stability(self, threshold: float = 1.0, window_size: int = 10,
                 feature_selector: Optional[str] = None, 
                 traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coordinate stability (inverse of transition rate) for each atom.
        
        Parameters
        ----------
        threshold : float, default=1.0
            Distance threshold for stability detection (in Ångstroms)
        window_size : int, default=10
            Window size for calculation
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Stability values per atom (0=unstable, 1=stable)
            
        Examples
        --------
        >>> # Find most stable atoms
        >>> pipeline.analysis.features.coordinates.stability(
        ...     threshold=1.0, window_size=10
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_stability(data, threshold, window_size)
    
    # === COMPARISON METHODS ===
    
    def differences(self, feature_selector2: Optional[str] = None, 
                   traj_selection2: Optional[Union[str, int, List]] = None,
                   preprocessing_func: Optional[Callable] = None,
                   feature_selector: Optional[str] = None, 
                   traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute differences between two coordinate datasets.
        
        Parameters
        ----------
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
            
        Returns
        -------
        np.ndarray
            Coordinate differences between datasets
            
        Examples
        --------
        >>> # Compare structures between conditions
        >>> pipeline.analysis.features.coordinates.differences(
        ...     traj_selection=[0, 1],      # Native state
        ...     traj_selection2=[2, 3]      # Denatured state
        ... )
        
        >>> # Compare different atom selections
        >>> pipeline.analysis.features.coordinates.differences(
        ...     feature_selector="ca_atoms",
        ...     feature_selector2="ca_atoms",
        ...     preprocessing_func=lambda x: np.mean(x, axis=0)  # Compare mean structures
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
    
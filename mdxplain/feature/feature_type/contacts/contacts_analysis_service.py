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

"""Factory for contacts-specific analysis operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING, Callable
import numpy as np

from ...services.helpers.analysis_data_helper import AnalysisDataHelper
from .contact_calculator_analysis import ContactCalculatorAnalysis
from ..interfaces.analysis_service_base import AnalysisServiceBase

if TYPE_CHECKING:
    from ....pipeline.entities.pipeline_data import PipelineData


class ContactsAnalysisService(AnalysisServiceBase):
    """
    Service for contacts-specific analysis operations.
    
    Provides a simplified interface for contact analysis methods by automatically
    handling data selection and forwarding method calls to the underlying
    calculator_analysis class. All methods from ContactCalculatorAnalysis
    are automatically available with additional feature_selector and 
    traj_selection parameters.
    """
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """Initialize contacts analysis service."""
        super().__init__(pipeline_data)
        self._feature_type = "contacts"
        self._calculator = ContactCalculatorAnalysis(
            use_memmap=pipeline_data.use_memmap,
            chunk_size=pipeline_data.chunk_size
        )
    
    # === BASIC CONTACT ANALYSIS ===
    
    def frequency(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute contact frequency per pair.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Contact frequency for each pair (0.0 to 1.0) with shape (n_pairs,)
            
        Examples:
        ---------
        >>> # Compute frequency for all contact pairs
        >>> pipeline.analysis.features.contacts.frequency()
        
        >>> # Compute frequency for selected contacts
        >>> pipeline.analysis.features.contacts.frequency(feature_selector="stable_contacts")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_frequency(data)
    
    # === PER-FRAME ANALYSIS ===
    
    def contacts_per_frame_abs(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute absolute number of contacts per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Total contact count per frame with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Count total contacts per frame
        >>> pipeline.analysis.features.contacts.contacts_per_frame_abs()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.contacts_per_frame_abs(data)
    
    def contacts_per_frame_percentage(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute percentage of contacts per frame.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Fraction of pairs in contact per frame (0.0 to 1.0) with shape (n_frames,)
            
        Examples:
        ---------
        >>> # Get contact percentage per frame
        >>> pipeline.analysis.features.contacts.contacts_per_frame_percentage()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.contacts_per_frame_percentage(data)
    
    # === PER-RESIDUE ANALYSIS ===
    
    def per_residue_mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean contact frequency per residue.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Mean contact frequency per residue
            
        Examples:
        ---------
        >>> # Find most connected residues
        >>> pipeline.analysis.features.contacts.per_residue_mean()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_mean(data)
    
    def per_residue_std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation of contacts per residue. Auto-converts condensed to squareform.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Standard deviation of contacts per residue
            
        Examples:
        ---------
        >>> # Find residues with variable contact patterns
        >>> pipeline.analysis.features.contacts.per_residue_std()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_std(data)
    
    def per_residue_sum(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute total contact count per residue. Auto-converts condensed to squareform.
        
        Parameters:
        -----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Total contact count per residue
            
        Examples:
        ---------
        >>> # Find most connected residues (total contacts)
        >>> pipeline.analysis.features.contacts.per_residue_sum()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_per_residue_sum(data)
    
    # === ADVANCED ANALYSIS ===
    
    def transitions_lagtime(self, threshold: int = 1, lag_time: int = 1,
                           feature_selector: Optional[str] = None, 
                           traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute contact transitions using lag time analysis.
        
        Parameters:
        -----------
        threshold : int, default=1
            Threshold for detecting transitions (contact changes)
        lag_time : int, default=1
            Number of frames to look ahead for transitions
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Transition counts per contact pair
            
        Examples:
        ---------
        >>> # Find dynamic contacts (frequent breaking/forming)
        >>> pipeline.analysis.features.contacts.transitions_lagtime()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_lagtime(data, threshold, lag_time)
    
    def transitions_window(self, threshold: int = 1, window_size: int = 10,
                          feature_selector: Optional[str] = None, 
                          traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute contact transitions within sliding window for each contact pair.
        
        Parameters:
        -----------
        threshold : int, default=1
            Threshold for detecting transitions (contact changes)
        window_size : int, default=10
            Size of sliding window for transition detection
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Transition counts per contact pair
            
        Examples:
        ---------
        >>> # Window-based transition detection for contacts
        >>> pipeline.analysis.features.contacts.transitions_window(
        ...     threshold=1, window_size=20
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_window(data, threshold, window_size)
    
    def stability(self, threshold: int = 1, window_size: int = 1,
                 feature_selector: Optional[str] = None, 
                 traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute contact stability (inverse of transition rate) for each contact pair.
        
        Parameters:
        -----------
        threshold : int, default=1
            Threshold for detecting stability
        window_size : int, default=1
            Window size for stability calculation
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns:
        --------
        np.ndarray
            Stability values per contact pair (higher = more stable)
            
        Examples:
        ---------
        >>> # Find most stable contacts
        >>> pipeline.analysis.features.contacts.stability(threshold=1, window_size=5)
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_stability(data, threshold, window_size)
    
    def differences(self, feature_selector2: Optional[str] = None, 
                   traj_selection2: Optional[Union[str, int, List]] = None,
                   preprocessing_func: Optional[Callable] = None,
                   feature_selector: Optional[str] = None, 
                   traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute differences between two contact data selections.
        
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
        >>> # Compare contacts between conditions
        >>> pipeline.analysis.features.contacts.differences(
        ...     traj_selection=[0, 1],      # Native state
        ...     traj_selection2=[2, 3]      # Denatured state
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
    
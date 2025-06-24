# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# contact_calculator - MD Trajectory Contact Analysis
#
# Utility class for computing contact maps from distance arrays.
# All methods are static and can be used without instantiation.
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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

from ..helper.FeatureShapeHelper import FeatureShapeHelper
from ..helper.CalculatorComputeHelper import CalculatorComputeHelper
from ..interfaces.CalculatorBase import CalculatorBase
from .contact_calculator_analysis import ContactCalculatorAnalysis

class ContactCalculator(CalculatorBase):
    """
    Utility class for computing contact maps from distance arrays.
    All methods are static and can be used without instantiation.
    """

    def __init__(self, use_memmap=False, cache_path=None, 
                chunk_size=None):
        super().__init__(use_memmap, cache_path, chunk_size)
        self.contacts_path = cache_path
        self.chunk_size = chunk_size
        self.use_memmap = use_memmap

        self.analysis = ContactCalculatorAnalysis(chunk_size=self.chunk_size)
    
    # ===== MAIN COMPUTATION METHOD =====
    
    def compute(self, distances, cutoff=4.5, squareform=False, k=0):
        """
        Compute contact maps from distance arrays.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (NxMxM for square form or NxP for condensed form)
        cutoff : float, default=4.5
            Distance cutoff for contacts (in Angstrom)
        use_memmap : bool, default=False
            Whether to use memory mapping
        contacts_path : str, optional
            Path for memory-mapped contact array
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)
        squareform : bool, default=True
            If True, output NxMxM. If False, output NxP (upper triangular)
        k : int, default=0
            Diagonal offset: 0=include diagonal, 1=exclude diagonal, >1=exclude additional diagonals
            
        Returns:
        --------
        numpy.ndarray
            Boolean contact array
        """
        # Calculate dimensions and conversion requirements
        output_shape, n_residues = CalculatorComputeHelper.calculate_output_dimensions(
            distances.shape, squareform, k)
        
        # Create output array
        contacts = CalculatorComputeHelper.create_output_array(self.use_memmap, self.contacts_path, output_shape, dtype=bool)
        
        # Process in chunks
        if self.chunk_size is None:
            self.chunk_size = distances.shape[0]
        
        for i in range(0, distances.shape[0], self.chunk_size):
            end_idx = min(i + self.chunk_size, distances.shape[0])
            chunk_contacts = distances[i:end_idx] <= cutoff
            
            # Convert format if needed
            if len(distances.shape) == 3 and not squareform:
                chunk_contacts = FeatureShapeHelper.squareform_to_condensed(chunk_contacts, k=k, chunk_size=self.chunk_size)
            elif len(distances.shape) == 2 and squareform:
                chunk_contacts = FeatureShapeHelper.condensed_to_squareform(chunk_contacts, n_residues, k=k, chunk_size=self.chunk_size)
            
            contacts[i:end_idx] = chunk_contacts
        
        return contacts

    def _compute_metric_values(self,contacts, metric, threshold, window_size, 
                              transition_mode='window', lag_time=1):
        """
        Compute metric values for contacts based on specified metric type.
        
        Parameters:
        -----------
        contacts : numpy.ndarray
            Contact array
        metric : str
            Metric type to compute
        threshold : float
            Threshold value (used for transitions metric)
        window_size : int
            Window size for transitions metric
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime'
        lag_time : int, default=1
            Lag time for transitions metric
        Returns:
        --------
        numpy.ndarray
            Computed metric values
        """
        if metric == 'frequency':
            return self.analysis.compute_frequency(contacts)
        elif metric == 'stability':
            return self.analysis.compute_stability(contacts)
        elif metric == 'transitions':
            # For transitions, use threshold as transition threshold (default 2.0 Å)
            if transition_mode == 'window':
                return self.analysis.compute_transitions_window(contacts, threshold=threshold, window_size=window_size)
            elif transition_mode == 'lagtime':
                return self.analysis.compute_transitions_lagtime(contacts, threshold=threshold, lag_time=lag_time)
            else:
                raise ValueError(f"Unknown transition mode: {transition_mode}. Supported: 'window', 'lagtime'")
        else:
            supported_metrics = ['frequency', 'stability', 'transitions']
            raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

    def compute_dynamic_values(self, input_data, metric='frequency', threshold_min=None, threshold_max=None, 
                              transition_threshold=2.0, window_size=10, feature_names=None, output_path=None,
                              transition_mode='window', lag_time=1):
        """Filter and select dynamic contacts based on specified criteria.
        
        Parameters:
        -----------
        contacts : numpy.ndarray
            Contact array
        metric : str, default='frequency'
            Metric to use for selection:
            - 'frequency': Frequency of contacts
            - 'stability': Stability of contacts
            - 'transitions': Number of transitions
        threshold_min : float, optional
            Minimum threshold for filtering (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold for filtering (metric_values <= threshold_max)
        transition_threshold : float, default=2.0
            Distance threshold for detecting transitions
            Only used for 'transitions' metric to compute the number of transitions
        feature_names : list, optional
            Names for distance pairs
        output_path : str, optional
            Path for memory-mapped output
        window_size : int, default=10
            Window size for transitions metric
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime'
        lag_time : int, default=1
            Lag time for transitions metric
            
        Returns:
        --------
        dict
            Dictionary containing filtered data and metadata
        """
        # Compute metric values using helper method
        metric_values = self._compute_metric_values(
            input_data, metric, transition_threshold, window_size, 
            transition_mode, lag_time
        )
        
        # Use the common helper
        return CalculatorComputeHelper.compute_dynamic_values(
            data=input_data,
            metric_values=metric_values,
            metric_name=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            feature_names=feature_names,
            use_memmap=self.use_memmap,
            output_path=output_path,
            chunk_size=self.chunk_size
        )

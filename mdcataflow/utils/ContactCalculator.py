# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# ContactCalculator - MD Trajectory Contact Analysis
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

import numpy as np
from .ArrayHandler import ArrayHandler
from .CalculatorStatHelper import CalculatorStatHelper
from .CalculatorComputeHelper import CalculatorComputeHelper

class ContactCalculator():
    """
    Utility class for computing contact maps from distance arrays.
    All methods are static and can be used without instantiation.
    """

    def __init__(self, use_memmap=False, contacts_path=None, 
                chunk_size=None, squareform=True, k=0):
        self.use_memmap = use_memmap
        self.contacts_path = contacts_path
        self.chunk_size = chunk_size
        self.squareform = squareform
        self.k = k
        
        # Create stat toolbox
        self.stat = self._create_stat_toolbox_registry()
    
    def _create_stat_toolbox_registry(self):
        """Create stat toolbox with methods bound to this instance."""
        return type('stat', (), {
            # === PAIR-BASED STATISTICS ===
            'compute_frequency': lambda contacts: CalculatorStatHelper.compute_func_per_pair(contacts, np.mean, self.chunk_size),
          
            # === FRAME-BASED STATISTICS ===
            'contacts_per_frame_abs': lambda contacts: CalculatorStatHelper.compute_func_per_frame(contacts, self.chunk_size, np.sum),
            'contacts_per_frame_percentage': lambda contacts: CalculatorStatHelper.compute_func_per_frame(contacts, self.chunk_size, np.mean),
            
            # === RESIDUE-BASED STATISTICS (only for square format) ===
            'compute_per_residue_mean': lambda contacts: CalculatorStatHelper.compute_func_per_residue(contacts, np.mean, self.chunk_size),
            'compute_per_residue_std': lambda contacts: CalculatorStatHelper.compute_func_per_residue(contacts, np.std, self.chunk_size),
            'compute_per_residue_sum': lambda contacts: CalculatorStatHelper.compute_func_per_residue(contacts, np.sum, self.chunk_size),
            
            # === TRANSITION ANALYSIS ===
            'compute_transitions_lagtime': lambda contacts, threshold=1, lag_time=1: CalculatorStatHelper.compute_transitions_within_lagtime(contacts, threshold, lag_time, self.chunk_size),
            'compute_transitions_window': lambda contacts, threshold=1, window_size=10: CalculatorStatHelper.compute_transitions_within_window(contacts, threshold, window_size, self.chunk_size),
            'compute_stability': lambda contacts, threshold=1, window_size=1: CalculatorStatHelper.compute_stability(contacts, threshold, window_size, self.chunk_size),
            
            # === COMPARISON METHODS ===
            'compute_differences': lambda contacts1, contacts2, preprocessing_func=None: CalculatorStatHelper.compute_differences(contacts1, contacts2, self.chunk_size, preprocessing_func),
        })()
    
    # ===== MAIN COMPUTATION METHOD =====
    
    def compute(self, distances, cutoff=4.5):
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
            distances.shape, self.squareform, self.k, "bool")
        
        # Create output array
        contacts = CalculatorComputeHelper.create_output_array(self.use_memmap, self.contacts_path, output_shape)
        
        # Process in chunks
        if self.chunk_size is None:
            self.chunk_size = distances.shape[0]
        
        for i in range(0, distances.shape[0], self.chunk_size):
            end_idx = min(i + self.chunk_size, distances.shape[0])
            chunk_contacts = distances[i:end_idx] <= cutoff
            
            # Convert format if needed
            if len(distances.shape) == 3 and not self.squareform:
                chunk_contacts = ArrayHandler.squareform_to_condensed(chunk_contacts, k=self.k, chunk_size=self.chunk_size)
            elif len(distances.shape) == 2 and self.squareform:
                chunk_contacts = ArrayHandler.condensed_to_squareform(chunk_contacts, n_residues, k=self.k, chunk_size=self.chunk_size)
            
            contacts[i:end_idx] = chunk_contacts
        
        return contacts

    def _compute_metric_values(self,contacts, metric):
        """
        Compute metric values for contacts based on specified metric type.
        
        Parameters:
        -----------
        contacts : numpy.ndarray
            Contact array
        metric : str
            Metric type to compute
        chunk_size : int
            Chunk size for processing
            
        Returns:
        --------
        numpy.ndarray
            Computed metric values
        """
        if metric == 'frequency':
            return self.stat.compute_frequency(contacts, np.mean, self.chunk_size)
        elif metric == 'stability':
            return self.stat.compute_stability(contacts, self.chunk_size)
        else:
            supported_metrics = ['frequency', 'stability']
            raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

    def compute_dynamic_values(self, contacts, metric='frequency', threshold_min=0.1, threshold_max=0.9, 
                              feature_names=None, reduced_contacts_path=None):
        """Filter and select dynamic contacts based on specified criteria.
        
        Parameters:
        -----------
        contacts : numpy.ndarray
            Contact array
        metric : str, default='frequency'
            Metric to use for selection:
            - 'frequency': Frequency of contacts
            - 'stability': Stability of contacts
        threshold_min : float, default=0.1
            Minimum threshold for filtering
        threshold_max : float, default=0.9
            Maximum threshold for filtering
        feature_names : list, optional
            Names for contact pairs
        use_memmap : bool, default=False
            Whether to use memory mapping
        reduced_contacts_path : str, optional
            Path for memory-mapped output
        chunk_size : int, default=1000
            Chunk size for processing
            
        Returns:
        --------
        dict
            Dictionary containing filtered data and metadata
        """
        # Compute metric values using helper method
        metric_values = self._compute_metric_values(contacts, metric)
        
        # Use the common helper
        return CalculatorComputeHelper.compute_dynamic_values(
            data=contacts,
            metric_values=metric_values,
            metric_name=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            feature_names=feature_names,
            use_memmap=self.use_memmap,
            output_path=reduced_contacts_path,
            chunk_size=self.chunk_size
        )

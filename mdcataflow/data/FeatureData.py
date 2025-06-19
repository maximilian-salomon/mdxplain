# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# FeatureData - Feature Data Container
#
# Container for feature data (distances, contacts) with associated calculator.
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

import os
from ..utils.DistanceCalculator import DistanceCalculator
from ..utils.ContactCalculator import ContactCalculator


class FeatureData:
    """
    Container for feature data with associated calculator and statistics helper.
    """
    
    def __init__(self, feature_type, use_memmap=False, cache_path=None, 
                chunk_size=None, squareform=True, k=0):
        """
        Initialize feature data container.
        
        Parameters:
        -----------
        feature_type : str
            Type of feature ('distances', 'contacts')
        use_memmap : bool, default=False
            Whether to use memory mapping for data storage
        cache_path : str, optional
            Path for cache file. If None and use_memmap=True, uses ./cache/<feature_type>.dat
        """
        if feature_type not in ['distances', 'contacts', 'cci']:
            raise ValueError(f"feature_type must be one of ['distances', 'contacts', 'cci'], got {feature_type}")
        
        self.feature_type = feature_type
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.squareform = squareform
        self.k = k
        
        # Set calculator based on feature type
        self.calculator_factory()
        
        # Handle cache path
        if use_memmap:
            if cache_path is None:
                os.makedirs("./cache", exist_ok=True)
                cache_path = f"./cache/{feature_type}.dat"
            self.cache_path = cache_path
            self.reduced_cache_path = f"{os.path.splitext(self.cache_path)[0]}_reduced.dat"
        else:
            self.cache_path = None
            self.reduced_cache_path = None
            
        # Initialize data as None
        self.data = None
        self.res_list = None
        self.reduced_data = None
        self.reduced_res_list = None
        self.reduction_info = None
        
        # Bind stats methods from calculator
        self._bind_stats_methods()

    def calculator_factory(self):
        if self.feature_type == 'distances':
            self.calculator = DistanceCalculator(use_memmap=self.use_memmap, 
                                                 cache_path=self.cache_path, 
                                                 chunk_size=self.chunk_size, 
                                                 squareform=self.squareform, 
                                                 k=self.k)
        elif self.feature_type == 'contacts':
            self.calculator = ContactCalculator(use_memmap=self.use_memmap, 
                                                cache_path=self.cache_path, 
                                                chunk_size=self.chunk_size, 
                                                squareform=self.squareform, 
                                                k=self.k)
        elif self.feature_type == 'cci':
            self.calculator = CCICalculator(use_memmap=self.use_memmap, 
                                                cache_path=self.cache_path, 
                                                chunk_size=self.chunk_size, 
                                                squareform=self.squareform, 
                                                k=self.k)
        
    def _bind_stats_methods(self):
        """Bind stat methods from calculator to self.stats with automatic parameter passing."""
        if hasattr(self.calculator, 'stat'):
            # Create a simple object to hold bound methods
            self.stats = type('BoundStats', (), {})()
            
            # Bind each method from calculator.stat
            for method_name in dir(self.calculator.stat):
                if not method_name.startswith('_') and callable(getattr(self.calculator.stat, method_name)):
                    method = getattr(self.calculator.stat, method_name)
                    # Create bound method that automatically uses reduced_data if available, else data
                    bound_method = lambda m=method: m(self.reduced_data if self.reduced_data is not None else self.data)
                    setattr(self.stats, method_name, bound_method)

    def compute(self, **kwargs):
        """
        Compute feature data using the associated calculator.
        
        Parameters:
        -----------
        *args : arguments
            Arguments to pass to calculator.compute()
        **kwargs : keyword arguments  
            Keyword arguments to pass to calculator.compute()
        """
        if self.feature_type == 'cci' or self.feature_type == 'contacts':
            if 'distances' not in kwargs:
                raise ValueError("Distances must be provided for cci or contacts")
            else:
                kwargs['distances'] = kwargs['distances'].data
                self.res_list = kwargs['distances'].res_list

        if self.feature_type == 'contacts':
            if 'cutoff' not in kwargs:
                raise ValueError("For contacts, cutoff must be provided")
        
        # Call the compute method of the associated calculator
        result = self.calculator.compute(**kwargs)
        
        # Handle different return types based on feature type
        if self.feature_type == 'distances':
            self.data, self.res_list = result
        elif self.feature_type == 'contacts' or self.feature_type == 'cci':
            self.data = result

    def reduce_data(self, metric='cv', threshold_min=None, threshold_max=None,
                    transition_threshold=2.0, window_size=10, transition_mode='window', lag_time=1):
        """
        Reduce data using dynamic value filtering and store in self.reduced_data.
        
        Parameters:
        -----------
        metric : str, default='cv'
            Metric to use for selection. Depending on feature type
            For distances: 'cv', 'std', 'variance', 'range' and 'transitions'
            For contacts: 'frequency' and 'stability'
        threshold_min : float, optional
            Minimum threshold for filtering
        threshold_max : float, optional
            Maximum threshold for filtering
        transition_threshold : float, default=2.0
            Distance threshold for detecting transitions (only for 'transitions' metric)
        window_size : int, default=10
            Window size for transitions metric (only for 'transitions' metric)
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime' (only for 'transitions' metric)
        lag_time : int, default=1
            Lag time for transitions metric (only for 'transitions' metric)
        """
        if self.data is None:
            raise ValueError("No data available. Call compute() first.")
        
        # Get reduction results from calculator
        results = self.calculator.compute_dynamic_values(
            distances=self.data,
            metric=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            feature_names=self.res_list,
            output_path=self.reduced_cache_path,
            transition_mode=transition_mode,
            lag_time=lag_time
        )
        
        # Simply set reduced_data
        self.reduced_data = results['dynamic_data']
        self.reduced_res_list = results['feature_names']
        self.reduction_info = results['n_dynamic'] / results['total_pairs']
        
        print(f"Now using reduced data. "
              f"Data reduced from {self.data.shape} to {self.reduced_data.shape}. "
              f"({self.reduction_info:.1%} retained).")

    def reset_reduction(self):
        """
        Reset feature reduction and return to using full original data.
        
        Examples:
        --------
        # Apply reduction
        features.reduce_features(metric='cv', threshold_min=0.5)
        print(features.active_data.shape)  # Reduced shape
        
        # Reset to full data
        features.reset_reduction()
        print(features.active_data.shape)  # Original shape
        """
        if self.reduced_data is None:
            print("No reduction to reset - already using full data.")
            return
        
        # Clear reduced data
        original_shape = self.data.shape
        reduced_shape = self.reduced_data.shape
        
        self.reduced_data = None
        self.reduced_res_list = None
        old_info = self.reduction_info
        self.reduction_info = None
        
        print(f"Reset reduction: Now using full data {original_shape}. "
              f"(Data was reduced to {reduced_shape}, {old_info:.1%})")

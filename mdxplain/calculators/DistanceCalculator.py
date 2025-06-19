# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# DistanceCalculator - MD Trajectory Distance Analysis
#
# Utility class for computing distances between residues in MD trajectories.
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
import os
import mdtraj as md
from tqdm import tqdm
from .CalculatorComputeHelper import CalculatorComputeHelper
from .CalculatorStatHelper import CalculatorStatHelper
from .FeatureShapeHelper import FeatureShapeHelper


class DistanceCalculator():
    """
    Utility class for computing distances between residues in MD trajectories.
    """
    
    def __init__(self, use_memmap=False, cache_path=None, 
                chunk_size=None, squareform=True, k=0):
        self.use_memmap = use_memmap
        self.distances_path = cache_path
        self.chunk_size = chunk_size
        self.squareform = squareform
        self.k = k
        self.pairs = None  # Will be computed from reference trajectory
        
        # Create stat toolbox
        self.stat = self._create_stat_toolbox_registry()
    
    def _create_stat_toolbox_registry(self):
        """Create stat toolbox with methods bound to this instance."""
        return type('stat', (), {
            # === PAIR-BASED STATISTICS ===
            'compute_mean': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, np.mean, self.chunk_size),
            'compute_std': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, np.std, self.chunk_size),
            'compute_min': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, np.min, self.chunk_size),
            'compute_max': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, np.max, self.chunk_size),
            'compute_median': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, np.median, self.chunk_size),
            'compute_variance': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, np.var, self.chunk_size),
            'compute_range': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, np.ptp, self.chunk_size),  # Peak-to-peak (range)
            'compute_q25': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, lambda x, axis: np.percentile(x, 25, axis=axis), self.chunk_size),
            'compute_q75': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, lambda x, axis: np.percentile(x, 75, axis=axis), self.chunk_size),
            'compute_iqr': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, lambda x, axis: np.percentile(x, 75, axis=axis) - np.percentile(x, 25, axis=axis), self.chunk_size),
            'compute_mad': lambda distances: CalculatorStatHelper.compute_func_per_pair(distances, lambda x, axis: np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis), self.chunk_size),  # Median Absolute Deviation
          
            # === FRAME-BASED STATISTICS ===
            'distances_per_frame_mean': lambda distances: CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.mean),
            'distances_per_frame_std': lambda distances: CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.std),
            'distances_per_frame_min': lambda distances: CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.min),
            'distances_per_frame_max': lambda distances: CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.max),
            'distances_per_frame_median': lambda distances: CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.median),
            'distances_per_frame_range': lambda distances: CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.ptp),
            'distances_per_frame_sum': lambda distances: CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.sum),
            
            # === RESIDUE-BASED STATISTICS (only for square format) ===
            'compute_per_residue_mean': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.mean, self.chunk_size),
            'compute_per_residue_std': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.std, self.chunk_size),
            'compute_per_residue_min': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.min, self.chunk_size),
            'compute_per_residue_max': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.max, self.chunk_size),
            'compute_per_residue_median': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.median, self.chunk_size),
            'compute_per_residue_sum': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.sum, self.chunk_size),
            'compute_per_residue_variance': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.var, self.chunk_size),
            'compute_per_residue_range': lambda distances: CalculatorStatHelper.compute_func_per_residue(distances, np.ptp, self.chunk_size),
            
            # === TRANSITION ANALYSIS ===
            'compute_transitions_lagtime': lambda distances, threshold=2.0, lag_time=10: CalculatorStatHelper.compute_transitions_within_lagtime(distances, threshold, lag_time, self.chunk_size),
            'compute_transitions_window': lambda distances, threshold=2.0, window_size=10: CalculatorStatHelper.compute_transitions_within_window(distances, threshold, window_size, self.chunk_size),
            'compute_stability': lambda distances, threshold=2.0, window_size=10, mode='window': CalculatorStatHelper.compute_stability(distances, threshold, window_size, self.chunk_size, mode),
            
            # === COMPARISON METHODS ===
            'compute_differences': lambda distances1, distances2, preprocessing_func=None: CalculatorStatHelper.compute_differences(distances1, distances2, self.chunk_size, preprocessing_func),
        })()

    # ===== MAIN COMPUTATION METHOD =====
    
    def compute(self, trajectories, ref=None):
        """
        Compute pairwise distances between all residues for all frames in trajectories.
        
        Parameters:
        -----------
        trajectories : list
            List of MDTraj trajectory objects
        ref : mdtraj.Trajectory, optional
            Reference trajectory for residue information. If None, uses first trajectory.
            
        Returns:
        --------
        tuple : (distances, res_list)
            distances : numpy.ndarray
                Distance array in condensed format (total_frames, n_pairs) or 
                square format (total_frames, n_residues, n_residues) based on self.squareform
            res_list : list
                List of residue pairs used for distance computation
        """
        # Setup computation parameters and arrays
        ref, total_frames, distances, condensed_path = self._setup_computation(trajectories, ref)
        
        # Process all trajectories
        distances, res_list = self._process_all_trajectories(trajectories, distances, total_frames)
        
        # Finalize output (convert units, format, cleanup)
        distances = self._finalize_output(distances, res_list, total_frames, condensed_path)
        
        return distances, res_list

    def _setup_computation(self, trajectories, ref):
        """Setup computation parameters and create output arrays."""
        total_frames = sum(traj.n_frames for traj in trajectories)

        if ref is None:
            ref = trajectories[0]
        
        # Generate pairs once from reference trajectory
        if self.pairs is None:
            self.pairs = self._generate_residue_pairs(ref.n_residues)
            print(f"Generated {len(self.pairs)} residue pairs for {ref.n_residues} residues")
        
        # Calculate condensed output shape (natural md.compute_contacts format)
        self.n_pairs = len(self.pairs)
        
        # Determine paths for memmap handling
        condensed_path = self.distances_path
        if self.squareform and self.use_memmap and self.distances_path is not None:
            # If squareform is requested, use temporary path for condensed
            condensed_path = self.distances_path.replace('.dat', '_condensed.dat')
        
        # Create output array in condensed format
        distances = CalculatorComputeHelper.create_output_array(
            self.use_memmap, condensed_path, (total_frames, self.n_pairs), dtype='float32'
        )
        
        return ref, total_frames, distances, condensed_path

    def _process_all_trajectories(self, trajectories, distances, total_frames):
        """Process all trajectories and compute distances."""
        frame_idx = 0
        res_list = None
        
        with tqdm(total=total_frames, desc="Computing distances") as pbar:
            for traj in trajectories:
                distances, frame_idx, res_list = self._process_trajectory(
                    traj, distances, frame_idx, pbar
                )
        
        return distances, res_list

    def _finalize_output(self, distances, res_list, total_frames, condensed_path):
        """Convert units, format output, and cleanup temporary files."""
        # Convert to Angstrom
        self._convert_to_angstrom(distances, total_frames)

        # Convert to squareform only if requested
        if self.squareform:
            distances = FeatureShapeHelper.condensed_to_squareform(
                distances, res_list, chunk_size=self.chunk_size, output_path=self.distances_path
            )
            
            # Clean up temporary condensed memmap if it was created
            if self.use_memmap and condensed_path != self.distances_path:
                if os.path.exists(condensed_path):
                    os.remove(condensed_path)
        
        return distances
       
    # ===== PRIVATE HELPER METHODS =====
    def _generate_residue_pairs(self, n_residues):
        """
        Generate all unique residue pairs (excluding self-pairs i==j).
        
        Parameters:
        -----------
        n_residues : int
            Number of residues
            
        Returns:
        --------
        list
            List of [i, j] pairs where i < j
        """
        pairs = []
        for i in range(n_residues):
            for j in range(i + 1, n_residues):  # i < j, excludes self-pairs
                pairs.append([i, j])
        return pairs

    def _convert_to_angstrom(self, distances, total_frames):
        """Convert distances from nm to Angstrom."""
        if FeatureShapeHelper.is_memmap(distances) and self.chunk_size is not None:
            for i in range(0, total_frames, self.chunk_size):
                end_idx = min(i + self.chunk_size, total_frames)
                distances[i:end_idx] *= 10
        else:
            distances *= 10
    
    def _process_trajectory(self, traj, distances, frame_idx, pbar = None):
        """Process a single trajectory in batches."""
        for k in range(0, traj.n_frames, self.chunk_size):
            frames_to_process = min(self.chunk_size, traj.n_frames - k)
            
            # Use our precomputed pairs list for ALL residue pairs (except self-pairs)
            dist, res_list = md.compute_contacts(
                traj[k:k + frames_to_process], 
                contacts=self.pairs,  # Use our generated pairs list
                scheme="closest-heavy"
            )
            
            # Direct assignment - dist is already in condensed format
            distances[frame_idx:frame_idx + frames_to_process] = dist
            
            frame_idx += frames_to_process
            if pbar is not None:
                pbar.update(frames_to_process)
        
        return distances, frame_idx, res_list
    
    def _compute_metric_values(self, distances, metric, threshold, window_size, 
                              transition_mode='window', lag_time=1):
        """
        Compute metric values for distances based on specified metric type.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
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
        if metric == 'cv':
            # CV = std/mean
            mean_vals = self.stat.compute_mean(distances)
            std_vals = self.stat.compute_std(distances)
            return std_vals / (mean_vals + 1e-10)  # Avoid division by zero
        elif metric == 'std':
            return self.stat.compute_std(distances)
        elif metric == 'variance':
            return self.stat.compute_variance(distances)
        elif metric == 'range':
            max_vals = self.stat.compute_max(distances)
            min_vals = self.stat.compute_min(distances)
            return max_vals - min_vals
        elif metric == 'transitions':
            # For transitions, use threshold as transition threshold (default 2.0 Å)
            if transition_mode == 'window':
                return self.stat.compute_transitions_window(distances, threshold=threshold, window_size=window_size)
            elif transition_mode == 'lagtime':
                return self.stat.compute_transitions_lagtime(distances, threshold=threshold, lag_time=lag_time)
            else:
                raise ValueError(f"Unknown transition mode: {transition_mode}. Supported: 'window', 'lagtime'")
        else:
            supported_metrics = ['cv', 'std', 'variance', 'range', 'transitions']
        raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

    def compute_dynamic_values(self, distances, metric='cv', threshold_min=None, threshold_max=None, 
                              transition_threshold=2.0, window_size=10, feature_names=None, output_path=None,
                              transition_mode='window', lag_time=1):
        """
        Filter and select variable/dynamic distances based on specified criteria.
            
            Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        metric : str, default='cv'
            Metric to use for selection:
            - 'cv': Coefficient of variation
            - 'std': Standard deviation
            - 'variance': Variance
            - 'range': Range (max - min)
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
            
        Examples:
        --------
        # Select highly variable distances (CV >= 0.5)
        result = calculator.compute_dynamic_values(distances, metric='cv', threshold_min=0.5)
        
        # Select moderately variable distances (0.2 <= CV <= 0.8)
        result = calculator.compute_dynamic_values(distances, metric='cv', 
                                                 threshold_min=0.2, threshold_max=0.8)
        
        # Select low variability distances (CV <= 0.3)
        result = calculator.compute_dynamic_values(distances, metric='cv', threshold_max=0.3)
        
        # Select highly transitioning contacts with window mode 
        (>= 5 transitions, 3.0 Å change in 10 frame windows means transition)
        result = calculator.compute_dynamic_values(distances, metric='transitions', 
                                                 threshold_min=5, transition_threshold=3.0, 
                                                 window_size=10, transition_mode='window')

        # Select highly transitioning contacts with lag-time mode 
        (>= 5 transitions, 3.0 Å change in 10 frame lag-time means transition)
        result = calculator.compute_dynamic_values(distances, metric='transitions', 
                                                    threshold_min=5, transition_threshold=3.0, 
                                                    lag_time=10, transition_mode='lagtime')
        """
        # Compute metric values using helper method
        metric_values = self._compute_metric_values(
            distances, metric, transition_threshold, window_size, 
            transition_mode, lag_time
        )
        
        # Use the common helper
        return CalculatorComputeHelper.compute_dynamic_values(
            data=distances,
            metric_values=metric_values,
            metric_name=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            chunk_size=self.chunk_size,
            feature_names=feature_names,
            use_memmap=self.use_memmap,
            output_path=output_path
        )

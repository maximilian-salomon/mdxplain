"""
DistanceCalculator - MD Trajectory Distance Analysis

Author: Maximilian Salomon
Version: 0.1.0
Created with assistance from Claude-4-Sonnet and Cursor AI.
"""

import numpy as np
import mdtraj as md
from tqdm import tqdm
from .ArrayHandler import ArrayHandler


class DistanceCalculator:
    """
    Utility class for computing distances between residues in MD trajectories.
    All methods are static and can be used without instantiation.
    """
    
    @staticmethod
    def _convert_to_angstrom(distances, total_frames, batch_size=None):
        """Convert distances from nm to Angstrom."""
        if ArrayHandler.is_memmap(distances) and batch_size is not None:
            for i in range(0, total_frames, batch_size):
                end_idx = min(i + batch_size, total_frames)
                distances[i:end_idx] *= 10
        else:
            distances *= 10
    
    @staticmethod
    def compute_distances(trajectories, ref, batch_size=2500, use_memmap=False, distances_path=None,
                         squareform=True, k=0):
        """
        Compute pairwise distances between all residues for all frames in trajectories.
        
        Parameters:
        -----------
        trajectories : list
            List of MDTraj trajectory objects
        ref : mdtraj.Trajectory
            Reference trajectory for residue information
        batch_size : int, default=2500
            Number of frames to process at once
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        distances_path : str, optional
            Path for memory-mapped distance array
        squareform : bool, default=True
            If True, output NxMxM. If False, output NxP (condensed format)
        k : int, default=0
            Diagonal offset: 0=include diagonal, 1=exclude diagonal, >1=exclude additional diagonals
            
        Returns:
        --------
        tuple : (distances, res_list)
            distances : numpy.ndarray
                Distance array with shape (total_frames, n_residues, n_residues) or (total_frames, n_contacts)
            res_list : list
                List of residue pairs
        """
        total_frames = sum(traj.n_frames for traj in trajectories)
        
        # Initialize distance array
        distances = DistanceCalculator._initialize_distance_array(
            total_frames, ref.n_residues, use_memmap, distances_path
        )
        
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="Computing distances") as pbar:
            for traj in trajectories:
                distances, frame_idx, res_list = DistanceCalculator._process_trajectory(
                    traj, distances, ref.n_residues, frame_idx, batch_size, pbar
                )
        
        DistanceCalculator._convert_to_angstrom(distances, total_frames, batch_size)

        if not squareform:
            return ArrayHandler.squareform_to_condensed(distances, k=k), res_list
        return distances, res_list
    
    @staticmethod
    def _initialize_distance_array(total_frames, n_residues, use_memmap, distances_path):
        """Initialize the distance array based on memory mapping preference."""
        if use_memmap:
            if distances_path is None:
                raise ValueError("distances_path must be provided when use_memmap=True")
            return np.memmap(distances_path, dtype='float32', mode='w+', 
                           shape=(total_frames, n_residues, n_residues))
        else:
            # Pre-allocate full array instead of concatenating
            return np.zeros((total_frames, n_residues, n_residues), dtype=np.float32)
    
    @staticmethod
    def _process_trajectory(traj, distances, n_residues, frame_idx, batch_size, pbar = None):
        """Process a single trajectory in batches."""
        for k in range(0, traj.n_frames, batch_size):
            frames_to_process = min(batch_size, traj.n_frames - k)
            dist, res_list = md.compute_contacts(traj[k:k + frames_to_process], scheme="closest-heavy")
            
            batch_dists = DistanceCalculator._build_distance_matrix(
                dist, res_list, frames_to_process, n_residues
            )
            
            # Direct assignment instead of concatenation
            distances[frame_idx:frame_idx + frames_to_process] = batch_dists
            
            frame_idx += frames_to_process
            if pbar is not None:
                pbar.update(frames_to_process)
        
        return distances, frame_idx, res_list
    
    @staticmethod
    def _build_distance_matrix(distances, res_list, frames_to_process, n_residues):
        """
        Build symmetric distance matrix from contact distances.
        
        Parameters:
        -----------
        dist : numpy.ndarray
            Distance array from md.compute_contacts
        res_list : list
            List of residue pairs
        frames_to_process : int
            Number of frames in this batch
        n_residues : int
            Total number of residues
            
        Returns:
        --------
        numpy.ndarray
            Symmetric distance matrix
        """
        batch_dists = np.zeros((frames_to_process, n_residues, n_residues), dtype=np.float32)
        
        # Convert res_list to numpy arrays for vectorized indexing
        res_array = np.array(res_list)
        i_indices = res_array[:, 0]
        j_indices = res_array[:, 1]
        
        # Vectorized assignment for all frames at once
        for frame_idx in range(frames_to_process):
            # Set upper triangle
            batch_dists[frame_idx, i_indices, j_indices] = distances[frame_idx]
            # Set lower triangle (symmetric)
            batch_dists[frame_idx, j_indices, i_indices] = distances[frame_idx]
        
        return batch_dists
    
    # ===== STATISTICAL ANALYSIS METHODS =====
    
    @staticmethod
    def compute_distance_mean(distances, chunk_size=None):
        """
        Compute mean distances across all frames.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Mean distances
        """
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            total_frames = distances.shape[0]
            result_shape = distances.shape[1:]
            accumulated = np.zeros(result_shape, dtype=np.float64)
            frame_count = 0
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = distances[i:end_idx]
                accumulated += np.sum(chunk, axis=0)
                frame_count += len(chunk)
            
            return accumulated / frame_count
        else:
            return np.mean(distances, axis=0)
    
    @staticmethod
    def compute_distance_std(distances, mean_distances=None, chunk_size=None):
        """
        Compute standard deviation of distances.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        mean_distances : numpy.ndarray, optional
            Pre-computed mean distances
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Standard deviation of distances
        """
        if mean_distances is None:
            mean_distances = DistanceCalculator.compute_distance_mean(distances, chunk_size)
        
        variance = DistanceCalculator.compute_distance_variance(distances, mean_distances, chunk_size)
        return np.sqrt(variance)
    
    @staticmethod
    def compute_distance_variance(distances, mean_distances=None, chunk_size=None):
        """
        Compute variance of distances.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        mean_distances : numpy.ndarray, optional
            Pre-computed mean distances
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Variance of distances
        """
        if mean_distances is None:
            mean_distances = DistanceCalculator.compute_distance_mean(distances, chunk_size)
        
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            total_frames = distances.shape[0]
            result_shape = distances.shape[1:]
            accumulated_sq_diff = np.zeros(result_shape, dtype=np.float64)
            frame_count = 0
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = distances[i:end_idx]
                sq_diff = (chunk - mean_distances) ** 2
                accumulated_sq_diff += np.sum(sq_diff, axis=0)
                frame_count += len(chunk)
            
            return accumulated_sq_diff / frame_count
        else:
            return np.var(distances, axis=0)

    @staticmethod
    def compute_distance_cv(distances, chunk_size=None):
        """
        Compute coefficient of variation for distances.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Coefficient of variation (std/mean)
        """
        mean_distances = DistanceCalculator.compute_distance_mean(distances, chunk_size)
        std_distances = DistanceCalculator.compute_distance_std(distances, mean_distances, chunk_size)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = std_distances / mean_distances
            cv[mean_distances == 0] = 0.0
        
        return cv

    @staticmethod
    def compute_avg_distance_per_frame(distances, chunk_size=None):
        """
        Compute average distance per frame.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)
            
        Returns:
        --------
        numpy.ndarray
            Average distance per frame
        """
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            total_frames = distances.shape[0]
            avg_per_frame = np.zeros(total_frames, dtype=np.float32)
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = distances[i:end_idx]
                
                if len(chunk.shape) == 3:
                    # Square format: average over residue pairs
                    avg_per_frame[i:end_idx] = np.mean(chunk, axis=(1, 2))
                else:
                    # Condensed format: average over contacts
                    avg_per_frame[i:end_idx] = np.mean(chunk, axis=1)
            
            return avg_per_frame
        else:
            if len(distances.shape) == 3:
                return np.mean(distances, axis=(1, 2))
            else:
                return np.mean(distances, axis=1)
    
    @staticmethod
    def compute_avg_distance_per_residue(distances, chunk_size=None):
        """
        Compute average distance per residue. Only works with square format.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (must be square format NxMxM)
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)
            
        Returns:
        --------
        numpy.ndarray
            Average distance per residue
        """
        if len(distances.shape) != 3:
            raise ValueError("This method only works with square format (NxMxM)")
        
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            total_frames = distances.shape[0]
            n_residues = distances.shape[1]
            distance_per_residue_sum = np.zeros(n_residues, dtype=np.float32)
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = distances[i:end_idx]
                # Sum over all distances involving each residue, then average over frames
                distance_per_residue_sum += np.sum(np.mean(chunk, axis=2), axis=0)
            
            return distance_per_residue_sum / total_frames
        else:
            # Average distance for each residue across all pairs and all frames
            return np.mean(np.mean(distances, axis=2), axis=0)
    
    @staticmethod
    def compute_distance_differences(distances1, distances2, chunk_size=None):
        """
        Compute differences in mean distances between two sets.
        
        Parameters:
        -----------
        distances1, distances2 : numpy.ndarray
            Distance arrays to compare
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Difference in mean distances (distances1 - distances2)
        """
        mean1 = DistanceCalculator.compute_distance_mean(distances1, chunk_size=chunk_size)
        mean2 = DistanceCalculator.compute_distance_mean(distances2, chunk_size=chunk_size)
        
        return mean1 - mean2
    
    # ===== DYNAMICS ANALYSIS METHODS =====
    
    @staticmethod
    def compute_distance_transitions_lagtime(distances, threshold=2.0, lag_time=1, chunk_size=None):
        """
        Compute number of significant distance transitions using lag-time analysis.
        Compares frame t with frame t+lag_time.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
        threshold : float, default=2.0
            Minimum change in distance (Angstrom) to count as transition
        lag_time : int, default=1
            Lag time for detecting transitions (compares t with t+window_size)
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Number of transitions for each distance pair
        """
        if lag_time < 1:
            raise ValueError("window_size must be >= 1")
        
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            return DistanceCalculator._compute_transitions_unified(distances, threshold, lag_time, chunk_size, mode='lagtime')
        else:
            return DistanceCalculator._compute_transitions_unified(distances, threshold, lag_time, None, mode='lagtime')
    
    @staticmethod
    def compute_distance_transitions_within_window(distances, threshold=2.0, window_size=10, chunk_size=None):
        """
        Check if ANY transition occurs within sliding windows (binary result).
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
        threshold : float, default=2.0
            Minimum change in distance (Angstrom) to count as transition
        window_size : int, default=10
            Window size for checking transitions
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Binary array (0/1) indicating if any transition occurred in each window
        """
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            return DistanceCalculator._compute_transitions_unified(distances, threshold, window_size, chunk_size, mode='within_window')
        else:
            return DistanceCalculator._compute_transitions_unified(distances, threshold, window_size, None, mode='within_window')
    
    @staticmethod
    def _compute_transitions_unified(distances, threshold, window_size, chunk_size, mode='lagtime'):
        """
        Unified helper method for computing transitions with different modes.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
        threshold : float
            Minimum change in distance to count as transition
        window_size : int
            Window/lag size
        chunk_size : int or None
            Chunk size for memmap processing
        mode : str
            'lagtime' or 'within_window'
            
        Returns:
        --------
        numpy.ndarray
            Transition counts or binary indicators
        """
        result_dtype = np.int32
        result_init = np.zeros(distances.shape[1:], dtype=result_dtype)
        
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            return DistanceCalculator._compute_transitions_chunks(distances, threshold, window_size, chunk_size, mode, result_init)
        else:
            return DistanceCalculator._compute_transitions_direct(distances, threshold, window_size, mode, result_init)
    
    @staticmethod
    def _compute_transitions_chunks(distances, threshold, window_size, chunk_size, mode, result):
        """Helper for chunk processing."""
        total_frames = distances.shape[0]
        
        if mode == 'lagtime':
            for i in range(0, total_frames - window_size, chunk_size):
                end_idx = min(i + chunk_size + window_size, total_frames)
                chunk = distances[i:end_idx]
                
                for j in range(len(chunk) - window_size):
                    start_frame = chunk[j]
                    end_frame = chunk[j + window_size]
                    distance_changes = np.abs(end_frame - start_frame)
                    result += (distance_changes > threshold).astype(np.int32)
        
        else:  # within_window
            for i in range(0, total_frames - window_size + 1, chunk_size):
                end_idx = min(i + chunk_size + window_size - 1, total_frames)
                chunk = distances[i:end_idx]
                
                for j in range(len(chunk) - window_size + 1):
                    window = chunk[j:j + window_size]
                    for k in range(len(window) - 1):
                        distance_changes = np.abs(window[k + 1] - window[k])
                        window_has_transition = (distance_changes > threshold).astype(np.int32)
                        result = np.maximum(result, window_has_transition)
        
        return result
    
    @staticmethod
    def _compute_transitions_direct(distances, threshold, window_size, mode, result):
        """Helper for direct processing (non-memmap)."""
        if mode == 'lagtime':
            for i in range(distances.shape[0] - window_size):
                start_frame = distances[i]
                end_frame = distances[i + window_size]
                distance_changes = np.abs(end_frame - start_frame)
                result += (distance_changes > threshold).astype(np.int32)
        
        else:  # within_window
            for i in range(distances.shape[0] - window_size + 1):
                window = distances[i:i + window_size]
                for j in range(len(window) - 1):
                    distance_changes = np.abs(window[j + 1] - window[j])
                    window_has_transition = (distance_changes > threshold).astype(np.int32)
                    result = np.maximum(result, window_has_transition)
        
        return result

    @staticmethod
    def compute_distance_stability(distances, threshold=2.0, window_size=1, chunk_size=None, mode='lagtime'):
        """
        Compute stability score based on distance transitions.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
        threshold : float, default=2.0
            Minimum change in distance (Angstrom) to count as transition
        window_size : int, default=1
            Window/lag size for detecting transitions
        chunk_size : int, default=None
            Chunk size for memmap processing
        mode : str, default='lagtime'
            'lagtime' or 'within_window'
            
        Returns:
        --------
        numpy.ndarray
            Stability score (1.0 = perfectly stable, 0.0 = maximum transitions)
        """
        if mode == 'lagtime':
            transitions = DistanceCalculator.compute_distance_transitions_lagtime(
                distances, threshold, window_size, chunk_size)
            max_transitions = distances.shape[0] - window_size
        else:  # within_window
            transitions = DistanceCalculator.compute_distance_transitions_within_window(
                distances, threshold, window_size, chunk_size)
            max_transitions = distances.shape[0] - window_size + 1
        
        if max_transitions > 0:
            return 1.0 - (transitions / max_transitions)
        else:
            return np.ones_like(transitions, dtype=np.float32)
    
    @staticmethod
    def compute_distance_variability(distances, threshold=2.0, window_size=1, chunk_size=None, mode='lagtime'):
        """
        Compute comprehensive distance variability metrics.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
        threshold : float, default=2.0
            Threshold for transition detection (Angstrom)
        window_size : int, default=1
            Window/lag size for transition analysis
        chunk_size : int, default=None
            Chunk size for memmap processing
        mode : str, default='lagtime'
            'lagtime' or 'within_window'
            
        Returns:
        --------
        dict
            Dictionary containing variability metrics
        """
        mean_distances = DistanceCalculator.compute_distance_mean(distances, chunk_size=chunk_size)
        variance = DistanceCalculator.compute_distance_variance(distances, mean_distances, chunk_size)
        cv = DistanceCalculator.compute_distance_cv(distances, chunk_size)
        
        if mode == 'lagtime':
            transitions = DistanceCalculator.compute_distance_transitions_lagtime(
                distances, threshold, window_size, chunk_size)
        else:
            transitions = DistanceCalculator.compute_distance_transitions_within_window(
                distances, threshold, window_size, chunk_size)
        
        stability = DistanceCalculator.compute_distance_stability(
            distances, threshold, window_size, chunk_size, mode)
        
        results = {
            'mean': mean_distances,
            'variance': variance,
            'cv': cv,
            'transitions': transitions,
            'stability': stability,
            'window_size': window_size,
            'mode': mode
        }
        
        return results
    
    @staticmethod
    def compute_distance_transitions_multi_window(distances, threshold=2.0, 
                                                window_sizes=[1, 5, 10], chunk_size=None, mode='lagtime'):
        """
        Compute transitions with multiple window sizes for comprehensive analysis.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
        threshold : float, default=2.0
            Minimum change in distance (Angstrom) to count as transition
        window_sizes : list, default=[1, 5, 10]
            List of window sizes to analyze
        chunk_size : int, default=None
            Chunk size for memmap processing
        mode : str, default='lagtime'
            'lagtime' or 'within_window'
            
        Returns:
        --------
        dict
            Dictionary with window_size as keys and transition counts as values
        """
        results = {}
        for window_size in window_sizes:
            if mode == 'lagtime':
                results[f'window_{window_size}'] = DistanceCalculator.compute_distance_transitions_lagtime(
                    distances, threshold, window_size, chunk_size)
            else:  # within_window
                results[f'window_{window_size}'] = DistanceCalculator.compute_distance_transitions_within_window(
                    distances, threshold, window_size, chunk_size)
        
        return results

    @staticmethod
    def compute_distance_matrix_stats(distances, chunk_size=None):
        """
        Compute comprehensive matrix statistics for distances.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        dict
            Dictionary containing comprehensive statistics:
            - 'mean': Mean distances
            - 'std': Standard deviation
            - 'variance': Variance
            - 'cv': Coefficient of variation
            - 'min': Minimum distances
            - 'max': Maximum distances
            - 'median': Median distances
            - 'q25': 25th percentile
            - 'q75': 75th percentile
            - 'range': Range (max - min)
        """
        mean_distances = DistanceCalculator.compute_distance_mean(distances, chunk_size)
        std_distances = DistanceCalculator.compute_distance_std(distances, mean_distances, chunk_size)
        variance = std_distances ** 2
        cv = DistanceCalculator.compute_distance_cv(distances, chunk_size)
        
        # Min/Max operations
        min_distances = DistanceCalculator.compute_distance_min(distances, chunk_size)
        max_distances = DistanceCalculator.compute_distance_max(distances, chunk_size)
        
        # Percentiles
        percentiles = DistanceCalculator.compute_distance_percentiles(distances, [25, 50, 75], chunk_size)
        
        range_distances = max_distances - min_distances
        
        return {
            'mean': mean_distances,
            'std': std_distances,
            'variance': variance,
            'cv': cv,
            'min': min_distances,
            'max': max_distances,
            'median': percentiles['p50'],
            'q25': percentiles['p25'],
            'q75': percentiles['p75'],
            'range': range_distances
        }
    
    @staticmethod
    def compute_distance_percentiles(distances, percentiles=[25, 50, 75], chunk_size=None):
        """
        Compute percentiles for distances (memory-intensive for memmap).
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        percentiles : list, default=[25, 50, 75]
            List of percentiles to compute
        chunk_size : int, default=None
            Chunk size for memmap processing (loads all data for percentiles)
            
        Returns:
        --------
        dict
            Dictionary containing percentiles (e.g., 'p25', 'p50', 'p75')
        """
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            # Warning: This loads all data into memory
            all_chunks = []
            total_frames = distances.shape[0]
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = distances[i:end_idx]
                all_chunks.append(chunk)
            
            combined = np.concatenate(all_chunks, axis=0)
            result = {}
            for p in percentiles:
                result[f'p{p}'] = np.percentile(combined, p, axis=0)
        else:
            result = {}
            for p in percentiles:
                result[f'p{p}'] = np.percentile(distances, p, axis=0)
        
        return result

    @staticmethod
    def compute_dynamic_distances(distances, metric='cv', threshold=0.5, chunk_size=None):
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
            - 'transitions': Number of transitions (requires threshold=2.0)
        threshold : float, default=0.5
            Threshold value for filtering
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'indices': Indices of dynamic distances
            - 'values': Values of the selected metric
            - 'dynamic_distances': Filtered distance data
            - 'metric_used': Which metric was used
            - 'threshold_used': Which threshold was used
        """
        if metric == 'cv':
            metric_values = DistanceCalculator.compute_distance_cv(distances, chunk_size)
        elif metric == 'std':
            metric_values = DistanceCalculator.compute_distance_std(distances, chunk_size=chunk_size)
        elif metric == 'variance':
            metric_values = DistanceCalculator.compute_distance_variance(distances, chunk_size=chunk_size)
        elif metric == 'range':
            stats = DistanceCalculator.compute_distance_matrix_stats(distances, chunk_size)
            metric_values = stats['range']
        elif metric == 'transitions':
            # For transitions, use threshold as transition threshold (default 2.0 Å)
            transition_threshold = threshold if threshold > 1.0 else 2.0
            metric_values = DistanceCalculator.compute_distance_transitions_lagtime(distances, threshold=transition_threshold, 
                                                                                    chunk_size=chunk_size)
            # Reset threshold for selection
            threshold = np.median(metric_values) if threshold <= 1.0 else threshold
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cv', 'std', 'variance', 'range', or 'transitions'")
        
        # Find indices where metric exceeds threshold
        dynamic_indices = np.where(metric_values > threshold)
        
        # Extract dynamic distances
        if len(distances.shape) == 3:  # Square format
            dynamic_distances = distances[:, dynamic_indices[0], dynamic_indices[1]]
        else:  # Condensed format
            dynamic_distances = distances[:, dynamic_indices[0]]
        
        return {
            'indices': dynamic_indices,
            'values': metric_values[dynamic_indices],
            'dynamic_distances': dynamic_distances,
            'metric_used': metric,
            'threshold_used': threshold,
            'n_dynamic': len(dynamic_indices[0]) if len(distances.shape) == 3 else len(dynamic_indices[0]),
            'total_pairs': metric_values.size
        }

    @staticmethod
    def compute_distance_min(distances, chunk_size=None):
        """
        Compute minimum distances across all frames.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Minimum distances
        """
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            total_frames = distances.shape[0]
            result_shape = distances.shape[1:]
            result = np.full(result_shape, np.inf, dtype=np.float64)
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = distances[i:end_idx]
                chunk_min = np.min(chunk, axis=0)
                result = np.minimum(result, chunk_min)
            
            return result
        else:
            return np.min(distances, axis=0)

    @staticmethod
    def compute_distance_max(distances, chunk_size=None):
        """
        Compute maximum distances across all frames.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (square or condensed format)
        chunk_size : int, default=None
            Chunk size for memmap processing
            
        Returns:
        --------
        numpy.ndarray
            Maximum distances
        """
        if ArrayHandler.is_memmap(distances) and chunk_size is not None:
            total_frames = distances.shape[0]
            result_shape = distances.shape[1:]
            result = np.full(result_shape, -np.inf, dtype=np.float64)
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = distances[i:end_idx]
                chunk_max = np.max(chunk, axis=0)
                result = np.maximum(result, chunk_max)
            
            return result
        else:
            return np.max(distances, axis=0)

    @staticmethod
    def detect_distance_unit(trajectories, distance_matrix):
        """
        Detect the unit of a distance matrix by comparing with MDTraj reference.
        
        Parameters:
        -----------
        trajectories : mdtraj.Trajectory or list
            Trajectory or list of trajectories
        distance_matrix : numpy.ndarray
            Distance matrix to check units
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'factor': Conversion factor (distance_matrix / mdtraj_reference)
            - 'unit': Detected unit ('angstrom', 'nanometer', or 'unknown')
        """
        # Handle single trajectory or list
        if not isinstance(trajectories, list):
            traj = trajectories
        else:
            traj = trajectories[0]
        
        # Use only first frame for efficiency
        input_first_frame = distance_matrix[0] if len(distance_matrix.shape) > 2 else distance_matrix[0:1]
        
        # Compute reference distances in same format
        calc_distances = DistanceCalculator._compute_reference_distances(traj, input_first_frame.shape)
        
        # Find matching non-zero values and calculate factor
        factor, matrix_distance, mdtraj_distance, first_idx = DistanceCalculator._find_matching_values(
            input_first_frame, calc_distances)
        
        # Determine unit from factor
        unit = DistanceCalculator._determine_unit(factor)
        
        return {
            'factor': factor,
            'unit': unit,
            'mdtraj_reference_nm': mdtraj_distance,
            'matrix_value': matrix_distance,
            'array_index': first_idx,
            'shapes_match': True,
            'note': f"Compared index {first_idx}: matrix={matrix_distance:.3f}, mdtraj={mdtraj_distance:.3f} nm"
        }
    
    @staticmethod
    def _compute_reference_distances(traj, input_shape):
        """Compute reference distances in same format as input."""
        if len(input_shape) == 2:  # Square format
            ref_distances, pairs = md.compute_contacts(traj[0], scheme="closest-heavy")
            ref_square = md.geometry.squareform(ref_distances, pairs)
            return ref_square
        else:  # Condensed format
            calc_distances, _ = md.compute_contacts(traj[0], scheme="closest-heavy")
            return calc_distances
    
    @staticmethod
    def _find_matching_values(input_array, calc_array):
        """Find first matching non-zero values in both arrays."""
        if input_array.shape != calc_array.shape:
            raise ValueError(f"Shape mismatch: input {input_array.shape} vs calculated {calc_array.shape}")
        
        input_flat = input_array.flatten()
        calc_flat = calc_array.flatten()
        
        non_zero_mask = (input_flat != 0) & (calc_flat != 0)
        
        if not np.any(non_zero_mask):
            raise ValueError("No matching non-zero values found in both arrays")
        
        first_idx = np.where(non_zero_mask)[0][0]
        matrix_distance = input_flat[first_idx]
        mdtraj_distance = calc_flat[first_idx]
        factor = matrix_distance / mdtraj_distance
        
        return factor, matrix_distance, mdtraj_distance, first_idx
    
    @staticmethod
    def _determine_unit(factor):
        """Determine unit from conversion factor."""
        if np.isclose(factor, 10.0, rtol=0.1):
            return 'angstrom'
        elif np.isclose(factor, 1.0, rtol=0.1):
            return 'nanometer'
        else:
            return 'unknown'

    @staticmethod
    def convert_to_angstrom(distance_matrix, trajectories=None, conversion_factor=None):
        """
        Convert distance matrix to Angstrom units.
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix to convert
        trajectories : mdtraj.Trajectory or list, optional
            Trajectories for unit detection (if conversion_factor not provided)
        conversion_factor : float, optional
            Direct conversion factor (matrix_unit / nanometer)
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix in Angstrom units
        """
        if conversion_factor is None:
            if trajectories is None:
                raise ValueError("Either trajectories or conversion_factor must be provided")
            
            unit_info = DistanceCalculator.detect_distance_unit(trajectories, distance_matrix)
            conversion_factor = unit_info['factor']
        
        # Convert to Angstrom: factor * 10 (because MDTraj is in nm, Angstrom = nm * 10)
        angstrom_factor = 10.0 / conversion_factor
        
        return distance_matrix * angstrom_factor
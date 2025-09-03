# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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

"""
Distance calculator for molecular dynamics trajectory analysis.

Utility class for computing pairwise distances between residues in MD trajectory.
Supports memory mapping for large datasets and provides statistical analysis capabilities.
"""


from typing import Dict, Optional, Tuple, Any, List, Union
import mdtraj as md
import numpy as np

from ..helper.calculator_compute_helper import CalculatorComputeHelper
from ..helper.feature_shape_helper import FeatureShapeHelper
from ..interfaces.calculator_base import CalculatorBase
from .distance_calculator_analysis import DistanceCalculatorAnalysis


class DistanceCalculator(CalculatorBase):
    """
    Calculator for computing pairwise distances between atoms/residues in MD trajectory.

    Computes all pairwise distances using MDTraj's distance calculation functions
    with support for memory-mapped arrays, chunked processing, and various output
    formats. Includes statistical analysis capabilities for variability analysis,
    transition detection, and comparative studies.

    Examples:
    ---------
    >>> # Basic distance calculation
    >>> calculator = DistanceCalculator()
    >>> distances, pairs = calculator.compute(trajectory)

    >>> # With memory mapping for large datasets
    >>> calculator = DistanceCalculator(use_memmap=True, cache_path='./cache/')
    >>> distances, pairs = calculator.compute(trajectory)
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize distance calculator with configuration parameters.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files
        chunk_size : int, optional
            Number of frames to process per chunk

        Returns:
        --------
        None

        Examples:
        ---------
        >>> # Basic initialization
        >>> calculator = DistanceCalculator()

        >>> # With memory mapping
        >>> calculator = DistanceCalculator(use_memmap=True, cache_path='./cache/')
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self.distances_path = cache_path

        self.pairs = None  # Will be computed
        self.n_pairs = None  # Will be set after pairs are generated

        self.analysis = DistanceCalculatorAnalysis(
            use_memmap=self.use_memmap, chunk_size=self.chunk_size
        )

    # ===== MAIN COMPUTATION METHOD =====

    def compute(self, input_data: md.Trajectory, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute pairwise distances between all atoms/residues for all frames.

        Parameters:
        -----------
        input_data : mdtraj.Trajectory
            MDTraj trajectory object to process
        **kwargs : dict
            Additional parameters:
            - excluded_neighbors : int, default=1 - Diagonal offset (excluded_neighbors=1 excludes diagonal,
              excluded_neighbors=2 excludes first two diagonals)
              Chain Breaks are automatically excluded. Meassured by jump in the seqid of a residue.
            - res_metadata : dict - Residue metadata for feature naming

        Returns:
        --------
        tuple[numpy.ndarray, dict]
            Tuple containing (distances, feature_metadata) where distances is the
            distance matrix in Angstrom and feature_metadata contains structured metadata

        Examples:
        ---------
        >>> # Basic distance calculation (excludes diagonal)
        >>> distances, metadata = calculator.compute(trajectory)
        >>> print(f"Distance matrix shape: {distances.shape}")

        >>> # With custom diagonal offset
        >>> distances, metadata = calculator.compute(trajectory, excluded_neighbors=2)

        >>> # With reference trajectory for consistent naming
        >>> distances, metadata = calculator.compute(trajectory, ref=ref_traj)
        """
        # Extract parameters from kwargs
        trajectory = input_data 
        excluded_neighbors = kwargs.get("excluded_neighbors", 1)  # Default excluded_neighbors=1 excludes diagonal  
        res_metadata = kwargs.get("res_metadata", None)

        # Setup computation parameters and arrays
        total_frames, distances = self._setup_computation(trajectory, excluded_neighbors, res_metadata)

        # Process single trajectory
        distances, res_list = self._process_trajectory(
            trajectory, distances
        )

        # Consistency check: res_list should match self.pairs
        self._validate_pair_consistency(res_list)

        # Generate feature metadata using labels if available
        feature_metadata = self._generate_feature_metadata(res_metadata)

        # Finalize output (convert units, cleanup)
        distances = self._finalize_output(distances, total_frames)

        return distances, feature_metadata

    def _setup_computation(self, trajectory: md.Trajectory, excluded_neighbors: int, res_metadata: Dict[str, Any]) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
        """
        Set up computation parameters and create output arrays.

        Parameters:
        -----------
        trajectory : mdtraj.Trajectory
            MDTraj trajectory object to process
        excluded_neighbors : int
            Sequential seqid distance for neighbor exclusion
        res_metadata : dict
            Residue metadata containing seqid information

        Returns:
        --------
        tuple
            (total_frames, distances_array)
        """
        total_frames = trajectory.n_frames

        # Generate pairs for THIS trajectory (not cached globally)
        # Each trajectory has different n_residues, so pairs must be trajectory-specific
        self.pairs = self._generate_residue_pairs(trajectory.n_residues, excluded_neighbors, res_metadata)
        print(
            f"Generated {len(self.pairs)} residue pairs for {trajectory.n_residues} residues"
        )

        # Calculate condensed output shape (natural md.compute_contacts format)
        self.n_pairs = len(self.pairs)

        # Create output array in condensed format
        distances = CalculatorComputeHelper.create_output_array(
            self.use_memmap,
            self.distances_path,
            (total_frames, self.n_pairs),
            dtype="float32",
        )

        return total_frames, distances

    def _finalize_output(self, distances: np.ndarray, total_frames: int) -> np.ndarray:
        """
        Convert units, format output, and cleanup temporary files.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format
        total_frames : int
            Total number of frames processed

        Returns:
        --------
        numpy.ndarray
            Distance array in condensed format
        """
        # Convert to Angstrom
        self._convert_to_angstrom(distances, total_frames)

        return distances

    def _validate_pair_consistency(self, res_list: List[int]) -> None:
        """
        Validate that res_list from MDTraj matches our generated pairs.

        Parameters:
        -----------
        res_list : list
            List of residue pair indices from MDTraj

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If res_list and self.pairs are inconsistent
        """
        if res_list is None:
            return

        self._validate_pair_counts(res_list)
        self._validate_individual_pairs(res_list)

    def _validate_pair_counts(self, res_list: List[int]) -> None:
        """
        Validate that the number of pairs matches.

        Parameters:
        -----------
        res_list : list
            List of residue pair indices from MDTraj

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If the number of pairs does not match
        """
        if len(res_list) != len(self.pairs):
            raise ValueError(
                f"Inconsistency detected: res_list has {len(res_list)} pairs "
                f"but self.pairs has {len(self.pairs)} pairs. "
                f"This indicates a problem with pair generation."
            )

    def _validate_individual_pairs(self, res_list: List[int]) -> None:
        """
        Validate that individual pairs match between res_list and self.pairs.

        Parameters:
        -----------
        res_list : list
            List of residue pair indices from MDTraj

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If the number of pairs does not match
        """
        for i, (mdtraj_pair, our_pair) in enumerate(zip(res_list, self.pairs)):
            converted_pair = self._convert_mdtraj_pair_format(mdtraj_pair)
            if converted_pair is not None:
                self._check_pair_equality(converted_pair, our_pair, i, mdtraj_pair)

    def _convert_mdtraj_pair_format(self, mdtraj_pair: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert MDTraj pair to comparable format.

        Parameters:
        -----------
        mdtraj_pair : tuple or str
            MDTraj pair to convert

        Returns:
        --------
        list
            List of residue pair indices
        """
        if isinstance(mdtraj_pair, tuple):
            return list(mdtraj_pair)
        if isinstance(mdtraj_pair, str):
            return None  # Skip string format pairs
        return mdtraj_pair

    def _check_pair_equality(self, converted_pair: Tuple[int, int], our_pair: Tuple[int, int], index: int, original_pair: Tuple[int, int]) -> None:
        """
        Check if converted pair equals our generated pair.

        Parameters:
        -----------
        converted_pair : list
            Converted pair from MDTraj
        our_pair : list
            Our generated pair
        index : int
            Index of the pair
        original_pair : tuple or str
            Original pair from MDTraj

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If the pair does not match
        """
        if not np.array_equal(converted_pair, our_pair):
            raise ValueError(
                f"Pair mismatch at index {index}: "
                f"MDTraj returned {original_pair}, "
                f"but we generated {our_pair}. "
                f"This indicates inconsistent pair ordering."
            )

    # ===== PRIVATE HELPER METHODS =====
    def _generate_residue_pairs(self, n_residues: int, excluded_neighbors: int, res_metadata: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Generate residue pairs using chain-aware metadata-based neighbor exclusion.

        Parameters:
        -----------
        n_residues : int
            Number of residues
        excluded_neighbors : int
            Sequential seqid distance for neighbor exclusion within chains
        res_metadata : dict
            Residue metadata containing seqid information for chain detection

        Returns:
        --------
        list
            List of [i, j] pairs where i < j, excluding consecutive residues in same chain

        Raises:
        -------
        ValueError
            If res_metadata is None
        """
        if res_metadata is None:
            raise ValueError(
                "res_metadata is required for seqid-based neighbor exclusion. "
                "Ensure trajectory has been processed with add_labels()."
            )
        
        # Validate input consistency
        if len(res_metadata) != n_residues:
            raise ValueError(
                f"Data inconsistency: n_residues={n_residues} but len(res_metadata)={len(res_metadata)}. "
                "The trajectory and metadata must have the same number of residues."
            )

        pairs = []
        if excluded_neighbors < 0:
            raise ValueError(
                "excluded_neighbors must be a positive integer or 0."
            )
        
        for i in range(n_residues):
            for j in range(i, n_residues):  # Check ALL pairs first
                # Check if residues i and j are consecutive in same chain
                if self._is_consecutive_in_sequence(i, j, res_metadata, excluded_neighbors):
                    continue  # Skip consecutive residues in same chain
                
                pairs.append([i, j])  # Keep all other pairs
                
        return pairs

    def _is_consecutive_in_sequence(self, i: int, j: int, res_metadata: Dict[str, Any], excluded_neighbors: int) -> bool:
        """
        Check if residues i and j are consecutive within same chain sequence.

        Parameters:
        -----------
        i : int
            First residue index
        j : int
            Second residue index (j > i)
        res_metadata : dict
            Residue metadata containing seqid information
        excluded_neighbors : int
            Maximum sequential seqid distance for exclusion

        Returns:
        --------
        bool
            True if residues are consecutive in same chain with distance <= excluded_neighbors
        """
        # Check all residues between i and j for chain breaks
        for k in range(i, j):
            current_seqid = res_metadata[k]['seqid']
            next_seqid = res_metadata[k + 1]['seqid']
            
            # Chain break detected → residues are not consecutive in same chain
            if abs(next_seqid - current_seqid) != 1:
                return False
        
        # All residues form continuous sequence → check distance threshold
        seqid_distance = abs(res_metadata[j]['seqid'] - res_metadata[i]['seqid'])
        return seqid_distance <= excluded_neighbors

    def _convert_to_angstrom(self, distances: np.ndarray, total_frames: int) -> np.ndarray:
        """
        Convert distances from nm to Angstrom.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in nanometers
        total_frames : int
            Total number of frames for chunked processing

        Returns:
        --------
        None
        """
        if FeatureShapeHelper.is_memmap(distances) or self.use_memmap:
            for i in range(0, total_frames, self.chunk_size):
                end_idx = min(i + self.chunk_size, total_frames)
                distances[i:end_idx] *= 10
        else:
            distances *= 10

    def _generate_feature_metadata(self, res_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature metadata for residue pairs using structured labels.

        Parameters:
        -----------
        feature_metadata : dict, optional
            Structured trajectory labels from res_label_data

        Returns:
        --------
        dict
            Feature metadata dictionary with 'is_pair' and 'features' keys

        Raises:
        -------
        ValueError
            If res_list and self.pairs are inconsistent
        """
        # Generate metadata using structured labels
        features = []
        for pair in self.pairs:
            i, j = pair
            if i < len(res_metadata) and j < len(res_metadata):
                # Create partners from structured labels
                partner1 = {
                    "residue": res_metadata[i],
                    "special_label": None,  # No special label for distances
                    "full_name": res_metadata[i]["full_name"],
                }
                partner2 = {
                    "residue": res_metadata[j],
                    "special_label": None,  # No special label for distances
                    "full_name": res_metadata[j]["full_name"],
                }
                features.append([partner1, partner2])
            else:
                # This should not happen if FeatureManager validates properly
                raise ValueError(
                    f"Residue indices {i}, {j} not found in res_metadata "
                    f"(length: {len(res_metadata)})"
                )

        return {"is_pair": True, "features": features}

    def _process_trajectory(self, traj: md.Trajectory, distances: np.ndarray) -> None:
        """
        Process a single trajectory in batches.

        Parameters:
        -----------
        traj : mdtraj.Trajectory
            Trajectory object to process
        distances : np.ndarray or np.memmap
            Output array for distance values

        Returns:
        --------
        tuple
            (distances_array, residue_list)
        """
        if self.chunk_size is None:
            dist, res_list = md.compute_contacts(
                traj,
                contacts=self.pairs,
                scheme="closest-heavy",  # Use our generated pairs list
            )
            distances[:] = dist  # Direct assignment
        else:
            for frame_start in range(0, traj.n_frames, self.chunk_size):
                frames_to_process = min(self.chunk_size, traj.n_frames - frame_start)

                # Use our precomputed pairs list for ALL residue pairs (except self-pairs)
                dist, res_list = md.compute_contacts(
                    traj[frame_start : frame_start + frames_to_process],
                    contacts=self.pairs,  # Use our generated pairs list
                    scheme="closest-heavy",
                )

                # Direct assignment - dist is already in condensed format
                distances[frame_start : frame_start + frames_to_process] = dist

        return distances, res_list

    def _compute_metric_values(
        self,
        distances: np.ndarray,
        metric: str,
        threshold: float,
        window_size: int,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
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
        # Define simple metrics that map directly to analysis methods
        simple_metrics = {
            "std": self.analysis.compute_std,
            "variance": self.analysis.compute_variance,
            "min": self.analysis.compute_min,
            "mad": self.analysis.compute_mad,
            "mean": self.analysis.compute_mean,
        }

        # Handle simple metrics
        if metric in simple_metrics:
            return simple_metrics[metric](distances)

        # Handle complex metrics
        if metric == "cv":
            mean_vals = self.analysis.compute_mean(distances)
            std_vals = self.analysis.compute_std(distances)
            return std_vals / (mean_vals + 1e-10)
        if metric == "range":
            max_vals = self.analysis.compute_max(distances)
            min_vals = self.analysis.compute_min(distances)
            return max_vals - min_vals
        if metric == "transitions":
            return self._compute_transitions_metric(
                distances, threshold, window_size, transition_mode, lag_time
            )

        # Unknown metric
        supported_metrics = [
            "cv",
            "std",
            "variance",
            "range",
            "transitions",
            "min",
            "mad",
        ]
        raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

    def _compute_transitions_metric(
        self, 
        distances: np.ndarray, 
        threshold: float, 
        window_size: int, 
        transition_mode: str, 
        lag_time: int
    ) -> np.ndarray:
        """
        Compute transitions metric based on specified mode and parameters.

        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array
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
        int
            Number of transitions
        """
        if transition_mode == "window":
            return self.analysis.compute_transitions_window(
                distances, threshold=threshold, window_size=window_size
            )
        if transition_mode == "lagtime":
            return self.analysis.compute_transitions_lagtime(
                distances, threshold=threshold, lag_time=lag_time
            )
        raise ValueError(
            f"Unknown transition mode: {transition_mode}. "
            f"Supported: 'window', 'lagtime'"
        )

    def compute_dynamic_values(
        self,
        input_data: np.ndarray,
        metric: str = "cv",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        feature_metadata: Optional[List[Any]] = None,
        output_path: Optional[str] = None,
        transition_threshold: float = 2.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> Dict[str, Any]:
        """
        Filter and select variable/dynamic distances based on specified criteria.

        Parameters:
        -----------
        input_data : numpy.ndarray
            Distance array (square or condensed format)
        metric : str, default='cv'
            Metric to use for selection:
            - 'cv': Coefficient of variation
            - 'std': Standard deviation
            - 'min': Minimum distance
            - 'mad': Median absolute deviation
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
        feature_metadata : list, optional
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
            input_data,
            metric,
            transition_threshold,
            window_size,
            transition_mode,
            lag_time,
        )

        # Use the common helper
        return CalculatorComputeHelper.compute_dynamic_values(
            data=input_data,
            metric_values=metric_values,
            metric_name=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            chunk_size=self.chunk_size,
            feature_metadata=feature_metadata,
            use_memmap=self.use_memmap,
            output_path=output_path,
        )

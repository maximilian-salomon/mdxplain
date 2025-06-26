# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
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

"""
Distance calculator for molecular dynamics trajectory analysis.

Utility class for computing pairwise distances between residues in MD trajectories.
Supports memory mapping for large datasets and provides statistical analysis capabilities.
"""

import os

import mdtraj as md
from tqdm import tqdm

from ..helper.calculator_compute_helper import CalculatorComputeHelper
from ..helper.feature_shape_helper import FeatureShapeHelper
from ..interfaces.calculator_base import CalculatorBase
from .distance_calculator_analysis import DistanceCalculatorAnalysis


class DistanceCalculator(CalculatorBase):
    """
    Calculator for computing pairwise distances between atoms/residues in MD trajectories.

    Computes all pairwise distances using MDTraj's distance calculation functions
    with support for memory-mapped arrays, chunked processing, and various output
    formats. Includes statistical analysis capabilities for variability analysis,
    transition detection, and comparative studies.

    Examples:
    ---------
    >>> # Basic distance calculation
    >>> calculator = DistanceCalculator()
    >>> distances, pairs = calculator.compute(trajectories)

    >>> # With memory mapping for large datasets
    >>> calculator = DistanceCalculator(use_memmap=True, cache_path='./cache/')
    >>> distances, pairs = calculator.compute(trajectories)
    """

    def __init__(self, use_memmap=False, cache_path=None, chunk_size=None):
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

        Examples:
        ---------
        >>> # Basic initialization
        >>> calculator = DistanceCalculator()

        >>> # With memory mapping
        >>> calculator = DistanceCalculator(use_memmap=True, cache_path='./cache/')
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self.distances_path = cache_path
        self.pairs = None  # Will be computed from reference trajectory
        self.n_pairs = None  # Will be set after pairs are generated

        self.analysis = DistanceCalculatorAnalysis(chunk_size=self.chunk_size)

    # ===== MAIN COMPUTATION METHOD =====

    def compute(self, input_data, **kwargs):
        """
        Compute pairwise distances between all atoms/residues for all frames.

        Parameters:
        -----------
        input_data : list
            List of MDTraj trajectory objects to process
        **kwargs : dict
            Additional parameters:
            - ref : mdtraj.Trajectory, optional - Reference trajectory for pair determination
            - squareform : bool, default=False - If True, output NxMxM format
            - k : int, default=0 - Diagonal offset
            - labels : list, optional - Residue labels for feature naming

        Returns:
        --------
        tuple[numpy.ndarray, list]
            Tuple containing (distances, pair_list) where distances is the
            distance matrix in Angstrom and pair_list contains pair identifiers

        Examples:
        ---------
        >>> # Basic distance calculation
        >>> distances, pairs = calculator.compute(trajectories)
        >>> print(f"Distance matrix shape: {distances.shape}")

        >>> # With reference trajectory for consistent naming
        >>> distances, pairs = calculator.compute(trajectories, ref=ref_traj)

        >>> # With residue labels for meaningful feature names
        >>> distances, pairs = calculator.compute(trajectories, labels=residue_labels)
        """
        # Extract parameters from kwargs
        trajectories = input_data
        ref = kwargs.get("ref", None)
        squareform = kwargs.get("squareform", False)
        k = kwargs.get("k", 0)
        labels = kwargs.get("labels", None)

        # Setup computation parameters and arrays
        ref, total_frames, distances, condensed_path = self._setup_computation(
            trajectories, ref, squareform, k
        )

        # Process all trajectories
        distances, res_list = self._process_all_trajectories(
            trajectories, distances, total_frames
        )

        # Generate feature names using labels if available
        feature_names = self._generate_feature_names(res_list, labels)

        # Finalize output (convert units, format, cleanup)
        distances = self._finalize_output(
            distances, res_list, total_frames, condensed_path, squareform
        )

        return distances, feature_names

    def _setup_computation(self, trajectories, ref, squareform, k):
        """
        Set up computation parameters and create output arrays.

        Parameters:
        -----------
        trajectories : list
            List of MDTraj trajectory objects to process
        ref : mdtraj.Trajectory or None
            Reference trajectory for pair determination
        squareform : bool
            Whether to output in square format
        k : int
            Diagonal offset for pair generation

        Returns:
        --------
        tuple
            (ref_trajectory, total_frames, distances_array, condensed_path)
        """
        total_frames = sum(traj.n_frames for traj in trajectories)

        if ref is None:
            ref = trajectories[0]

        # Generate pairs once from reference trajectory
        if self.pairs is None:
            self.pairs = self._generate_residue_pairs(ref.n_residues, k)
            print(
                f"Generated {len(self.pairs)} residue pairs for {ref.n_residues} residues"
            )

        # Calculate condensed output shape (natural md.compute_contacts format)
        self.n_pairs = len(self.pairs)

        # Determine paths for memmap handling
        condensed_path = self.distances_path
        if squareform and self.use_memmap and self.distances_path is not None:
            # If squareform is requested, use temporary path for condensed
            condensed_path = self.distances_path.replace(
                ".dat", "_condensed.dat")

        # Create output array in condensed format
        distances = CalculatorComputeHelper.create_output_array(
            self.use_memmap,
            condensed_path,
            (total_frames, self.n_pairs),
            dtype="float32",
        )

        return ref, total_frames, distances, condensed_path

    def _process_all_trajectories(self, trajectories, distances, total_frames):
        """
        Process all trajectories and compute distances.

        Parameters:
        -----------
        trajectories : list
            List of MDTraj trajectory objects to process
        distances : np.ndarray or np.memmap
            Output array for distance values
        total_frames : int
            Total number of frames across all trajectories

        Returns:
        --------
        tuple
            (distances_array, residue_list) with computed distances and residue names
        """
        frame_idx = 0
        res_list = None

        with tqdm(total=total_frames, desc="Computing distances") as pbar:
            for traj in trajectories:
                distances, frame_idx, res_list = self._process_trajectory(
                    traj, distances, frame_idx, pbar
                )

        return distances, res_list

    def _finalize_output(
        self, distances, res_list, total_frames, condensed_path, squareform
    ):
        """
        Convert units, format output, and cleanup temporary files.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format
        res_list : list
            List of residue pair names
        total_frames : int
            Total number of frames processed
        condensed_path : str
            Path to condensed format file (for cleanup)
        squareform : bool
            Whether to convert to square format

        Returns:
        --------
        np.ndarray or np.memmap
            Final distance array in requested format
        """
        # Convert to Angstrom
        self._convert_to_angstrom(distances, total_frames)

        # Convert to squareform only if requested
        if squareform:
            # Calculate n_residues from pairs
            n_residues = max(max(pair) for pair in self.pairs) + 1
            distances = FeatureShapeHelper.condensed_to_squareform(
                distances,
                res_list,
                n_residues,
                chunk_size=self.chunk_size,
                output_path=self.distances_path,
            )

            # Clean up temporary condensed memmap if it was created
            if self.use_memmap and condensed_path != self.distances_path:
                if os.path.exists(condensed_path):
                    os.remove(condensed_path)

        return distances

    # ===== PRIVATE HELPER METHODS =====
    def _generate_residue_pairs(self, n_residues, k):
        """
        Generate all unique residue pairs (excluding self-pairs i==j).

        Parameters:
        -----------
        n_residues : int
            Number of residues
        k : int
            Diagonal offset (k=1 excludes diagonal, k=2 excludes first two diagonals)

        Returns:
        --------
        list
            List of [i, j] pairs where i < j
        """
        pairs = []
        if k <= 0:
            k = 1
        for i in range(n_residues):
            for j in range(
                i + k, n_residues
            ):  # i < j, excludes self-pairs and k diagonals
                pairs.append([i, j])
        return pairs

    def _convert_to_angstrom(self, distances, total_frames):
        """
        Convert distances from nm to Angstrom.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in nanometers
        total_frames : int
            Total number of frames for chunked processing
        """
        if FeatureShapeHelper.is_memmap(distances) and self.chunk_size is not None:
            for i in range(0, total_frames, self.chunk_size):
                end_idx = min(i + self.chunk_size, total_frames)
                distances[i:end_idx] *= 10
        else:
            distances *= 10

    def _generate_feature_names(self, res_list, labels):
        """
        Generate feature names for residue pairs using labels if available.

        Parameters:
        -----------
        res_list : list
            List of residue pair indices from MDTraj
        labels : list, optional
            Residue labels for naming

        Returns:
        --------
        list
            List of feature names for each pair
        """
        if labels is None:
            # Use original residue pair indices
            return res_list

        # Convert pairs to label-based names
        feature_names = []
        for pair in self.pairs:
            i, j = pair
            if i < len(labels) and j < len(labels):
                feature_names.append(f"{labels[i]}-{labels[j]}")
            else:
                # Fallback to indices if labels don't cover all residues
                feature_names.append(f"{i}-{j}")

        return feature_names

    def _process_trajectory(self, traj, distances, frame_idx, pbar=None):
        """
        Process a single trajectory in batches.

        Parameters:
        -----------
        traj : mdtraj.Trajectory
            Trajectory object to process
        distances : np.ndarray or np.memmap
            Output array for distance values
        frame_idx : int
            Starting frame index for this trajectory
        pbar : tqdm.tqdm, optional
            Progress bar for updates

        Returns:
        --------
        tuple
            (distances_array, updated_frame_idx, residue_list)
        """
        if self.chunk_size is None:
            dist, res_list = md.compute_contacts(
                traj,
                contacts=self.pairs,
                scheme="closest-heavy",  # Use our generated pairs list
            )
            distances = dist
            frame_idx += traj.n_frames
            if pbar is not None:
                pbar.update(traj.n_frames)
        else:
            for k in range(0, traj.n_frames, self.chunk_size):
                frames_to_process = min(self.chunk_size, traj.n_frames - k)

                # Use our precomputed pairs list for ALL residue pairs (except self-pairs)
                dist, res_list = md.compute_contacts(
                    traj[k: k + frames_to_process],
                    contacts=self.pairs,  # Use our generated pairs list
                    scheme="closest-heavy",
                )

                # Direct assignment - dist is already in condensed format
                distances[frame_idx: frame_idx + frames_to_process] = dist

                frame_idx += frames_to_process
                if pbar is not None:
                    pbar.update(frames_to_process)

        return distances, frame_idx, res_list

    def _compute_metric_values(
        self,
        distances,
        metric,
        threshold,
        window_size,
        transition_mode="window",
        lag_time=1,
    ):
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
        if metric == "cv":
            # CV = std/mean
            mean_vals = self.analysis.compute_mean(distances)
            std_vals = self.analysis.compute_std(distances)
            return std_vals / (mean_vals + 1e-10)  # Avoid division by zero
        if metric == "std":
            return self.analysis.compute_std(distances)
        if metric == "variance":
            return self.analysis.compute_variance(distances)
        if metric == "range":
            max_vals = self.analysis.compute_max(distances)
            min_vals = self.analysis.compute_min(distances)
            return max_vals - min_vals
        if metric == "transitions":
            # For transitions, use threshold as transition threshold (default 2.0 Å)
            if transition_mode == "window":
                return self.analysis.compute_transitions_window(
                    distances, threshold=threshold, window_size=window_size
                )
            if transition_mode == "lagtime":
                return self.analysis.compute_transitions_lagtime(
                    distances, threshold=threshold, lag_time=lag_time
                )
            raise ValueError(
                f"Unknown transition mode: {transition_mode}. Supported: 'window', 'lagtime'"
            )

        supported_metrics = ["cv", "std", "variance", "range", "transitions"]
        raise ValueError(
            f"Unknown metric: {metric}. Supported: {supported_metrics}")

    def compute_dynamic_values(
        self,
        input_data,
        metric="cv",
        threshold_min=None,
        threshold_max=None,
        feature_names=None,
        output_path=None,
        transition_threshold=2.0,
        window_size=10,
        transition_mode="window",
        lag_time=1,
    ):
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
            feature_names=feature_names,
            use_memmap=self.use_memmap,
            output_path=output_path,
        )

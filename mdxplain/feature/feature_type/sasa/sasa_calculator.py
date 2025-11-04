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
SASA calculator for molecular dynamics trajectory analysis.

Utility class for computing Solvent Accessible Surface Area using the
Shrake-Rupley algorithm with support for memory mapping and chunked processing.
"""

from typing import Dict, Tuple, Any
import mdtraj as md
import numpy as np
from tqdm import tqdm

from ..helper.calculator_compute_helper import CalculatorComputeHelper
from ..interfaces.calculator_base import CalculatorBase
from .sasa_calculator_analysis import SASACalculatorAnalysis


class SASACalculator(CalculatorBase):
    """
    Calculator for computing Solvent Accessible Surface Area from MD trajectories.

    Uses the Shrake-Rupley algorithm to compute SASA at residue or atom level.
    The algorithm places points on the surface of each atom and counts how many
    are accessible to a spherical probe of specified radius.

    Uses mdtraj for SASA calculation under the hood.

    Examples
    --------
    >>> # Basic SASA calculation
    >>> calculator = SASACalculator()
    >>> sasa, metadata = calculator.compute(trajectory, mode='residue')

    >>> # With memory mapping for large datasets
    >>> calculator = SASACalculator(use_memmap=True, cache_path='./cache/')
    >>> sasa, metadata = calculator.compute(trajectory, mode='atom')
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize SASA calculator with configuration parameters.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files (includes trajectory name)
        chunk_size : int, optional
            Number of frames to process per chunk

        Returns
        -------
        None

        Examples
        --------
        >>> # Basic initialization
        >>> calculator = SASACalculator()

        >>> # With memory mapping
        >>> calculator = SASACalculator(use_memmap=True, cache_path='./cache/')
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self.sasa_path = cache_path

        self.analysis = SASACalculatorAnalysis(
            use_memmap=self.use_memmap, chunk_size=self.chunk_size
        )

    def compute(self, input_data: md.Trajectory, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute SASA using Shrake-Rupley algorithm from trajectory.

        Parameters
        ----------
        input_data : mdtraj.Trajectory
            MDTraj trajectory object to process
        kwargs : dict
            Additional parameters:

            - mode : str - 'residue' or 'atom' level SASA
            - probe_radius : float - Probe sphere radius in nm
            - res_metadata : dict - Residue metadata for naming

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Tuple containing (sasa_array, feature_metadata) where sasa_array
            has shape (n_frames, n_residues) or (n_frames, n_atoms) with
            SASA values in nm²

        Examples
        --------
        >>> # Residue-level SASA
        >>> sasa, metadata = calculator.compute(trajectory, mode='residue', probe_radius=0.14)

        >>> # Atom-level SASA with custom probe
        >>> sasa, metadata = calculator.compute(trajectory, mode='atom', probe_radius=0.12)
        """
        # Extract parameters from kwargs
        trajectory = input_data
        mode = kwargs.get('mode', 'residue')
        probe_radius = kwargs.get('probe_radius', 0.14)
        res_metadata = kwargs.get('res_metadata', None)

        # Determine output dimensions
        n_features = trajectory.n_residues if mode == 'residue' else trajectory.n_atoms

        # Setup output array
        sasa_array = self._setup_output_array(trajectory, n_features)

        # Compute SASA
        sasa_array = self._compute_sasa(trajectory, sasa_array, mode, probe_radius)

        # Generate feature metadata
        feature_metadata = self._generate_feature_metadata(trajectory, mode, probe_radius, res_metadata)

        return sasa_array, feature_metadata

    def _setup_output_array(self, trajectory: md.Trajectory, n_features: int) -> np.ndarray:
        """
        Create output array for SASA storage.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Trajectory object for size information
        n_features : int
            Number of SASA features (residues or atoms)

        Returns
        -------
        numpy.ndarray or numpy.memmap
            Output array for SASA storage
        """
        sasa_array = CalculatorComputeHelper.create_output_array(
            self.use_memmap,
            self.sasa_path,
            (trajectory.n_frames, n_features),
            dtype="float32"
        )

        return sasa_array

    def _compute_sasa(self, trajectory: md.Trajectory, sasa_array: np.ndarray, mode: str, probe_radius: float) -> np.ndarray:
        """
        Compute SASA using MDTraj's Shrake-Rupley implementation.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory
        sasa_array : numpy.ndarray or numpy.memmap
            Output array for SASA values
        mode : str
            'residue' or 'atom' level calculation
        probe_radius : float
            Probe radius in nanometers

        Returns
        -------
        numpy.ndarray
            Filled SASA array
        """
        if self.use_memmap:
            # Chunk-wise processing for memory efficiency
            for i in tqdm(range(0, trajectory.n_frames, self.chunk_size),
                         desc="Computing SASA", unit="chunks"):
                end = min(i + self.chunk_size, trajectory.n_frames)
                chunk = trajectory[i:end]
                
                # Compute SASA for chunk using Shrake-Rupley algorithm
                chunk_sasa = md.shrake_rupley(
                    chunk,
                    mode=mode,
                    probe_radius=probe_radius
                )
                
                sasa_array[i:end] = chunk_sasa
        else:
            # In-memory processing for smaller datasets
            sasa_values = md.shrake_rupley(
                trajectory,
                mode=mode,
                probe_radius=probe_radius
            )
            sasa_array[:] = sasa_values

        return sasa_array

    def _generate_feature_metadata(self, trajectory: md.Trajectory, mode: str, probe_radius: float, res_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature metadata for SASA calculations.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory for topology information
        mode : str
            'residue' or 'atom' level calculation
        probe_radius : float
            Probe radius used in calculation
        res_metadata : dict or None
            Residue metadata for naming

        Returns
        -------
        dict
            Feature metadata dictionary with SASA calculation details
        """
        features = []

        if mode == 'residue':
            # Create residue-level features
            for res_idx, residue in enumerate(trajectory.topology.residues):
                residue_feature = self._generate_residue_feature(res_idx, residue, res_metadata)
                features.append([residue_feature])
        else:  # mode == 'atom'
            # Create atom-level features
            for atom in trajectory.topology.atoms:
                atom_feature = self._generate_atom_feature(atom)
                features.append([atom_feature])

        return {
            "is_pair": False,
            "features": features,
            "computation_params": {
                "mode": mode,
                "probe_radius_nm": probe_radius
            },
            "algorithm": "shrake_rupley",
            "units": "nm²",
            "n_features": len(features),
            "visualization": {
                "is_discrete": False,
                "axis_label": "SASA (Å²)"
            }
        }

    def _generate_residue_feature(self, res_idx: int, residue, res_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature metadata for a single residue.

        Parameters
        ----------
        res_idx : int
            Residue index
        residue : mdtraj residue object
            Residue topology information
        res_metadata : dict or None
            Existing residue metadata

        Returns
        -------
        dict
            Residue feature metadata
        """
        if res_metadata and res_idx < len(res_metadata):
            # Use existing residue metadata
            return {
                "residue": res_metadata[res_idx],
                "special_label": None,
                "full_name": f"{res_metadata[res_idx]['full_name']}_SASA"
            }
        else:
            # Create basic residue info
            return {
                "residue": {
                    "index": res_idx,
                    "name": residue.name,
                    "id": residue.resSeq
                },
                "special_label": None,
                "full_name": f"{residue.name}{residue.resSeq}_SASA"
            }

    def _generate_atom_feature(self, atom) -> Dict[str, Any]:
        """
        Generate feature metadata for a single atom.

        Parameters
        ----------
        atom : mdtraj atom object
            Atom topology information

        Returns
        -------
        dict
            Atom feature metadata
        """
        return {
            "atom": {
                "index": atom.index,
                "name": atom.name,
                "element": atom.element.symbol,
                "residue": {
                    "index": atom.residue.index,
                    "name": atom.residue.name,
                    "id": atom.residue.resSeq
                }
            },
            "special_label": None,
            "full_name": f"{atom.residue.name}{atom.residue.resSeq}:{atom.name}_SASA"
        }

    def compute_dynamic_values(
        self,
        input_data: np.ndarray,
        metric: str = "cv",
        threshold_min: float = None,
        threshold_max: float = None,
        feature_metadata: list = None,
        output_path: str = None,
        transition_threshold: float = 0.5,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> Dict[str, Any]:
        """
        Filter and select variable/dynamic SASA values based on statistical criteria.

        Parameters
        ----------
        input_data : numpy.ndarray
            SASA array (n_frames, n_features)
        metric : str, default='cv'
            Metric to use for selection:
            
            - 'cv': Coefficient of variation
            - 'std': Standard deviation
            - 'variance': Variance
            - 'range': Range (max - min)
            - 'mad': Median absolute deviation
            - 'mean': Mean SASA
            - 'min': Minimum SASA
            - 'max': Maximum SASA
            - 'burial_fraction': Fraction of time below min_threshold
            - 'exposure_fraction': Fraction of time above max_threshold
            - 'transitions': Number of transitions exceeding threshold
        threshold_min : float, optional
            Minimum threshold for filtering (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold for filtering (metric_values <= threshold_max)
        feature_metadata : list, optional
            SASA metadata corresponding to data columns
        output_path : str, optional
            Path for memory-mapped output
        transition_threshold : float, default=0.5
            SASA threshold for detecting transitions (in nm²)
        window_size : int, default=10
            Window size for transition analysis
        transition_mode : str, default='window'
            Transition analysis mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for transition analysis

        Returns
        -------
        dict
            Dictionary with keys: 'indices', 'values', 'dynamic_data', 'feature_metadata',
            'metric_used', 'n_dynamic', 'total_features', 'threshold_min', 'threshold_max'

        Examples
        --------
        >>> # Select highly variable SASA (CV >= 0.5)
        >>> result = calculator.compute_dynamic_values(sasa, metric='cv', threshold_min=0.5)
        >>> variable_sasa = result['dynamic_data']

        >>> # Select buried residues (burial_fraction >= 0.8, needs threshold_max as burial cutoff)
        >>> result = calculator.compute_dynamic_values(
        ...     sasa, metric='burial_fraction', threshold_min=0.8, threshold_max=0.5
        ... )

        >>> # Select SASA with many transitions
        >>> result = calculator.compute_dynamic_values(
        ...     sasa, metric='transitions', threshold_min=5,
        ...     transition_threshold=1.0, window_size=10
        ... )
        """
        # Compute metric values using helper method
        metric_values = self._compute_metric_values(
            input_data,
            metric,
            transition_threshold,
            window_size,
            transition_mode,
            lag_time,
            threshold_min,
            threshold_max,
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

    def _compute_metric_values(
        self,
        sasa_data: np.ndarray,
        metric: str,
        threshold: float,
        window_size: int,
        transition_mode: str = "window",
        lag_time: int = 1,
        threshold_min: float = None,
        threshold_max: float = None,
    ) -> np.ndarray:
        """
        Compute metric values for SASA based on specified metric type.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array (n_frames, n_features)
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
        threshold_min : float, optional
            Used for burial_fraction calculation
        threshold_max : float, optional
            Used as burial/exposure cutoff

        Returns
        -------
        numpy.ndarray
            Computed metric values per SASA feature
        """
        # Define simple metrics that map directly to analysis methods
        simple_metrics = {
            "std": self.analysis.compute_std,
            "variance": self.analysis.compute_variance,
            "min": self.analysis.compute_min,
            "max": self.analysis.compute_max,
            "mad": self.analysis.compute_mad,
            "mean": self.analysis.compute_mean,
        }

        # Handle simple metrics
        if metric in simple_metrics:
            return simple_metrics[metric](sasa_data)

        # Handle complex metrics
        if metric == "cv":
            mean_vals = self.analysis.compute_mean(sasa_data)
            std_vals = self.analysis.compute_std(sasa_data)
            return std_vals / (mean_vals + 1e-10)
        if metric == "range" or metric == "dynamic_range":
            max_vals = self.analysis.compute_max(sasa_data)
            min_vals = self.analysis.compute_min(sasa_data)
            return max_vals - min_vals
        if metric == "burial_fraction":
            cutoff = threshold_min if threshold_min is not None else 0.1
            return self.analysis.compute_burial_fraction(sasa_data, cutoff)
        if metric == "exposure_fraction":
            cutoff = threshold_max if threshold_max is not None else 1.0
            return self.analysis.compute_exposure_fraction(sasa_data, cutoff)
        if metric == "transitions":
            if transition_mode == "lagtime":
                return self.analysis.compute_transitions_lagtime(sasa_data, threshold, lag_time)
            else:  # window
                return self.analysis.compute_transitions_window(sasa_data, threshold, window_size)

        # Unknown metric
        supported_metrics = [
            "cv", "std", "variance", "range", "transitions", "min", "max",
            "mad", "mean", "burial_fraction", "exposure_fraction"
        ]
        raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")


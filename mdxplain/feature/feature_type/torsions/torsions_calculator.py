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
Torsions calculator for molecular dynamics trajectory analysis.

Utility class for computing dihedral torsion angles with support for
backbone and side chain angles, memory mapping, and chunked processing.
"""

from typing import Any, Dict, List, Tuple

import mdtraj as md
import numpy as np

from mdxplain.utils.progress_util import ProgressController

from ..helper.calculator_compute_helper import CalculatorComputeHelper
from ..helper.calculator_stat_helper import CalculatorStatHelper
from ..interfaces.calculator_base import CalculatorBase
from .torsions_calculator_analysis import TorsionsCalculatorAnalysis


class TorsionsCalculator(CalculatorBase):
    """
    Calculator for computing dihedral torsion angles from MD trajectories.

    Uses MDTraj's torsion angle calculation functions to compute backbone
    (phi, psi, omega) and side chain (chi1-4) dihedral angles with proper
    padding for residues where angles are not defined. All angles are
    computed and returned in degrees.

    Examples
    --------
    >>> # Basic torsions calculation with all angles
    >>> calculator = TorsionsCalculator()
    >>> torsions, metadata = calculator.compute(trajectory)

    >>> # Only phi and psi backbone angles
    >>> calculator = TorsionsCalculator()
    >>> torsions, metadata = calculator.compute(trajectory, 
    ...                                        calculate_phi=True, calculate_psi=True,
    ...                                        calculate_omega=False, calculate_chi=False)

    >>> # With memory mapping for large datasets
    >>> calculator = TorsionsCalculator(use_memmap=True, cache_path='./cache/')
    >>> torsions, metadata = calculator.compute(trajectory, calculate_chi=True)
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000, use_pbc: bool = True) -> None:
        """
        Initialize torsions calculator with configuration parameters.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files (includes trajectory name)
        chunk_size : int, optional
            Number of frames to process per chunk
        use_pbc : bool, default=True
            If True and the trajectory contains unitcell information,
            angles are computed under the minimum image convention

        Returns
        -------
        None

        Examples
        --------
        >>> # Basic initialization
        >>> calculator = TorsionsCalculator()

        >>> # With memory mapping
        >>> calculator = TorsionsCalculator(use_memmap=True, cache_path='./cache/')

        >>> # Without periodic boundary conditions
        >>> calculator = TorsionsCalculator(use_pbc=False)
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self.torsions_path = cache_path
        self.use_pbc = use_pbc

        self.analysis = TorsionsCalculatorAnalysis(
            use_memmap=self.use_memmap, chunk_size=self.chunk_size
        )

    def compute(self, input_data: md.Trajectory, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute torsion angles from trajectory.

        Parameters
        ----------
        input_data : mdtraj.Trajectory
            MDTraj trajectory object to process
        kwargs : dict
            Additional parameters:

            - calculate_phi : bool - Whether to compute phi backbone angles
            - calculate_psi : bool - Whether to compute psi backbone angles
            - calculate_omega : bool - Whether to compute omega backbone angles
            - calculate_chi : bool - Whether to compute side chain chi angles (chi1-4)
            - res_metadata : dict - Residue metadata for naming

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Tuple containing (torsions_array, feature_metadata) where torsions_array
            has shape (n_frames, n_residues * n_angle_types) with angles in degrees

        Examples
        --------
        >>> # Only phi and psi backbone torsions
        >>> torsions, metadata = calculator.compute(trajectory, 
        ...                                        calculate_phi=True, calculate_psi=True,
        ...                                        calculate_omega=False, calculate_chi=False)

        >>> # All angles including chi
        >>> torsions, metadata = calculator.compute(trajectory, 
        ...                                        calculate_phi=True, calculate_psi=True, 
        ...                                        calculate_omega=True, calculate_chi=True)
        """
        # Extract parameters from kwargs
        trajectory = input_data
        calculate_phi = kwargs.get('calculate_phi', True)
        calculate_psi = kwargs.get('calculate_psi', True)
        calculate_omega = kwargs.get('calculate_omega', True)
        calculate_chi = kwargs.get('calculate_chi', True)
        res_metadata = kwargs.get('res_metadata', None)

        # First compute to get the actual number of features
        test_angles, angle_info = self._compute_all_angles(
            trajectory[:1], calculate_phi, calculate_psi, calculate_omega, calculate_chi
        )
        n_features = test_angles.shape[1]

        # Setup output array with correct size
        torsions_array = self._setup_output_array(trajectory, n_features)

        # Compute torsion angles
        torsions_array = self._compute_torsion_angles(
            trajectory, torsions_array, calculate_phi, calculate_psi, calculate_omega, calculate_chi
        )

        # Generate feature metadata using angle_info from first computation
        feature_metadata = self._generate_feature_metadata(trajectory, angle_info, res_metadata)

        return torsions_array, feature_metadata

    def _setup_output_array(self, trajectory: md.Trajectory, n_features: int) -> np.ndarray:
        """
        Create output array for torsions storage.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Trajectory object for size information
        n_features : int
            Number of computed angle features

        Returns
        -------
        numpy.ndarray or numpy.memmap
            Output array for torsions storage
        """
        torsions_array = CalculatorComputeHelper.create_output_array(
            self.use_memmap,
            self.torsions_path,
            (trajectory.n_frames, n_features),
            dtype="float32"
        )

        return torsions_array

    def _compute_torsion_angles(self, trajectory: md.Trajectory, torsions_array: np.ndarray,
                               calculate_phi: bool, calculate_psi: bool, 
                               calculate_omega: bool, calculate_chi: bool) -> np.ndarray:
        """
        Compute torsion angles using MDTraj functions.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory
        torsions_array : numpy.ndarray or numpy.memmap
            Output array for torsion angles
        calculate_phi : bool
            Whether to compute phi angles
        calculate_psi : bool
            Whether to compute psi angles
        calculate_omega : bool
            Whether to compute omega angles
        calculate_chi : bool
            Whether to include chi angles

        Returns
        -------
        numpy.ndarray
            Filled torsions array
        """
        if self.use_memmap:
            # Chunk-wise processing for memory efficiency
            for i in ProgressController.iterate(
                range(0, trajectory.n_frames, self.chunk_size),
                desc="Computing torsions",
                unit="chunks",
            ):
                end = min(i + self.chunk_size, trajectory.n_frames)
                chunk = trajectory[i:end]
                
                # Compute angles for chunk
                chunk_angles, _ = self._compute_all_angles(
                    chunk, calculate_phi, calculate_psi, calculate_omega, calculate_chi
                )
                
                torsions_array[i:end] = chunk_angles
        else:
            # In-memory processing for smaller datasets
            all_angles, _ = self._compute_all_angles(
                trajectory, calculate_phi, calculate_psi, calculate_omega, calculate_chi
            )
            torsions_array[:] = all_angles

        return torsions_array

    def _compute_all_angles(self, trajectory: md.Trajectory, calculate_phi: bool, 
                           calculate_psi: bool, calculate_omega: bool, calculate_chi: bool) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
        """
        Compute all requested torsion angles directly.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Trajectory to process
        calculate_phi : bool
            Whether to compute phi angles
        calculate_psi : bool
            Whether to compute psi angles
        calculate_omega : bool
            Whether to compute omega angles
        calculate_chi : bool
            Whether to include chi angles

        Returns
        -------
        tuple[numpy.ndarray, list]
            Tuple of (angles_array, angle_info) where angle_info contains
            (angle_type, indices) pairs for metadata generation
        """
        angle_arrays = []
        angle_info = []

        # Compute phi angles
        if calculate_phi:
            indices, values = md.compute_phi(trajectory, periodic=self.use_pbc)
            if values.size > 0:
                angle_arrays.append(np.degrees(values))
                angle_info.append(('phi', indices))

        # Compute psi angles
        if calculate_psi:
            indices, values = md.compute_psi(trajectory, periodic=self.use_pbc)
            if values.size > 0:
                angle_arrays.append(np.degrees(values))
                angle_info.append(('psi', indices))

        # Compute omega angles
        if calculate_omega:
            indices, values = md.compute_omega(trajectory, periodic=self.use_pbc)
            if values.size > 0:
                angle_arrays.append(np.degrees(values))
                angle_info.append(('omega', indices))

        # Compute chi angles if requested
        if calculate_chi:
            for chi_num in [1, 2, 3, 4]:
                if chi_num == 1:
                    indices, values = md.compute_chi1(trajectory, periodic=self.use_pbc)
                elif chi_num == 2:
                    indices, values = md.compute_chi2(trajectory, periodic=self.use_pbc)
                elif chi_num == 3:
                    indices, values = md.compute_chi3(trajectory, periodic=self.use_pbc)
                elif chi_num == 4:
                    indices, values = md.compute_chi4(trajectory, periodic=self.use_pbc)

                if values.size > 0:
                    angle_arrays.append(np.degrees(values))
                    angle_info.append((f'chi{chi_num}', indices))

        return np.hstack(angle_arrays), angle_info

    def _generate_feature_metadata(self, trajectory: md.Trajectory, angle_info: List[Tuple[str, np.ndarray]],
                                  res_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature metadata for computed torsion angles.

        Creates metadata only for angles that were actually computed,
        using angle_info from the compute process.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory for topology information
        angle_info : list
            List of (angle_type, indices) tuples from computation
        res_metadata : dict or None
            Residue metadata for naming

        Returns
        -------
        dict
            Feature metadata dictionary with torsion calculation details
        """
        features = []
        backbone_angles = []
        chi_included = False

        # Process each angle type and its computed angles
        for angle_type, indices in angle_info:
            # Track what was computed
            if angle_type in ['phi', 'psi', 'omega']:
                backbone_angles.append(angle_type)
            elif angle_type.startswith('chi'):
                chi_included = True

            # Create metadata for each computed angle
            for atom_indices in indices:
                # Get central residue (typically atom index 1)
                residue = trajectory.topology.atom(atom_indices[2]).residue
                res_idx = residue.index

                if res_metadata and res_idx < len(res_metadata):
                    # Use existing residue metadata
                    feature_info = {
                        "residue": res_metadata[res_idx],
                        "angle_type": angle_type,
                        "special_label": None,
                        "full_name": f"{res_metadata[res_idx]['full_name']}_{angle_type}"
                    }
                else:
                    # Create basic residue info
                    feature_info = {
                        "residue": {
                            "index": res_idx,
                            "name": residue.name,
                            "id": residue.resSeq
                        },
                        "angle_type": angle_type,
                        "special_label": None,
                        "full_name": f"{residue.name}{residue.resSeq}_{angle_type}"
                    }
                features.append([feature_info])

        return {
            "is_pair": False,
            "features": features,
            "computation_params": {
                "backbone_angles": backbone_angles,
                "include_chi": chi_included
            },
            "units": "degrees",
            "angle_range": "(-180, +180)",
            "n_features": len(features),
            "algorithm": "mdtraj_dihedral",
            "visualization": {
                "is_discrete": False,
                "axis_label": "Angle (°)",
                "allow_hide_prefix": True
            }
        }

    def compute_dynamic_values(
        self,
        input_data: np.ndarray,
        metric: str = "cv",
        threshold_min: float = None,
        threshold_max: float = None,
        feature_metadata: list = None,
        output_path: str = None,
        transition_threshold: float = 30.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> Dict[str, Any]:
        """
        Filter and select variable/dynamic torsion angles based on statistical criteria.

        All statistical computations use circular statistics appropriate for torsion angles.
        Input data is assumed to be in degrees and all calculations handle circular
        properties (periodicity at ±180°) correctly.

        Parameters
        ----------
        input_data : numpy.ndarray
            Torsion angle array (n_frames, n_angles) in degrees
        metric : str, default='cv'
            Metric to use for selection:
            
            - 'cv': Coefficient of variation (circular)
            - 'std': Standard deviation (circular)
            - 'variance': Variance (circular)
            - 'mad': Median absolute deviation (circular)
            - 'mean': Mean torsion angle (circular)
            - 'range': Angular range accounting for circularity
            - 'min': Minimum value
            - 'max': Maximum value
            - 'transitions': Number of transitions exceeding threshold
        threshold_min : float, optional
            Minimum threshold for filtering (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold for filtering (metric_values <= threshold_max)
        feature_metadata : list, optional
            Torsion angle metadata corresponding to data columns
        output_path : str, optional
            Path for memory-mapped output
        transition_threshold : float, default=30.0
            Angular threshold for detecting transitions (in degrees)
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
            'metric_used', 'n_dynamic', 'total_angles', 'threshold_min', 'threshold_max'

        Examples
        --------
        >>> # Select highly flexible torsions (CV >= 0.5)
        >>> result = calculator.compute_dynamic_values(torsions, metric='cv', threshold_min=0.5)
        >>> flexible_torsions = result['dynamic_data']

        >>> # Select torsions with high circular variance (0.8-1.0 indicates high flexibility)
        >>> result = calculator.compute_dynamic_values(
        ...     torsions, metric='variance', threshold_min=0.8
        ... )

        >>> # Select torsions with many angular transitions (> 30° change)
        >>> result = calculator.compute_dynamic_values(
        ...     torsions, metric='transitions', threshold_min=5,
        ...     transition_threshold=30.0, window_size=10
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
        torsions_data: np.ndarray,
        metric: str,
        threshold: float,
        window_size: int,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute metric values for torsions based on specified metric type.

        Parameters
        ----------
        torsions_data : numpy.ndarray
            Torsion angle array (n_frames, n_angles)
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

        Returns
        -------
        numpy.ndarray
            Computed metric values per torsion angle
        """
        # All metrics mapped to analysis methods
        metrics_map = {
            "std": self.analysis.compute_std,
            "variance": self.analysis.compute_variance,
            "mad": self.analysis.compute_mad,
            "mean": self.analysis.compute_mean,
            "cv": self.analysis.compute_cv,
            "range": self.analysis.compute_range,
            "min": self.analysis.compute_min,
            "max": self.analysis.compute_max,
        }

        # Handle metrics directly from analysis
        if metric in metrics_map:
            return metrics_map[metric](torsions_data)

        # Handle transitions with parameters
        if metric == "transitions":
            if transition_mode == "lagtime":
                return self.analysis.compute_transitions_lagtime(torsions_data, threshold, lag_time)
            else:  # window
                return self.analysis.compute_transitions_window(torsions_data, threshold, window_size)

        # Unknown metric
        supported_metrics = [
            "cv", "std", "variance", "mad", "mean", "range", "min", "max", "transitions"
        ]
        raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

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
Coordinates calculator for molecular dynamics trajectory analysis.

Utility class for extracting XYZ coordinates from MD trajectories with
flexible atom selection. Supports memory mapping for large datasets.
"""

from typing import Dict, Tuple, Any
import mdtraj as md
import numpy as np
from tqdm import tqdm

from ..helper.calculator_compute_helper import CalculatorComputeHelper
from ..interfaces.calculator_base import CalculatorBase
from .coordinates_calculator_analysis import CoordinatesCalculatorAnalysis


class CoordinatesCalculator(CalculatorBase):
    """
    Calculator for extracting atomic coordinates from MD trajectories.

    Extracts XYZ coordinates for selected atoms using flexible selection
    strings. Supports memory-mapped arrays for large datasets, chunked
    processing for memory efficiency, and various atom selection modes.

    Examples
    --------
    >>> # Basic coordinate extraction
    >>> calculator = CoordinatesCalculator()
    >>> coords, metadata = calculator.compute(trajectory, selection='ca')

    >>> # With memory mapping for large datasets
    >>> calculator = CoordinatesCalculator(use_memmap=True, cache_path='./cache/')
    >>> coords, metadata = calculator.compute(trajectory, selection='backbone')
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize coordinates calculator with configuration parameters.

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
        >>> calculator = CoordinatesCalculator()

        >>> # With memory mapping
        >>> calculator = CoordinatesCalculator(use_memmap=True, cache_path='./cache/')
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self.coordinates_path = cache_path

        self.analysis = CoordinatesCalculatorAnalysis(
            use_memmap=self.use_memmap, chunk_size=self.chunk_size
        )

    def compute(self, input_data: md.Trajectory, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract coordinates for selected atoms from trajectory.

        Parameters
        ----------
        input_data : mdtraj.Trajectory
            MDTraj trajectory object to process
        '**'kwargs : dict
            Additional parameters:
            
            - selection : str, default='ca' - Atom selection string
            - res_metadata : dict - Residue metadata for naming

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Tuple containing (coordinates_array, feature_metadata) where coordinates_array
            has shape (n_frames, n_selected_atoms * 3) and feature_metadata contains
            atom information and selection details

        Examples
        --------
        >>> # Extract alpha carbon coordinates
        >>> coords, metadata = calculator.compute(trajectory, selection='ca')
        >>> print(f"Shape: {coords.shape}")  # (n_frames, n_residues * 3)

        >>> # Extract all atoms
        >>> coords, metadata = calculator.compute(trajectory, selection='all')

        >>> # Custom selection
        >>> coords, metadata = calculator.compute(trajectory, 
        ...                                      selection='resname ALA and name CA')
        """
        # Extract parameters from kwargs
        trajectory = input_data
        selection = kwargs.get('selection', 'ca')

        # Perform atom selection
        indices = self._perform_atom_selection(trajectory, selection)

        # Setup output array
        n_features = len(indices) * 3  # XYZ per atom
        coordinates = self._setup_output_array(trajectory, n_features)

        # Extract coordinates
        coordinates = self._extract_coordinates(trajectory, indices, coordinates)

        # Generate feature metadata
        feature_metadata = self._generate_feature_metadata(indices, trajectory.topology, selection)

        return coordinates, feature_metadata

    def _perform_atom_selection(self, trajectory: md.Trajectory, selection: str) -> np.ndarray:
        """
        Perform atom selection using MDTraj selection strings.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Trajectory to select atoms from
        selection : str
            Selection string ('all', 'ca', 'backbone', 'heavy', or MDTraj syntax)

        Returns
        -------
        numpy.ndarray
            Array of selected atom indices

        Raises
        ------
        ValueError
            If selection string is invalid or selects no atoms
        """
        if selection == 'all':
            indices = np.arange(trajectory.n_atoms)
        else:
            try:
                indices = trajectory.topology.select(selection)
            except Exception as e:
                raise ValueError(
                    f"Invalid selection string '{selection}': {e}. "
                    f"Use MDTraj selection syntax (e.g., 'name CA', 'backbone', 'not element H')."
                ) from e

        if len(indices) == 0:
            raise ValueError(
                f"Selection '{selection}' matched no atoms in trajectory. "
                f"Please check your selection string."
            )

        print(f"Selected {len(indices)} atoms using selection '{selection}'")
        return indices

    def _setup_output_array(self, trajectory: md.Trajectory, n_features: int) -> np.ndarray:
        """
        Create output array for coordinates storage.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Trajectory object for size information
        n_features : int
            Number of coordinate features (n_atoms * 3)

        Returns
        -------
        numpy.ndarray or numpy.memmap
            Output array for coordinates storage
        """
        coordinates = CalculatorComputeHelper.create_output_array(
            self.use_memmap,
            self.coordinates_path,
            (trajectory.n_frames, n_features),
            dtype="float32"
        )

        return coordinates

    def _extract_coordinates(self, trajectory: md.Trajectory, indices: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Extract coordinates for selected atoms from trajectory.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory
        indices : numpy.ndarray
            Selected atom indices
        coordinates : numpy.ndarray or numpy.memmap
            Output array for coordinates

        Returns
        -------
        numpy.ndarray
            Filled coordinates array
        """
        if self.use_memmap or hasattr(coordinates, 'flush'):
            # Chunk-wise processing for memory efficiency
            for i in tqdm(range(0, trajectory.n_frames, self.chunk_size),
                         desc="Extracting coordinates", unit="chunks"):
                end = min(i + self.chunk_size, trajectory.n_frames)
                
                # Extract coordinates for chunk
                chunk_coords = trajectory[i:end].xyz[:, indices, :]
                
                # Reshape to flat format (n_frames, n_atoms * 3)
                coordinates[i:end] = chunk_coords.reshape(end - i, -1)
                
                # Flush to disk if memory-mapped
                if hasattr(coordinates, 'flush'):
                    coordinates.flush()
        else:
            # In-memory processing for smaller datasets
            coords = trajectory.xyz[:, indices, :]
            coordinates[:] = coords.reshape(trajectory.n_frames, -1)

        # Convert from nanometers to Angstrom (MDTraj uses nm)
        coordinates *= 10.0

        return coordinates

    def _generate_feature_metadata(self, indices: np.ndarray, topology: md.Topology, selection: str) -> Dict[str, Any]:
        """
        Generate feature metadata for selected atoms.

        Parameters
        ----------
        indices : numpy.ndarray
            Selected atom indices
        topology : mdtraj.Topology
            Trajectory topology for atom information
        selection : str
            Original selection string

        Returns
        -------
        dict
            Feature metadata dictionary with atom information
        """
        # Create atom information for each selected atom
        features = []
        
        for atom_idx in indices:
            atom = topology.atom(atom_idx)
            
            # Create XYZ entries for this atom
            for coord in ['x', 'y', 'z']:
                atom_info = {
                    "atom": {
                        "index": int(atom_idx),
                        "name": atom.name,
                        "element": atom.element.symbol,
                        "residue": {
                            "index": atom.residue.index,
                            "name": atom.residue.name,
                            "id": atom.residue.resSeq
                        },
                        "coordinate": coord
                    },
                    "special_label": None,
                    "full_name": f"{atom.residue.name}{atom.residue.resSeq}:{atom.name}_{coord}"
                }
                features.append([atom_info])

        return {
            "is_pair": False,
            "features": features,
            "computation_params": {
                "selection": selection
            },
            "n_selected_atoms": len(indices),
            "n_features": len(features),
            "coordinate_system": "cartesian_xyz",
            "units": "angstrom",
            "visualization": {
                "is_discrete": False,
                "axis_label": "Position (Å)"
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
        transition_threshold: float = 1.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> Dict[str, Any]:
        """
        Filter and select variable/dynamic coordinates based on statistical criteria.

        Parameters
        ----------
        input_data : numpy.ndarray
            Coordinate array (n_frames, n_coordinates)
        metric : str, default='cv'
            Metric to use for selection:
            
            - 'cv': Coefficient of variation
            - 'std': Standard deviation
            - 'variance': Variance
            - 'range': Range (max - min)
            - 'mad': Median absolute deviation
            - 'mean': Mean coordinate
            - 'min': Minimum coordinate
            - 'max': Maximum coordinate
            - 'rmsf': Root mean square fluctuation per atom
            - 'transitions': Number of transitions exceeding threshold
        threshold_min : float, optional
            Minimum threshold for filtering (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold for filtering (metric_values <= threshold_max)
        feature_metadata : list, optional
            Coordinate metadata corresponding to data columns
        output_path : str, optional
            Path for memory-mapped output
        transition_threshold : float, default=1.0
            Distance threshold for detecting transitions (in Angstroms)
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
            'metric_used', 'n_dynamic', 'total_coordinates', 'threshold_min', 'threshold_max'

        Examples
        --------
        >>> # Select highly variable coordinates (CV >= 0.5)
        >>> result = calculator.compute_dynamic_values(coords, metric='cv', threshold_min=0.5)
        >>> variable_coords = result['dynamic_data']

        >>> # Select atoms with high RMSF (> 2.0 Å)
        >>> result = calculator.compute_dynamic_values(coords, metric='rmsf', threshold_min=2.0)

        >>> # Select coordinates with many positional transitions
        >>> result = calculator.compute_dynamic_values(
        ...     coords, metric='transitions', threshold_min=5,
        ...     transition_threshold=1.5, window_size=10
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
        coordinates: np.ndarray,
        metric: str,
        threshold: float,
        window_size: int,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute metric values for coordinates based on specified metric type.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array (n_frames, n_coordinates)
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
            Computed metric values per coordinate
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
            return simple_metrics[metric](coordinates)

        # Handle complex metrics
        if metric == "cv":
            return self.analysis.compute_cv(coordinates)
        if metric == "range":
            return self.analysis.compute_range(coordinates)
        if metric == "rmsf":
            return self.analysis.compute_rmsf(coordinates)
        if metric == "transitions":
            return self._compute_transitions_metric(
                coordinates, threshold, window_size, transition_mode, lag_time
            )

        # Unknown metric
        supported_metrics = [
            "cv", "std", "variance", "range", "transitions", "min", "max",
            "mad", "mean", "rmsf"
        ]
        raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

    def _compute_transitions_metric(
        self, 
        coordinates: np.ndarray, 
        threshold: float, 
        window_size: int, 
        transition_mode: str, 
        lag_time: int
    ) -> np.ndarray:
        """
        Compute transitions metric based on specified mode and parameters.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array (n_frames, n_coordinates)
        threshold : float
            Threshold value for detecting transitions (in Angstroms)
        window_size : int
            Window size for transitions metric
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime'
        lag_time : int, default=1
            Lag time for transitions metric

        Returns
        -------
        numpy.ndarray
            Number of transitions per coordinate
        """
        if transition_mode == "window":
            return self.analysis.compute_transitions_window(
                coordinates, threshold=threshold, window_size=window_size
            )
        if transition_mode == "lagtime":
            return self.analysis.compute_transitions_lagtime(
                coordinates, threshold=threshold, lag_time=lag_time
            )
        raise ValueError(
            f"Unknown transition mode: {transition_mode}. "
            f"Supported: 'window', 'lagtime'"
        )

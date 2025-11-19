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
DSSP calculator for molecular dynamics trajectory analysis.

Utility class for computing secondary structure assignments using DSSP algorithm
with support for multiple encoding formats and memory mapping.
"""

from typing import Dict, Tuple, Any
import mdtraj as md
import numpy as np
from tqdm import tqdm

from ..helper.calculator_compute_helper import CalculatorComputeHelper
from ..interfaces.calculator_base import CalculatorBase
from .dssp_calculator_analysis import DSSPCalculatorAnalysis
from .helper.dssp_encoding_helper import DSSPEncodingHelper


class DSSPCalculator(CalculatorBase):
    """
    Calculator for computing DSSP secondary structure assignments from MD trajectories.

    Uses MDTraj's DSSP implementation to assign secondary structure classifications
    with support for both simplified and full DSSP schemes. Multiple encoding
    formats support different downstream analysis requirements.

    Uses mdtraj for dssp calculations under the hood.

    Examples
    --------
    >>> # Basic DSSP calculation
    >>> calculator = DSSPCalculator()
    >>> dssp, metadata = calculator.compute(trajectory, simplified=True)

    >>> # With memory mapping for large datasets
    >>> calculator = DSSPCalculator(use_memmap=True, cache_path='./cache/')
    >>> dssp, metadata = calculator.compute(trajectory, encoding='onehot')
    """

    FULL_CLASSES = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C', 'NA']
    """Full DSSP classification after space to C conversion (9 classes)"""
    
    SIMPLIFIED_CLASSES = ['H', 'E', 'C', 'NA']
    """Simplified DSSP classification (4 classes including NA)"""

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize DSSP calculator with configuration parameters.

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
        >>> calculator = DSSPCalculator()

        >>> # With memory mapping
        >>> calculator = DSSPCalculator(use_memmap=True, cache_path='./cache/')
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self.dssp_path = cache_path

        self.analysis = DSSPCalculatorAnalysis(
            full_classes=self.FULL_CLASSES, simplified_classes=self.SIMPLIFIED_CLASSES,
            use_memmap=self.use_memmap, chunk_size=self.chunk_size, cache_path=self.cache_path
        )

    def compute(self, input_data: md.Trajectory, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute DSSP secondary structure assignments from trajectory.

        Parameters
        ----------
        input_data : mdtraj.Trajectory
            MDTraj trajectory object to process
        kwargs : dict
            Additional parameters:

            - simplified : bool - Use simplified (3-class) or full (8-class) DSSP
            - encoding : str - Output encoding ('onehot', 'integer', 'char')
            - res_metadata : dict - Residue metadata for naming

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Tuple containing (dssp_array, feature_metadata) where dssp_array
            format depends on encoding and classification level

        Examples
        --------
        >>> # Simplified DSSP with one-hot encoding
        >>> dssp, metadata = calculator.compute(trajectory, simplified=True, encoding='onehot')

        >>> # Full DSSP with integer encoding
        >>> dssp, metadata = calculator.compute(trajectory, simplified=False, encoding='integer')
        """
        # Extract parameters from kwargs
        trajectory = input_data
        simplified = kwargs.get('simplified', False)
        encoding = kwargs.get('encoding', 'onehot')
        res_metadata = kwargs.get('res_metadata', None)

        # Determine output dimensions and data type
        n_residues = trajectory.n_residues
        classes = self.SIMPLIFIED_CLASSES if simplified else self.FULL_CLASSES
        n_classes = len(classes)

        # Setup output array based on encoding
        dssp_array = self._setup_output_array(trajectory, n_residues, n_classes, encoding)

        # Compute DSSP assignments
        dssp_array = self._compute_dssp_assignments(trajectory, dssp_array, simplified, encoding, classes)

        # Generate feature metadata
        feature_metadata = self._generate_feature_metadata(trajectory, simplified, encoding, classes, res_metadata)

        return dssp_array, feature_metadata

    def _setup_output_array(self, trajectory: md.Trajectory, n_residues: int, n_classes: int, encoding: str) -> np.ndarray:
        """
        Create output array for DSSP storage based on encoding type.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Trajectory object for size information
        n_residues : int
            Number of residues
        n_classes : int
            Number of DSSP classes
        encoding : str
            Encoding type ('onehot', 'integer', 'char')

        Returns
        -------
        numpy.ndarray or numpy.memmap
            Output array for DSSP storage
        """
        if encoding == 'onehot':
            # One-hot encoding: (n_frames, n_residues * n_classes)
            shape = (trajectory.n_frames, n_residues * n_classes)
            dtype = "float32"
        elif encoding == 'integer':
            # Integer encoding: (n_frames, n_residues)
            shape = (trajectory.n_frames, n_residues)
            dtype = "int8"
        else:  # encoding == 'char'
            # Character encoding: (n_frames, n_residues)
            shape = (trajectory.n_frames, n_residues)
            dtype = "U1"

        dssp_array = CalculatorComputeHelper.create_output_array(
            self.use_memmap,
            self.dssp_path,
            shape,
            dtype=dtype
        )

        return dssp_array

    def _compute_dssp_assignments(self, trajectory: md.Trajectory, dssp_array: np.ndarray, 
                                 simplified: bool, encoding: str, classes: list) -> np.ndarray:
        """
        Compute DSSP assignments using MDTraj and encode according to specified format.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory
        dssp_array : numpy.ndarray or numpy.memmap
            Output array for DSSP assignments
        simplified : bool
            Whether to use simplified classification
        encoding : str
            Output encoding format
        classes : list
            List of class labels

        Returns
        -------
        numpy.ndarray
            Filled DSSP array
        """
        if self.use_memmap:
            # Chunk-wise processing for memory efficiency
            for i in tqdm(range(0, trajectory.n_frames, self.chunk_size),
                         desc="Computing DSSP", unit="chunks"):
                end = min(i + self.chunk_size, trajectory.n_frames)
                chunk = trajectory[i:end]

                # Compute DSSP for chunk
                chunk_dssp = md.compute_dssp(chunk, simplified=simplified)

                # MDTraj returns ' ' for loops/irregular elements
                chunk_dssp = np.where(chunk_dssp == ' ', 'C', chunk_dssp)

                # Encode chunk according to format
                encoded_chunk = self._encode_dssp_assignments(chunk_dssp, encoding, classes, simplified)

                dssp_array[i:end] = encoded_chunk
        else:
            # In-memory processing for smaller datasets
            dssp_assignments = md.compute_dssp(trajectory, simplified=simplified)

            # MDTraj returns ' ' for loops/irregular elements
            dssp_assignments = np.where(dssp_assignments == ' ', 'C', dssp_assignments)

            # Encode according to format
            dssp_array[:] = self._encode_dssp_assignments(dssp_assignments, encoding, classes, simplified)

        return dssp_array

    def _encode_dssp_assignments(self, dssp_data: np.ndarray, encoding: str, classes: list, simplified: bool) -> np.ndarray:
        """
        Encode DSSP assignments according to specified format.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            Raw DSSP assignments (n_frames, n_residues)
        encoding : str
            Encoding format ('onehot', 'integer', 'char')
        classes : list
            List of class labels
        simplified : bool
            Whether using simplified classification

        Returns
        -------
        numpy.ndarray
            Encoded DSSP assignments

        Notes
        -----
        Always uses direct encoding methods, not chunked variants.
        This is correct because:

        - When use_memmap=False: Data is small, called once for full array
        - When use_memmap=True: Data is chunk, called per chunk in loop
        
        The *_chunked methods are only for encoding large arrays in one call,
        but here we either have small arrays or are already processing chunks.

        Space conversion happens centrally in _compute_dssp_assignments() before
        this method is called, so dssp_data is already cleaned.
        """
        if encoding == 'char':
            return dssp_data.astype('U1')
        elif encoding == 'integer':
            return DSSPEncodingHelper.encode_integer(dssp_data, classes)
        else:  # onehot
            return DSSPEncodingHelper.encode_onehot_direct(dssp_data, classes)

    def _generate_feature_metadata(self, trajectory: md.Trajectory, simplified: bool, encoding: str, 
                                  classes: list, res_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature metadata for DSSP calculations.

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory for topology information
        simplified : bool
            Whether using simplified classification
        encoding : str
            Encoding format used
        classes : list
            List of class labels
        res_metadata : dict or None
            Residue metadata for naming

        Returns
        -------
        dict
            Feature metadata dictionary with DSSP calculation details
            
        Notes
        -----
        Delegates to helper methods for clean separation of concerns.
        """
        if encoding == 'onehot':
            features = self._create_onehot_features(trajectory, classes, res_metadata)
        else:
            features = self._create_standard_features(trajectory, res_metadata)
        
        # Prepare visualization metadata
        is_onehot = (encoding == 'onehot')
        axis_label = "Secondary Structure State" if is_onehot else "Secondary Structure"

        # Prepare tick labels for discrete features
        tick_labels = self._create_tick_labels(encoding, classes)

        # Create matrix_mapping for char => int conversion
        matrix_mapping = {char: idx for idx, char in enumerate(classes)}

        return {
            "is_pair": False,
            "features": features,
            "computation_params": {
                "simplified": simplified,
                "encoding": encoding,
                "classes": classes
            },
            "n_classes": len(classes),
            "n_features": len(features),
            "algorithm": "dssp",
            "matrix_mapping": matrix_mapping,
            "visualization": {
                "is_discrete": True,
                "is_binary": is_onehot,
                "axis_label": axis_label,
                "tick_labels": tick_labels,
                "allow_hide_prefix": True
            }
        }

    def _create_onehot_features(self, trajectory: md.Trajectory, classes: list, res_metadata: Dict[str, Any]) -> list:
        """
        Create feature metadata for one-hot encoding.
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory for topology information
        classes : list
            List of DSSP class labels
        res_metadata : dict or None
            Residue metadata for naming
            
        Returns
        -------
        list
            List of feature metadata for each residue-class combination
            
        Notes
        -----
        Creates one feature per residue per DSSP class.
        """
        features = []
        for res_idx, residue in enumerate(trajectory.topology.residues):
            res_info = self._get_residue_info(res_idx, residue, res_metadata)
            for class_label in classes:
                feature_info = {
                    "residue": res_info,
                    "dssp_class": class_label,
                    "special_label": None,
                    "full_name": f"{res_info.get('full_name', f'{residue.name}{residue.resSeq}')}_DSSP_{class_label}"
                }
                features.append([feature_info])
        return features

    def _create_standard_features(self, trajectory: md.Trajectory, res_metadata: Dict[str, Any]) -> list:
        """
        Create feature metadata for standard encoding.
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Source trajectory for topology information
        res_metadata : dict or None
            Residue metadata for naming
            
        Returns
        -------
        list
            List of feature metadata for each residue
            
        Notes
        -----
        Creates one feature per residue for char/integer encoding.
        """
        features = []
        for res_idx, residue in enumerate(trajectory.topology.residues):
            res_info = self._get_residue_info(res_idx, residue, res_metadata)
            feature_info = {
                "residue": res_info,
                "special_label": None,
                "full_name": f"{res_info.get('full_name', f'{residue.name}{residue.resSeq}')}_DSSP"
            }
            features.append([feature_info])
        return features

    def _get_residue_info(self, res_idx: int, residue, res_metadata: Dict[str, Any]) -> dict:
        """
        Get residue information from metadata or create default.
        
        Parameters
        ----------
        res_idx : int
            Residue index
        residue : mdtraj.Residue
            MDTraj residue object
        res_metadata : dict or None
            Existing residue metadata
            
        Returns
        -------
        dict
            Residue information dictionary
            
        Notes
        -----
        Uses existing metadata if available, otherwise creates default info.
        """
        if res_metadata and res_idx < len(res_metadata):
            return res_metadata[res_idx]
        else:
            return {
                "index": res_idx,
                "name": residue.name,
                "id": residue.resSeq,
                "full_name": f"{residue.name}{residue.resSeq}"
            }

    def _create_tick_labels(self, encoding: str, classes: list) -> Dict[str, list]:
        """
        Create tick labels for DSSP visualization.

        Parameters
        ----------
        encoding : str
            Encoding format ('onehot', 'integer', 'char')
        classes : list
            List of DSSP class labels

        Returns
        -------
        dict
            Dictionary with 'short' and 'long' label lists

        Notes
        -----
        For onehot encoding, creates binary labels.
        For integer/char encoding, creates full class labels with long descriptive names.
        """
        # Filter out 'NA' if present
        classes = [c for c in classes if c != 'NA']

        if encoding == 'onehot':
            # Binary labels (will be per-class specific in actual use)
            # This is generic, actual labels depend on specific dssp_class in feature_info
            return {
                'short': ['N', 'Y'],  # Generic placeholder
                'long': ['Not', 'Is']  # Generic placeholder
            }
        else:
            # Multi-class labels
            dssp_long_names = {
                'H': 'Alpha helix',
                'B': 'Isolated\nbeta-bridge',
                'E': 'Extended\nstrand',
                'G': '3/10 helix',
                'I': 'Pi helix',
                'T': 'Hydrogen\nbonded turn',
                'S': 'Bend',
                'C': 'Loop'
            }
            return {
                'short': classes,
                'long': [dssp_long_names.get(c, c) for c in classes]
            }

    def compute_dynamic_values(
        self,
        input_data: np.ndarray,
        metric: str = "transitions",
        threshold_min: float = None,
        threshold_max: float = None,
        feature_metadata: list = None,
        output_path: str = None,
        transition_threshold: float = 1,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> Dict[str, Any]:
        """
        Filter and select dynamic secondary structure features based on statistical criteria.

        Parameters
        ----------
        input_data : numpy.ndarray
            DSSP array (n_frames, n_residues) with encoded secondary structure classes
        metric : str, default='transitions'
            Metric to use for selection:
            
            - 'transitions': Number of secondary structure class transitions
            - 'transition_frequency': Frequency of transitions (same as transitions)
            - 'stability': Secondary structure stability (1 - transition_frequency)
            - 'class_frequencies': Frequency distribution of classes
        threshold_min : float, optional
            Minimum threshold for filtering (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold for filtering (metric_values <= threshold_max)
        feature_metadata : list, optional
            DSSP metadata corresponding to data columns
        output_path : str, optional
            Path for memory-mapped output
        transition_threshold : float, default=1
            Not used for DSSP (transitions are type changes, not threshold-based)
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
            'metric_used', 'n_dynamic', 'total_residues', 'threshold_min', 'threshold_max'

        Examples
        --------
        >>> # Select highly dynamic residues (>= 5 transitions)
        >>> result = calculator.compute_dynamic_values(dssp, metric='transitions', threshold_min=5)
        >>> dynamic_residues = result['dynamic_data']

        >>> # Select stable residues (transitions <= 2)
        >>> result = calculator.compute_dynamic_values(dssp, metric='transitions', threshold_max=2)

        >>> # Select residues with high structural stability
        >>> result = calculator.compute_dynamic_values(
        ...     dssp, metric='class_stability', threshold_min=0.8
        ... )
        """
        # Compute metric values using helper method
        metric_values = self._compute_metric_values(
            input_data,
            metric,
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
        dssp_data: np.ndarray,
        metric: str,
        window_size: int,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute metric values for DSSP based on specified metric type.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP array (n_frames, n_residues) with encoded secondary structure
        metric : str
            Metric type to compute
        window_size : int
            Window size for transitions metric
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime'
        lag_time : int, default=1
            Lag time for transitions metric
        metric : str
            Metric type to compute
        window_size : int
            Window size for transitions metric
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime'
        lag_time : int, default=1
            Lag time for transitions metric

        Returns
        -------
        numpy.ndarray
            Computed metric values per residue
        """
        # Handle complex metrics
        if metric == "transitions":
            if transition_mode == "lagtime":
                return self.analysis.compute_transitions_lagtime(dssp_data, lag_time=lag_time)
            else:  # window
                return self.analysis.compute_transitions_window(dssp_data, window_size=window_size)
        if metric == "transition_frequency":
            return self.analysis.compute_transition_frequency(dssp_data)
        if metric == "stability":
            return self.analysis.compute_stability(dssp_data)
        if metric == "class_frequencies":
            frequencies, _ = self.analysis.compute_class_frequencies(dssp_data, simplified=True)
            # Return dominant class frequency per residue (max frequency)
            return np.max(frequencies, axis=1)

        # Unknown metric
        supported_metrics = [
            "transitions", "transition_frequency", "stability", "class_frequencies"
        ]
        raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

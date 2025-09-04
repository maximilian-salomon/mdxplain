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
Coordinates feature type implementation for molecular dynamics analysis.

Coordinates feature type implementation for extracting XYZ positions from
molecular dynamics trajectories with flexible atom selection capabilities.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import mdtraj as md

from ..interfaces.feature_type_base import FeatureTypeBase
from .coordinates_calculator import CoordinatesCalculator
from .reduce_coordinates_metrics import ReduceCoordinatesMetrics


class Coordinates(FeatureTypeBase):
    """
    Coordinates feature type for extracting XYZ positions from trajectories.

    Extracts atomic coordinates from molecular dynamics trajectories using
    flexible atom selection strings. Supports various selection modes including
    all atoms, alpha carbons, backbone atoms, heavy atoms, or custom MDTraj
    selection strings.

    This is a base feature type with no dependencies that provides direct
    access to positional data for structural analysis.

    Examples:
    ---------
    >>> # Extract alpha carbon coordinates
    >>> coords = Coordinates(selection='ca')
    >>> pipeline.feature.add_feature(coords)

    >>> # Extract all heavy atoms
    >>> coords = Coordinates(selection='heavy')
    >>> pipeline.feature.add_feature(coords)

    >>> # Custom selection string
    >>> coords = Coordinates(selection='name CA and resid 1 to 100')
    >>> pipeline.feature.add_feature(coords)
    """

    ReduceMetrics = ReduceCoordinatesMetrics
    """Available reduce metrics for coordinates features."""

    def __init__(self, selection: str = 'ca') -> None:
        """
        Initialize coordinates feature type with atom selection.

        Parameters:
        -----------
        selection : str, default='ca'
            Atom selection string for coordinate extraction. Options:
            - 'all': All atoms in trajectory
            - 'ca': Alpha carbon atoms only  
            - 'backbone': Backbone atoms (N, CA, C, O)
            - 'heavy': All heavy (non-hydrogen) atoms
            - Custom MDTraj selection string (e.g., 'name CA and resid 1 to 100')

        Returns:
        --------
        None

        Examples:
        ---------
        >>> # Alpha carbons only (default)
        >>> coords = Coordinates()

        >>> # All atoms
        >>> coords = Coordinates(selection='all')

        >>> # Backbone atoms
        >>> coords = Coordinates(selection='backbone')

        >>> # Custom selection
        >>> coords = Coordinates(selection='resname ALA and name CA')
        """
        super().__init__()
        self.selection = selection

    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the coordinates calculator with specified configuration.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files when using memory mapping
        chunk_size : int, optional
            Number of frames to process per chunk for memory-efficient processing

        Returns:
        --------
        None

        Examples:
        ---------
        >>> # Basic initialization
        >>> coords.init_calculator()

        >>> # With memory mapping for large datasets
        >>> coords.init_calculator(use_memmap=True, cache_path='./cache/')

        >>> # With custom chunk size
        >>> coords.init_calculator(chunk_size=1000)
        """
        self.calculator = CoordinatesCalculator(
            use_memmap=use_memmap, 
            cache_path=cache_path, 
            chunk_size=chunk_size
        )

    def compute(self, input_data: md.Trajectory, feature_metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute coordinates for selected atoms from molecular dynamics trajectory.

        Parameters:
        -----------
        input_data : mdtraj.Trajectory
            MD trajectory to extract coordinates from
        feature_metadata : dict
            Residue metadata (passed through unchanged for coordinates)

        Returns:
        --------
        tuple[numpy.ndarray, dict]
            Tuple containing (coordinates_array, feature_metadata) where coordinates_array
            has shape (n_frames, n_selected_atoms * 3) with XYZ coordinates in Angstrom
            and feature_metadata contains atom selection information

        Examples:
        ---------
        >>> # Extract coordinates for selected atoms
        >>> coords = Coordinates(selection='ca')
        >>> coords.init_calculator()
        >>> data, metadata = coords.compute(trajectory, res_metadata)
        >>> print(f"Coordinate array shape: {data.shape}")
        >>> print(f"Number of selected atoms: {data.shape[1] // 3}")

        Raises:
        -------
        ValueError
            If calculator is not initialized or selection is invalid
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )
        
        return self.calculator.compute(
            input_data=input_data,
            selection=self.selection,
            res_metadata=feature_metadata
        )

    def get_dependencies(self) -> List[str]:
        """
        Get list of feature type dependencies for coordinates calculations.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[str]
            Empty list as coordinates are a base feature with no dependencies

        Examples:
        ---------
        >>> coords = Coordinates()
        >>> print(coords.get_dependencies())
        []
        """
        return []

    @classmethod
    def get_type_name(cls) -> str:  # noqa: vulture
        """
        Return unique string identifier for the coordinates feature type.

        Parameters:
        -----------
        None

        Returns:
        --------
        str
            String identifier 'coordinates' used as key in feature dictionaries

        Examples:
        ---------
        >>> print(Coordinates.get_type_name())
        'coordinates'
        """
        return "coordinates"

    def get_input(self):
        """
        Get the input feature type that coordinates depend on.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            None since coordinates are a base feature with no input dependencies

        Examples:
        ---------
        >>> coords = Coordinates()
        >>> print(coords.get_input())
        None
        """
        return None
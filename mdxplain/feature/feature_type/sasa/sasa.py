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
SASA feature type implementation for molecular dynamics analysis.

SASA feature type implementation for computing Solvent Accessible Surface Area
using the Shrake-Rupley algorithm with support for both residue and atom level.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import mdtraj as md

from ..interfaces.feature_type_base import FeatureTypeBase
from .sasa_calculator import SASACalculator


class SASA(FeatureTypeBase):
    """
    SASA feature type for computing Solvent Accessible Surface Area.

    Computes solvent accessible surface area using the Shrake-Rupley algorithm.
    Supports both residue-level and atom-level calculations with configurable
    probe radius. The algorithm determines the surface area accessible to a
    spherical probe of specified radius.

    This is a base feature type with no dependencies that provides insights
    into protein-solvent interactions and buried surface areas.

    Uses mdtraj for SASA calculation under the hood.

    Examples
    --------
    >>> # Residue-level SASA with default water probe
    >>> sasa = SASA(mode='residue')
    >>> pipeline.feature.add_feature(sasa)

    >>> # Atom-level SASA with custom probe radius
    >>> sasa = SASA(mode='atom', probe_radius=0.12)
    >>> pipeline.feature.add_feature(sasa)

    >>> # Large probe for detecting cavities
    >>> sasa = SASA(mode='residue', probe_radius=0.20)
    >>> pipeline.feature.add_feature(sasa)
    """

    """Available reduce metrics for SASA features."""

    def __init__(self, mode: str = 'residue', probe_radius: float = 0.14) -> None:
        """
        Initialize SASA feature type with calculation parameters.

        Parameters
        ----------
        mode : str, default='residue'
            Level of SASA calculation:
            - 'residue': SASA per residue (sum of constituent atoms)
            - 'atom': SASA per individual atom
        probe_radius : float, default=0.14
            Probe sphere radius in nanometers. Common values:
            - 0.14 nm: Water molecule (default, most common)
            - 0.12 nm: Small probe for tight cavities  
            - 0.20 nm: Large probe for detecting large cavities

        Returns
        -------
        None

        Examples
        --------
        >>> # Residue-level SASA (default)
        >>> sasa = SASA()

        >>> # Atom-level SASA
        >>> sasa = SASA(mode='atom')

        >>> # Custom probe radius for cavity detection
        >>> sasa = SASA(mode='residue', probe_radius=0.20)

        >>> # Small probe for detailed surface analysis
        >>> sasa = SASA(mode='residue', probe_radius=0.12)
        """
        super().__init__()
        
        if mode not in ['residue', 'atom']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'residue' or 'atom'.")
        
        if probe_radius <= 0:
            raise ValueError(f"Probe radius must be positive, got {probe_radius}")

        self.mode = mode
        self.probe_radius = probe_radius

    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the SASA calculator with specified configuration.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files when using memory mapping
        chunk_size : int, optional
            Number of frames to process per chunk for memory-efficient processing

        Returns
        -------
        None

        Examples
        --------
        >>> # Basic initialization
        >>> sasa.init_calculator()

        >>> # With memory mapping for large datasets
        >>> sasa.init_calculator(use_memmap=True, cache_path='./cache/')

        >>> # With custom chunk size
        >>> sasa.init_calculator(chunk_size=1000)
        """
        self.calculator = SASACalculator(
            use_memmap=use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size
        )

    def compute(self, input_data: md.Trajectory, feature_metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute SASA from molecular dynamics trajectory.

        Parameters
        ----------
        input_data : mdtraj.Trajectory
            MD trajectory to compute SASA from
        feature_metadata : dict
            Residue metadata (passed through for residue-level analysis)

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Tuple containing (sasa_array, feature_metadata) where sasa_array
            has shape (n_frames, n_residues) or (n_frames, n_atoms) depending
            on mode, with SASA values in nm²

        Examples
        --------
        >>> # Compute residue-level SASA
        >>> sasa = SASA(mode='residue')
        >>> sasa.init_calculator()
        >>> data, metadata = sasa.compute(trajectory, res_metadata)
        >>> print(f"SASA array shape: {data.shape}")
        >>> print(f"Mean SASA per residue: {data.mean(axis=0)}")

        >>> # Compute atom-level SASA with memory mapping
        >>> sasa = SASA(mode='atom')
        >>> sasa.init_calculator(use_memmap=True)
        >>> data, metadata = sasa.compute(trajectory, res_metadata)

        Raises
        ------
        ValueError
            If calculator is not initialized
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )
        
        return self.calculator.compute(
            input_data=input_data,
            mode=self.mode,
            probe_radius=self.probe_radius,
            res_metadata=feature_metadata
        )

    def get_dependencies(self) -> List[str]:
        """
        Get list of feature type dependencies for SASA calculations.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            Empty list as SASA is a base feature with no dependencies

        Examples
        --------
        >>> sasa = SASA()
        >>> print(sasa.get_dependencies())
        []
        """
        return []

    @classmethod
    def get_type_name(cls) -> str:  # noqa: vulture
        """
        Return unique string identifier for the SASA feature type.

        Parameters
        ----------
        None

        Returns
        -------
        str
            String identifier 'sasa' used as key in feature dictionaries

        Examples
        --------
        >>> print(SASA.get_type_name())
        'sasa'
        """
        return "sasa"

    def get_input(self):
        """
        Get the input feature type that SASA depends on.

        Parameters
        ----------
        None

        Returns
        -------
        None
            None since SASA is a base feature with no input dependencies

        Examples
        --------
        >>> sasa = SASA()
        >>> print(sasa.get_input())
        None
        """
        return None
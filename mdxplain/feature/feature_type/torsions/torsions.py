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
Torsions feature type implementation for molecular dynamics analysis.

Torsions feature type implementation for computing dihedral torsion angles
including backbone and side chain angles with flexible selection options.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import mdtraj as md

from ..interfaces.feature_type_base import FeatureTypeBase
from .torsions_calculator import TorsionsCalculator


class Torsions(FeatureTypeBase):
    """
    Torsions feature type for computing dihedral torsion angles.

    Computes dihedral torsion angles from molecular dynamics trajectories including
    backbone angles (phi, psi, omega) and side chain angles (chi1-4). Provides
    flexible selection of angle types with all angles returned in degrees.

    This is a base feature type with no dependencies that provides conformational
    information for protein flexibility and dynamics analysis.

    Uses mdtraj for torsion angle calculation under the hood.

    Examples
    --------
    >>> # All angles (default)
    >>> torsions = Torsions()
    >>> pipeline.feature.add_feature(torsions)

    >>> # Only backbone angles
    >>> torsions = Torsions(calculate_chi=False)
    >>> pipeline.feature.add_feature(torsions)

    >>> # Only phi and psi angles
    >>> torsions = Torsions(calculate_phi=True, calculate_psi=True, 
    ...                     calculate_omega=False, calculate_chi=False)
    >>> pipeline.feature.add_feature(torsions)

    >>> # Only side chain chi angles
    >>> torsions = Torsions(calculate_phi=False, calculate_psi=False, 
    ...                     calculate_omega=False, calculate_chi=True)
    >>> pipeline.feature.add_feature(torsions)
    """

    """Available reduce metrics for torsion features."""

    def __init__(self, calculate_phi: bool = True, calculate_psi: bool = True, 
                 calculate_omega: bool = True, calculate_chi: bool = True) -> None:
        """
        Initialize torsions feature type with angle selection parameters.

        Parameters
        ----------
        calculate_phi : bool, default=True
            Whether to compute phi backbone angles
        calculate_psi : bool, default=True
            Whether to compute psi backbone angles
        calculate_omega : bool, default=True
            Whether to compute omega backbone angles
        calculate_chi : bool, default=True
            Whether to compute side chain chi angles (chi1, chi2, chi3, chi4)

        Returns
        -------
        None

        Examples
        --------
        >>> # All angles (default)
        >>> torsions = Torsions()

        >>> # Only phi and psi angles
        >>> torsions = Torsions(calculate_phi=True, calculate_psi=True, 
        ...                     calculate_omega=False, calculate_chi=False)

        >>> # Only backbone angles
        >>> torsions = Torsions(calculate_chi=False)

        >>> # Only chi side chain angles
        >>> torsions = Torsions(calculate_phi=False, calculate_psi=False, 
        ...                     calculate_omega=False, calculate_chi=True)

        Notes
        -----
        All angles are computed and returned in degrees (-180 to +180).
        """
        super().__init__()
        
        self.calculate_phi = calculate_phi
        self.calculate_psi = calculate_psi
        self.calculate_omega = calculate_omega
        self.calculate_chi = calculate_chi

    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the torsions calculator with specified configuration.

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
        >>> torsions.init_calculator()

        >>> # With memory mapping for large datasets
        >>> torsions.init_calculator(use_memmap=True, cache_path='./cache/')

        >>> # With custom chunk size
        >>> torsions.init_calculator(chunk_size=1000)
        """
        self.calculator = TorsionsCalculator(
            use_memmap=use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size
        )

    def compute(self, input_data: md.Trajectory, feature_metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute torsion angles from molecular dynamics trajectory.

        Parameters
        ----------
        input_data : mdtraj.Trajectory
            MD trajectory to compute torsion angles from
        feature_metadata : dict
            Residue metadata (passed through for residue-level analysis)

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Tuple containing (torsions_array, feature_metadata) where torsions_array
            has shape (n_frames, n_residues * n_angle_types) with torsion angles
            in degrees (-180 to +180). Missing angles are filled with 0.0.

        Examples
        --------
        >>> # Compute only phi and psi backbone angles
        >>> torsions = Torsions(calculate_phi=True, calculate_psi=True, 
        ...                     calculate_omega=False, calculate_chi=False)
        >>> torsions.init_calculator()
        >>> data, metadata = torsions.compute(trajectory, res_metadata)
        >>> print(f"Torsions array shape: {data.shape}")  # (n_frames, n_residues * 2)

        >>> # Compute all angles with memory mapping
        >>> torsions = Torsions()
        >>> torsions.init_calculator(use_memmap=True)
        >>> data, metadata = torsions.compute(trajectory, res_metadata)

        Raises
        ------
        ValueError
            If calculator is not initialized
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        # Compute torsions
        torsion_data, metadata = self.calculator.compute(
            input_data=input_data,
            calculate_phi=self.calculate_phi,
            calculate_psi=self.calculate_psi,
            calculate_omega=self.calculate_omega,
            calculate_chi=self.calculate_chi,
            res_metadata=feature_metadata
        )

        # Add feature type information for representative frame finding
        metadata['feature_type_name'] = self.get_type_name()
        metadata['dependencies'] = self.get_dependencies()
        metadata['input_feature'] = self.get_input()
        metadata['is_periodic'] = True

        return torsion_data, metadata

    def get_dependencies(self) -> List[str]:
        """
        Get list of feature type dependencies for torsions calculations.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            Empty list as torsions are a base feature with no dependencies

        Examples
        --------
        >>> torsions = Torsions()
        >>> print(torsions.get_dependencies())
        []
        """
        return []

    @classmethod
    def get_type_name(cls) -> str:  # noqa: vulture
        """
        Return unique string identifier for the torsions feature type.

        Parameters
        ----------
        None

        Returns
        -------
        str
            String identifier 'torsions' used as key in feature dictionaries

        Examples
        --------
        >>> print(Torsions.get_type_name())
        'torsions'
        """
        return "torsions"

    def get_input(self):
        """
        Get the input feature type that torsions depend on.

        Parameters
        ----------
        None

        Returns
        -------
        None
            None since torsions are a base feature with no input dependencies

        Examples
        --------
        >>> torsions = Torsions()
        >>> print(torsions.get_input())
        None
        """
        return None

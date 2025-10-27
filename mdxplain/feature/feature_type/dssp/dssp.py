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
DSSP feature type implementation for molecular dynamics analysis.

DSSP feature type implementation for computing secondary structure assignments
using the DSSP algorithm with support for multiple encoding formats.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import mdtraj as md

from ..interfaces.feature_type_base import FeatureTypeBase
from .dssp_calculator import DSSPCalculator


class DSSP(FeatureTypeBase):
    """
    DSSP feature type for computing secondary structure assignments.

    Computes secondary structure using the DSSP (Dictionary of Secondary Structure
    in Proteins) algorithm. Supports both simplified classification (H/E/C for
    helix/sheet/coil) and full classification with all 8 DSSP classes.
    Multiple encoding formats are available for different analysis needs.

    This is a base feature type with no dependencies that provides structural
    classification information for protein analysis.

    Uses mdtraj for dssp calculations under the hood.

    Examples
    --------
    >>> # Simplified DSSP with one-hot encoding
    >>> dssp = DSSP(simplified=True, encoding='onehot')
    >>> pipeline.feature.add_feature(dssp)

    >>> # Full DSSP classification with integer encoding
    >>> dssp = DSSP(simplified=False, encoding='integer')
    >>> pipeline.feature.add_feature(dssp)

    >>> # Character encoding for visualization
    >>> dssp = DSSP(simplified=True, encoding='char')
    >>> pipeline.feature.add_feature(dssp)
    """

    """Available reduce metrics for DSSP features."""

    def __init__(self, simplified: bool = False, encoding: str = 'char') -> None:
        """
        Initialize DSSP feature type with classification and encoding parameters.

        Parameters
        ----------
        simplified : bool, default=False
            Secondary structure classification level:

            - True: Simplified 3-class (H=helix, E=sheet, C=coil/other)
            - False: Full 8-class DSSP (H, B, E, G, I, T, S, C)
        encoding : str, default='char'
            Output encoding format:
            
            - 'onehot': One-hot encoded binary vectors
            - 'integer': Integer class indices (0, 1, 2, ...)
            - 'char': Character codes ('H', 'E', 'C', etc.)

        Returns
        -------
        None

        Examples
        --------
        >>> # Default: Full classification with one-hot encoding
        >>> dssp = DSSP()

        >>> # Simplified for basic analysis
        >>> dssp = DSSP(simplified=True)

        >>> # Integer encoding for machine learning
        >>> dssp = DSSP(simplified=False, encoding='integer')

        >>> # Character codes for visualization
        >>> dssp = DSSP(simplified=True, encoding='char')

        Raises
        ------
        ValueError
            If encoding is not one of 'onehot', 'integer', or 'char'
        """
        super().__init__()
        
        if encoding not in ['onehot', 'integer', 'char']:
            raise ValueError(f"Invalid encoding '{encoding}'. Must be 'onehot', 'integer', or 'char'.")

        self.simplified = simplified
        self.encoding = encoding

    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the DSSP calculator with specified configuration.

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
        >>> dssp.init_calculator()

        >>> # With memory mapping for large datasets
        >>> dssp.init_calculator(use_memmap=True, cache_path='./cache/')

        >>> # With custom chunk size
        >>> dssp.init_calculator(chunk_size=1000)
        """
        self.calculator = DSSPCalculator(
            use_memmap=use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size
        )

    def compute(self, input_data: md.Trajectory, feature_metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute DSSP secondary structure assignments from trajectory.

        Parameters
        ----------
        input_data : mdtraj.Trajectory
            MD trajectory to compute DSSP from
        feature_metadata : dict
            Residue metadata (passed through for residue-level analysis)

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Tuple containing (dssp_array, feature_metadata) where dssp_array
            format depends on encoding:
            
            - 'onehot': (n_frames, n_residues * n_classes) 
            - 'integer': (n_frames, n_residues)
            - 'char': (n_frames, n_residues) with string dtype

        Examples
        --------
        >>> # Compute simplified DSSP with one-hot encoding
        >>> dssp = DSSP(simplified=True, encoding='onehot')
        >>> dssp.init_calculator()
        >>> data, metadata = dssp.compute(trajectory, res_metadata)
        >>> print(f"DSSP array shape: {data.shape}")  # (n_frames, n_residues * 3)

        >>> # Compute full DSSP with character encoding
        >>> dssp = DSSP(simplified=False, encoding='char')
        >>> dssp.init_calculator()
        >>> data, metadata = dssp.compute(trajectory, res_metadata)
        >>> print(f"Secondary structure codes: {data[0]}")  # ['H', 'H', 'E', 'C', ...]

        Raises
        ------
        ValueError
            If calculator is not initialized
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        # Compute DSSP
        dssp_data, metadata = self.calculator.compute(
            input_data=input_data,
            simplified=self.simplified,
            encoding=self.encoding,
            res_metadata=feature_metadata
        )

        # Add feature type information for representative frame finding
        metadata['feature_type_name'] = self.get_type_name()
        metadata['dependencies'] = self.get_dependencies()
        metadata['input_feature'] = self.get_input()
        metadata['is_periodic'] = False

        return dssp_data, metadata

    def get_dependencies(self) -> List[str]:
        """
        Get list of feature type dependencies for DSSP calculations.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            Empty list as DSSP is a base feature with no dependencies

        Examples
        --------
        >>> dssp = DSSP()
        >>> print(dssp.get_dependencies())
        []
        """
        return []

    @classmethod
    def get_type_name(cls) -> str:  # noqa: vulture
        """
        Return unique string identifier for the DSSP feature type.

        Parameters
        ----------
        None

        Returns
        -------
        str
            String identifier 'dssp' used as key in feature dictionaries

        Examples
        --------
        >>> print(DSSP.get_type_name())
        'dssp'
        """
        return "dssp"

    def get_input(self):
        """
        Get the input feature type that DSSP depends on.

        Parameters
        ----------
        None

        Returns
        -------
        None
            None since DSSP is a base feature with no input dependencies

        Examples
        --------
        >>> dssp = DSSP()
        >>> print(dssp.get_input())
        None
        """
        return None
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
Feature data container for trajectory-specific computed features.

Container for trajectory-specific feature data (distances, contacts) with 
associated calculator. Stores per-trajectory feature data enabling mixed 
systems with different proteins in a single pipeline.
"""

from typing import Dict, Optional, Any, List
import numpy as np
from ...utils.data_utils import DataUtils
from ..feature_type.interfaces.feature_type_base import FeatureTypeBase


class FeatureData:
    """
    Internal container for trajectory-specific computed feature data.

    Stores per-trajectory feature data enabling mixed systems with different
    proteins in a single pipeline. Each trajectory computes its own features
    which are combined at the selection level.
    
    Attributes:
    -----------
    data : np.ndarray
        Feature matrix for single trajectory
    feature_metadata : dict
        Feature metadata for single trajectory
    reduced_data : np.ndarray 
        Reduced feature matrix for single trajectory
    reduced_feature_metadata : dict
        Reduced metadata for single trajectory
    reduction_info : dict
        Reduction information for single trajectory
    """

    def __init__(
        self, feature_type: FeatureTypeBase, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 10000, trajectory_name: Optional[str] = None
    ) -> None:
        """
        Initialize feature data container.

        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object to compute data
        use_memmap : bool, default=False
            Whether to use memory mapping
        cache_path : str, optional
            Cache directory path for memmap
        chunk_size : int, optional
            Chunk size for processing large datasets. Smaller chunks use less
            memory but may be slower. If None, uses automatic chunking.
        trajectory_name : str, optional
            Trajectory name for unique cache filenames

        Returns:
        --------
        None
            Initializes feature data container
        """
        self.feature_type = feature_type
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        
        # Handle cache path with trajectory-specific naming
        if use_memmap:
            if trajectory_name:
                filename = f"{str(feature_type)}_{trajectory_name}.dat"
                reduced_filename = f"{str(feature_type)}_reduced_{trajectory_name}.dat"
            else:
                filename = f"{str(feature_type)}.dat"
                reduced_filename = f"{str(feature_type)}_reduced.dat"
                
            self.cache_path = DataUtils.get_cache_file_path(filename, cache_path)
            self.reduced_cache_path = DataUtils.get_cache_file_path(reduced_filename, cache_path)
        else:
            self.cache_path = None
            self.reduced_cache_path = None

        # Initialize single-trajectory data structures
        self.data = None
        self.feature_metadata = None
        self.reduced_data = None
        self.reduced_feature_metadata = None
        self.reduction_info = None

        self.feature_type.init_calculator(
            use_memmap=self.use_memmap,
            cache_path=self.cache_path,
            chunk_size=self.chunk_size,
        )

        self.analsis = None

    def get_data(self, force_original: bool = False) -> np.ndarray:
        """
        Get current dataset (reduced if available, else original).

        Parameters:
        -----------
        force_original : bool, default=False
            Whether to force using the original data instead of the reduced data

        Returns:
        --------
        numpy.ndarray
            Feature data array for this trajectory

        Examples:
        ---------
        >>> # Get current data
        >>> data = feature_data.get_data()
        >>> print(f"Data shape: {data.shape}")
        """
        if self.reduced_data is not None and not force_original:
            return self.reduced_data
        return self.data

    def get_feature_metadata(self, force_original: bool = False) -> Dict[str, Any]:
        """
        Get current feature metadata (reduced if available, else original).

        This method returns the metadata corresponding to the current
        active dataset. If reduced data is available, it returns the
        reduced metadata. Otherwise, it returns the original metadata.

        Parameters:
        -----------
        force_original : bool, default=False
            Whether to force using the original data instead of the reduced data

        Returns:
        --------
        dict or None
            Feature metadata dict for this trajectory
            If traj_idx is None: Dict mapping trajectory indices to metadata dicts
            Returns None if no metadata available

        Examples:
        ---------
        >>> # Get current metadata
        >>> metadata = feature_data.get_feature_metadata()
        >>> print(f"Features: {len(metadata['features'])}")
        """
        if self.reduced_feature_metadata is not None and not force_original:
            return self.reduced_feature_metadata
        return self.feature_metadata

    def get_feature_names(self, force_original: bool = False) -> List[str]:
        """
        Extract feature names from metadata.

        This method generates human-readable feature names from the
        structured metadata. For pair-based features, it creates names
        by joining the full names of the involved partners.

        Parameters:
        -----------
        force_original : bool, default=False
            Whether to return the feature names for the original data

        Returns:
        --------
        list or None
            List of feature names extracted from metadata, or None if not available

        Examples:
        ---------
        >>> feature_data = traj.get_feature("distances")
        >>> names = feature_data.get_feature_names()
        >>> print(f"First few feature names: {names[:3]}")
        >>> # Output: ['ALA1-VAL2', 'ALA1-GLY3', 'VAL2-GLY3']
        """
        metadata = (
            self.feature_metadata if force_original else self.get_feature_metadata()
        )

        if metadata is None:
            return None

        return [
            "-".join(partner["full_name"] for partner in feature)
            for feature in metadata.get("features", [])
        ]

    def save(self, save_path: str) -> None:
        """
        Save FeatureData object to disk.

        Parameters:
        -----------
        save_path : str
            Path where to save the FeatureData object

        Returns:
        --------
        None
            Saves the FeatureData object to the specified path

        Examples:
        ---------
        >>> feature_data.save('analysis_results/distances.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load FeatureData object from disk.

        Parameters:
        -----------
        load_path : str
            Path to the saved FeatureData file

        Returns:
        --------
        None
            Loads the FeatureData object from the specified path

        Examples:
        ---------
        >>> feature_data.load('analysis_results/distances.pkl')
        """
        DataUtils.load_object(self, load_path)

    def print_info(self) -> None:
        """
        Print comprehensive feature information.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Prints feature information to console

        Examples:
        ---------
        >>> feature_data.print_info()
        === FeatureData ===
        Feature Type: Distances
        Original Data: 1000 frames x 250 features
        Reduced Data: 1000 frames x 100 features (PCA)
        """
        if self._has_no_data():
            print("No feature data available.")
            return

        self._print_feature_header()
        self._print_feature_details()
        if self.reduced_data is not None:
            self._print_reduction_info()

    def _has_no_data(self) -> bool:
        """
        Check if no feature data is available.

        Returns:
        --------
        bool
            True if no data is available, False otherwise
        """
        return self.data is None and self.reduced_data is None

    def _print_feature_header(self) -> None:
        """
        Print header with feature type information.

        Returns:
        --------
        None
        """
        print("=== FeatureData ===")
        feature_type_name = getattr(self.feature_type, '__class__', str(self.feature_type)).__name__
        print(f"Feature Type: {feature_type_name}")

    def _print_feature_details(self) -> None:
        """
        Print detailed feature information.

        Returns:
        --------
        None
        """
        if self.data is not None:
            print(f"Original Data: {self.data.shape[0]} frames x {self.data.shape[1]} features")
            
        if self.feature_metadata is not None:
            n_features = len(self.feature_metadata.get("features", []))
            if n_features > 0:
                print(f"Feature Metadata: {n_features} feature definitions")

    def _print_reduction_info(self) -> None:
        """
        Print information about data reduction.

        Returns:
        --------
        None
        """
        if self.reduced_data is not None:
            reduction_method = "Unknown"
            if self.reduction_info and "reduction_method" in self.reduction_info:
                reduction_method = self.reduction_info["reduction_method"]
            print(f"Reduced Data: {self.reduced_data.shape[0]} frames x {self.reduced_data.shape[1]} features ({reduction_method})")
            
        if self.reduced_feature_metadata is not None:
            n_reduced_features = len(self.reduced_feature_metadata.get("features", []))
            if n_reduced_features > 0:
                print(f"Reduced Metadata: {n_reduced_features} feature definitions")

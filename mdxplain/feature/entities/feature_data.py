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
Feature data container for computed features with analysis methods.

Container for feature data (distances, contacts) with associated calculator.
Stores feature data and provides bound analysis methods from calculators.
"""

from ...utils.data_utils import DataUtils


class FeatureData:
    """
    Internal container for computed feature data with analysis methods.

    Stores feature data and provides bound analysis methods from calculators.
    """

    def __init__(
        self, feature_type, use_memmap=False, cache_path="./cache", chunk_size=10000
    ):
        """
        Initialize feature data container.

        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object to compute data
        use_memmap : bool, default=False
            Whether to use memory mapping
        cache_path : str, optional
            Cache file path for memmap
        chunk_size : int, optional
            Chunk size for processing large datasets. Smaller chunks use less
            memory but may be slower. If None, uses automatic chunking.

        Returns:
        --------
        None
            Initializes feature data container
        """
        self.feature_type = feature_type
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

        # Handle cache path
        if use_memmap:
            self.cache_path = DataUtils.get_cache_file_path(
                f"{str(feature_type)}.dat", cache_path
            )
            self.reduced_cache_path = DataUtils.get_cache_file_path(
                f"{str(feature_type)}_reduced.dat", cache_path
            )
        else:
            self.cache_path = None
            self.reduced_cache_path = None

        # Initialize data as None
        self.data = None
        self.feature_metadata = None  # Structured feature metadata
        self.reduced_data = None
        self.reduced_feature_metadata = None  # Reduced structured metadata
        self.reduction_info = None

        self.feature_type.init_calculator(
            use_memmap=self.use_memmap,
            cache_path=self.cache_path,
            chunk_size=self.chunk_size,
        )

        self.analsis = None

    def get_data(self, force_original=False):
        """
        Get current dataset (reduced if available, else original).

        Parameters:
        -----------
        force_original : bool, default=False
            Whether to force using the original data instead of the reduced data

        Returns:
        --------
        numpy.ndarray
            Feature data array, shape (n_frames, n_features)

        Examples:
        ---------
        >>> feature_data = traj.get_feature("distances")
        >>> data = feature_data.get_data()
        >>> print(f"Data shape: {data.shape}")
        """
        if self.reduced_data is not None and not force_original:
            return self.reduced_data
        return self.data

    def get_feature_metadata(self, force_original=False):
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
            Feature metadata dictionary with 'is_pair' and 'features' keys,
            or None if not available

        Examples:
        ---------
        >>> feature_data = traj.get_feature("distances")
        >>> metadata = feature_data.get_feature_metadata()
        >>> print(f"Number of features: {len(metadata['features'])}")
        """
        if self.reduced_feature_metadata is not None and not force_original:
            return self.reduced_feature_metadata
        return self.feature_metadata

    def get_feature_names(self, force_original=False):
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

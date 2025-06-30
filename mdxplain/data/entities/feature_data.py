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

import os


class FeatureData:
    """
    Internal container for computed feature data with analysis methods.

    Stores feature data and provides bound analysis methods from calculators.
    """

    def __init__(
        self, feature_type, use_memmap=False, cache_path=None, chunk_size=None
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
        """
        self.feature_type = feature_type
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

        # Handle cache path
        if use_memmap:
            if cache_path is None:
                os.makedirs("./cache", exist_ok=True)
                cache_path = f"./cache/{feature_type}.dat"
            self.cache_path = cache_path
            self.reduced_cache_path = (
                f"{os.path.splitext(self.cache_path)[0]}_reduced.dat"
            )
        else:
            self.cache_path = None
            self.reduced_cache_path = None

        # Initialize data as None
        self.data = None
        self.feature_names = None
        self.reduced_data = None
        self.reduced_feature_names = None
        self.reduction_info = None

        self.feature_type.init_calculator(
            use_memmap=self.use_memmap,
            cache_path=self.cache_path,
            chunk_size=self.chunk_size,
        )

        self.analsis = None

    def get_data(self):
        """
        Get current dataset (reduced if available, else original).

        Returns:
        --------
        numpy.ndarray
            Feature data array, shape (n_frames, n_features)
        """
        if self.reduced_data is not None:
            return self.reduced_data
        return self.data

    def get_feature_names(self):
        """
        Get current feature names (reduced if available, else original).

        Returns:
        --------
        numpy.ndarray
            Feature names array, shape (n_features, 2) with residue pair indices
        """
        if self.reduced_feature_names is not None:
            return self.reduced_feature_names
        return self.feature_names

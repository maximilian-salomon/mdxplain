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
Decomposition data container for computed dimensionality reduction results.

Container for decomposition results (PCA, KernelPCA) with associated metadata
and hyperparameters. Stores decomposed data with transformation information.
"""

from ...utils.data_utils import DataUtils


class DecompositionData:
    """
    Container for decomposition results with metadata and hyperparameters.

    Stores results from dimensionality reduction methods (PCA, KernelPCA) along
    with transformation metadata and hyperparameters used for computation.
    """

    def __init__(self, decomposition_type, use_memmap=False, cache_path="./cache"):
        """
        Initialize decomposition data container.

        Parameters:
        -----------
        decomposition_type : str
            Type of decomposition used (e.g., "pca", "kernel_pca")
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Path for memory-mapped cache files

        Returns:
        --------
        None
            Initializes decomposition data container

        Examples:
        ---------
        >>> # Basic initialization
        >>> decomp_data = DecompositionData("pca")

        >>> # With memory mapping
        >>> decomp_data = DecompositionData(
        ...     "kernel_pca",
        ...     use_memmap=True,
        ...     cache_path="./cache/kernel_pca.dat"
        ... )
        """
        self.decomposition_type = decomposition_type
        self.use_memmap = use_memmap

        self.cache_path = DataUtils.get_cache_file_path(
            f"{self.decomposition_type}.dat", cache_path
        )

        self.data = None
        self.metadata = None

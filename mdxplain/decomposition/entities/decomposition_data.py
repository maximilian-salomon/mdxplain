# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Kiro AI (Claude Sonnet 4.0).
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
        self.cache_path = cache_path

        self.data = None
        self.metadata = None
        self.frame_mapping = None

    def get_frame_mapping(self):
        """
        Get frame mapping from global frame indices to trajectory origins.

        Returns:
        --------
        dict or None
            Mapping from global_frame_index to (trajectory_index, local_frame_index),
            or None if decomposition has not been computed or mapping is not available

        Examples:
        ---------
        >>> decomp_data = DecompositionData("pca")
        >>> frame_mapping = decomp_data.get_frame_mapping()
        >>> if frame_mapping is not None:
        ...     print(f"Frame 0 comes from: {frame_mapping[0]}")  # (traj_idx, local_frame_idx)
        """
        return self.frame_mapping
        
    def set_frame_mapping(self, frame_mapping):
        """
        Set frame mapping from global frame indices to trajectory origins.

        Parameters:
        -----------
        frame_mapping : dict
            Mapping from global_frame_index to (trajectory_index, local_frame_index)

        Returns:
        --------
        None
            Sets the frame mapping for trajectory tracking

        Examples:
        ---------
        >>> decomp_data = DecompositionData("pca")
        >>> mapping = {0: (0, 10), 1: (0, 11), 2: (1, 5)}  # global -> (traj, local)
        >>> decomp_data.set_frame_mapping(mapping)
        """
        self.frame_mapping = frame_mapping

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

from typing import Dict, Any
from ...utils.data_utils import DataUtils


class DecompositionData:
    """
    Container for decomposition results with metadata and hyperparameters.

    Stores results from dimensionality reduction methods (PCA, KernelPCA) along
    with transformation metadata and hyperparameters used for computation.
    """

    def __init__(self, decomposition_type: str, use_memmap: bool = False, cache_path: str = "./cache") -> None:
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
        
    def set_frame_mapping(self, frame_mapping: Dict[int, int]) -> None:
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

    def save(self, save_path: str) -> None:
        """
        Save DecompositionData object to disk.

        Parameters:
        -----------
        save_path : str
            Path where to save the DecompositionData object

        Returns:
        --------
        None
            Saves the DecompositionData object to the specified path

        Examples:
        ---------
        >>> decomposition_data.save('analysis_results/pca_decomposition.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load DecompositionData object from disk.

        Parameters:
        -----------
        load_path : str
            Path to the saved DecompositionData file

        Returns:
        --------
        None
            Loads the DecompositionData object from the specified path

        Examples:
        ---------
        >>> decomposition_data.load('analysis_results/pca_decomposition.pkl')
        """
        DataUtils.load_object(self, load_path)

    def print_info(self) -> None:
        """
        Print comprehensive decomposition information.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Prints decomposition information to console

        Examples:
        ---------
        >>> decomposition_data.print_info()
        === DecompositionData ===
        Decomposition Type: PCA
        Transformed Data: 1000 frames x 10 components
        Explained Variance Ratio: [0.45, 0.23, 0.12, 0.08, ...]
        """
        if self._has_no_data():
            print("No decomposition data available.")
            return

        self._print_decomposition_header()
        self._print_decomposition_details()
        if self.frame_mapping is not None:
            self._print_frame_mapping_info()

    def _has_no_data(self) -> bool:
        """
        Check if no decomposition data is available.

        Returns:
        --------
        bool
            True if no data is available, False otherwise
        """
        return self.data is None

    def _print_decomposition_header(self) -> None:
        """
        Print header with decomposition type.

        Returns:
        --------
        None
        """
        print("=== DecompositionData ===")
        print(f"Decomposition Type: {self.decomposition_type.upper()}")

    def _print_decomposition_details(self) -> None:
        """
        Print detailed decomposition information.

        Returns:
        --------
        None
        """
        if self.data is not None:
            print(f"Transformed Data: {self.data.shape[0]} frames x {self.data.shape[1]} components")

        if self.metadata is not None:
            # Show explained variance for PCA
            if "explained_variance_ratio" in self.metadata:
                variance_ratios = self.metadata["explained_variance_ratio"]
                if len(variance_ratios) > 5:
                    shown_ratios = variance_ratios[:5]
                    print(f"Explained Variance Ratio: {[f'{r:.3f}' for r in shown_ratios]}... (showing first 5)")
                else:
                    print(f"Explained Variance Ratio: {[f'{r:.3f}' for r in variance_ratios]}")
                
                total_variance = sum(variance_ratios)
                print(f"Total Explained Variance: {total_variance:.3f} ({total_variance*100:.1f}%)")

            # Show hyperparameters if available
            hyperparams = self.metadata.get("hyperparameters", {})
            if hyperparams:
                param_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
                print(f"Hyperparameters: {param_str}")

            # Show selection information if available
            if "selection_name" in self.metadata:
                print(f"Based on Selection: {self.metadata['selection_name']}")

    def _print_frame_mapping_info(self) -> None:
        """
        Print information about frame mapping.

        Returns:
        --------
        None
        """
        if self.frame_mapping:
            n_mapped_frames = len(self.frame_mapping)
            # Count unique trajectories
            trajectory_indices = set()
            for traj_idx, _ in self.frame_mapping.values():
                trajectory_indices.add(traj_idx)
            n_trajectories = len(trajectory_indices)
            print(f"Frame Mapping: {n_mapped_frames} frames from {n_trajectories} trajectories")

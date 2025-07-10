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
DecompositionManager for managing decomposition data objects.

Manager for creating and managing decomposition results from feature matrices.
Used to add, reset, and manage decomposition data in trajectory data objects.
"""

from ..entities.decomposition_data import DecompositionData
from ...utils.data_utils import DataUtils


class DecompositionManager:
    """
    Manager for decomposition data objects.

    Manages the creation and storage of decomposition results from feature
    matrices. Works with TrajectoryData objects to perform dimensionality
    reduction using various decomposition methods (PCA, KernelPCA, etc.).

    Examples:
    ---------
    >>> # Create manager and add PCA decomposition
    >>> from mdxplain.decomposition import decomposition_type
    >>> manager = DecompositionManager()
    >>> manager.add_decomposition(
    ...     traj_data, "feature_selection", decomposition_type.PCA,
    ...     n_components=10
    ... )

    >>> # Manager with memory mapping for large datasets
    >>> manager = DecompositionManager(use_memmap=True, chunk_size=1000)
    >>> manager.add_decomposition(
    ...     traj_data, "contact_selection", decomposition_type.KernelPCA,
    ...     n_components=20, kernel='rbf'
    ... )
    """

    def __init__(self, use_memmap=False, chunk_size=10000, cache_dir="./cache"):
        """
        Initialize decomposition manager.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for decomposition data
        chunk_size : int, optional
            Processing chunk size for incremental computation
        cache_dir : str, optional
            Cache directory path for decomposition data

        Returns:
        --------
        None
            Initializes DecompositionManager instance with specified configuration

        Examples:
        ---------
        >>> # Basic manager
        >>> manager = DecompositionManager()

        >>> # Manager with memory mapping
        >>> manager = DecompositionManager(
        ...     use_memmap=True,
        ...     chunk_size=1000,
        ...     cache_dir="./cache/decomposition"
        ... )
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir

    def add(self, traj_data, selection_name, decomposition_type, force=False):
        """
        Add and compute a decomposition for selected feature data.

        This method creates a DecompositionData instance for the specified
        decomposition type, retrieves the selected feature matrix, performs
        the decomposition computation, and stores the result in the
        TrajectoryData object.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object containing feature selections
        selection_name : str
            Name of the feature selection to decompose
        decomposition_type : DecompositionTypeBase instance
            Decomposition type instance with parameters (e.g., PCA(n_components=10))
        force : bool, default=False
            Whether to force recomputation if decomposition already exists

        Returns:
        --------
        None
            Adds computed decomposition to trajectory data

        Raises:
        -------
        ValueError
            If the decomposition already exists, if required selection is missing,
            or if the decomposition computation fails

        Examples:
        ---------
        >>> # Add PCA decomposition
        >>> from mdxplain.decomposition import decomposition_type
        >>> manager = DecompositionManager()
        >>> manager.add(
        ...     traj_data, "feature_selection", decomposition_type.PCA(n_components=10)
        ... )

        >>> # Add KernelPCA with custom parameters
        >>> manager.add(
        ...     traj_data, "any_selection", decomposition_type.KernelPCA(n_components=15, gamma=0.1)
        ... )

        >>> # Add ContactKernelPCA for contact features
        >>> manager.add(
        ...     traj_data, "contact_selection", decomposition_type.ContactKernelPCA(n_components=20)
        ... )

        >>> # Force recomputation of existing decomposition
        >>> manager.add(
        ...     traj_data, "feature_selection", decomposition_type.PCA(n_components=20), force=True
        ... )
        """
        decomposition_key = DataUtils.get_type_key(decomposition_type)
        full_key = f"{selection_name}_{decomposition_key}"

        self._check_decomposition_existence(traj_data, full_key, force)

        data_matrix = traj_data.get_selected_matrix(selection_name)

        decomposition_data = DecompositionData(
            decomposition_type=decomposition_key,
            use_memmap=self.use_memmap,
            cache_path=self._get_selection_cache_path(selection_name),
        )

        self._compute_decomposition(
            decomposition_data, decomposition_type, data_matrix, selection_name
        )

        self._store_decomposition_results(
            traj_data, full_key, decomposition_data, data_matrix.shape
        )

    def _check_decomposition_existence(self, traj_data, full_key, force):
        """
        Check if decomposition already exists and handle accordingly.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        full_key : str
            Full decomposition key
        force : bool
            Whether to force recomputation

        Returns:
        --------
        None
            Validates decomposition status

        Raises:
        -------
        ValueError
            If decomposition exists and force is False
        """
        if not hasattr(traj_data, "decomposition_data"):
            traj_data.decomposition_data = {}

        if full_key in traj_data.decomposition_data:
            if force:
                print(
                    f"WARNING: Decomposition '{full_key}' already exists. Forcing recomputation."
                )
                del traj_data.decomposition_data[full_key]
            else:
                raise ValueError(f"Decomposition '{full_key}' already exists.")

    def _get_selection_cache_path(self, selection_name):
        """
        Get selection-specific cache path for decomposition data.

        Creates cache path structure: base_cache_dir/selection_name/
        This allows multiple decomposition types for the same selection.

        Parameters:
        -----------
        selection_name : str
            Name of the feature selection

        Returns:
        --------
        str or None
            Cache directory path for the selection
        """
        if self.use_memmap and self.cache_dir:
            return f"{self.cache_dir}/{selection_name}"
        return None

    def _compute_decomposition(
        self, decomposition_data, decomposition_type, data_matrix, selection_name
    ):
        """
        Compute the decomposition using the specified type and parameters.

        Parameters:
        -----------
        decomposition_data : DecompositionData
            Decomposition data container
        decomposition_type : DecompositionTypeBase instance
            Decomposition type instance with parameters
        data_matrix : numpy.ndarray
            Data matrix to decompose
        selection_name : str
            Name of the feature selection used

        Returns:
        --------
        None
            Performs decomposition computation
        """
        if not hasattr(decomposition_type, "init_calculator"):
            raise ValueError(
                f"Invalid decomposition type '{decomposition_type}'. "
                "Please provide a decomposition type instance."
            )

        decomposition_type.init_calculator(
            use_memmap=self.use_memmap,
            cache_path=decomposition_data.cache_path,
            chunk_size=self.chunk_size,
        )

        transformed_data, metadata = decomposition_type.compute(data_matrix)
        metadata["selection_name"] = selection_name

        decomposition_data.data = transformed_data
        decomposition_data.metadata = metadata

    def _store_decomposition_results(
        self, traj_data, full_key, decomposition_data, original_shape
    ):
        """
        Store decomposition results in trajectory data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        full_key : str
            Full decomposition key
        decomposition_data : DecompositionData
            Computed decomposition data
        original_shape : tuple
            Shape of original data matrix

        Returns:
        --------
        None
            Stores decomposition results
        """
        traj_data.decomposition_data[full_key] = decomposition_data

        print(
            f"Decomposition '{full_key}' computed successfully. "
            f"Data reduced from {original_shape} to {decomposition_data.data.shape}."
        )

    def reset_decompositions(self, traj_data):
        """
        Reset all computed decompositions and clear decomposition data.

        This method removes all computed decompositions and their associated data,
        requiring decompositions to be recalculated from scratch.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        None
            Clears all decomposition data from traj_data.decomposition_data

        Examples:
        ---------
        >>> manager = DecompositionManager()
        >>> manager.reset_decompositions(traj_data)
        """
        if (
            not hasattr(traj_data, "decomposition_data")
            or not traj_data.decomposition_data
        ):
            print("No decompositions to reset.")
            return

        decomposition_list = list(traj_data.decomposition_data.keys())
        traj_data.decomposition_data.clear()

        print(
            f"Reset {len(decomposition_list)} decomposition(s): {', '.join(decomposition_list)}"
        )
        print(
            "All decomposition data has been cleared. Decompositions must be recalculated."
        )

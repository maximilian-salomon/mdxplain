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
DecompositionManager for managing decomposition data objects.

Manager for creating and managing decomposition results from feature matrices.
Used to add, reset, and manage decomposition data in trajectory data objects.
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, TYPE_CHECKING
import os
import numpy as np

from ..entities.decomposition_data import DecompositionData
from ..decomposition_type.interfaces.decomposition_type_base import DecompositionTypeBase
from ...utils.data_utils import DataUtils

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


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
    ...     pipeline_data, "feature_selection", decomposition_type.PCA,
    ...     n_components=10
    ... )

    >>> # Manager with memory mapping for large datasets
    >>> manager = DecompositionManager(use_memmap=True, chunk_size=1000)
    >>> manager.add_decomposition(
    ...     pipeline_data, "contact_selection", decomposition_type.KernelPCA,
    ...     n_components=20, kernel='rbf'
    ... )
    """

    def __init__(self, use_memmap: bool = False, chunk_size: int = 10000, cache_dir: str = "./cache") -> None:
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
        os.makedirs(self.cache_dir, exist_ok=True)

        if chunk_size <= 0 and not isinstance(chunk_size, int):
            raise ValueError("Chunk size must be a positive integer.")

    def add(
        self,
        pipeline_data: PipelineData,
        selection_name: str,
        decomposition_type: DecompositionTypeBase,
        decomposition_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Add and compute a decomposition for selected feature data.

        This method creates a DecompositionData instance for the specified
        decomposition type, retrieves the selected feature matrix, performs
        the decomposition computation, and stores the result in the
        TrajectoryData object.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.add("selection", decomposition_type.PCA())  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.add(pipeline_data, "selection", decomposition_type.PCA())  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object containing feature selections
        selection_name : str
            Name of the feature selection to decompose
        decomposition_type : DecompositionTypeBase instance
            Decomposition type instance with parameters (e.g., PCA(n_components=10))
        decomposition_name : str
            Name to save the decomposition. If None (default),
            it is "selection_name_{str(decomposition_type)}"
        data_selector_name : str, optional
            Name of DataSelector to apply frame filtering before decomposition.
            If None, uses all frames from the selection.
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
        ...     pipeline_data, "feature_selection", decomposition_type.PCA(n_components=10)
        ... )

        >>> # Add KernelPCA with custom parameters
        >>> manager.add(
        ...     pipeline_data, "any_selection", decomposition_type.KernelPCA(n_components=15, gamma=0.1)
        ... )

        >>> # Add ContactKernelPCA for contact features
        >>> manager.add(
        ...     pipeline_data, "contact_selection", decomposition_type.ContactKernelPCA(n_components=20)
        ... )

        >>> # Force recomputation of existing decomposition
        >>> manager.add(
        ...     pipeline_data, "feature_selection", decomposition_type.PCA(n_components=20), force=True
        ... )
        """
        decomposition_key = DataUtils.get_type_key(decomposition_type)
        if decomposition_name is None:
            decomposition_name = f"{selection_name}_{decomposition_key}"

        self._check_decomposition_existence(pipeline_data, decomposition_name, force)

        # Get data with frame mapping
        data_matrix, frame_mapping = pipeline_data.get_selected_data(
            selection_name, data_selector_name, return_frame_mapping=True
        )

        decomposition_data = DecompositionData(
            decomposition_type=decomposition_key,
            use_memmap=self.use_memmap,
            cache_path=self._get_selection_cache_path(decomposition_name),
        )

        self._compute_decomposition(
            decomposition_data, decomposition_type, data_matrix, decomposition_name
        )

        # Store frame mapping in decomposition data
        decomposition_data.set_frame_mapping(frame_mapping)

        self._store_decomposition_results(
            pipeline_data,
            selection_name,
            decomposition_name,
            decomposition_data,
            data_matrix.shape,
            decomposition_key,
        )

    def _check_decomposition_existence(self, pipeline_data: PipelineData, selection_name: str, force: bool) -> None:
        """
        Check if decomposition already exists and handle accordingly.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object
        selection_name : str
            Selection name used as decomposition key
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
        if selection_name in pipeline_data.decomposition_data:
            if force:
                print(
                    f"WARNING: Decomposition for selection '{selection_name}' already exists. Forcing recomputation."
                )
                del pipeline_data.decomposition_data[selection_name]
            else:
                raise ValueError(
                    f"Decomposition for selection '{selection_name}' already exists."
                )

    def _get_selection_cache_path(self, selection_name: str) -> str:
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
        self, decomposition_data: DecompositionData, decomposition_type: DecompositionTypeBase, data_matrix: np.ndarray, decomposition_name: str
    ) -> Any:
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
        decomposition_name : str
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
        metadata["decomposition_name"] = decomposition_name

        decomposition_data.data = transformed_data
        decomposition_data.metadata = metadata

    def _store_decomposition_results(
        self,
        pipeline_data: PipelineData,
        selection_name: str,
        decomposition_name: str,
        decomposition_data: DecompositionData,
        original_shape: Tuple[int, ...],
        decomposition_key: str,
    ) -> None:
        """
        Store decomposition results in trajectory data.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object
        selection_name : str
            Name of the used selection
        decomposition_name : str
            Name of the decomposition
        decomposition_data : DecompositionData
            Computed decomposition data
        original_shape : tuple
            Shape of original data matrix
        decomposition_key : str
            Type of decomposition for logging

        Returns:
        --------
        None
            Stores decomposition results
        """
        pipeline_data.decomposition_data[decomposition_name] = decomposition_data

        print(
            f"Decomposition '{decomposition_key}' with name '{decomposition_name}' for selection '{selection_name}' computed successfully. "
            f"Data reduced from {original_shape} to {decomposition_data.data.shape}."
        )

    def reset_decompositions(self, pipeline_data: PipelineData) -> None:
        """
        Reset all computed decompositions and clear decomposition data.

        This method removes all computed decompositions and their associated data,
        requiring decompositions to be recalculated from scratch.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.reset_decompositions()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.reset_decompositions(pipeline_data)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object

        Returns:
        --------
        None
            Clears all decomposition data from pipeline_data.decomposition_data

        Examples:
        ---------
        >>> manager = DecompositionManager()
        >>> manager.reset_decompositions(pipeline_data)
        """
        if not pipeline_data.decomposition_data:
            print("No decompositions to reset.")
            return

        decomposition_list = list(pipeline_data.decomposition_data.keys())
        pipeline_data.decomposition_data.clear()

        print(
            f"Reset {len(decomposition_list)} decomposition(s): {', '.join(decomposition_list)}"
        )
        print(
            "All decomposition data has been cleared. Decompositions must be recalculated."
        )

    def save(self, pipeline_data: PipelineData, save_path: str) -> None:
        """
        Save all decomposition data to single file.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.save('decomposition.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.save(pipeline_data, 'decomposition.npy')  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container with decomposition data
        save_path : str
            Path where to save all decomposition data in one file

        Returns:
        --------
        None
            Saves all decomposition data to the specified file
            
        Examples:
        ---------
        >>> manager.save(pipeline_data, 'decomposition.npy')
        """
        DataUtils.save_object(pipeline_data.decomposition_data, save_path)

    def load(self, pipeline_data: PipelineData, load_path: str) -> None:
        """
        Load all decomposition data from single file.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.load('decomposition.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.load(pipeline_data, 'decomposition.npy')  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container to load decomposition data into
        load_path : str
            Path to saved decomposition data file

        Returns:
        --------
        None
            Loads all decomposition data from the specified file
            
        Examples:
        ---------
        >>> manager.load(pipeline_data, 'decomposition.npy')
        """
        temp_dict = {}
        DataUtils.load_object(temp_dict, load_path)
        pipeline_data.decomposition_data = temp_dict

    def print_info(self, pipeline_data: PipelineData) -> None:
        """
        Print decomposition data information.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.print_info()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.print_info(pipeline_data)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container with decomposition data

        Returns:
        --------
        None
            Prints decomposition data information to console

        Examples:
        ---------
        >>> manager.print_info(pipeline_data)
        """
        if len(pipeline_data.decomposition_data) == 0:
            print("No decompositiondata data available.")
            return

        print("=== DecompositionData Information ===")
        data_names = list(pipeline_data.decomposition_data.keys())
        print(f"DecompositionData Names: {len(data_names)} ({", ".join(data_names)})")
        
        for name, data in pipeline_data.decomposition_data.items():
            print(f"\n--- {name} ---")
            data.print_info()

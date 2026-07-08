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

from typing import Optional, Any, Dict, Tuple, TYPE_CHECKING
import gc
import os
import numpy as np

from ..entities.decomposition_data import DecompositionData
from ..decomposition_type.interfaces.decomposition_type_base import DecompositionTypeBase
from ..helper.decomposition_validation_helper import (
    DecompositionValidationHelper,
)
from ..services.decomposition_add_service import DecompositionAddService
from ...utils.data_utils import DataUtils
from ...utils.cleanup_utils import CleanupUtils
from ...utils.memmap_utils import MemmapUtils
from ...utils.path_utils import PathUtils

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class DecompositionManager:
    """
    Manager for decomposition data objects.

    Manages the creation and storage of decomposition results from feature
    matrices. Works with TrajectoryData objects to perform dimensionality
    reduction using various decomposition methods (PCA, KernelPCA, etc.).

    Examples
    --------
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

    def __init__(self, use_memmap: bool = False, chunk_size: int = 2000, cache_dir: str = "./cache") -> None:
        """
        Initialize decomposition manager.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for decomposition data
        chunk_size : int, optional
            Processing chunk size for incremental computation
        cache_dir : str, optional
            Cache directory path for decomposition data

        Returns
        -------
        None
            Initializes DecompositionManager instance with specified configuration

        Examples
        --------
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
        self.cache_dir = PathUtils.prepare_directory_path(
            cache_dir,
            create=True,
            purpose="cache directory",
        )

        DecompositionValidationHelper.validate_chunk_size(chunk_size)

    def add_decomposition(
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

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:

        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.add("selection", decomposition_type.PCA())  # NO pipeline_data parameter

        Standalone mode:

        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.add_decomposition(pipeline_data, "selection", decomposition_type.PCA())  # pipeline_data required

        Parameters
        ----------
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

        Returns
        -------
        None
            Adds computed decomposition to trajectory data

        Raises
        ------
        ValueError
            If the decomposition already exists, if required selection is missing,
            or if the decomposition computation fails

        Examples
        --------
        >>> # Add PCA decomposition
        >>> from mdxplain.decomposition import decomposition_type
        >>> manager = DecompositionManager()
        >>> manager.add_decomposition(
        ...     pipeline_data, "feature_selection", decomposition_type.PCA(n_components=10)
        ... )

        >>> # Add KernelPCA with custom parameters
        >>> manager.add_decomposition(
        ...     pipeline_data, "any_selection", decomposition_type.KernelPCA(n_components=15, gamma=0.1)
        ... )

        >>> # Add ContactKernelPCA for contact features
        >>> manager.add_decomposition(
        ...     pipeline_data, "contact_selection", decomposition_type.ContactKernelPCA(n_components=20)
        ... )

        >>> # Force recomputation of existing decomposition
        >>> manager.add_decomposition(
        ...     pipeline_data, "feature_selection", decomposition_type.PCA(n_components=20), force=True
        ... )
        """
        decomposition_key = DataUtils.get_type_key(decomposition_type)
        if decomposition_name is None:
            decomposition_name = f"{selection_name}_{decomposition_key}"

        self._check_decomposition_existence(pipeline_data, decomposition_name, force)

        # Validate feature type requirements
        DecompositionValidationHelper.validate_feature_type_compatibility(
            pipeline_data, selection_name, decomposition_type
        )
        
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

        Delegates the name-availability check to the validation helper. When
        the name is taken and force is set, the existing decomposition and its
        cache are released so it can be recomputed.

        Parameters
        ----------
        pipeline_data : PipelineData
            Trajectory data object
        selection_name : str
            Selection name used as decomposition key
        force : bool
            Whether to force recomputation

        Returns
        -------
        None
            Validates decomposition status

        Raises
        ------
        ValueError
            If decomposition exists and force is False
        """
        DecompositionValidationHelper.validate_target_available(
            pipeline_data, selection_name, force
        )
        if selection_name not in pipeline_data.decomposition_data:
            return
        print(
            f"WARNING: Decomposition for selection '{selection_name}' already exists. Forcing recomputation."
        )
        old_data = pipeline_data.decomposition_data[selection_name]
        cache_path = old_data.cache_path
        old_array = old_data.data
        old_data.data = None
        del pipeline_data.decomposition_data[selection_name]
        gc.collect()
        MemmapUtils.close_memmap_view(old_array)
        old_array = None
        gc.collect()
        if cache_path and os.path.exists(cache_path):
            CleanupUtils.remove_path(
                cache_path,
                purpose="decomposition cache path",
            )

    def _get_selection_cache_path(self, selection_name: str) -> Optional[str]:
        """
        Get selection-specific cache path for decomposition data.

        Creates cache path structure: base_cache_dir/selection_name/
        This allows multiple decomposition types for the same selection.

        Parameters
        ----------
        selection_name : str
            Name of the feature selection

        Returns
        -------
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

        Parameters
        ----------
        decomposition_data : DecompositionData
            Decomposition data container
        decomposition_type : DecompositionTypeBase instance
            Decomposition type instance with parameters
        data_matrix : numpy.ndarray
            Data matrix to decompose
        decomposition_name : str
            Name of the feature selection used

        Returns
        -------
        None
            Performs decomposition computation
        """
        DecompositionValidationHelper.validate_decomposition_type(
            decomposition_type
        )

        decomposition_type.init_calculator(
            use_memmap=self.use_memmap,
            cache_path=decomposition_data.cache_path or "./cache",
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

        Parameters
        ----------
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

        Returns
        -------
        None
            Stores decomposition results
        """
        pipeline_data.decomposition_data[decomposition_name] = decomposition_data

        assert decomposition_data.data is not None
        print(
            f"Decomposition '{decomposition_key}' with name '{decomposition_name}' for selection '{selection_name}' computed successfully. "
            f"Data reduced from {original_shape} to {decomposition_data.data.shape}."
        )

    def reduce_components(
        self,
        pipeline_data: PipelineData,
        source_name: str,
        new_name: str,
        n_components: int,
        force: bool = False,
    ) -> None:
        """
        Keep only the first n_components of an existing decomposition.

        PCA and Kernel PCA components are ordered by descending variance, so the
        leading ``n_components`` are exactly the ones a fresh run with that count
        would produce. This truncation is a cheap slice of the stored transformed
        data -- it does NOT recompute the (expensive) eigendecomposition. The
        original decomposition is left untouched; a reduced clone is stored under
        ``new_name`` and can be used downstream (clustering, plots) like any
        other decomposition.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        source_name : str
            Name of the decomposition to reduce
        new_name : str
            Name to store the reduced clone under
        n_components : int
            Number of leading components to keep
        force : bool, default=False
            Whether to overwrite an existing decomposition with the same name

        Returns
        -------
        None
            Stores the reduced decomposition in pipeline data

        Examples
        --------
        >>> # An auto-selected decomposition yielded 30 components; keep 5
        >>> pipeline.decomposition.reduce_components(
        ...     "ContactKernelPCA", "ContactKernelPCA_5", n_components=5
        ... )
        >>> pipeline.clustering.add.dpa("ContactKernelPCA_5", Z=2.5)
        """
        DecompositionValidationHelper.validate_source_exists(
            pipeline_data, source_name
        )
        DecompositionValidationHelper.validate_target_available(
            pipeline_data, new_name, force
        )
        source = pipeline_data.decomposition_data[source_name]
        DecompositionValidationHelper.validate_component_count(
            source, n_components
        )
        reduced = self._build_reduced_decomposition(
            source, n_components, source_name
        )
        pipeline_data.decomposition_data[new_name] = reduced
        print(
            f"Reduced '{source_name}' to {n_components} components, "
            f"stored as '{new_name}'."
        )

    @staticmethod
    def _build_reduced_decomposition(
        source: DecompositionData, n_components: int, source_name: str
    ) -> DecompositionData:
        """
        Build a truncated clone of a decomposition.

        Copies the leading ``n_components`` columns of the transformed data into
        a small owned array and adjusts the metadata; the frame mapping is
        shared with the source.

        Parameters
        ----------
        source : DecompositionData
            Source decomposition to reduce
        n_components : int
            Number of leading components to keep
        source_name : str
            Name of the source, recorded in the metadata

        Returns
        -------
        DecompositionData
            The reduced clone
        """
        assert source.data is not None
        reduced = DecompositionData(source.decomposition_type)
        reduced.data = np.array(source.data[:, :n_components])
        reduced.metadata = DecompositionManager._reduced_metadata(
            source.metadata, n_components, source_name
        )
        reduced.frame_mapping = source.frame_mapping
        return reduced

    @staticmethod
    def _reduced_metadata(
        metadata: Optional[Dict[str, Any]],
        n_components: int,
        source_name: str,
    ) -> Dict[str, Any]:
        """
        Build the metadata for a reduced decomposition.

        Parameters
        ----------
        metadata : dict or None
            Source metadata
        n_components : int
            Number of leading components kept
        source_name : str
            Name of the source decomposition

        Returns
        -------
        Dict[str, Any]
            Metadata with the component count, variance arrays truncated to
            n_components, and the source recorded under ``reduced_from``
        """
        reduced: Dict[str, Any] = dict(metadata) if metadata else {}
        reduced["n_components"] = n_components
        reduced["reduced_from"] = source_name
        reduced["auto_selected"] = False
        for key in ("explained_variance_ratio", "explained_variance"):
            values = reduced.get(key)
            if values is not None:
                reduced[key] = np.asarray(values)[:n_components]
        return reduced

    def reset_decompositions(self, pipeline_data: PipelineData) -> None:
        """
        Reset all computed decompositions and clear decomposition data.

        This method removes all computed decompositions and their associated data,
        requiring decompositions to be recalculated from scratch.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:

        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.reset_decompositions()  # NO pipeline_data parameter

        Standalone mode:

        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.reset_decompositions(pipeline_data)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Trajectory data object

        Returns
        -------
        None
            Clears all decomposition data from pipeline_data.decomposition_data

        Examples
        --------
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

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:

        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.save('decomposition.npy')  # NO pipeline_data parameter

        Standalone mode:

        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.save(pipeline_data, 'decomposition.npy')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with decomposition data
        save_path : str
            Path where to save all decomposition data in one file

        Returns
        -------
        None
            Saves all decomposition data to the specified file
            
        Examples
        --------
        >>> manager.save(pipeline_data, 'decomposition.npy')
        """
        DataUtils.save_object(pipeline_data.decomposition_data, save_path)

    def load(self, pipeline_data: PipelineData, load_path: str) -> None:
        """
        Load all decomposition data from single file.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:

        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.load('decomposition.npy')  # NO pipeline_data parameter

        Standalone mode:

        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.load(pipeline_data, 'decomposition.npy')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container to load decomposition data into
        load_path : str
            Path to saved decomposition data file

        Returns
        -------
        None
            Loads all decomposition data from the specified file
            
        Examples
        --------
        >>> manager.load(pipeline_data, 'decomposition.npy')
        """
        temp_dict = {}
        DataUtils.load_object(temp_dict, load_path)
        pipeline_data.decomposition_data = temp_dict

    def print_info(self, pipeline_data: PipelineData) -> None:
        """
        Print decomposition data information.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:

        >>> pipeline = PipelineManager()
        >>> pipeline.decomposition.print_info()  # NO pipeline_data parameter

        Standalone mode:
        
        >>> pipeline_data = PipelineData()
        >>> manager = DecompositionManager()
        >>> manager.print_info(pipeline_data)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with decomposition data

        Returns
        -------
        None
            Prints decomposition data information to console

        Examples
        --------
        >>> manager.print_info(pipeline_data)
        """
        if len(pipeline_data.decomposition_data) == 0:
            print("No decompositiondata data available.")
            return

        print("=== DecompositionData Information ===")
        data_names = list(pipeline_data.decomposition_data.keys())
        print(f"DecompositionData Names: {len(data_names)} ({', '.join(data_names)})")
        
        for name, data in pipeline_data.decomposition_data.items():
            print(f"\n--- {name} ---")
            data.print_info()

    @property
    def add(self) -> DecompositionAddService:
        """
        Service for adding decomposition algorithms with simplified syntax.

        Provides an intuitive interface for adding decomposition algorithms without
        requiring explicit decomposition type instantiation or imports.

        Returns
        -------
        DecompositionAddService
            Service instance for adding decomposition algorithms with combined parameters
            
        Examples
        --------
        >>> # Add different decomposition algorithms
        >>> pipeline.decomposition.add.pca("my_features", n_components=10)
        >>> pipeline.decomposition.add.kernel_pca("contact_features", kernel='rbf', n_components=20)
        >>> pipeline.decomposition.add.contact_kernel_pca("contact_features", n_components=15)
        >>> pipeline.decomposition.add.diffusion_maps("distance_features", n_components=12)
        
        Notes
        -----
        Pipeline data is automatically injected by AutoInjectProxy.
        All decomposition type parameters are combined with manager.add parameters.
        """
        return DecompositionAddService(self, None)

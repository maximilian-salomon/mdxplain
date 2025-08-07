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
MD trajectory data container and analysis interface.

Container for MD trajectory data and analysis results. Uses a pluggable
Loader class for flexible loading strategies. Provides unified interface
for handling multiple trajectories and their derived features.
"""

from typing import Optional

import numpy as np

from ...utils.data_utils import DataUtils


class TrajectoryData:
    """
    Container for MD trajectory data and analysis results.

    This class serves as the main interface for loading molecular dynamics trajectories
    and computing various features such as distances and contacts. It provides a
    unified interface for handling multiple trajectories and their derived features
    with memory-efficient processing capabilities.

    The class uses a pluggable architecture where different feature types can be
    added and computed independently or with dependencies on other features.

    Examples:
    ---------
    Basic usage with distances and contacts:

    >>> import mdxplain.data.feature_type as feature_type
    >>> traj = TrajectoryData()
    >>> traj.load_trajectories('../data')
    >>> traj.add_feature(feature_type.Distances())
    >>> traj.add_feature(feature_type.Contacts(cutoff=4.5))
    >>> distances = traj.get_feature(feature_type.Distances())
    >>> contacts = traj.get_feature(feature_type.Contacts())

    Memory-mapped processing for large datasets:

    >>> traj = TrajectoryData(use_memmap=True, cache_dir='./cache')
    >>> traj.load_trajectories('../large_data')
    >>> traj.add_feature(feature_type.Distances(), chunk_size=1000)

    Attributes:
    -----------
    use_memmap : bool
        Whether to use memory mapping for large datasets
    cache_dir : str or None
        Directory for cache files when using memory mapping
    trajectories : list or None
        List of loaded MD trajectory objects
    trajectory_names : list or None
        List of trajectory names
    features : dict
        Dictionary storing FeatureData instances by feature type string
    """

    def __init__(self, use_memmap=False, cache_dir="./cache", chunk_size=10000):
        """
        Initialize trajectory data container.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets. When True,
            large arrays are stored on disk and accessed via memory mapping
            to reduce RAM usage.
        cache_dir : str, default="./cache"
            Directory for cache files when using memory mapping.
        chunk_size : int, default=10000
            Chunk size for memory-efficient processing of large datasets.

        Returns:
        --------
        None
            Initializes trajectory data container

        Examples:
        ---------
        >>> # Standard initialization
        >>> traj = TrajectoryData()

        >>> # For large datasets with memory mapping
        >>> traj = TrajectoryData(use_memmap=True, cache_dir='/tmp/mdxplain_cache')
        """
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size

        self.trajectories = []
        self.trajectory_names = []  # List of trajectory names
        self.features = {}  # Dictionary to store FeatureData instances by feature type
        self.res_label_data = None  # res_labels for trajectory residues
        self.selected_data = {}  # selected data for trajectory residues
        self.decomposition_data = {}  # Dictionary to store DecompositionData instances
        self.cluster_data = {}  # Dictionary to store ClusterData instances

    def get_feature(self, feature_type):
        """
        Retrieve a computed feature by its type.

        This method returns the FeatureData instance for a previously computed
        feature. The returned object provides access to the computed data,
        feature names, analysis methods, and data reduction capabilities.

        Supports all three input variants:
        - feature_type.Distances() (instance)
        - feature_type.Distances (class with metaclass)
        - "distances" (string)

        Parameters:
        -----------
        feature_type : FeatureTypeBase, FeatureTypeBase class, or str
            Feature type instance, class, or string (e.g., Distances(), Distances, "distances")

        Returns:
        --------
        FeatureData
            The FeatureData instance containing computed data and analysis methods.

        Raises:
        -------
        ValueError
            If the requested feature type has not been computed yet.

        Examples:
        ---------
        >>> # Get distances feature - all variants work:
        >>> distances = traj.get_feature(feature_type.Distances())  # instance
        >>> distances = traj.get_feature(feature_type.Distances)    # class
        >>> distances = traj.get_feature("distances")               # string
        >>> distance_data = distances.get_data()
        >>> feature_names = distances.get_feature_names()

        >>> # Get contacts and apply analysis
        >>> contacts = traj.get_feature(feature_type.Contacts)
        >>> frequency = contacts.analysis.compute_frequency()
        """
        # Convert to feature key using same logic as FeatureManager
        if isinstance(feature_type, str):
            key = feature_type
        elif hasattr(feature_type, "get_type_name"):
            key = feature_type.get_type_name()
        else:
            # Fallback: try to convert to string (handles metaclass)
            key = str(feature_type)

        feature_data = self.features.get(key)
        if feature_data is None:
            raise ValueError(f"Feature {key} not found.")
        return feature_data

    def get_trajectory_names(self):
        """
        Get list of trajectory names.

        This method returns the names of all loaded trajectories as a list.
        The names are generated during trajectory loading and can be used
        for identification and debugging purposes.

        Parameters:
        -----------
        None

        Returns:
        --------
        list or None
            List of trajectory names, or None if trajectories not loaded

        Examples:
        ---------
        >>> names = traj.get_trajectory_names()
        >>> print(names)
        ['system1_prot_traj1', 'system1_prot_traj2', 'system2_prot_traj1']
        """
        return self.trajectory_names

    def print_trajectory_info(self):
        """
        Print information about loaded trajectories.

        This method displays a formatted list of all loaded trajectories
        including their indices, names, and basic information. Useful for
        debugging and getting an overview of the current trajectory data.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Prints the trajectory names

        Examples:
        ---------
        >>> traj.print_trajectory_info()
        Loaded 3 trajectories:
          [0] system1_prot_traj1: 1000 frames
          [1] system1_prot_traj2: 1500 frames
          [2] system2_prot_traj1: 800 frames
        """
        if self.trajectories is None or self.trajectory_names is None:
            print("No trajectories loaded.")
            return

        print(f"Loaded {len(self.trajectories)} trajectories:")
        for i, (traj, name) in enumerate(zip(self.trajectories, self.trajectory_names)):
            print(f"  [{i}] {name}: {traj}")

    def save(self, save_path):
        """
        Save the complete TrajectoryData object to disk.

        This method serializes the entire TrajectoryData object including
        all computed features, trajectories, and metadata to a file. The
        saved object can be loaded later to restore the complete analysis
        state without recomputation.

        Parameters:
        -----------
        save_path : str
            Path where to save the TrajectoryData object. Should have a
            .pkl or .npy extension. The directory will be created if it
            doesn't exist.

        Returns:
        --------
        None
            Saves the TrajectoryData object to the specified path

        Examples:
        ---------
        >>> # Save after computing features
        >>> traj.add_feature(feature_type.Distances())
        >>> traj.add_feature(feature_type.Contacts(cutoff=4.5))
        >>> traj.save('analysis_results/trajectory_data.pkl')

        >>> # Save with specific path structure
        >>> import os
        >>> save_dir = 'project_results/session_001'
        >>> os.makedirs(save_dir, exist_ok=True)
        >>> traj.save(f'{save_dir}/traj_analysis.pkl')

        Notes:
        -----
        - All computed features and their reduced versions are saved
        - Memory-mapped data files remain separate and are referenced
        - Trajectories are included in the saved object
        - Analysis method bindings are restored upon loading
        - Cache paths and memory mapping settings are preserved
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path):
        """
        Load a previously saved TrajectoryData object from disk.

        This method deserializes a TrajectoryData object from a file,
        restoring all computed features, trajectories, and analysis state.
        After loading, the object is ready for immediate use without
        requiring recomputation of features.

        Parameters:
        -----------
        load_path : str
            Path to the saved TrajectoryData file (.pkl or .npy).
            The file must have been created using the save() method.

        Returns:
        --------
        None
            Loads the TrajectoryData object from the specified path

        Examples:
        ---------
        >>> # Load previously saved analysis
        >>> traj = TrajectoryData()
        >>> traj.load('analysis_results/trajectory_data.pkl')
        >>>
        >>> # Access loaded features immediately
        >>> distances = traj.get_feature(feature_type.Distances())
        >>> contacts = traj.get_feature(feature_type.Contacts())
        >>>
        >>> # Continue analysis where you left off
        >>> mean_distances = distances.analysis.compute_mean()

        >>> # Load and continue with different analysis
        >>> traj_loaded = TrajectoryData()
        >>> traj_loaded.load('project_results/session_001/traj_analysis.pkl')
        >>> contacts = traj_loaded.get_feature(feature_type.Contacts())
        >>> contacts.reduce_data(
        ...     metric=feature_type.Contacts.ReduceMetrics.STABILITY,
        ...     threshold_min=0.5
        ... )

        Raises:
        -------
        FileNotFoundError
            If the specified file doesn't exist
        ValueError
            If the file is corrupted or not a valid TrajectoryData save file

        Notes:
        -----
        - All previously computed features are restored
        - Analysis method bindings are automatically recreated
        - Memory mapping settings and cache paths are preserved
        - If memory-mapped data files are missing, an error will occur
        - Trajectories are fully restored with all metadata
        """
        DataUtils.load_object(self, load_path)

    def get_selected_matrix(self, name):
        """
        Return merged matrix from all selected features.

        This method retrieves a previously created feature selection and
        returns a merged matrix containing all selected columns from the
        different feature types. The matrix combines both reduced and
        original data as specified during selection creation.

        Parameters:
        -----------
        name : str
            Name of the selection to retrieve

        Returns:
        --------
        numpy.ndarray
            Merged matrix with selected columns from all features

        Raises:
        -------
        ValueError
            If selection not found or no data available

        Examples:
        ---------
        >>> # After creating a selection with FeatureSelector
        >>> selector = FeatureSelector()
        >>> selector.add("distances", "res ALA")
        >>> selector.select(traj_data, "ala_analysis")
        >>>
        >>> # Get the merged matrix
        >>> matrix = traj_data.get_selected_matrix("ala_analysis")
        >>> print(f"Selected data shape: {matrix.shape}")
        """
        self._validate_selection_exists(name)
        matrices = self._collect_matrices_for_selection(name)
        return self._merge_matrices(matrices, name)

    def _validate_selection_exists(self, name: str):
        """
        Validate that the selection exists.

        Parameters:
        -----------
        name : str
            Name of the selection to validate

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If the selection does not exist
        """
        if not hasattr(self, "selected_data") or name not in self.selected_data:
            raise ValueError(f"No selection named '{name}' found.")

    def _collect_matrices_for_selection(self, name: str) -> list:
        """
        Collect matrices for all features in the selection.

        Parameters:
        -----------
        name : str
            Name of the selection to collect matrices for

        Returns:
        --------
        list
            List of matrices for all features in the selection

        Raises:
        -------
        ValueError
            If the selection does not exist
        """
        matrices = []

        for feature_type, selection_info in self.selected_data[name].items():
            feature_matrices = self._get_matrices_for_feature(
                feature_type, selection_info, name
            )
            matrices.extend(feature_matrices)

        return matrices

    def _get_matrices_for_feature(
        self, feature_type: str, selection_info: dict, name: str
    ) -> list:
        """
        Get matrices for a single feature type.

        Parameters:
        -----------
        feature_type : str
            Name of the feature type to get matrices for
        selection_info : dict
            Selection information for the feature type
        name : str
            Name of the selection to get matrices for

        Returns:
        --------
        list
            List of matrices for the feature type

        Raises:
        -------
        ValueError
            If the feature type does not exist
        """
        feature_data = self.features[feature_type]
        indices = selection_info["indices"]
        use_reduced_flags = selection_info["use_reduced"]

        reduced_indices, original_indices = self._group_indices_by_type(
            indices, use_reduced_flags
        )

        matrices = []
        matrices.extend(
            self._get_data_matrices(
                feature_data,
                reduced_indices,
                "reduced_data",
                "Reduced",
                feature_type,
                name,
            )
        )
        matrices.extend(
            self._get_data_matrices(
                feature_data, original_indices, "data", "Original", feature_type, name
            )
        )

        return matrices

    def _group_indices_by_type(self, indices: list, use_reduced_flags: list) -> tuple:
        """
        Group indices by whether they use reduced or original data.

        Parameters:
        -----------
        indices : list
            List of indices to group
        use_reduced_flags : list
            List of flags indicating whether the indices use reduced data

        Returns:
        --------
        tuple
            Tuple containing two lists: reduced_indices and original_indices
        """
        reduced_indices = []
        original_indices = []

        for col_idx, use_reduced in zip(indices, use_reduced_flags):
            if use_reduced:
                reduced_indices.append(col_idx)
            else:
                original_indices.append(col_idx)

        return reduced_indices, original_indices

    def _create_memmap_selection(
        self, data, indices: list, name: str, data_type: str, feature_type: str
    ):
        """
        Create memory-efficient selection using chunk-wise processing.

        This method avoids loading entire columns into RAM by processing
        data in chunks and writing directly to a memmap output file.

        Parameters:
        -----------
        data : numpy.ndarray or memmap
            Source data to select from
        indices : list
            Column indices to select
        name : str
            Selection name for cache file naming
        data_type : str
            Type of data (for cache naming)
        feature_type : str
            Feature type name (for cache naming)

        Returns:
        --------
        list
            List containing the memmap-selected data matrix
        """
        n_rows, _ = data.shape
        n_cols = len(indices)

        cache_path = DataUtils.get_cache_file_path(
            f"{name}_{feature_type}_{data_type}_selection.dat", self.cache_dir
        )

        result = np.memmap(
            cache_path,
            dtype=data.dtype,
            mode="w+",
            shape=(n_rows, n_cols),
        )

        chunk_size = self.chunk_size

        for row_start in range(0, n_rows, chunk_size):
            row_end = min(row_start + chunk_size, n_rows)
            result[row_start:row_end, :] = data[row_start:row_end, indices]

        return [result]

    def _get_data_matrices(
        self,
        feature_data,
        indices: list,
        data_attr: str,
        data_type: str,
        feature_type: str,
        name: str,
    ) -> list:
        """
        Get matrices from specified data attribute.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        indices : list
            List of indices to get matrices for
        data_attr : str
            Attribute of the feature data object to get matrices from
        data_type : str
            Type of data to get matrices from
        feature_type : str
            Name of the feature type to get matrices for
        name : str
            Name of the selection to get matrices for

        Returns:
        --------
        list
            List of matrices for the feature type

        Raises:
        -------
        ValueError
            If the data attribute is not available
        """
        if not indices:
            return []

        data = getattr(feature_data, data_attr)
        if data is None:
            raise ValueError(
                f"{data_type} data not available for feature '{feature_type}' "
                f"in selection '{name}'."
            )

        if self.use_memmap:
            return self._create_memmap_selection(
                data, indices, name, data_type, feature_type
            )
        return [data[:, indices]]

    def _merge_matrices(self, matrices: list, name: str) -> np.ndarray:
        """
        Merge collected matrices horizontally.

        Parameters:
        -----------
        matrices : list
            List of matrices to merge
        name : str
            Name of the selection to merge matrices for

        Returns:
        --------
        numpy.ndarray
            Merged matrix with selected columns from all features

        Raises:
        -------
        ValueError
            If no valid data is found for the selection
        """
        if not matrices:
            raise ValueError(f"No valid data found for selection '{name}'.")

        # Check if any matrix is memmap and preserve memmap nature
        if self.use_memmap:
            return self._memmap_hstack(matrices, name)
        else:
            return np.hstack(matrices)

    def get_selected_feature_metadata(self, name):
        """
        Return metadata for all selected features.

        This method retrieves the metadata for all features in a selection,
        providing detailed information about each column in the corresponding
        selected matrix. The metadata includes feature definitions and types,
        allowing for proper interpretation of the selected data.

        Parameters:
        -----------
        name : str
            Name of the selection to retrieve

        Returns:
        --------
        numpy.ndarray
            Array of dictionaries, one for each column in the selected matrix.
            Each dictionary has the structure:
            {
                'features': original feature metadata entry,
                'type': feature type name as string
            }

        Raises:
        -------
        ValueError
            If selection not found or no metadata available

        Examples:
        ---------
        >>> # Get metadata for a selection
        >>> metadata = traj_data.get_selected_feature_metadata("ala_analysis")
        >>> print(f"Number of selected features: {len(metadata)}")
        >>>
        >>> # Examine first feature
        >>> first_feature = metadata[0]
        >>> print(f"Feature type: {first_feature['type']}")
        >>> print(f"Feature details: {first_feature['features']}")
        """
        self._validate_selection_exists(name)
        selected_metadata = self._collect_metadata_for_selection(name)
        return self._finalize_metadata_result(selected_metadata, name)

    def _collect_metadata_for_selection(self, name: str) -> list:
        """
        Collect metadata for all features in the selection.

        Parameters:
        -----------
        name : str
            Name of the selection to collect metadata for

        Returns:
        --------
        list
            List of metadata for all features in the selection

        Raises:
        -------
        ValueError
            If the selection does not exist
        """
        selected_metadata = []

        for feature_type, selection_info in self.selected_data[name].items():
            feature_metadata = self._get_metadata_for_feature(
                feature_type, selection_info, name
            )
            selected_metadata.extend(feature_metadata)

        return selected_metadata

    def _get_metadata_for_feature(
        self, feature_type: str, selection_info: dict, name: str
    ) -> list:
        """
        Get metadata for a single feature type.

        Parameters:
        -----------
        feature_type : str
            Name of the feature type to get metadata for
        selection_info : dict
            Selection information for the feature type
        name : str
            Name of the selection to get metadata for

        Returns:
        --------
        list
            List of metadata for the feature type

        Raises:
        -------
        ValueError
            If the feature type does not exist
        """
        feature_data = self.features[feature_type]
        indices = selection_info["indices"]
        use_reduced_flags = selection_info["use_reduced"]

        feature_metadata = []

        for col_idx, use_reduced in zip(indices, use_reduced_flags):
            metadata_entry = self._get_single_metadata_entry(
                feature_data, col_idx, use_reduced, feature_type, name
            )
            if metadata_entry:
                feature_metadata.append(metadata_entry)

        return feature_metadata

    def _get_single_metadata_entry(
        self,
        feature_data,
        col_idx: int,
        use_reduced: bool,
        feature_type: str,
        name: str,
    ) -> Optional[dict]:
        """
        Get metadata entry for a single column.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        col_idx : int
            Index of the column to get metadata for
        use_reduced : bool
            Whether the column uses reduced data
        feature_type : str
            Name of the feature type to get metadata for
        name : str
            Name of the selection to get metadata for

        Returns:
        --------
        Optional[dict]
            Metadata entry for the column, or None if not available

        Raises:
        -------
        ValueError
            If the feature type does not exist
        """
        metadata = self._get_feature_metadata_by_type(
            feature_data, use_reduced, feature_type, name
        )

        if "features" in metadata:
            return {"features": metadata["features"][col_idx], "type": feature_type}
        return None

    def _get_feature_metadata_by_type(
        self, feature_data, use_reduced: bool, feature_type: str, name: str
    ) -> dict:
        """
        Get the appropriate metadata based on use_reduced flag.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        use_reduced : bool
            Whether the column uses reduced data
        feature_type : str
            Name of the feature type to get metadata for
        name : str
            Name of the selection to get metadata for

        Returns:
        --------
        dict
            Metadata for the feature type

        Raises:
        -------
        ValueError
            If the metadata is not available
        """
        if use_reduced:
            if feature_data.reduced_feature_metadata is None:
                raise ValueError(
                    f"Reduced metadata not available for feature '{feature_type}' "
                    f"in selection '{name}'."
                )
            return feature_data.reduced_feature_metadata
        else:
            if feature_data.feature_metadata is None:
                raise ValueError(
                    f"Metadata not available for feature '{feature_type}' "
                    f"in selection '{name}'."
                )
            return feature_data.feature_metadata

    def _finalize_metadata_result(
        self, selected_metadata: list, name: str
    ) -> np.ndarray:
        """
        Finalize and validate the metadata result.

        Parameters:
        -----------
        selected_metadata : list
            List of metadata for all features in the selection
        name : str
            Name of the selection to get metadata for

        Returns:
        --------
        numpy.ndarray
            Array of dictionaries, one for each column in the selected matrix.

        Raises:
        -------
        ValueError
            If no valid metadata is found for the selection
        """
        if not selected_metadata:
            raise ValueError(f"No valid metadata found for selection '{name}'.")

        return np.array(selected_metadata)

    def get_decomposition(self, decomposition_name):
        """
        Retrieve a computed decomposition by selection name.

        This method returns the DecompositionData instance for a previously
        computed decomposition. The returned object provides access to the
        decomposed data, metadata, hyperparameters, and transformation details.

        Parameters:
        -----------
        decomposition_name : str
            Name of the decomposition

        Returns:
        --------
        DecompositionData
            The DecompositionData instance containing decomposed data and metadata

        Raises:
        -------
        ValueError
            If the requested decomposition has not been computed yet

        Examples:
        ---------
        >>> # Get decomposition for a selection
        >>> decomp_data = traj.get_decomposition("feature_sel")
        >>> transformed = decomp_data.get_data()
        >>> metadata = decomp_data.get_metadata()

        >>> # Get decomposition type from metadata
        >>> decomp_type = decomp_data.metadata.get('decomposition_type', 'unknown')
        >>> print(f"Decomposition type: {decomp_type}")
        """
        if decomposition_name not in self.decomposition_data:
            available_decompositions = list(self.decomposition_data.keys())
            raise ValueError(
                f"Decomposition with name '{decomposition_name}' not found. "
                f"Available decompositions: {available_decompositions}"
            )

        return self.decomposition_data[decomposition_name]

    def list_decompositions(self):
        """
        List all computed decompositions.

        Returns a list of all computed decompositions with their selection
        names and decomposition types for easy overview.

        Parameters:
        -----------
        None

        Returns:
        --------
        list
            List of dictionaries containing decomposition information

        Examples:
        ---------
        >>> decompositions = traj.list_decompositions()
        >>> for decomp in decompositions:
        ...     print(f"Selection: {decomp['selection']}, Type: {decomp['type']}")
        """
        decomposition_list = []

        for decomposition_name, decomp_data in self.decomposition_data.items():
            # Get decomposition type from metadata
            decomp_type = 'unknown'
            if hasattr(decomp_data, 'metadata') and decomp_data.metadata:
                decomp_type = decomp_data.metadata.get('decomposition_type', 'unknown')
            elif hasattr(decomp_data, 'decomposition_type'):
                decomp_type = decomp_data.decomposition_type
            
            decomposition_list.append(
                {
                    "decomposition_name": decomposition_name,
                    "type": decomp_type,
                }
            )

        return decomposition_list

    def get_cluster(self, cluster_name):
        """
        Retrieve a computed clustering result by cluster name.

        This method returns the ClusterData instance for a previously
        computed clustering analysis. The returned object provides access to the
        cluster labels, metadata, and clustering parameters.

        Parameters:
        -----------
        cluster_name : str
            Name of the clustering result to retrieve

        Returns:
        --------
        ClusterData
            The ClusterData instance containing cluster labels and metadata

        Raises:
        -------
        ValueError
            If the requested clustering result has not been computed yet

        Examples:
        ---------
        >>> # Get clustering result by name
        >>> cluster_data = traj.get_cluster("dbscan_analysis")
        >>> labels = cluster_data.labels
        >>> metadata = cluster_data.metadata

        >>> # Get clustering result with default name
        >>> cluster_data = traj.get_cluster("dbscan_eps0.5_min5")
        >>> n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        """
        if cluster_name not in self.cluster_data:
            available_clusters = list(self.cluster_data.keys())
            raise ValueError(
                f"Clustering result '{cluster_name}' not found. "
                f"Available clusters: {available_clusters}"
            )

        return self.cluster_data[cluster_name]

    def list_clusters(self):
        """
        List all computed clustering results.

        Returns a list of all computed clustering results with their names
        and basic information for easy overview.

        Parameters:
        -----------
        None

        Returns:
        --------
        list
            List of dictionaries containing clustering information

        Examples:
        ---------
        >>> clusters = traj.list_clusters()
        >>> for cluster in clusters:
        ...     print(f"Name: {cluster['name']}, Type: {cluster['type']}, "
        ...           f"Clusters: {cluster['n_clusters']}")
        """
        cluster_list = []

        for cluster_name, cluster_data in self.cluster_data.items():
            # Get basic information from cluster data
            labels = cluster_data.labels if hasattr(cluster_data, 'labels') else None
            cluster_type = cluster_data.cluster_type if hasattr(cluster_data, 'cluster_type') else 'unknown'
            
            # Calculate number of clusters (excluding noise points labeled as -1)
            n_clusters = 0
            if labels is not None:
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            cluster_list.append(
                {
                    "name": cluster_name,
                    "type": cluster_type,
                    "n_clusters": n_clusters,
                    "n_points": len(labels) if labels is not None else 0,
                }
            )

        return cluster_list

    def _memmap_hstack(self, matrices: list, name: str) -> np.ndarray:
        """
        Horizontally stack matrices while preserving memmap nature.

        Parameters:
        -----------
        matrices : list
            List of matrices to stack (all are memmap)
        name : str
            Name of the selection for cache file naming

        Returns:
        --------
        numpy.ndarray
            Stacked matrix stored as memmap
        """
        # Calculate total shape
        total_samples = matrices[0].shape[0]
        total_features = sum(matrix.shape[1] for matrix in matrices)

        # Determine the appropriate dtype for mixed data types
        dtypes = [matrix.dtype for matrix in matrices]
        result_dtype = np.result_type(*dtypes)

        # Create cache path for the stacked matrix
        cache_path = DataUtils.get_cache_file_path(f"{name}.dat", self.cache_dir)

        # Create memmap for the result
        result = np.memmap(
            cache_path,
            dtype=result_dtype,
            mode="w+",
            shape=(total_samples, total_features),
        )

        # Fill the result matrix column by column, chunk by chunk
        col_start = 0
        for matrix in matrices:
            col_end = col_start + matrix.shape[1]

            # Process in chunks to avoid loading entire matrix into memory
            for row_start in range(0, total_samples, self.chunk_size):
                row_end = min(row_start + self.chunk_size, total_samples)
                result[row_start:row_end, col_start:col_end] = matrix[
                    row_start:row_end, :
                ]

            col_start = col_end

        return result

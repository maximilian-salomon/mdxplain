# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
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
ClusterManager for managing clustering data objects.

Manager for creating and managing clustering results from feature matrices
or decomposition results. Used to add, reset, and manage clustering data
in trajectory data objects.
"""

from ..entities.cluster_data import ClusterData
from ...utils.data_utils import DataUtils
import shutil
import os


class ClusterManager:
    """
    Manager for clustering data objects.

    Manages the creation and storage of clustering results from feature
    matrices or decomposition results. Works with TrajectoryData objects
    to perform clustering analysis using various clustering methods
    (DBSCAN, HDBSCAN, DPA).

    Examples:
    ---------
    >>> # Create manager and add DBSCAN clustering
    >>> from mdxplain.clustering import cluster_type
    >>> manager = ClusterManager()
    >>> manager.add(
    ...     pipeline_data, "feature_selection", cluster_type.DBSCAN(eps=0.5),
    ...     use_decomposed=False
    ... )

    >>> # Manager with custom cache directory
    >>> manager = ClusterManager(cache_dir="./cache/clustering")
    >>> manager.add(
    ...     pipeline_data, "pca_decomposition", cluster_type.HDBSCAN(min_cluster_size=10),
    ...     use_decomposed=True
    ... )
    """

    def __init__(self, cache_dir="./cache"):
        """
        Initialize cluster manager.

        Parameters:
        -----------
        cache_dir : str, optional
            Cache directory path for clustering data, default="./cache"

        Returns:
        --------
        None
            Initializes ClusterManager instance with specified configuration

        Examples:
        ---------
        >>> # Basic manager
        >>> manager = ClusterManager()

        >>> # Manager with custom cache directory
        >>> manager = ClusterManager(cache_dir="./cache/clustering")
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def reset_clusters(self, pipeline_data):
        """
        Reset all computed clustering results and clear clustering data.

        This method removes all computed clustering results and their associated data,
        requiring clustering to be recalculated from scratch.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.clustering.reset_clusters()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = ClusterManager()
        >>> manager.reset_clusters(pipeline_data)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns:
        --------
        None
            Clears all clustering data from pipeline_data.cluster_data

        Examples:
        ---------
        >>> manager = ClusterManager()
        >>> manager.reset_clusters(pipeline_data)
        """
        if not pipeline_data.cluster_data:
            print("No clustering results to reset.")
            return

        cluster_list = list(pipeline_data.cluster_data.keys())
        pipeline_data.cluster_data.clear()

        print(
            f"Reset {len(cluster_list)} clustering result(s): {', '.join(cluster_list)}"
        )
        print("All clustering data has been cleared. Clustering must be recalculated.")

    def _check_cluster_existence(self, pipeline_data, cluster_name, force):
        """
        Check if clustering already exists and handle accordingly.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object
        cluster_name : str
            Cluster name/key
        force : bool
            Whether to force recomputation

        Returns:
        --------
        None
            Validates clustering status

        Raises:
        -------
        ValueError
            If clustering exists and force is False
        """
        if cluster_name in pipeline_data.cluster_data:
            if force:
                print(
                    f"WARNING: Clustering '{cluster_name}' already exists. Forcing recomputation."
                )
                del pipeline_data.cluster_data[cluster_name]
            else:
                raise ValueError(f"Clustering '{cluster_name}' already exists.")

    def _validate_cluster_type(self, cluster_type):
        """
        Validate cluster type instance and parameters.

        Parameters:
        -----------
        cluster_type : ClusterTypeBase instance
            Cluster type instance with parameters

        Returns:
        --------
        None
            Validates cluster type

        Raises:
        -------
        ValueError
            If cluster type is invalid or missing required methods
        """
        if not hasattr(cluster_type, "init_calculator"):
            raise ValueError(
                f"Invalid cluster type '{cluster_type}'. "
                "Please provide a cluster type instance."
            )

        if not hasattr(cluster_type, "get_type_name"):
            raise ValueError(
                f"Invalid cluster type '{cluster_type}'. "
                "Cluster type must implement get_type_name method."
            )

    def _validate_data_matrix(self, data_matrix, selection_name):
        """
        Validate data matrix for clustering.

        Parameters:
        -----------
        data_matrix : numpy.ndarray
            Data matrix to validate
        selection_name : str
            Name of the selection for error messages

        Returns:
        --------
        None
            Validates data matrix

        Raises:
        -------
        ValueError
            If data matrix is invalid for clustering
        """
        if data_matrix is None:
            raise ValueError(f"No data found for selection '{selection_name}'.")

        if data_matrix.size == 0:
            raise ValueError(f"Empty data matrix for selection '{selection_name}'.")

        if len(data_matrix.shape) != 2:
            raise ValueError(
                f"Data matrix for selection '{selection_name}' must be 2D. "
                f"Got shape: {data_matrix.shape}"
            )

        if data_matrix.shape[0] < 2:
            raise ValueError(
                f"Data matrix for selection '{selection_name}' must have at least 2 samples. "
                f"Got {data_matrix.shape[0]} samples."
            )

    def _get_decomposition_data(self, pipeline_data, decomposition_name):
        """
        Retrieve decomposition data matrix for clustering.

        Uses the new simplified approach where decompositions are stored
        with selection_name as the key (not selection_name_decomposition_type).

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object containing decomposition results
        decomposition_name : str
            Name of the decomposition

        Returns:
        --------
        numpy.ndarray
            Decomposition data matrix for clustering

        Raises:
        -------
        ValueError
            If decomposition data is not found or invalid
        """
        if not pipeline_data.decomposition_data:
            raise ValueError(
                f"No decomposition data found. Available decompositions: []"
            )

        decomposition_data = pipeline_data.get_decomposition(decomposition_name)
        data_matrix = decomposition_data.data

        if data_matrix is None:
            raise ValueError(
                f"Decomposition data matrix is None for selection '{decomposition_name}'"
            )

        return data_matrix

    def _get_feature_selection_data(self, pipeline_data, selection_name):
        """
        Retrieve feature selection data matrix for clustering.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object containing feature selections
        selection_name : str
            Name of the feature selection

        Returns:
        --------
        numpy.ndarray
            Feature selection data matrix for clustering

        Raises:
        -------
        ValueError
            If feature selection data is not found or invalid
        """
        pipeline_data.validate_selection_exists(selection_name)
        data_matrix = pipeline_data.get_selected_matrix(selection_name)

        if data_matrix is None:
            raise ValueError(
                f"Feature selection data matrix is None for selection '{selection_name}'"
            )

        return data_matrix

    def _get_data_matrix(self, pipeline_data, name, use_decomposed):
        """
        Retrieve data matrix for clustering based on use_decomposed flag.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object
        name : str
            Name to select data
        use_decomposed : bool
            Whether to use decomposition data (True) or feature selection data (False)

        Returns:
        --------
        numpy.ndarray
            Data matrix for clustering

        Raises:
        -------
        ValueError
            If data retrieval fails or data is invalid
        """
        if use_decomposed:
            return self._get_decomposition_data(pipeline_data, name)
        else:
            return self._get_feature_selection_data(pipeline_data, name)

    def add(
        self,
        pipeline_data,
        selection_name,
        cluster_type,
        use_decomposed=True,
        cluster_name=None,
        force=False,
        override_cache=False,
    ):
        """
        Add clustering analysis to trajectory data.

        This method performs clustering analysis on the specified data selection
        using the provided cluster type. Results are stored in the PipelineData
        object's cluster_data dictionary with the specified or default cluster name.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.clustering.add("selection", cluster_type.DBSCAN())  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = ClusterManager()
        >>> manager.add(pipeline_data, "selection", cluster_type.DBSCAN())  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature selections or decomposition results
        selection_name : str
            Name of the feature selection (if use_decomposed=False) or
            decomposition result (if use_decomposed=True) to cluster
        cluster_type : ClusterTypeBase instance
            Cluster type instance with parameters (e.g., DBSCAN(eps=0.5))
        use_decomposed : bool, optional
            Whether to use decomposition results (True) or feature selections (False),
            default=True
        cluster_name : str, optional
            Custom name for storing clustering results. If None, defaults to str(cluster_type)
        force : bool, optional
            Whether to force recomputation if clustering already exists, default=False
        override_cache : bool, optional
            Whether to clear entire cluster_name subdirectory before computation, default=False

        Returns:
        --------
        None
            Stores ClusterData object in pipeline_data.cluster_data dictionary

        Examples:
        ---------
        >>> # Cluster decomposition results with custom name
        >>> from mdxplain.clustering import cluster_type
        >>> manager = ClusterManager()
        >>> manager.add(
        ...     pipeline_data, "pca_selection", cluster_type.DBSCAN(eps=0.5),
        ...     use_decomposed=True, cluster_name="my_dbscan"
        ... )

        >>> # Cluster feature selection with default name
        >>> manager.add(
        ...     pipeline_data, "distance_selection", cluster_type.HDBSCAN(min_cluster_size=10),
        ...     use_decomposed=False
        ... )

        Raises:
        -------
        ValueError
            If selection not found, cluster type invalid, or clustering computation fails
        """
        self._validate_cluster_type(cluster_type)
        cluster_name = self._determine_cluster_name(cluster_name, cluster_type)

        cache_path = DataUtils.get_cache_file_path(cluster_name, self.cache_dir)

        if override_cache:
            self._clear_cache_directory(cache_path)

        cluster_type.init_calculator(cache_path=cache_path)

        self._check_cluster_existence(pipeline_data, cluster_name, force)

        data_matrix = self._prepare_data_for_clustering(
            pipeline_data, selection_name, use_decomposed
        )
        cluster_labels, metadata = self._perform_clustering(cluster_type, data_matrix)

        cluster_data = self._create_cluster_data(
            cluster_type, cluster_labels, metadata, selection_name
        )
        self._store_clustering_results(pipeline_data, cluster_name, cluster_data)

    def _determine_cluster_name(self, cluster_name, cluster_type):
        """
        Determine cluster name (use provided or default to str(cluster_type)).

        Parameters:
        -----------
        cluster_name : str or None
            Custom cluster name provided by user
        cluster_type : ClusterTypeBase instance
            Cluster type instance to get default name from

        Returns:
        --------
        str
            Final cluster name to use for storage
        """
        return (
            cluster_name
            if cluster_name is not None
            else DataUtils.get_type_key(cluster_type)
        )

    def _clear_cache_directory(self, cache_path):
        """
        Clear entire cache directory for cluster_name subdirectory.

        Parameters:
        -----------
        cache_path : str
            Path to the cache directory to clear

        Returns:
        --------
        None
            Removes all files in the cache directory
        """
        shutil.rmtree(cache_path)
        print(f"Cleared cache directory: {cache_path}")

    def _prepare_data_for_clustering(
        self, pipeline_data, selection_name, use_decomposed
    ):
        """
        Prepare and validate data matrix for clustering.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object
        selection_name : str
            Name of the selection to prepare data for
        use_decomposed : bool
            Whether to use decomposition data or feature selection data

        Returns:
        --------
        numpy.ndarray
            Validated data matrix ready for clustering
        """
        data_matrix = self._get_data_matrix(
            pipeline_data, selection_name, use_decomposed
        )
        self._validate_data_matrix(data_matrix, selection_name)
        return data_matrix

    def _perform_clustering(self, cluster_type, data_matrix):
        """
        Perform clustering computation with error handling.

        Parameters:
        -----------
        cluster_type : ClusterTypeBase instance
            Initialized cluster type instance
        data_matrix : numpy.ndarray
            Data matrix to cluster

        Returns:
        --------
        tuple
            Tuple containing (cluster_labels, metadata)
        """
        cluster_labels, metadata = cluster_type.compute(data_matrix)
        self._validate_cluster_labels(cluster_labels, data_matrix)

        return cluster_labels, metadata

    def _validate_cluster_labels(self, cluster_labels, data_matrix):
        """
        Ensure cluster labels correspond to trajectory frame indices.

        Parameters:
        -----------
        cluster_labels : numpy.ndarray
            Array of cluster labels for each sample
        data_matrix : numpy.ndarray
            Original data matrix that was clustered

        Returns:
        --------
        None
            Validates cluster labels

        Raises:
        -------
        ValueError
            If cluster labels length doesn't match number of trajectory frames
        """
        if len(cluster_labels) != data_matrix.shape[0]:
            raise ValueError(
                f"Cluster labels length ({len(cluster_labels)}) does not match "
                f"number of trajectory frames ({data_matrix.shape[0]})"
            )

    def _create_cluster_data(
        self, cluster_type, cluster_labels, metadata, selection_name
    ):
        """
        Create ClusterData object with results.

        Parameters:
        -----------
        cluster_type : ClusterTypeBase instance
            Cluster type instance used for clustering
        cluster_labels : numpy.ndarray
            Array of cluster labels for each sample
        metadata : dict
            Dictionary containing clustering metadata
        selection_name : str
            Name of the selection that was clustered

        Returns:
        --------
        ClusterData
            Initialized ClusterData object with results
        """
        cache_path = DataUtils.get_cache_file_path(selection_name, self.cache_dir)
        cluster_data = ClusterData(
            cluster_type=cluster_type.get_type_name(), cache_path=cache_path
        )
        cluster_data.labels = cluster_labels
        cluster_data.metadata = metadata
        return cluster_data

    def _store_clustering_results(self, pipeline_data, cluster_name, cluster_data):
        """
        Store clustering results and print success message.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Trajectory data object to store results in
        cluster_name : str
            Name to use as key for storing clustering results
        cluster_data : ClusterData
            ClusterData object containing results and metadata

        Returns:
        --------
        None
            Stores results in pipeline_data.cluster_data and prints success message
        """
        pipeline_data.cluster_data[cluster_name] = cluster_data

        n_clusters = cluster_data.metadata.get("n_clusters", "unknown")
        n_frames = len(cluster_data.labels)

        print(f"Clustering '{cluster_name}' completed successfully.")
        print(f"Found {n_clusters} clusters for {n_frames} frames.")

    def _handle_clustering_error(self, error, cluster_type, selection_name):
        """
        Handle clustering computation errors with informative messages.

        Parameters:
        -----------
        error : Exception
            The original error that occurred
        cluster_type : ClusterTypeBase instance
            Cluster type that failed
        selection_name : str
            Name of the selection being clustered

        Returns:
        --------
        None
            Re-raises error with additional context

        Raises:
        -------
        ValueError
            Enhanced error message with clustering context
        """
        cluster_type_name = getattr(
            cluster_type, "get_type_name", lambda: str(cluster_type)
        )()

        raise ValueError(
            f"Clustering computation failed for '{cluster_type_name}' on selection '{selection_name}'. "
            f"Original error: {str(error)}"
        ) from error

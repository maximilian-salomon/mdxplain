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
Central data container for the Pipeline orchestration system.

This module provides the PipelineData class that serves as the central
data container orchestrating all analysis data including trajectories,
features, clustering results, and decomposition results.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ...trajectory.entities.trajectory_data import TrajectoryData
from ...feature.entities.feature_data import FeatureData
from ...feature_selection.entities.feature_selector_data import FeatureSelectorData
from ...clustering.entities.cluster_data import ClusterData
from ...decomposition.entities.decomposition_data import DecompositionData
from ...data_selector.entities.data_selector_data import DataSelectorData
from ...comparison.entities.comparison_data import ComparisonData
from ...feature_importance.entities.feature_importance_data import FeatureImportanceData
from ...utils.data_utils import DataUtils
from ..helper.selection_matrix_helper import SelectionMatrixHelper
from ..helper.selection_metadata_helper import SelectionMetadataHelper
from ..helper.comparison_data_helper import ComparisonDataHelper
from ..helper.selection_memmap_helper import SelectionMemmapHelper


class PipelineData:
    """
    Central data container orchestrating all analysis data.

    This class serves as the central data hub for the pipeline system,
    containing all trajectory data, computed features, clustering results,
    decomposition results, and future analysis modules.

    The PipelineData serves as the central "God-Object"
    that gets passed around to managers, following the
    builder pattern while providing separation of concerns.

    Examples:
    ---------
    Pipeline mode (automatic):

    >>> pipeline = PipelineManager()
    >>> # PipelineData is managed automatically
    >>> pipeline.trajectory.load_trajectories('../data')

    Standalone mode (manual):

    >>> pipeline_data = PipelineData()
    >>> manager = TrajectoryManager()
    >>> manager.load_trajectories(pipeline_data, '../data')
    """

    def __init__(
        self,
        use_memmap: bool = False,
        cache_dir: str = "./cache",
        chunk_size: int = 10000,
    ):
        """
        Initialize the central pipeline data container.

        Creates empty containers for all analysis data types that will
        be populated through the respective manager interfaces.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_dir : str, default="./cache"
            Directory for cache files when using memory mapping
        chunk_size : int, default=10000
            Chunk size for memory-efficient processing

        Returns:
        --------
        None
            Initializes PipelineData instance with empty data containers
        """
        # Validate chunk_size parameter
        if not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be an integer, got {type(chunk_size).__name__}"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        # Memory management configuration
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size

        # Core trajectory data
        self.trajectory_data: TrajectoryData = TrajectoryData()

        # Analysis data containers
        self.feature_data: Dict[str, FeatureData] = {}
        self.selected_feature_data: Dict[str, FeatureSelectorData] = {}
        self.decomposition_data: Dict[str, DecompositionData] = {}
        self.cluster_data: Dict[str, ClusterData] = {}

        # New analysis modules
        self.data_selector_data: Dict[str, DataSelectorData] = {}
        self.comparison_data: Dict[str, ComparisonData] = {}
        self.feature_importance_data: Dict[str, FeatureImportanceData] = {}

    def clear_all_data(self) -> None:
        """
        Clear all stored analysis data.

        Resets all data containers to empty state, effectively clearing
        all computed results while preserving the container structure.
        Useful for starting fresh analysis or freeing memory.

        Returns:
        --------
        None
            Clears all data containers in-place

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> # ... after computations ...
        >>> pipeline_data.clear_all_data()
        """
        self.trajectory_data = TrajectoryData()
        self.feature_data.clear()
        self.selected_feature_data.clear()
        self.decomposition_data.clear()
        self.cluster_data.clear()
        self.data_selector_data.clear()
        self.comparison_data.clear()
        self.feature_importance_data.clear()

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary information about all stored data.

        Provides an overview of all data containers with counts and
        availability information. Useful for debugging and monitoring
        the state of the pipeline.

        Returns:
        --------
        dict
            Summary dictionary with data container information

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> summary = pipeline_data.get_data_summary()
        >>> print(summary['trajectories_loaded'])
        >>> print(summary['features_computed'])
        """
        return {
            "trajectories_loaded": len(self.trajectory_data.trajectories),
            "trajectory_names": len(self.trajectory_data.trajectory_names),
            "features_computed": len(self.feature_data),
            "feature_selections": len(self.selected_feature_data),
            "clusterings_performed": len(self.cluster_data),
            "decompositions_computed": len(self.decomposition_data),
            "data_selectors_created": len(self.data_selector_data),
            "comparisons_created": len(self.comparison_data),
            "feature_importance_analyses": len(self.feature_importance_data),
        }

    def has_trajectories(self) -> bool:
        """
        Check if trajectory data is available.

        Returns:
        --------
        bool
            True if trajectories are loaded, False otherwise
        """
        return len(self.trajectory_data.trajectories) > 0

    def has_features(self) -> bool:
        """
        Check if any feature data is available.

        Returns:
        --------
        bool
            True if features are computed, False otherwise
        """
        return len(self.feature_data) > 0

    def has_clusterings(self) -> bool:
        """
        Check if any clustering results are available.

        Returns:
        --------
        bool
            True if clustering results exist, False otherwise
        """
        return len(self.cluster_data) > 0

    def has_decompositions(self) -> bool:
        """
        Check if any decomposition results are available.

        Returns:
        --------
        bool
            True if decomposition results exist, False otherwise
        """
        return len(self.decomposition_data) > 0

    # =============================================================================
    # FEATURE ACCESS METHODS
    # =============================================================================

    def get_feature(self, feature_type):
        """
        Retrieve a computed feature by its type.

        This method returns the FeatureData instance for a previously computed
        feature. The returned object provides access to the computed data,
        feature names, analysis methods, and data reduction capabilities.

        Supports three input variants:
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
            The FeatureData instance containing computed data and analysis methods

        Raises:
        -------
        ValueError
            If the requested feature type has not been computed yet

        Examples:
        ---------
        >>> # Get distances feature - all variants work:
        >>> distances = pipeline_data.get_feature("distances")
        >>> distance_data = distances.get_data()
        >>> feature_names = distances.get_feature_names()

        >>> # Get contacts and apply analysis
        >>> contacts = pipeline_data.get_feature("contacts")
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

        feature_data = self.feature_data.get(key)
        if feature_data is None:
            available_features = list(self.feature_data.keys())
            raise ValueError(
                f"Feature '{key}' not found. Available features: {available_features}"
            )
        return feature_data

    # =============================================================================
    # DECOMPOSITION ACCESS METHODS
    # =============================================================================

    def get_decomposition(self, decomposition_name: str):
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
        >>> decomp_data = pipeline_data.get_decomposition("feature_sel")
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
        >>> decompositions = pipeline_data.list_decompositions()
        >>> for decomp in decompositions:
        ...     print(f"Selection: {decomp['decomposition_name']}, Type: {decomp['type']}")
        """
        decomposition_list = []

        for decomposition_name, decomp_data in self.decomposition_data.items():
            # Get decomposition type from metadata
            decomp_type = "unknown"
            if hasattr(decomp_data, "metadata") and decomp_data.metadata:
                decomp_type = decomp_data.metadata.get("decomposition_type", "unknown")
            elif hasattr(decomp_data, "decomposition_type"):
                decomp_type = decomp_data.decomposition_type

            decomposition_list.append(
                {
                    "decomposition_name": decomposition_name,
                    "type": decomp_type,
                }
            )

        return decomposition_list

    # =============================================================================
    # CLUSTERING ACCESS METHODS
    # =============================================================================

    def get_cluster(self, cluster_name: str):
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
        >>> cluster_data = pipeline_data.get_cluster("dbscan_analysis")
        >>> labels = cluster_data.labels
        >>> metadata = cluster_data.metadata

        >>> # Get clustering result with default name
        >>> cluster_data = pipeline_data.get_cluster("dbscan_eps0.5_min5")
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
        >>> clusters = pipeline_data.list_clusters()
        >>> for cluster in clusters:
        ...     print(f"Name: {cluster['name']}, Type: {cluster['type']}, "
        ...           f"Clusters: {cluster['n_clusters']}")
        """
        cluster_list = []

        for cluster_name, cluster_data in self.cluster_data.items():
            # Get basic information from cluster data
            labels = cluster_data.labels if hasattr(cluster_data, "labels") else None
            cluster_type = (
                cluster_data.cluster_type
                if hasattr(cluster_data, "cluster_type")
                else "unknown"
            )

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

    # =============================================================================
    # PIPELINE MANAGEMENT METHODS
    # =============================================================================

    def save(self, save_path: str) -> None:
        """
        Save the complete PipelineData object to disk.

        This method serializes the entire PipelineData object including
        all computed features, trajectories, clusterings, decompositions,
        and metadata to a file. The saved object can be loaded later to
        restore the complete analysis state without recomputation.

        Parameters:
        -----------
        save_path : str
            Path where to save the PipelineData object. Should have a
            .pkl extension. The directory will be created if it doesn't exist.

        Returns:
        --------
        None
            Saves the PipelineData object to the specified path

        Examples:
        ---------
        >>> # Save after computing features
        >>> pipeline_data.save('analysis_results/pipeline_data.pkl')

        >>> # Save with specific path structure
        >>> import os
        >>> save_dir = 'project_results/session_001'
        >>> os.makedirs(save_dir, exist_ok=True)
        >>> pipeline_data.save(f'{save_dir}/pipeline_analysis.pkl')

        Notes:
        -----
        - All computed features, clusterings, and decompositions are saved
        - Memory-mapped data files remain separate and are referenced
        - Complete pipeline state is preserved including configuration
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load a previously saved PipelineData object from disk.

        This method deserializes a PipelineData object from a file,
        restoring all computed features, trajectories, and analysis state.
        After loading, the object is ready for immediate use without
        requiring recomputation.

        Parameters:
        -----------
        load_path : str
            Path to the saved PipelineData file (.pkl).
            The file must have been created using the save() method.

        Returns:
        --------
        None
            Loads the PipelineData object from the specified path

        Examples:
        ---------
        >>> # Load previously saved analysis
        >>> pipeline_data = PipelineData()
        >>> pipeline_data.load('analysis_results/pipeline_data.pkl')
        >>>
        >>> # Access loaded features immediately
        >>> distances = pipeline_data.get_feature("distances")
        >>> contacts = pipeline_data.get_feature("contacts")
        >>>
        >>> # Continue analysis where you left off
        >>> mean_distances = distances.analysis.compute_mean()

        Raises:
        -------
        FileNotFoundError
            If the specified file doesn't exist
        ValueError
            If the file is corrupted or not a valid PipelineData save file

        Notes:
        -----
        - All previously computed features are restored
        - Memory mapping settings and cache paths are preserved
        - If memory-mapped data files are missing, an error will occur
        - Complete pipeline state including configuration is restored
        """
        DataUtils.load_object(self, load_path)

    # =============================================================================
    # FEATURE SELECTION SYSTEM METHODS
    # =============================================================================

    def get_selected_metadata(self, name):
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
        >>> metadata = pipeline_data.get_selected_metadata("ala_analysis")
        >>> print(f"Number of selected features: {len(metadata)}")
        >>>
        >>> # Examine first feature
        >>> first_feature = metadata[0]
        >>> print(f"Feature type: {first_feature['type']}")
        >>> print(f"Feature details: {first_feature['features']}")
        """
        self.validate_selection_exists(name)
        selected_metadata = SelectionMetadataHelper.collect_metadata_for_selection(
            self, name
        )
        return SelectionMetadataHelper.finalize_metadata_result(selected_metadata, name)

    def validate_selection_exists(self, name: str):
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
        if name not in self.selected_feature_data:
            raise ValueError(f"No selection named '{name}' found.")

        selector_data = self.selected_feature_data[name]
        if not selector_data.has_results():
            raise ValueError(
                f"Selection '{name}' has not been processed yet. Run select() first."
            )
        
    def get_selected_data(
        self,
        feature_selector: str,
        data_selector: Optional[str] = None
    ) -> np.ndarray:
        """
        Get data matrix with selected features and optionally selected frames.

        This method combines feature selection (columns) and data selection (rows)
        to create a matrix with the desired subset of data. Feature selection is
        required to define which columns to include.

        Parameters:
        -----------
        feature_selector : str
            Name of the feature selector (which columns to include).
            Must be provided - cannot be None.
        data_selector : str, optional
            Name of the data selector (which rows to include).
            If None, uses all available frames.

        Returns:
        --------
        np.ndarray
            Matrix with selected columns and optionally selected rows.
            - With data_selector: (n_selected_frames, n_selected_features)
            - Without data_selector: (n_all_frames, n_selected_features)

        Raises:
        -------
        ValueError
            If feature_selector doesn't exist, data_selector doesn't exist, or no data available

        Examples:
        ---------
        >>> # Get data with both feature and frame selection
        >>> data = pipeline_data.get_selected_data(
        ...     feature_selector="key_distances",
        ...     data_selector="folded_frames"
        ... )
        >>> print(f"Selected data shape: {data.shape}")

        >>> # Get all frames but only selected features
        >>> data = pipeline_data.get_selected_data(
        ...     feature_selector="important_features"
        ... )
        """
        # Validate feature selector exists
        self.validate_selection_exists(feature_selector)
        
        # Get full feature matrix using SelectionMatrixHelper
        matrices = SelectionMatrixHelper.collect_matrices_for_selection(self, feature_selector)
        full_matrix = SelectionMatrixHelper.merge_matrices(
            matrices, feature_selector, self.use_memmap, self.cache_dir, self.chunk_size
        )

        # If no data selector specified, return full matrix
        if data_selector is None:
            return full_matrix

        # Validate data selector exists  
        if data_selector not in self.data_selector_data:
            available = list(self.data_selector_data.keys())
            raise ValueError(
                f"Data selector '{data_selector}' not found. "
                f"Available data selectors: {available}"
            )
            
        # Get selected frame indices
        selector_data = self.data_selector_data[data_selector]
        frame_indices = selector_data.get_frame_indices()
        
        if not frame_indices:
            raise ValueError(f"Data selector '{data_selector}' has no selected frames")
        
        # Use memmap-safe frame selection for large datasets
        if self.use_memmap and len(frame_indices) > self.chunk_size:
            return SelectionMemmapHelper.create_memmap_frame_selection(
                full_matrix, frame_indices, f"{feature_selector}_{data_selector}",
                self.cache_dir, self.chunk_size
            )
        
        # For small selections, direct indexing is fine
        return full_matrix[frame_indices]

    # =============================================================================
    # COMPARISON DATA METHODS
    # =============================================================================

    def get_comparison_data(
        self, comparison_name: str, sub_comparison_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get X (features) and y (labels) for a specific comparison sub-comparison.

        This method provides the central access point for comparison data,
        combining ComparisonData metadata with efficient data processing.
        Used by modules to get ready-to-use datasets for analysis.

        Parameters:
        -----------
        comparison_name : str
            Name of the comparison to retrieve data from
        sub_comparison_name : str
            Name of the specific sub-comparison within the comparison

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (X, y) where:
            - X is the feature matrix with selected features and frames
            - y is the label array for the comparison groups

        Raises:
        -------
        ValueError
            If comparison not found, sub-comparison not found, or no data available

        Examples:
        ---------
        >>> # Get data for a binary comparison
        >>> X, y = pipeline_data.get_comparison_data("folded_vs_unfolded", "main")
        >>> print(f"Features shape: {X.shape}")
        >>> print(f"Labels: {np.unique(y)}")

        >>> # Get data for one-vs-rest comparison
        >>> X, y = pipeline_data.get_comparison_data("conformations", "folded_vs_rest")
        >>> print(f"Data shape: {X.shape}, Labels: {np.unique(y)}")
        """
        # Validate comparison exists
        if comparison_name not in self.comparison_data:
            available = list(self.comparison_data.keys())
            raise ValueError(
                f"Comparison '{comparison_name}' not found. "
                f"Available comparisons: {available}"
            )

        # Get comparison metadata
        comparison_data = self.comparison_data[comparison_name]

        # Use ComparisonDataHelper for data processing
        return ComparisonDataHelper.get_sub_comparison_data(
            self, comparison_data, sub_comparison_name
        )

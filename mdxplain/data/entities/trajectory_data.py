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

    def __init__(self, use_memmap=False, cache_dir=None):
        """
        Initialize trajectory data container.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets. When True,
            large arrays are stored on disk and accessed via memory mapping
            to reduce RAM usage.
        cache_dir : str, optional
            Directory for cache files when using memory mapping. If None
            and use_memmap is True, defaults to './cache'.

        Examples:
        ---------
        >>> # Standard initialization
        >>> traj = TrajectoryData()

        >>> # For large datasets with memory mapping
        >>> traj = TrajectoryData(use_memmap=True, cache_dir='/tmp/mdxplain_cache')
        """
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir

        if use_memmap and cache_dir is None:
            self.cache_dir = "./cache"

        self.trajectories = []
        self.trajectory_names = []  # List of trajectory names
        self.features = {}  # Dictionary to store FeatureData instances by feature type
        self.res_label_data = None  # res_labels for trajectory residues
        self.selected_data = {}  # selected data for trajectory residues

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
        elif hasattr(feature_type, 'get_type_name'):
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

        Parameters:
        -----------
        save_path : str
            Path where to save the TrajectoryData object. Should have a
            .pkl or .npy extension. The directory will be created if it
            doesn't exist.

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

        Parameters:
        -----------
        load_path : str
            Path to the saved TrajectoryData file (.pkl or .npy).
            The file must have been created using the save() method.

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
        """
        if not hasattr(self, 'selected_data') or name not in self.selected_data:
            raise ValueError(f"No selection named '{name}' found.")
        
        matrices = []
        
        for feature_type, selection_info in self.selected_data[name].items():
            feature_data = self.features[feature_type]
            indices = selection_info['indices']
            use_reduced_flags = selection_info['use_reduced']
            
            # Initialize arrays for both data types
            reduced_indices = []
            original_indices = []
            
            # Sort indices by data type
            for _, (col_idx, use_reduced) in enumerate(zip(indices, use_reduced_flags)):
                if use_reduced:
                    reduced_indices.append(col_idx)
                else:
                    original_indices.append(col_idx)
            
            # Get reduced data if needed
            if reduced_indices:
                if feature_data.reduced_data is None:
                    raise ValueError(f"Reduced data not available for feature '{feature_type}' in selection '{name}'.")
                matrices.append(feature_data.reduced_data[:, reduced_indices])
            
            # Get original data if needed
            if original_indices:
                if feature_data.data is None:
                    raise ValueError(f"Original data not available for feature '{feature_type}' in selection '{name}'.")
                matrices.append(feature_data.data[:, original_indices])
        
        # Merge horizontally (along columns)
        if not matrices:
            raise ValueError(f"No valid data found for selection '{name}'.")
        
        return np.hstack(matrices)

    def get_selected_feature_metadata(self, name):
        """
        Return metadata for all selected features.
        
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
        """
        if not hasattr(self, 'selected_data') or name not in self.selected_data:
            raise ValueError(f"No selection named '{name}' found.")
        
        selected_metadata = []
        
        for feature_type, selection_info in self.selected_data[name].items():
            feature_data = self.features[feature_type]
            indices = selection_info['indices']
            use_reduced_flags = selection_info['use_reduced']
            
            for _, (col_idx, use_reduced) in enumerate(zip(indices, use_reduced_flags)):
                # Get the appropriate metadata based on use_reduced flag
                if use_reduced:
                    if feature_data.reduced_feature_metadata is None:
                        raise ValueError(f"Reduced metadata not available for feature '{feature_type}' in selection '{name}'.")
                    metadata = feature_data.reduced_feature_metadata
                else:
                    if feature_data.feature_metadata is None:
                        raise ValueError(f"Metadata not available for feature '{feature_type}' in selection '{name}'.")
                    metadata = feature_data.feature_metadata
                
                # Add selected feature metadata
                if 'features' in metadata:
                    selected_metadata.append({
                        'features': metadata['features'][col_idx],
                        'type': feature_type
                    })
        
        if not selected_metadata:
            raise ValueError(f"No valid metadata found for selection '{name}'.")
        
        return np.array(selected_metadata)

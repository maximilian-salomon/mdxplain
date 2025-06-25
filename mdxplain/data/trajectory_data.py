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

from ..utils.data_utils import DataUtils
from .feature_data import FeatureData
from .trajectory_loader import TrajectoryLoader


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

        self.trajectories = None
        self.features = {}  # Dictionary to store FeatureData instances by feature type

    def add_feature(self, feature_type, cache_path=None, chunk_size=None):
        """
        Add and compute a feature for the loaded trajectories.

        This method creates a FeatureData instance for the specified feature type,
        handles dependency checking, and computes the feature data. Features with
        dependencies (like Contacts depending on Distances) will automatically
        use the required input data.

        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object (e.g., Distances(), Contacts()). The feature
            type determines what kind of analysis will be performed.
        cache_path : str, optional
            Specific cache path for this feature's data when using memory mapping.
            If None, uses the default cache directory structure.
        chunk_size : int, optional
            Chunk size for processing large datasets. Smaller chunks use less
            memory but may be slower. If None, uses automatic chunking.

        Raises:
        -------
        ValueError
            If the feature already exists with computed data, if required
            dependencies are missing, or if trajectories are not loaded.

        Examples:
        ---------
        >>> # Add basic distance feature
        >>> traj.add_feature(feature_type.Distances())

        >>> # Add contacts with custom cutoff
        >>> traj.add_feature(feature_type.Contacts(cutoff=3.5))

        >>> # Add feature with memory mapping and chunking
        >>> traj.add_feature(
        ...     feature_type.Distances(),
        ...     cache_path='/tmp/distances.dat',
        ...     chunk_size=500
        ... )
        """
        feature_key = str(feature_type)

        # Check if feature already exists
        if feature_key in self.features and self.features[feature_key].data is not None:
            raise ValueError(
                f"{feature_key.capitalize()} FeatureData already exists.")

        # Check dependencies
        dependencies = feature_type.get_dependencies()
        for dep in dependencies:
            dep_key = str(dep)
            if dep_key not in self.features or self.features[dep_key].data is None:
                raise ValueError(
                    f"Dependency '{dep_key}' must be computed before '{feature_key}'."
                )

        # Create FeatureData instance
        feature_data = FeatureData(
            feature_type=feature_type,
            use_memmap=self.use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size,
        )

        # If the feature type uses another feature as input,
        # compute the feature with the input feature data
        # Otherwise, compute the feature with the trajectories
        if feature_type.get_input() is not None:
            feature_data.compute(
                self.features[feature_type.get_input()].get_data(),
                self.features[feature_type.get_input()].get_feature_names(),
            )
        else:
            if self.trajectories is None:
                raise ValueError(
                    "Trajectories must be loaded before computing features."
                )
            feature_data.compute(self.trajectories, feature_names=None)

        # Store the feature data
        self.features[feature_key] = feature_data

    def get_feature(self, feature_type):
        """
        Retrieve a computed feature by its type.

        This method returns the FeatureData instance for a previously computed
        feature. The returned object provides access to the computed data,
        feature names, analysis methods, and data reduction capabilities.

        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object (e.g., Distances(), Contacts()). Must be
            the same type as used when adding the feature.

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
        >>> # Get distances feature
        >>> distances = traj.get_feature(feature_type.Distances())
        >>> distance_data = distances.get_data()
        >>> feature_names = distances.get_feature_names()

        >>> # Get contacts and apply analysis
        >>> contacts = traj.get_feature(feature_type.Contacts())
        >>> frequency = contacts.analysis.compute_frequency()

        >>> # Apply data reduction
        >>> contacts.reduce_data(
        ...     contacts.ReduceMetrics.FREQUENCY,
        ...     threshold_min=0.1,
        ...     threshold_max=0.9
        ... )
        """
        feature_data = self.features.get(str(feature_type))
        if feature_data is None:
            raise ValueError(f"Feature {str(feature_type)} not found.")
        return feature_data

    def load_trajectories(self, data_input, concat=False, stride=1):
        """
        Load molecular dynamics trajectories from files or directories.

        This method handles loading of MD trajectories in various formats
        (e.g., .xtc, .dcd, .trr) along with their topology files. The loading
        is performed using the TrajectoryLoader class which supports automatic
        format detection and multiple trajectory handling.

        Parameters:
        -----------
        data_input : str or list
            Path to directory containing trajectory files, or list of trajectory
            file paths. When a directory is provided, all supported trajectory
            files in that directory will be loaded.
        concat : bool, default=False
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. Useful when dealing with trajectory splits.
        stride : int, default=1
            Load every stride-th frame from trajectories. Use values > 1 to
            reduce memory usage and computation time by subsampling frames.

        Examples:
        ---------
        >>> # Load from directory
        >>> traj.load_trajectories('../data')

        >>> # Load specific files with striding
        >>> traj.load_trajectories(['traj1.xtc', 'traj2.xtc'], stride=10)

        >>> # Load and concatenate trajectories per system
        >>> traj.load_trajectories('../data', concat=True, stride=5)

        Notes:
        -----
        - Supported formats depend on MDTraj capabilities
        - Topology files (.pdb, .gro, .psf) should be in the same directory
        - Large trajectories benefit from striding to reduce memory usage
        """
        self.trajectories = TrajectoryLoader.load_trajectories(
            data_input, concat, stride
        )

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

# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# trajectory_data - MD Trajectory Loading and Processing
#
# Container for MD trajectory data and analysis results.
# Uses a pluggable Loader class for flexible loading strategies.
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

from .trajectory_loader import TrajectoryLoader
from .feature_data import FeatureData
from ..utils.data_utils import DataUtils


class TrajectoryData:
    """
    Container for MD trajectory data and analysis results.
    Uses a pluggable Loader class for flexible loading strategies.
    """
    
    def __init__(self, use_memmap=False, cache_dir=None):
        """
        Initialize trajectory data container.
        
        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_dir : str, optional
            Directory for cache files when using memory mapping
        """
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir
        
        if use_memmap and cache_dir is None:
            self.cache_dir = "./cache"
        
        self.trajectories = None
        self.features = {}  # Dictionary to store FeatureData instances by feature type

    def add_feature(self, feature_type, cache_path=None,
                    chunk_size=None):
        """
        Add a FeatureData instance for the specified feature type.
        
        Parameters:
        -----------
        feature_type : object
            Feature type object (e.g., Distances(), Contacts())
        cache_path : str, optional
            Specific cache path for this feature
        chunk_size : int, optional
            Chunk size for processing
        """
        feature_key = str(feature_type)
        
        # Check if feature already exists
        if feature_key in self.features and self.features[feature_key].data is not None:
            raise ValueError(f"{feature_key.capitalize()} FeatureData already exists.")
        
        # Check dependencies
        dependencies = feature_type.get_dependencies()
        for dep in dependencies:
            dep_key = str(dep)
            if dep_key not in self.features or self.features[dep_key].data is None:
                raise ValueError(f"Dependency '{dep_key}' must be computed before '{feature_key}'.")
        
        # Create FeatureData instance
        feature_data = FeatureData(
            feature_type=feature_type,
            use_memmap=self.use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size
        )

        # If the feature type uses another feature as input, 
        # compute the feature with the input feature data
        # Otherwise, compute the feature with the trajectories
        if feature_type.get_input() != None:
            feature_data.compute(self.features[feature_type.get_input()].get_data(), 
                                 self.features[feature_type.get_input()].get_feature_names()
            )
        else:
            if self.trajectories is None:
                raise ValueError("Trajectories must be loaded before computing features.")
            feature_data.compute(self.trajectories, feature_names=None)

        # Store the feature data
        self.features[feature_key] = feature_data

    def get_feature(self, feature_type):
        """
        Get FeatureData instance for the specified feature type.
        
        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object (e.g., Distances(), Contacts())
            
        Returns:
        --------
        FeatureData or None
            The FeatureData instance if it exists, None otherwise
        """
        feature_data = self.features.get(str(feature_type))
        if feature_data is None:
            raise ValueError(f"Feature {str(feature_type)} not found.")
        return feature_data

    def load_trajectories(self, data_input, concat=False, stride=1):
        """
        Load trajectories using the TrajectoryLoader class.

        Parameters:
        -----------
        data_input : list or str
            List of trajectory objects or path to directory
        concat : bool, default=False
            Whether to concatenate trajectories per system
        stride : int, default=1
            Load every stride-th frame from trajectories
        """
        self.trajectories = TrajectoryLoader.load_trajectories(data_input, concat, stride)

    def save(self, save_path):
        """
        Save the TrajectoryData object using DataUtils.
        
        Parameters:
        -----------
        save_path : str
            Path where to save the object (should end with .npy)
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path):
        """
        Load data into this TrajectoryData object using DataUtils.
        
        Parameters:
        -----------
        load_path : str
            Path to the saved TrajectoryData .npy file
        """
        DataUtils.load_object(self, load_path)
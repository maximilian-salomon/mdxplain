# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# TrajectoryData - MD Trajectory Loading and Processing
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

import os
from .TrajectoryLoader import TrajectoryLoader
from .FeatureData import FeatureData
from ..utils.DataUtils import DataUtils


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
                    chunk_size=None, squareform=True, k=0, **kwargs):
        """
        Add a FeatureData instance for the specified feature type.
        
        Parameters:
        -----------
        feature_type : str
            Type of feature ('distances', 'contacts', 'cci')
        cache_path : str, optional
            Specific cache path for this feature
        """
        if feature_type == 'distances' and self.features['distances'].data is not None:
            raise ValueError("Distances FeatureData already exists.")
        
        if feature_type == 'contacts' or feature_type == 'cci':
            if self.features['distances'].data is None:
                raise ValueError("First add a distances FeatureData before adding contacts or cci.")
            else:
                kwargs['distances'] = self.features['distances']

        feature_data = FeatureData(
            feature_type=feature_type,
            use_memmap=self.use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size,
            squareform=squareform,
            k=k
        )

        feature_data.compute(**kwargs)

        self.features[feature_type] = feature_data

    def get_feature(self, feature_type):
        """
        Get FeatureData instance for the specified feature type.
        
        Parameters:
        -----------
        feature_type : str
            Type of feature ('distances', 'contacts', 'cci')
            
        Returns:
        --------
        FeatureData or None
            The FeatureData instance if it exists, None otherwise
        """
        return self.features.get(feature_type)

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
        DataUtils.save_trajectory_data(self, save_path)

    def load(self, load_path):
        """
        Load data into this TrajectoryData object using DataUtils.
        
        Parameters:
        -----------
        load_path : str
            Path to the saved TrajectoryData .npy file
        """
        DataUtils.load_trajectory_data(self, load_path)
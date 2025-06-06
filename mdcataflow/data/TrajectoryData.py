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
from ..utils.DistanceCalculator import DistanceCalculator
from ..utils.ContactCalculator import ContactCalculator
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
        
        if use_memmap:
            if cache_dir is None:
                cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            self.distances_path = os.path.join(cache_dir, "distances.dat")
            self.contacts_path = os.path.join(cache_dir, "contacts.dat")
        
        self.trajectories = None
        self.distances = None
        self.contacts = None
        self.res_list = None

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

    def compute_distances(self, ref, batch_size=2500):
        """
        Compute pairwise distances using the DistanceCalculator utility class.
        
        Parameters:
        -----------
        ref : mdtraj.Trajectory
            Reference trajectory for residue information
        batch_size : int, default=2500
            Number of frames to process at once
        """
        self.distances, self.res_list = DistanceCalculator.compute_distances(
            trajectories=self.trajectories,
            ref=ref,
            batch_size=batch_size,
            use_memmap=self.use_memmap,
            distances_path=getattr(self, 'distances_path', None)
        )

    def compute_contacts(self, cutoff=4.5):
        """
        Compute contact maps using the ContactCalculator utility class.
        
        Parameters:
        -----------
        cutoff : float, default=4.5
            Distance cutoff for contacts (in Angstrom)
        """
        if self.distances is None:
            raise ValueError("Distances must be computed first. Call compute_distances() before compute_contacts().")
        
        self.contacts = ContactCalculator.compute_contacts(
            distances=self.distances,
            cutoff=cutoff,
            use_memmap=self.use_memmap,
            contacts_path=getattr(self, 'contacts_path', None)
        )

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
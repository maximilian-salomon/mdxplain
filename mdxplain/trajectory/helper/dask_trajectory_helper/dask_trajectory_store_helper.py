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
DaskMDTrajectory Store Manager - Zarr store management and slicing operations.

Handles complex zarr store operations like path resolution, slicing, 
and store creation for DaskMDTrajectory instances.
"""

from typing import Union
import os
import tempfile

import numpy as np
import zarr

from .zarr_cache_helper import ZarrCacheHelper


class DaskMDTrajectoryStoreHelper:
    """
    Store manager for DaskMDTrajectory zarr operations.
    
    Handles zarr store creation, path resolution, and slicing operations.
    """
    
    def __init__(self, cache_dir: str = './cache'):
        """
        Initialize DaskMDTrajectoryStoreHelper.
        
        Parameters:
        -----------
        cache_dir : str, default='./cache'
            Directory for temporary files during store operations
            
        Returns:
        --------
        None
            Initializes store manager instance
        """
        self.cache_dir = cache_dir
    
    def create_slice_store(self, dask_traj, key: Union[slice, np.ndarray]) -> zarr.Group:
        """
        Create new zarr store with sliced trajectory data.
        
        Parameters:
        -----------
        dask_traj : DaskMDTrajectory
            Source trajectory to slice
        key : Union[slice, np.ndarray]
            Slice or array indices to extract
            
        Returns:
        --------
        zarr.Group
            New zarr store with sliced data
            
        Examples:
        ---------
        >>> store_mgr = DaskMDTrajectoryStoreHelper()
        >>> zarr_store = store_mgr.create_slice_store(dask_traj, slice(0, 100))
        >>> # Creates store with first 100 frames
        """
        # Create temporary zarr store with secure temporary directory
        # Use a temporary file that persists until the returned store is used        
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', dir=self.cache_dir)
        os.close(temp_fd)  # Close file descriptor, but keep file
        os.remove(temp_path)  # Remove file so zarr can create directory
        temp_store = zarr.open(temp_path, mode='w')
        
        # Get sliced data
        sliced_coords = dask_traj._dask_coords[key]
        sliced_time = dask_traj._dask_time[key]
        
        # Determine new dimensions
        if isinstance(key, slice):
            start, stop, step = key.indices(dask_traj.n_frames)
            n_frames_new = len(range(start, stop, step))
        else:
            n_frames_new = len(key)
        
        # Store sliced data
        temp_store.create_array(
            'coordinates',
            data=sliced_coords.compute(),
            chunks=(min(dask_traj.chunk_size, n_frames_new), dask_traj.n_atoms, 3)
        )
        temp_store.create_array(
            'time',
            data=sliced_time.compute(),
            chunks=(min(dask_traj.chunk_size, n_frames_new),)
        )
        
        # Handle unitcell data
        if dask_traj._has_unitcell:
            temp_store.create_array(
                'unitcell_vectors',
                data=dask_traj._dask_unitcell_vectors[key].compute(),
                chunks=(min(dask_traj.chunk_size, n_frames_new), 3, 3)
            )
            temp_store.create_array(
                'unitcell_lengths', 
                data=dask_traj._dask_unitcell_lengths[key].compute(),
                chunks=(min(dask_traj.chunk_size, n_frames_new), 3)
            )
            temp_store.create_array(
                'unitcell_angles',
                data=dask_traj._dask_unitcell_angles[key].compute(),
                chunks=(min(dask_traj.chunk_size, n_frames_new), 3)
            )
        
        # Update metadata
        new_metadata = dask_traj.metadata.copy()
        new_metadata['n_frames'] = n_frames_new
        temp_store.attrs['metadata'] = new_metadata
        
        # Store topology
        ZarrCacheHelper.store_topology(temp_store, dask_traj._topology)
        
        return temp_store

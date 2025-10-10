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
DaskMDTrajectory Builder - Complex initialization logic extraction.

Handles the complex initialization process for DaskMDTrajectory instances,
including cache management, Zarr store setup, and Dask array initialization.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import os
import multiprocessing

import zarr
import dask.array as da
import mdtraj as md

from .zarr_cache_helper import ZarrCacheHelper
from .parallel_operations_helper import ParallelOperationsHelper

if TYPE_CHECKING:
    from ...entities.dask_md_trajectory import DaskMDTrajectory


class DaskMDTrajectoryBuildHelper:
    """
    Builder class for creating and initializing DaskMDTrajectory instances.
    
    Handles the complex initialization process by breaking it down into
    manageable steps while maintaining full compatibility with the original API.
    """
    
    def __init__(self) -> None:
        """
        Initialize DaskMDTrajectoryBuildHelper.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Initializes builder instance
        """
        pass
    
    def initialize_instance(self, instance: DaskMDTrajectory, trajectory_file: str, 
                           topology_file: Optional[str] = None,
                           zarr_cache_path: Optional[str] = None, 
                           chunk_size: int = 1000,
                           n_workers: Optional[int] = None) -> None:
        """
        Initialize a DaskMDTrajectory instance with all required components.
        
        Parameters
        ----------
        instance : DaskMDTrajectory
            Instance to initialize
        trajectory_file : str
            Path to trajectory file (.xtc, .dcd, etc.)
        topology_file : str, optional
            Path to topology file (.pdb, .gro, etc.)
        zarr_cache_path : str, optional
            Custom path for Zarr cache file
        chunk_size : int, default=1000
            Number of frames per chunk (optimized for performance)
        n_workers : int, optional
            Number of parallel workers (defaults to CPU count)
            
        Returns
        -------
        None
            Initializes instance in-place
            
        Examples
        --------
        >>> builder = DaskMDTrajectoryBuildHelper()
        >>> instance = DaskMDTrajectory.__new__(DaskMDTrajectory)
        >>> builder.initialize_instance(instance, 'traj.xtc', 'topology.pdb')
        """
        instance.trajectory_file = trajectory_file
        instance.topology_file = topology_file
        instance.chunk_size = chunk_size
        instance.n_workers = n_workers if n_workers is not None else multiprocessing.cpu_count()
        
        # Extract cache directory from zarr_cache_path and set as instance property
        if zarr_cache_path:
            instance._cache_dir = os.path.dirname(zarr_cache_path)
        else:
            instance._cache_dir = './cache'
        
        # Initialize cache manager with cache_dir
        instance.cache_manager = ZarrCacheHelper(chunk_size=chunk_size, cache_dir=instance._cache_dir)
        
        # Get or create Zarr cache
        instance.zarr_cache_path, instance.metadata = instance.cache_manager.get_or_create_cache(
            trajectory_file, topology_file, zarr_cache_path
        )
        
        # Open Zarr store
        instance._zarr_store = zarr.open(instance.zarr_cache_path, mode='r')
        
        # Load topology directly from store
        instance._topology = ZarrCacheHelper.load_topology(instance._zarr_store)
        
        # Create Dask arrays
        instance._dask_coords = da.from_zarr(instance.zarr_cache_path, component='coordinates')
        instance._dask_time = da.from_zarr(instance.zarr_cache_path, component='time')
        
        # Optional unitcell data
        instance._have_unitcell = 'unitcell_vectors' in instance._zarr_store
        if instance._have_unitcell:
            instance._dask_unitcell_vectors = da.from_zarr(instance.zarr_cache_path, component='unitcell_vectors')
            instance._dask_unitcell_lengths = da.from_zarr(instance.zarr_cache_path, component='unitcell_lengths')
            instance._dask_unitcell_angles = da.from_zarr(instance.zarr_cache_path, component='unitcell_angles')
        
        # Initialize parallel operations with cache_dir (already set as instance property)
        instance._parallel_ops = ParallelOperationsHelper(
            zarr_path=instance.zarr_cache_path, 
            topology=instance._topology, 
            n_workers=instance.n_workers, 
            chunk_size=chunk_size,
            cache_dir=instance._cache_dir
        )
        
        # Cache for frequently accessed data
        instance._xyz_cache = None
        instance._time_cache = None
    
    def initialize_from_mdtraj(self, instance: DaskMDTrajectory, mdtraj: md.Trajectory,
                               zarr_cache_path: Optional[str] = None,
                               chunk_size: int = 1000,
                               n_workers: Optional[int] = None) -> None:
        """
        Initialize DaskMDTrajectory instance from MDTraj trajectory object.
        
        Parameters
        ----------
        instance : DaskMDTrajectory
            Instance to initialize
        mdtraj : md.Trajectory
            MDTraj trajectory object
        zarr_cache_path : str, optional
            Path for zarr cache. If None, creates temporary cache.
        chunk_size : int, default=1000
            Number of frames per chunk
        n_workers : int, optional
            Number of parallel workers

        Returns
        -------
        None
            Initializes instance in-place
        """
        # Set basic trajectory info (no actual files)
        instance.trajectory_file = f"<mdtraj_object_{id(mdtraj)}>"
        instance.topology_file = None
        instance.chunk_size = chunk_size
        instance.n_workers = n_workers if n_workers is not None else multiprocessing.cpu_count()
        
        # Extract cache directory from zarr_cache_path and set as instance property
        if zarr_cache_path:
            instance._cache_dir = os.path.dirname(zarr_cache_path)
        else:
            instance._cache_dir = './cache'
        
        # Initialize cache manager with cache_dir
        instance.cache_manager = ZarrCacheHelper(chunk_size=chunk_size, cache_dir=instance._cache_dir)
        
        # Create cache from mdtraj object
        instance.zarr_cache_path, instance.metadata = instance.cache_manager.create_cache_from_mdtraj(
            mdtraj, zarr_cache_path
        )
        
        # Track if this is a temporary cache for cleanup
        if zarr_cache_path is None:
            instance._is_temp_store = True
            instance._temp_zarr_path = instance.zarr_cache_path
        else:
            instance._is_temp_store = False
            instance._temp_zarr_path = None
        
        # Open Zarr store
        instance._zarr_store = zarr.open(instance.zarr_cache_path, mode='r')
        
        # Load topology directly from store
        instance._topology = ZarrCacheHelper.load_topology(instance._zarr_store)
        
        # Create Dask arrays
        instance._dask_coords = da.from_zarr(instance.zarr_cache_path, component='coordinates')
        instance._dask_time = da.from_zarr(instance.zarr_cache_path, component='time')
        
        # Optional unitcell data
        instance._have_unitcell = 'unitcell_vectors' in instance._zarr_store
        if instance._have_unitcell:
            instance._dask_unitcell_vectors = da.from_zarr(instance.zarr_cache_path, component='unitcell_vectors')
            instance._dask_unitcell_lengths = da.from_zarr(instance.zarr_cache_path, component='unitcell_lengths')
            instance._dask_unitcell_angles = da.from_zarr(instance.zarr_cache_path, component='unitcell_angles')
        
        # Initialize parallel operations with cache_dir (already set as instance property)
        instance._parallel_ops = ParallelOperationsHelper(
            zarr_path=instance.zarr_cache_path, 
            topology=instance._topology, 
            n_workers=instance.n_workers, 
            chunk_size=chunk_size,
            cache_dir=instance._cache_dir
        )
        
        # Cache for frequently accessed data
        instance._xyz_cache = None
        instance._time_cache = None
        
    def create_from_zarr_store(self, zarr_store: zarr.Group, 
                              chunk_size: int = 1000,
                              n_workers: Optional[int] = None) -> dict:
        """
        Prepare configuration data from existing Zarr store.
        
        Parameters
        ----------
        zarr_store : zarr.Group
            Zarr group containing trajectory data
        chunk_size : int, default=1000
            Chunk size for new instance
        n_workers : int, optional
            Number of workers for parallel operations
            
        Returns
        -------
        dict
            Configuration dictionary for DaskMDTrajectory creation
            
        Examples
        --------
        >>> builder = DaskMDTrajectoryBuildHelper()
        >>> config = builder.create_from_zarr_store(zarr_store)
        >>> # Use config to create instance in DaskMDTrajectory
        """        
        # Extract metadata and topology
        metadata = dict(zarr_store.attrs['metadata'])
        topology = ZarrCacheHelper.load_topology(zarr_store)
        
        # Setup basic properties
        n_workers_final = n_workers if n_workers is not None else multiprocessing.cpu_count()
        
        # Extract zarr path from store - all file-based stores have paths
        zarr_path = self._extract_path_from_store(zarr_store)
        
        # Return configuration data for DaskMDTrajectory to create instance
        return {
            'zarr_store': zarr_store,
            'metadata': metadata,
            'topology': topology,
            'chunk_size': chunk_size,
            'n_workers': n_workers_final,
            'zarr_path': zarr_path,
            'has_unitcell': 'unitcell_vectors' in zarr_store
        }
    
    def _extract_path_from_store(self, zarr_store: zarr.Group) -> str:
        """
        Extract usable path from zarr store.
        
        Parameters
        ----------
        zarr_store : zarr.Group
            Zarr group to extract path from
            
        Returns
        -------
        str
            File system path for da.from_zarr
            
        Raises
        ------
        ValueError
            If store has no extractable file system path
        """
        # Check if store has a string representation with file path
        store_str = str(zarr_store)
        if 'file://' in store_str:
            # Extract path from file:// URL
            path = store_str.replace('<Group file://', '').replace('>', '')
            return path
        
        # Try to get path from store attributes
        if hasattr(zarr_store, 'store') and hasattr(zarr_store.store, 'path'):
            return str(zarr_store.store.path)
        
        # All our stores should be file-based, if not it's an error
        raise ValueError(f"Cannot extract path from zarr store: {type(zarr_store.store)}")

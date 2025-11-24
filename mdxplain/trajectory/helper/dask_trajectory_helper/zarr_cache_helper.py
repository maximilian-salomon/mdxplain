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
Zarr cache manager for memory-efficient trajectory storage.

Handles conversion from trajectory files to optimized Zarr format using md.iterload().
"""

import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mdtraj as md
import numpy as np
import zarr
from mdxplain.utils.progress_util import ProgressController
from zarr.codecs import BloscCodec

# Default compression for all zarr operations  
DEFAULT_COMPRESSOR = BloscCodec(cname='lz4', clevel=1)


class ZarrCacheHelper:
    """
    Manages Zarr cache files for efficient trajectory storage and access.
    
    Uses md.iterload() for memory-efficient conversion and stores trajectories
    in optimized Zarr format with 1000-frame chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, compression: str = 'lz4', cache_dir: str = './cache'):
        """
        Initialize Zarr cache manager.
        
        Parameters
        ----------
        chunk_size : int, default=1000
            Number of frames per chunk (optimized for DaskZarr)
        compression : str, default='lz4'
            Compression algorithm for Zarr storage
        cache_dir : str, default='./cache'
            Default cache directory for zarr files

        Returns
        -------
        None
            Initializes Zarr cache manager
        """
        self.chunk_size = chunk_size
        self.compression = compression
        self.cache_dir = cache_dir
        
    def get_cache_path(self, trajectory_file: str, cache_dir: Optional[str] = None) -> str:
        """
        Generate cache path for trajectory file.
        
        Parameters
        ----------
        trajectory_file : str
            Path to trajectory file
        cache_dir : str, optional
            Directory for cache files (default: ./cache)
            
        Returns
        -------
        str
            Path to Zarr cache file
            
        Examples
        --------
        >>> cache_manager = ZarrCacheHelper()
        >>> # Default cache directory
        >>> path = cache_manager.get_cache_path('/data/traj.xtc')
        >>> print(path)  # './cache/traj.dask.zarr'
        >>> # Custom cache directory
        >>> path = cache_manager.get_cache_path('/data/traj.xtc', '/tmp/cache')
        >>> print(path)  # '/tmp/cache/traj.dask.zarr'
        """
        # Determine cache directory
        if cache_dir is None:
            cache_dir = self.cache_dir
        
        # Generate cache filename
        traj_name = Path(trajectory_file).stem
        cache_filename = f"{traj_name}.dask.zarr"
        
        return os.path.join(cache_dir, cache_filename)
        
    def cache_exists(self, cache_path: str) -> bool:
        """
        Check if valid Zarr cache exists.
        
        Parameters
        ----------
        cache_path : str
            Path to Zarr cache file
            
        Returns
        -------
        bool
            True if valid cache exists
            
        Raises
        ------
        Exception
            If cache file is corrupted or unreadable (errors bubble up for debugging)
            
        Examples
        --------
        >>> cache_manager = ZarrCacheHelper()
        >>> cache_path = './cache/trajectory.dask.zarr'
        >>> if cache_manager.cache_exists(cache_path):
        ...     print("Cache found!")
        ... else:
        ...     print("Need to create cache")
        """
        if not os.path.exists(cache_path):
            return False
            
        # Try to open and validate cache - let errors bubble up for debugging
        store = zarr.open(cache_path, mode='r')
        if 'coordinates' not in store:
            return False
        if 'metadata' not in store.attrs:
            return False
        return True
    
    def create_cache(self, trajectory_file: str, topology_file: Optional[str], 
                    cache_path: str) -> Dict[str, Any]:
        """
        Create Zarr cache from trajectory file using md.iterload().
        
        Parameters
        ----------
        trajectory_file : str
            Path to trajectory file
        topology_file : str, optional
            Path to topology file
        cache_path : str
            Path for Zarr cache file
            
        Returns
        -------
        dict
            Metadata about the cached trajectory
            
        Raises
        ------
        FileNotFoundError
            If trajectory_file or topology_file doesn't exist
        ValueError
            If trajectory file format is unsupported
        OSError
            If cache directory is not writable
            
        Examples
        --------
        >>> cache_manager = ZarrCacheHelper(chunk_size=1000)
        >>> metadata = cache_manager.create_cache(
        ...     'trajectory.xtc', 'topology.pdb', 'cache/traj.zarr'
        ... )
        >>> print(f"Cached {metadata['n_frames']} frames")
        """
        print(f"📦 Creating Zarr cache: {cache_path}")
        
        # Load topology
        topology = self._load_topology(trajectory_file, topology_file)
        
        # Analyze trajectory dimensions
        traj_info = self._analyze_trajectory(trajectory_file, topology)
        
        # Create zarr store with arrays
        store = self._create_zarr_store(cache_path, traj_info)
        
        # Fill store with trajectory data
        self._fill_zarr_store(trajectory_file, topology, store, traj_info)
        
        # Store metadata and topology
        metadata = self._create_metadata(trajectory_file, topology_file, traj_info, topology)
        self._store_final_data(store, metadata, topology)
        
        print(f"  ✅ Cache created: {cache_path}")
        print(f"  📊 Size: {self._get_cache_size(cache_path):.1f} MB")
        
        return metadata
    
    def _load_topology(self, trajectory_file: str, topology_file: Optional[str]) -> md.Topology:
        """
        Load topology from file or trajectory.
        
        Parameters
        ----------
        trajectory_file : str
            Path to trajectory file
        topology_file : Optional[str]
            Path to topology file, or None to extract from trajectory
        
        Returns
        -------
        md.Topology
            MDTraj topology object
        """
        if topology_file:
            return md.load_topology(topology_file)
        else:
            temp_traj = md.load_frame(trajectory_file, 0)
            return temp_traj.topology
    
    def _analyze_trajectory(self, trajectory_file: str, topology: md.Topology) -> dict:
        """
        Analyze trajectory dimensions and properties.
        
        Parameters
        ----------
        trajectory_file : str
            Path to trajectory file
        topology : md.Topology
            MDTraj topology object
        
        Returns
        -------
        dict
            Dictionary with trajectory info (n_frames, n_atoms, has_unitcell)
        """
        print("  🔍 Analyzing trajectory dimensions...")
        n_frames = 0
        frame_iter = md.iterload(trajectory_file, top=topology, chunk=self.chunk_size)
        for chunk in ProgressController.iterate(
            frame_iter, desc="Counting frames"
        ):
            n_frames += chunk.n_frames
        
        traj_info = {
            'n_frames': n_frames,
            'n_atoms': chunk.n_atoms,
            'has_unitcell': chunk.unitcell_vectors is not None
        }
        
        print(f"  📐 Trajectory: {n_frames} frames × {chunk.n_atoms} atoms")
        return traj_info
    
    def _create_zarr_store(self, cache_path: str, traj_info: dict) -> zarr.Group:
        """
        Create zarr store with all required arrays.
        
        Parameters
        ----------
        cache_path : str
            Path for the new zarr store
        traj_info : dict
            Dictionary with trajectory info (n_frames, n_atoms, has_unitcell)
            
        Returns
        -------
        zarr.Group
            Newly created zarr store with coordinate and time arrays
        """
        store = zarr.open(cache_path, mode='w')
        compressor = DEFAULT_COMPRESSOR
        
        # Create coordinate and time arrays
        store.create_array(
            'coordinates',
            shape=(traj_info['n_frames'], traj_info['n_atoms'], 3),
            chunks=(self.chunk_size, traj_info['n_atoms'], 3),
            dtype=np.float32,
            compressors=compressor
        )
        
        store.create_array(
            'time',
            shape=(traj_info['n_frames'],),
            chunks=(self.chunk_size,),
            dtype=np.float32,
            compressors=compressor
        )
        
        # Create unitcell arrays if needed
        if traj_info['has_unitcell']:
            self._create_unitcell_arrays(store, traj_info, compressor)
            
        return store
    
    def _create_unitcell_arrays(self, store: zarr.Group, traj_info: Dict[str, Any], compressor: BloscCodec) -> None:
        """
        Create unitcell arrays in zarr store.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to create arrays in
        traj_info : dict
            Dictionary with trajectory dimensions
        compressor : object
            Compression codec for the arrays
            
        Returns
        -------
        None
            Creates unitcell_vectors, unitcell_lengths, unitcell_angles arrays
        """
        for array_name, shape_suffix in [('unitcell_vectors', (3, 3)), 
                                         ('unitcell_lengths', (3,)), 
                                         ('unitcell_angles', (3,))]:
            store.create_array(
                array_name,
                shape=(traj_info['n_frames'],) + shape_suffix,
                chunks=(self.chunk_size,) + shape_suffix,
                dtype=np.float32,
                compressors=compressor
            )
    
    def _fill_zarr_store(self, trajectory_file: str, topology: md.Topology, 
                        store: zarr.Group, traj_info: dict) -> None:
        """
        Fill zarr store with trajectory data.
        
        Parameters
        ----------
        trajectory_file : str
            Path to source trajectory file
        topology : md.Topology
            MDTraj topology object
        store : zarr.Group
            Target zarr store to fill with data
        traj_info : dict
            Dictionary with trajectory dimensions
            
        Returns
        -------
        None
            Fills store with trajectory data using md.iterload
        """
        print("  💾 Writing trajectory data...")
        frame_idx = 0
        
        frame_iter = md.iterload(trajectory_file, top=topology, chunk=self.chunk_size)
        total_chunks = (traj_info['n_frames'] + self.chunk_size - 1) // self.chunk_size
        
        for chunk in ProgressController.iterate(
            frame_iter, desc="Processing chunks", total=total_chunks
        ):
            end_idx = frame_idx + chunk.n_frames
            
            self._store_chunk_data(store, chunk, frame_idx, end_idx)
            frame_idx = end_idx
    
    def _store_chunk_data(self, store: zarr.Group, chunk: md.Trajectory, frame_idx: int, end_idx: int) -> None:
        """
        Store single chunk data to zarr arrays.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to write data to
        chunk : md.Trajectory
            MDTraj trajectory chunk with data
        frame_idx : int
            Starting frame index for this chunk
        end_idx : int
            Ending frame index for this chunk
            
        Returns
        -------
        None
            Stores chunk coordinate, time, and unitcell data to zarr arrays
        """
        store['coordinates'][frame_idx:end_idx] = chunk.xyz.astype(np.float32)
        
        # Handle time data
        if chunk.time is not None:
            store['time'][frame_idx:end_idx] = chunk.time.astype(np.float32)
        else:
            store['time'][frame_idx:end_idx] = np.arange(frame_idx, end_idx, dtype=np.float32)
        
        # Handle unitcell data
        if chunk.unitcell_vectors is not None:
            store['unitcell_vectors'][frame_idx:end_idx] = chunk.unitcell_vectors.astype(np.float32)
            store['unitcell_lengths'][frame_idx:end_idx] = chunk.unitcell_lengths.astype(np.float32)
            store['unitcell_angles'][frame_idx:end_idx] = chunk.unitcell_angles.astype(np.float32)
    
    def _create_metadata(self, trajectory_file: str, topology_file: Optional[str], 
                        traj_info: dict, topology: md.Topology) -> dict:
        """
        Create metadata dictionary for trajectory.
        
        Parameters
        ----------
        trajectory_file : str
            Path to source trajectory file
        topology_file : str, optional
            Path to topology file
        traj_info : dict
            Dictionary with trajectory dimensions
        topology : md.Topology
            MDTraj topology object
            
        Returns
        -------
        dict
            Metadata dictionary with trajectory information
        """
        return {
            'n_frames': traj_info['n_frames'],
            'n_atoms': traj_info['n_atoms'],
            'n_residues': topology.n_residues,
            'chunk_size': self.chunk_size,
            'trajectory_file': os.path.abspath(trajectory_file),
            'topology_file': os.path.abspath(topology_file) if topology_file else None,
            'created_time': time.time(),
            'has_unitcell': traj_info['has_unitcell']
        }
    
    def _store_final_data(self, store: zarr.Group, metadata: dict, topology: md.Topology) -> None:
        """
        Store metadata and topology to zarr store.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to write final data to
        metadata : dict
            Trajectory metadata dictionary
        topology : md.Topology
            MDTraj topology object to store
            
        Returns
        -------
        None
            Stores metadata attributes and topology in store
        """
        store.attrs['metadata'] = metadata
        compressor = DEFAULT_COMPRESSOR
        self.store_topology(store, topology, compressor)
    
    def load_cache_metadata(self, cache_path: str) -> Dict[str, Any]:
        """
        Load metadata from Zarr cache.
        
        Parameters
        ----------
        cache_path : str
            Path to Zarr cache file
            
        Returns
        -------
        dict
            Trajectory metadata
            
        Raises
        ------
        FileNotFoundError
            If cache file doesn't exist
        KeyError
            If metadata is missing from cache file
            
        Examples
        --------
        >>> cache_manager = ZarrCacheHelper()
        >>> metadata = cache_manager.load_cache_metadata('cache/traj.zarr')
        >>> print(f"Cache contains {metadata['n_frames']} frames")
        """
        store = zarr.open(cache_path, mode='r')
        return dict(store.attrs['metadata'])
    
    def get_or_create_cache(self, trajectory_file: str, topology_file: Optional[str] = None,
                           cache_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get existing cache or create new one.
        
        Parameters
        ----------
        trajectory_file : str
            Path to trajectory file
        topology_file : str, optional
            Path to topology file
        cache_path : str, optional
            Custom cache path
            
        Returns
        -------
        tuple
            (cache_path, metadata)
            
        Raises
        ------
        FileNotFoundError
            If trajectory_file doesn't exist
        ValueError
            If trajectory file format is unsupported
        OSError
            If cache directory is not writable
            
        Examples
        --------
        >>> cache_manager = ZarrCacheHelper()
        >>> cache_path, metadata = cache_manager.get_or_create_cache(
        ...     'trajectory.xtc', 'topology.pdb'
        ... )
        >>> print(f"Using cache at {cache_path}")
        >>> print(f"Contains {metadata['n_frames']} frames")
        """
        if cache_path is None:
            cache_path = self.get_cache_path(trajectory_file)
        
        if self.cache_exists(cache_path):
            print(f"📁 Using existing cache: {cache_path}")
            metadata = self.load_cache_metadata(cache_path)
        else:
            metadata = self.create_cache(trajectory_file, topology_file, cache_path)
        
        return cache_path, metadata
    
    def _get_cache_size(self, cache_path: str) -> float:
        """
        Get cache file size in MB.
        
        Parameters
        ----------
        cache_path : str
            Path to cache directory or file
            
        Returns
        -------
        float
            Total cache size in megabytes
        """
        if os.path.isdir(cache_path):
            total_size = 0
            for dirpath, _, filenames in os.walk(cache_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024**2)
        else:
            return os.path.getsize(cache_path) / (1024**2)
    
    @staticmethod
    def store_topology(store: zarr.Group, topology: md.Topology, compressor: Optional[Any] = None) -> None:
        """
        Store topology in Zarr store using pickle serialization.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to store topology in
        topology : md.Topology
            MDTraj topology object to serialize and store
        compressor : object, optional
            Compression codec for topology storage
            
        Returns
        -------
        None
            Stores topology as pickled array in zarr store
            
        Examples
        --------
        >>> import zarr
        >>> import mdtraj as md
        >>> store = zarr.open('trajectory.zarr', mode='w')
        >>> topology = md.load_topology('protein.pdb')
        >>> ZarrCacheHelper.store_topology(store, topology)
        """
        topology_bytes = pickle.dumps(topology)
        
        # Create array for topology storage
        topology_data = np.frombuffer(topology_bytes, dtype=np.uint8)
        store.create_array(
            'topology_pickle',
            data=topology_data,
            chunks=(len(topology_data),),  # Single chunk for topology
            compressors=compressor
        )
    
    @staticmethod  
    def load_topology(store: zarr.Group) -> md.Topology:
        """
        Load topology from Zarr store.
        
        Parameters
        ----------
        store : zarr.Group
            Zarr store containing pickled topology data
            
        Returns
        -------
        md.Topology
            Loaded MDTraj topology object
            
        Examples
        --------
        >>> import zarr
        >>> store = zarr.open('trajectory.zarr', mode='r')
        >>> topology = ZarrCacheHelper.load_topology(store)
        >>> print(f\"Loaded {topology.n_atoms} atoms\")
        """
        # Load topology from array 
        topology_uint8 = store['topology_pickle'][:]
        topology_bytes = topology_uint8.tobytes()        
        return pickle.loads(topology_bytes)
    
    def create_cache_from_mdtraj(self, mdtraj: 'md.Trajectory', 
                                 cache_path: Optional[str] = None) -> Tuple[str, dict]:
        """
        Create Zarr cache directly from MDTraj trajectory object.
        
        Parameters
        ----------
        mdtraj : md.Trajectory
            MDTraj trajectory object to cache
        cache_path : str, optional
            Path for cache. If None, creates temporary cache.
            
        Returns
        -------
        tuple
            (cache_path, metadata_dict) containing cache location and info
            
        Examples
        --------
        >>> import mdtraj as md
        >>> traj = md.load('trajectory.xtc', top='topology.pdb')
        >>> cache_helper = ZarrCacheHelper()
        >>> cache_path, metadata = cache_helper.create_cache_from_mdtraj(traj)
        """       
        if cache_path is None:
            # Create temporary cache
            temp_dir = tempfile.mkdtemp(prefix='dask_traj_', suffix='.zarr')
            cache_path = temp_dir
        
        print(f"📦 Creating Zarr cache from MDTraj: {cache_path}")
        
        # Create trajectory info from mdtraj
        traj_info = {
            'n_frames': mdtraj.n_frames,
            'n_atoms': mdtraj.n_atoms,
            'has_unitcell': mdtraj.unitcell_vectors is not None
        }
        print(f"  📐 Trajectory: {traj_info['n_frames']} frames × {traj_info['n_atoms']} atoms")
        
        # Create zarr store
        store = self._create_zarr_store(cache_path, traj_info)
        
        # Write data directly from mdtraj
        print("  💾 Writing trajectory data...")
        store['coordinates'][:] = mdtraj.xyz
        store['time'][:] = mdtraj.time if mdtraj.time is not None else np.arange(mdtraj.n_frames)
        
        # Write unitcell data if present
        if traj_info['has_unitcell']:
            store['unitcell_vectors'][:] = mdtraj.unitcell_vectors
            store['unitcell_lengths'][:] = mdtraj.unitcell_lengths
            store['unitcell_angles'][:] = mdtraj.unitcell_angles
        
        # Save topology 
        self.store_topology(store, mdtraj.topology)
        
        # Create metadata
        metadata = {
            'n_frames': traj_info['n_frames'],
            'n_atoms': traj_info['n_atoms'],
            'has_unitcell': traj_info['has_unitcell'],
            'source': 'mdtraj_conversion',
            'chunk_size': self.chunk_size
        }
        
        # Save metadata
        store.attrs['metadata'] = metadata
        
        # Calculate cache size
        cache_size_mb = self._get_cache_size(cache_path)
        print(f"  ✅ Cache created: {cache_path}")
        print(f"  📊 Size: {cache_size_mb:.1f} MB")
        
        return cache_path, metadata

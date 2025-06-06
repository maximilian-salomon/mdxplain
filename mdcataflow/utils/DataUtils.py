"""
DataUtils - Utility functions for saving and loading TrajectoryData objects

Author: Maximilian Salomon
Version: 0.1.0
Created with assistance from Claude-4-Sonnet and Cursor AI.
"""

import os
import numpy as np


class DataUtils:
    """
    Utility class for saving and loading TrajectoryData objects.
    Preserves memmap properties correctly.
    """
    
    @staticmethod
    def save_trajectory_data(trajectory_data, save_path):
        """
        Save a TrajectoryData object preserving memmap properties.
        
        Parameters:
        -----------
        trajectory_data : TrajectoryData
            The TrajectoryData object to save
        save_path : str
            Path where to save the object (should end with .npy)
        """
        # Prepare object copy with memmap info
        save_obj = DataUtils._prepare_save_object(trajectory_data)
        
        # Save with numpy
        np.save(save_path, save_obj, allow_pickle=True)

    @staticmethod
    def load_trajectory_data(trajectory_data, load_path):
        """
        Load data into an existing TrajectoryData object preserving memmap properties.
        
        Parameters:
        -----------
        trajectory_data : TrajectoryData
            The TrajectoryData object to load data into
        load_path : str
            Path to the saved TrajectoryData .npy file
        """
        # Load the saved object
        loaded_obj = np.load(load_path, allow_pickle=True).item()
        
        # Restore all attributes
        DataUtils._restore_object_attributes(trajectory_data, loaded_obj)

    @staticmethod
    def _prepare_save_object(trajectory_data):
        """Prepare object for saving, handling memmaps specially."""
        save_obj = {}
        
        # Copy all attributes
        for attr_name in dir(trajectory_data):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(trajectory_data, attr_name)
            
            # Handle memmaps specially
            if isinstance(attr_value, np.memmap):
                save_obj[attr_name] = DataUtils._save_memmap_info(attr_value, attr_name)
            else:
                save_obj[attr_name] = attr_value
        
        return save_obj

    @staticmethod
    def _save_memmap_info(memmap_array, attr_name):
        """Save memmap info - every memmap must have a filename."""
        
        # Every memmap must have a filename
        if not (hasattr(memmap_array, 'filename') and memmap_array.filename):
            raise ValueError(f"Memmap {attr_name} has no filename - this should not happen!")
        
        # Memmap has a file - just save the metadata
        return {
            '_is_memmap': True,
            'dtype': memmap_array.dtype,
            'shape': memmap_array.shape,
            'mode': getattr(memmap_array, 'mode', 'r'),
            'original_path': memmap_array.filename
        }

    @staticmethod
    def _restore_object_attributes(trajectory_data, loaded_obj):
        """Restore object attributes, recreating memmaps."""
        for attr_name, attr_value in loaded_obj.items():
            
            # Check if this was a memmap
            if isinstance(attr_value, dict) and attr_value.get('_is_memmap', False):
                # Recreate memmap
                restored_memmap = DataUtils._restore_memmap(trajectory_data, attr_value, attr_name)
                setattr(trajectory_data, attr_name, restored_memmap)
            else:
                # Regular attribute
                setattr(trajectory_data, attr_name, attr_value)

    @staticmethod
    def _restore_memmap(trajectory_data, memmap_info, attr_name):
        """Restore a memmap from saved info."""
        original_path = memmap_info['original_path']
        
        # Check if original file still exists
        if os.path.exists(original_path):
            # Original file exists - use it directly
            print(f"Loading memmap {attr_name} from original file: {original_path}")
            return np.memmap(original_path, dtype=memmap_info['dtype'], 
                           mode='r', shape=tuple(memmap_info['shape']))
        
        # Original file doesn't exist - try to use trajectory's cache path if available
        if hasattr(trajectory_data, 'use_memmap') and trajectory_data.use_memmap:
            if hasattr(trajectory_data, f"{attr_name}_path"):
                target_path = getattr(trajectory_data, f"{attr_name}_path")
                
                # If target path is different from original and target exists, use it
                if target_path != original_path and os.path.exists(target_path):
                    print(f"Loading memmap {attr_name} from cache: {target_path}")
                    return np.memmap(target_path, dtype=memmap_info['dtype'], 
                                   mode='r', shape=tuple(memmap_info['shape']))
        
        # File not found anywhere
        print(f"Warning: Could not restore memmap {attr_name} - file {original_path} not found")
        return None 
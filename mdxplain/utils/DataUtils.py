# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# DataUtils - Utility functions for saving and loading Python objects with memmap support
#
# Utility class for saving and loading Python objects with memmap support.
# Works with any Python object, not just TrajectoryData.
# Preserves memmap properties correctly.
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
import numpy as np


class DataUtils:
    """
    Utility class for saving and loading Python objects with memmap support.
    Works with any Python object, not just TrajectoryData.
    Preserves memmap properties correctly.
    """
    
    @staticmethod
    def save_object(obj, save_path):
        """
        Save any Python object preserving memmap properties.
        
        Parameters:
        -----------
        obj : object
            Any Python object to save
        save_path : str
            Path where to save the object (should end with .npy)
        """
        save_obj = DataUtils._prepare_save_object(obj)
        np.save(save_path, save_obj, allow_pickle=True)

    @staticmethod
    def load_object(obj, load_path):
        """
        Load data into any existing Python object preserving memmap properties.
        
        Parameters:
        -----------
        obj : object
            Any Python object to load data into
        load_path : str
            Path to the saved object .npy file
        """
        loaded_obj = np.load(load_path, allow_pickle=True).item()
        DataUtils._restore_object_attributes(obj, loaded_obj)

    @staticmethod
    def _prepare_save_object(obj):
        """Prepare any object for saving, handling memmaps specially."""
        save_obj = {}
        
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(obj, attr_name)
            
            if isinstance(attr_value, np.memmap):
                save_obj[attr_name] = DataUtils._save_memmap_info(attr_value, attr_name)
            else:
                save_obj[attr_name] = attr_value
        
        return save_obj

    @staticmethod
    def _save_memmap_info(memmap_array, attr_name):
        """Save memmap info - every memmap must have a filename."""
        if not (hasattr(memmap_array, 'filename') and memmap_array.filename):
            raise ValueError(f"Memmap {attr_name} has no filename - this should not happen!")
        
        return {
            '_is_memmap': True,
            'dtype': memmap_array.dtype,
            'shape': memmap_array.shape,
            'mode': getattr(memmap_array, 'mode', 'r'),
            'original_path': memmap_array.filename
        }

    @staticmethod
    def _restore_object_attributes(obj, loaded_obj):
        """Restore object attributes, recreating memmaps."""
        for attr_name, attr_value in loaded_obj.items():
            if isinstance(attr_value, dict) and attr_value.get('_is_memmap', False):
                restored_memmap = DataUtils._restore_memmap(obj, attr_value, attr_name)
                setattr(obj, attr_name, restored_memmap)
            else:
                setattr(obj, attr_name, attr_value)

    @staticmethod
    def _restore_memmap(obj, memmap_info, attr_name):
        """Restore a memmap from saved info."""
        original_path = memmap_info['original_path']
        
        if os.path.exists(original_path):
            return np.memmap(original_path, dtype=memmap_info['dtype'], 
                           mode='r', shape=tuple(memmap_info['shape']))
        
        if hasattr(obj, 'use_memmap') and obj.use_memmap:
            if hasattr(obj, f"{attr_name}_path"):
                target_path = getattr(obj, f"{attr_name}_path")
                if target_path != original_path and os.path.exists(target_path):
                    return np.memmap(target_path, dtype=memmap_info['dtype'], 
                                   mode='r', shape=tuple(memmap_info['shape']))
        
        return None 
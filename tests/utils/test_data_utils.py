#!/usr/bin/env python3
# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# Unit tests for DataUtils public methods
#
# Comprehensive tests for DataUtils public API: save_object and load_object
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

import warnings
# Suppress pkg_resources warnings before any imports that might trigger them
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=DeprecationWarning)

import os
import pickle
import tempfile
import numpy as np
import pytest
from unittest.mock import Mock

from mdxplain.utils.data_utils import DataUtils


class SimpleTestObject:
    """Simple test object for DataUtils testing."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __dir__(self):
        return [attr for attr in self.__dict__.keys() if not attr.startswith('_')]


class MemmapTestObject:
    """Test object with memmap support."""
    
    def __init__(self, use_memmap=False, **kwargs):
        self.use_memmap = use_memmap
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __dir__(self):
        return [attr for attr in self.__dict__.keys() if not attr.startswith('_')]


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_object():
    """Object with basic Python types."""
    return SimpleTestObject(
        string_attr="test_string",
        int_attr=42,
        float_attr=3.14,
        list_attr=[1, 2, 3],
        dict_attr={"key": "value"},
        none_attr=None,
        bool_attr=True
    )


@pytest.fixture
def array_object():
    """Object with numpy arrays."""
    np.random.seed(42)
    return SimpleTestObject(
        small_array=np.array([1, 2, 3, 4, 5]),
        large_array=np.random.random((10, 5)),
        int_array=np.array([[1, 2], [3, 4]], dtype=np.int32),
        complex_array=np.array([1+2j, 3+4j])
    )


@pytest.fixture
def memmap_object(temp_dir):
    """Object with memory-mapped arrays."""
    # Create memmap files
    file1 = os.path.join(temp_dir, "test1.dat")
    file2 = os.path.join(temp_dir, "test2.dat")
    
    # Create test data and memmaps
    np.random.seed(42)
    memmap1 = np.memmap(file1, dtype=np.float64, mode='w+', shape=(20, 10))
    memmap1[:] = np.random.random((20, 10))
    memmap1.flush()
    
    memmap2 = np.memmap(file2, dtype=np.float32, mode='w+', shape=(5, 8))
    memmap2[:] = np.random.random((5, 8)).astype(np.float32)
    memmap2.flush()
    
    return MemmapTestObject(
        use_memmap=True,
        data1=memmap1,
        data2=memmap2,
        data1_path=file1,
        data2_path=file2,
        regular_attr="normal_value"
    )


class TestDataUtilsSaveObject:
    """Test DataUtils.save_object() method."""
    
    def test_save_object_creates_file(self, simple_object, temp_dir):
        """Test that saving creates the output file."""
        save_path = os.path.join(temp_dir, "test.npy")
        DataUtils.save_object(simple_object, save_path)
        assert os.path.exists(save_path)
    
    def test_save_object_with_private_attributes(self, temp_dir):
        """
        Test that private attributes are excluded from saving.
        
        Private attributes (_attr, __attr) are implementation details that:
        - May change between versions (API stability)
        - Often contain sensitive data (security)  
        - May be non-serializable (file handles, connections)
        - Can be large temporary data (performance)
        """
        obj = SimpleTestObject()
        obj.public_attr = "public"
        obj._private_attr = "private" 
        obj.__very_private = "very_private"
        
        save_path = os.path.join(temp_dir, "private.npy")
        DataUtils.save_object(obj, save_path)
        
        # Load and verify only public attributes are saved
        loaded_obj = SimpleTestObject()
        DataUtils.load_object(loaded_obj, save_path)
        
        assert hasattr(loaded_obj, 'public_attr')
        assert not hasattr(loaded_obj, '_private_attr')
        assert not hasattr(loaded_obj, '__very_private')
    
    def test_save_memmap_without_filename_raises_error(self, temp_dir):
        """Test that memmaps without filename raise ValueError."""
        obj = SimpleTestObject()
        
        # Create mock memmap without filename
        mock_memmap = Mock(spec=np.memmap)
        mock_memmap.filename = None
        mock_memmap.dtype = np.float32
        mock_memmap.shape = (5, 5)
        obj.bad_memmap = mock_memmap
        
        save_path = os.path.join(temp_dir, "bad_memmap.npy")
        
        with pytest.raises(ValueError, match="Memmap bad_memmap has no filename"):
            DataUtils.save_object(obj, save_path)

class TestDataUtilsLoadObject:
    """Test DataUtils.load_object() method."""
    
    def test_load_simple_object_preserves_all_attributes(self, simple_object, temp_dir):
        """Test that loading preserves all simple data types."""
        save_path = os.path.join(temp_dir, "simple.npy")
        DataUtils.save_object(simple_object, save_path)
        
        loaded_obj = SimpleTestObject()
        DataUtils.load_object(loaded_obj, save_path)
        
        assert loaded_obj.string_attr == "test_string"
        assert loaded_obj.int_attr == 42
        assert loaded_obj.float_attr == 3.14
        assert loaded_obj.list_attr == [1, 2, 3]
        assert loaded_obj.dict_attr == {"key": "value"}
        assert loaded_obj.none_attr is None
        assert loaded_obj.bool_attr is True
    
    def test_load_array_object_preserves_arrays(self, array_object, temp_dir):
        """Test that loading preserves numpy arrays correctly."""
        save_path = os.path.join(temp_dir, "arrays.npy")
        DataUtils.save_object(array_object, save_path)
        
        loaded_obj = SimpleTestObject()
        DataUtils.load_object(loaded_obj, save_path)
        
        assert np.array_equal(loaded_obj.small_array, array_object.small_array)
        assert np.allclose(loaded_obj.large_array, array_object.large_array)
        assert np.array_equal(loaded_obj.int_array, array_object.int_array)
        assert np.array_equal(loaded_obj.complex_array, array_object.complex_array)
    
    def test_load_memmap_object_preserves_memmaps(self, memmap_object, temp_dir):
        """Test that loading preserves memmaps as memmaps with correct data."""
        save_path = os.path.join(temp_dir, "memmaps.npy")
        original_data1 = np.array(memmap_object.data1)
        original_data2 = np.array(memmap_object.data2)
        
        DataUtils.save_object(memmap_object, save_path)
        
        loaded_obj = MemmapTestObject(use_memmap=True)
        loaded_obj.data1_path = memmap_object.data1_path
        loaded_obj.data2_path = memmap_object.data2_path
        DataUtils.load_object(loaded_obj, save_path)
        
        # Check memmap types are preserved
        assert isinstance(loaded_obj.data1, np.memmap)
        assert isinstance(loaded_obj.data2, np.memmap)
        
        # Check data integrity
        assert np.allclose(loaded_obj.data1, original_data1)
        assert np.allclose(loaded_obj.data2, original_data2)
        
        # Check regular attributes
        assert loaded_obj.regular_attr == "normal_value"
    
    def test_load_memmap_with_missing_file_returns_none(self, temp_dir):
        """Test that memmaps with missing files are set to None."""
        # Create object with memmap info but no actual file
        obj = MemmapTestObject(use_memmap=True)
        fake_memmap_info = {
            'missing_data': {
                '_is_memmap': True,
                'dtype': np.float64,
                'shape': (10, 10),
                'mode': 'r',
                'original_path': '/nonexistent/path.dat'
            }
        }
        
        save_path = os.path.join(temp_dir, "missing.npy")
        np.save(save_path, fake_memmap_info, allow_pickle=True)
        
        loaded_obj = MemmapTestObject(use_memmap=True)
        DataUtils.load_object(loaded_obj, save_path)
        
        assert loaded_obj.missing_data is None
    
    def test_load_memmap_with_fallback_path(self, temp_dir):
        """Test that memmaps can use fallback paths when original is missing."""
        # Create fallback file
        fallback_file = os.path.join(temp_dir, "fallback.dat")
        test_data = np.random.random((4, 5))
        fallback_memmap = np.memmap(fallback_file, dtype=np.float32, mode='w+', shape=(4, 5))
        fallback_memmap[:] = test_data.astype(np.float32)
        fallback_memmap.flush()
        
        # Create saved data with missing original path
        memmap_info = {
            'test_data': {
                '_is_memmap': True,
                'dtype': np.float32,
                'shape': (4, 5),
                'mode': 'r',
                'original_path': '/nonexistent/original.dat'
            }
        }
        
        save_path = os.path.join(temp_dir, "fallback_test.npy")
        np.save(save_path, memmap_info, allow_pickle=True)
        
        # Load with fallback path
        loaded_obj = MemmapTestObject(use_memmap=True)
        loaded_obj.test_data_path = fallback_file
        DataUtils.load_object(loaded_obj, save_path)
        
        assert isinstance(loaded_obj.test_data, np.memmap)
        assert np.allclose(loaded_obj.test_data, test_data.astype(np.float32))
    
    def test_load_nested_object_structure(self, temp_dir):
        """Test that loading preserves nested object structures."""
        inner_obj = SimpleTestObject(inner_attr="inner_value", inner_num=99)
        nested_obj = SimpleTestObject(
            outer_attr="outer_value",
            nested_object=inner_obj,
            nested_dict={"inner": {"deep": "value"}},
            nested_list=[{"a": 1}, {"b": 2}]
        )
        
        save_path = os.path.join(temp_dir, "nested.npy")
        DataUtils.save_object(nested_obj, save_path)
        
        loaded_obj = SimpleTestObject()
        DataUtils.load_object(loaded_obj, save_path)
        
        assert loaded_obj.outer_attr == "outer_value"
        assert loaded_obj.nested_object.inner_attr == "inner_value"
        assert loaded_obj.nested_object.inner_num == 99
        assert loaded_obj.nested_dict == {"inner": {"deep": "value"}}
        assert loaded_obj.nested_list == [{"a": 1}, {"b": 2}]
    
    def test_load_nonexistent_file_raises_error(self, temp_dir):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        obj = SimpleTestObject()
        nonexistent_path = os.path.join(temp_dir, "nonexistent.npy")
        
        with pytest.raises(FileNotFoundError):
            DataUtils.load_object(obj, nonexistent_path)
    
    def test_load_corrupted_file_raises_error(self, temp_dir):
        """Test that loading corrupted files raises appropriate error."""
        obj = SimpleTestObject()
        corrupted_path = os.path.join(temp_dir, "corrupted.npy")
        
        # Create corrupted file
        with open(corrupted_path, 'wb') as f:
            f.write(b"This is not a valid numpy file")
        
        with pytest.raises((ValueError, OSError, pickle.UnpicklingError)):
            DataUtils.load_object(obj, corrupted_path)
    
    def test_load_invalid_data_format_raises_error(self, temp_dir):
        """Test that loading file with wrong data format raises error."""
        obj = SimpleTestObject()
        invalid_path = os.path.join(temp_dir, "invalid.npy")
        
        # Save array instead of dictionary
        np.save(invalid_path, np.array([1, 2, 3]), allow_pickle=True)
        
        with pytest.raises(ValueError):
            DataUtils.load_object(obj, invalid_path)
    
    def test_load_memmap_creation_fails_gracefully(self, temp_dir):
        """Test that memmap creation failure is handled gracefully."""
        obj = MemmapTestObject(use_memmap=True)
        invalid_memmap_path = os.path.join(temp_dir, "invalid_memmap.npy")
        
        # Create memmap info with invalid dtype
        invalid_memmap_info = {
            'test_data': {
                '_is_memmap': True,
                'dtype': 'invalid_dtype',  # Invalid dtype
                'shape': (10, 10),
                'mode': 'r',
                'original_path': '/nonexistent/path.dat'
            }
        }
        
        np.save(invalid_memmap_path, invalid_memmap_info, allow_pickle=True)
        
        # Should not crash, but set attribute to None
        DataUtils.load_object(obj, invalid_memmap_path)
        assert obj.test_data is None
    
    def test_load_overwrites_existing_attributes(self, temp_dir):
        """Test that loading overwrites existing object attributes."""
        obj_with_existing = SimpleTestObject(existing_attr="old_value", new_attr="will_be_overwritten")
        
        # Save different data
        save_data = {"existing_attr": "new_value", "additional_attr": "added"}
        save_path = os.path.join(temp_dir, "overwrite.npy")
        np.save(save_path, save_data, allow_pickle=True)
        
        DataUtils.load_object(obj_with_existing, save_path)
        
        assert obj_with_existing.existing_attr == "new_value"
        assert obj_with_existing.additional_attr == "added"
        # Note: new_attr will still exist as it's not removed, only added to

class TestDataUtilsIntegration:
    """Integration tests for complete save/load workflows."""
    
    def test_round_trip_preserves_all_data(self, simple_object, temp_dir):
        """Test that complete save/load cycle preserves all data perfectly."""
        save_path = os.path.join(temp_dir, "roundtrip.npy")
        
        # Save
        DataUtils.save_object(simple_object, save_path)
        
        # Load
        loaded_obj = SimpleTestObject()
        DataUtils.load_object(loaded_obj, save_path)
        
        # Verify every attribute matches exactly
        for attr_name in dir(simple_object):
            if not attr_name.startswith('_'):
                original_value = getattr(simple_object, attr_name)
                loaded_value = getattr(loaded_obj, attr_name)
                assert loaded_value == original_value
    
    def test_multiple_save_load_cycles_no_corruption(self, simple_object, temp_dir):
        """Test that multiple save/load cycles don't corrupt data."""
        save_path = os.path.join(temp_dir, "multi_cycle.npy")
        current_obj = simple_object
        
        for _ in range(3):
            # Save current object
            DataUtils.save_object(current_obj, save_path)
            
            # Load into new object
            loaded_obj = SimpleTestObject()
            DataUtils.load_object(loaded_obj, save_path)
            
            # Verify data integrity
            assert loaded_obj.string_attr == "test_string"
            assert loaded_obj.int_attr == 42
            assert loaded_obj.list_attr == [1, 2, 3]
            
            current_obj = loaded_obj
    
    def test_loaded_objects_are_independent_copies(self, temp_dir):
        """Test that loaded objects are independent copies, not references."""
        obj1 = SimpleTestObject(mutable_list=[1, 2, 3])
        save_path = os.path.join(temp_dir, "independence.npy")
        
        # Save and load
        DataUtils.save_object(obj1, save_path)
        obj2 = SimpleTestObject()
        DataUtils.load_object(obj2, save_path)
        
        # Modify original object
        obj1.mutable_list.append(4)
        
        # Verify loaded object is unaffected
        assert obj2.mutable_list == [1, 2, 3]
        assert obj1.mutable_list == [1, 2, 3, 4]

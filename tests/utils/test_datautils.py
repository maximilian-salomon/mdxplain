#!/usr/bin/env python3
"""
Unit tests for DataUtils save/load functionality.
"""

import warnings
# Suppress pkg_resources warnings before any imports that might trigger them
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=DeprecationWarning)

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import Mock, patch

from mdcataflow.utils.DataUtils import DataUtils


class SimpleMockTrajectory:
    """Simple mock trajectory class that works with DataUtils."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __dir__(self):
        return [attr for attr in self.__dict__.keys() if not attr.startswith('_')]


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory with automatic cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_trajectory():
    """Fixture providing a mock trajectory object with regular arrays."""
    np.random.seed(42)  # Use fixed seed for reproducible tests
    return SimpleMockTrajectory(
        use_memmap=False,
        trajectories=["traj1", "traj2"],
        res_list=[1, 2, 3, 4, 5],
        distances=np.random.random((10, 5)),
        contacts=np.random.random((5, 5))
    )


@pytest.fixture
def memmap_files(temp_dir):
    """Fixture creating temporary memmap files with automatic cleanup."""
    cache_dir = os.path.join(temp_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    distances_file = os.path.join(cache_dir, "test_distances.dat")
    contacts_file = os.path.join(cache_dir, "test_contacts.dat")
    
    # Create test memmaps with reproducible data
    np.random.seed(42)
    distances_data = np.random.random((20, 10))
    contacts_data = np.random.random((10, 10))
    
    distances_memmap = np.memmap(distances_file, dtype='float64', mode='w+', shape=(20, 10))
    distances_memmap[:] = distances_data
    distances_memmap.flush()
    
    contacts_memmap = np.memmap(contacts_file, dtype='float64', mode='w+', shape=(10, 10))
    contacts_memmap[:] = contacts_data
    contacts_memmap.flush()
    
    yield {
        'distances_file': distances_file,
        'contacts_file': contacts_file,
        'distances_memmap': distances_memmap,
        'contacts_memmap': contacts_memmap,
        'cache_dir': cache_dir
    }
    
    # Proper memmap cleanup: close file handles before temp dir cleanup
    distances_memmap.flush()
    contacts_memmap.flush()


@pytest.fixture
def mock_memmap_trajectory(memmap_files):
    """Fixture providing a mock trajectory with memmaps."""
    return SimpleMockTrajectory(
        use_memmap=True,
        trajectories=["traj1", "traj2"],
        res_list=[1, 2, 3, 4, 5],
        distances=memmap_files['distances_memmap'],
        contacts=memmap_files['contacts_memmap'],
        distances_path=os.path.join(memmap_files['cache_dir'], "distances.dat"),
        contacts_path=os.path.join(memmap_files['cache_dir'], "contacts.dat")
    )


class TestDataUtilsRegularArrays:
    """Test class for regular numpy arrays functionality."""
    
    def test_save_creates_main_file(self, mock_trajectory, temp_dir):
        """Test that saving regular arrays creates the main .npy file."""
        save_path = os.path.join(temp_dir, "test_regular.npy")
        DataUtils.save_object(mock_trajectory, save_path)
        assert os.path.exists(save_path)
    
    def test_save_does_not_create_data_directory(self, mock_trajectory, temp_dir):
        """Test that saving regular arrays does not create data directory."""
        save_path = os.path.join(temp_dir, "test_regular.npy")
        DataUtils.save_object(mock_trajectory, save_path)
        data_dir = os.path.splitext(save_path)[0] + "_data"
        assert not os.path.exists(data_dir)
    
    def test_load_preserves_trajectories(self, mock_trajectory, temp_dir):
        """Test that loading preserves trajectories list."""
        save_path = os.path.join(temp_dir, "test_regular.npy")
        DataUtils.save_object(mock_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.trajectories == mock_trajectory.trajectories
    
    def test_load_preserves_res_list(self, mock_trajectory, temp_dir):
        """Test that loading preserves res_list."""
        save_path = os.path.join(temp_dir, "test_regular.npy")
        DataUtils.save_object(mock_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.res_list == mock_trajectory.res_list
    
    def test_load_preserves_distances_data(self, mock_trajectory, temp_dir):
        """Test that loading preserves distances array data."""
        save_path = os.path.join(temp_dir, "test_regular.npy")
        DataUtils.save_object(mock_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert np.allclose(loaded_traj.distances, mock_trajectory.distances)
    
    def test_load_preserves_contacts_data(self, mock_trajectory, temp_dir):
        """Test that loading preserves contacts array data."""
        save_path = os.path.join(temp_dir, "test_regular.npy")
        DataUtils.save_object(mock_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert np.allclose(loaded_traj.contacts, mock_trajectory.contacts)
    
    def test_load_nonexistent_file_raises_error(self, temp_dir):
        """Test proper error handling for missing files."""
        traj = SimpleMockTrajectory()
        nonexistent_path = os.path.join(temp_dir, "nonexistent.npy")
        
        with pytest.raises(FileNotFoundError):
            DataUtils.load_object(traj, nonexistent_path)


class TestDataUtilsMemmaps:
    """Test class for memmap functionality."""
    
    def test_save_creates_main_file(self, mock_memmap_trajectory, temp_dir):
        """Test that saving memmaps creates main .npy file."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        assert os.path.exists(save_path)
    
    def test_save_does_not_create_data_directory(self, mock_memmap_trajectory, temp_dir):
        """Test that saving memmaps does NOT create data directory (new behavior)."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        data_dir = os.path.splitext(save_path)[0] + "_data"
        assert not os.path.exists(data_dir)
    
    def test_save_preserves_original_memmap_files(self, mock_memmap_trajectory, temp_dir):
        """Test that original memmap files are preserved (not copied)."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        original_distances_path = mock_memmap_trajectory.distances.filename
        original_contacts_path = mock_memmap_trajectory.contacts.filename
        
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        # Original files should still exist
        assert os.path.exists(original_distances_path)
        assert os.path.exists(original_contacts_path)
    
    def test_load_preserves_distances_memmap_type(self, mock_memmap_trajectory, temp_dir):
        """Test that loading preserves distances as memmap type."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        assert isinstance(loaded_traj.distances, np.memmap)
    
    def test_load_preserves_contacts_memmap_type(self, mock_memmap_trajectory, temp_dir):
        """Test that loading preserves contacts as memmap type."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        assert isinstance(loaded_traj.contacts, np.memmap)
    
    def test_load_preserves_distances_data_integrity(self, mock_memmap_trajectory, temp_dir):
        """Test that distances data values are preserved exactly."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        original_distances = np.array(mock_memmap_trajectory.distances)
        
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        assert np.allclose(loaded_traj.distances, original_distances)
    
    def test_load_preserves_contacts_data_integrity(self, mock_memmap_trajectory, temp_dir):
        """Test that contacts data values are preserved exactly."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        original_contacts = np.array(mock_memmap_trajectory.contacts)
        
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        assert np.allclose(loaded_traj.contacts, original_contacts)
    
    def test_load_preserves_distances_shape(self, mock_memmap_trajectory, temp_dir):
        """Test that distances shape is preserved during save/load."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        original_shape = mock_memmap_trajectory.distances.shape
        
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.distances.shape == original_shape
    
    def test_load_preserves_contacts_shape(self, mock_memmap_trajectory, temp_dir):
        """Test that contacts shape is preserved during save/load."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        original_shape = mock_memmap_trajectory.contacts.shape
        
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.contacts.shape == original_shape
    
    def test_memmap_references_original_file(self, mock_memmap_trajectory, temp_dir):
        """Test that loaded memmap points to the original file."""
        save_path = os.path.join(temp_dir, "test_memmap.npy")
        original_distances_path = mock_memmap_trajectory.distances.filename
        
        DataUtils.save_object(mock_memmap_trajectory, save_path)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.distances.filename == original_distances_path


class TestDataUtilsMixedAttributes:
    """Test class for mixed attribute types."""
    
    def test_string_attribute_preservation(self, temp_dir):
        """Test that string attributes are preserved."""
        traj = SimpleMockTrajectory(string_attr="test_string")
        save_path = os.path.join(temp_dir, "test_string.npy")
        
        DataUtils.save_object(traj, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.string_attr == "test_string"
    
    def test_integer_attribute_preservation(self, temp_dir):
        """Test that integer attributes are preserved."""
        traj = SimpleMockTrajectory(int_attr=42)
        save_path = os.path.join(temp_dir, "test_int.npy")
        
        DataUtils.save_object(traj, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.int_attr == 42
    
    def test_list_attribute_preservation(self, temp_dir):
        """Test that list attributes are preserved."""
        traj = SimpleMockTrajectory(list_attr=[1, 2, 3])
        save_path = os.path.join(temp_dir, "test_list.npy")
        
        DataUtils.save_object(traj, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.list_attr == [1, 2, 3]
    
    def test_dict_attribute_preservation(self, temp_dir):
        """Test that dict attributes are preserved."""
        traj = SimpleMockTrajectory(dict_attr={"key": "value"})
        save_path = os.path.join(temp_dir, "test_dict.npy")
        
        DataUtils.save_object(traj, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.dict_attr == {"key": "value"}
    
    def test_numpy_array_preservation(self, temp_dir):
        """Test that numpy arrays are preserved."""
        traj = SimpleMockTrajectory(array_attr=np.array([1, 2, 3, 4, 5]))
        save_path = os.path.join(temp_dir, "test_array.npy")
        
        DataUtils.save_object(traj, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert np.array_equal(loaded_traj.array_attr, np.array([1, 2, 3, 4, 5]))
    
    def test_none_attribute_preservation(self, temp_dir):
        """Test that None attributes are preserved."""
        traj = SimpleMockTrajectory(none_attr=None)
        save_path = os.path.join(temp_dir, "test_none.npy")
        
        DataUtils.save_object(traj, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        assert loaded_traj.none_attr is None


class TestDataUtilsEdgeCases:
    """Test class for edge cases and error conditions."""
    
    @patch('mdcataflow.utils.DataUtils.np.save')
    def test_save_failure_is_propagated(self, mock_save, mock_trajectory, temp_dir):
        """Test that save failures are properly propagated."""
        mock_save.side_effect = IOError("Disk full")
        save_path = os.path.join(temp_dir, "test_fail.npy")
        
        with pytest.raises(IOError):
            DataUtils.save_object(mock_trajectory, save_path)
    
    @patch('mdcataflow.utils.DataUtils.np.save')
    def test_save_is_called_once(self, mock_save, mock_trajectory, temp_dir):
        """Test that np.save is called exactly once."""
        save_path = os.path.join(temp_dir, "test_call.npy")
        DataUtils.save_object(mock_trajectory, save_path)
        mock_save.assert_called_once()
    
    def test_memmap_without_filename_raises_error(self, temp_dir):
        """Test that memmaps without filename raise an error."""
        # Create a mock memmap without filename
        mock_memmap = Mock(spec=np.memmap)
        mock_memmap.filename = None
        mock_memmap.dtype = np.float64
        mock_memmap.shape = (10, 10)
        
        traj = SimpleMockTrajectory(bad_memmap=mock_memmap)
        save_path = os.path.join(temp_dir, "test_bad_memmap.npy")
        
        with pytest.raises(ValueError, match="has no filename"):
            DataUtils.save_object(traj, save_path)
    
    def test_empty_trajectory_saves_successfully(self, temp_dir):
        """Test that empty trajectory objects can be saved."""
        traj = SimpleMockTrajectory()
        save_path = os.path.join(temp_dir, "test_empty.npy")
        
        DataUtils.save_object(traj, save_path)
        assert os.path.exists(save_path)
    
    def test_empty_trajectory_loads_successfully(self, temp_dir):
        """Test that empty trajectory objects can be loaded."""
        traj = SimpleMockTrajectory()
        save_path = os.path.join(temp_dir, "test_empty.npy")
        
        DataUtils.save_object(traj, save_path)
        
        loaded_traj = SimpleMockTrajectory()
        DataUtils.load_object(loaded_traj, save_path)
        
        # Should complete without error
        assert True
    
    def test_missing_memmap_file_returns_none(self, temp_dir):
        """Test that missing memmap files are handled gracefully."""
        # Create a trajectory object with memmap info pointing to non-existent file
        save_path = os.path.join(temp_dir, "test_missing.npy")
        fake_memmap_info = {
            'distances': {
                '_is_memmap': True,
                'dtype': np.float64,
                'shape': (10, 10),
                'mode': 'r',
                'original_path': '/non/existent/path.dat'
            }
        }
        
        # Save the fake info
        np.save(save_path, fake_memmap_info, allow_pickle=True)
        
        loaded_traj = SimpleMockTrajectory(use_memmap=True)
        DataUtils.load_object(loaded_traj, save_path)
        
        # Should return None for missing file
        assert loaded_traj.distances is None 
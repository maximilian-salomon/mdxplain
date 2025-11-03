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

import warnings
# Suppress pkg_resources warnings before any imports that might trigger them
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=DeprecationWarning)

import os
import pickle
import tempfile
import numpy as np
import pytest
import mdtraj as md
from unittest.mock import Mock

from mdxplain.utils.data_utils import DataUtils
from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory


class SimpleTestObject:
    """Simple test object for DataUtils testing."""
    
    def __init__(self, **kwargs):
        """
        Initialize SimpleTestObject with arbitrary keyword arguments.
        
        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments to set as object attributes.
            Each key-value pair becomes an attribute of the object.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __dir__(self):
        """
        Return list of public attributes for object introspection.
        
        Returns
        -------
        list of str
            List of attribute names that don't start with underscore.
            Used by dir() function to show only public attributes.
        """
        return [attr for attr in self.__dict__.keys() if not attr.startswith('_')]


class MemmapTestObject:
    """Test object with memmap support."""
    
    def __init__(self, use_memmap=False, **kwargs):
        """
        Initialize MemmapTestObject with memmap support flag and attributes.
        
        Parameters
        ----------
        use_memmap : bool, default=False
            Flag indicating whether this object supports memory-mapped arrays.
            Used by DataUtils to determine memmap handling during save/load.
        **kwargs : dict
            Arbitrary keyword arguments to set as object attributes.
            Each key-value pair becomes an attribute of the object.
        """
        self.use_memmap = use_memmap
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __dir__(self):
        """
        Return list of public attributes for object introspection.
        
        Returns
        -------
        list of str
            List of attribute names that don't start with underscore.
            Used by dir() function to show only public attributes.
        """
        return [attr for attr in self.__dict__.keys() if not attr.startswith('_')]


class TrajectoryContainer:
    """Container mimicking trajectory storage in PipelineData."""
    def __init__(self):
        """
        Initialize TrajectoryContainer with empty data structures.
        
        Creates empty lists and dictionaries for storing trajectory data,
        metadata, and analysis results in a PipelineData-like structure.
        """
        self.trajectories = []
        self.metadata = {}
        self.analysis_results = {}


class ComplexAnalysis:
    """Complex analysis object with many nested levels."""
    def __init__(self):
        """
        Initialize ComplexAnalysis with deeply nested test data structures.
        
        Creates 4-level nested dictionaries with mixed data types including
        numpy arrays, dicts, lists, and scalar values to test DataUtils
        handling of complex nested structures.
        """
        # 4-level nesting with mixed types
        self.experiment_data = {
            "systems": {
                "system_A": {
                    "replicates": {
                        "rep1": {"frames": np.array([1, 2, 3]), "score": 0.85},
                        "rep2": {"frames": np.array([4, 5, 6]), "score": 0.78}
                    },
                    "metadata": {"temperature": 300, "pH": 7.4}
                },
                "system_B": {
                    "replicates": {
                        "rep1": {"frames": np.array([7, 8, 9]), "score": 0.92}
                    },
                    "metadata": {"temperature": 310, "pH": 6.8}
                }
            },
            "analysis_methods": [
                {"name": "PCA", "components": np.random.rand(5, 100)},
                {"name": "t-SNE", "embedding": np.random.rand(1000, 2)}
            ]
        }
        
        # List of dicts with arrays
        self.time_series_data = [
            {"time": np.linspace(0, 100, 1000), "observable": "rmsd", "values": np.random.rand(1000)},
            {"time": np.linspace(0, 100, 1000), "observable": "rg", "values": np.random.rand(1000)}
        ]


class MockTopology:
    """Lightweight mock topology."""
    def __init__(self, n_atoms, n_residues):
        """
        Initialize MockTopology with atom and residue counts.
        
        Parameters
        ----------
        n_atoms : int
            Number of atoms in the mock topology.
        n_residues : int
            Number of residues in the mock topology.
        """
        self.n_atoms = n_atoms
        self.n_residues = n_residues


class MockTrajectory:
    """Lightweight mock trajectory for testing without file dependencies."""
    def __init__(self, n_frames=10, n_atoms=5):
        """
        Initialize MockTrajectory with synthetic trajectory data.
        
        Parameters
        ----------
        n_frames : int, default=10
            Number of frames in the mock trajectory.
        n_atoms : int, default=5
            Number of atoms in the mock trajectory.
            
        Notes
        -----
        Creates synthetic xyz coordinates and time arrays with deterministic
        random data (seed=42) for reproducible testing.
        """
        self.n_frames = n_frames
        self.n_atoms = n_atoms
        self.trajectory_file = "mock_trajectory.xtc"
        self.topology_file = "mock_topology.pdb" 
        self.zarr_cache_path = "mock_cache.zarr"
        self.chunk_size = 100
        
        # Mock xyz data
        np.random.seed(42)
        self.xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
        self.time = np.arange(n_frames, dtype=np.float32)
        
        # Mock topology
        self.topology = MockTopology(n_atoms, max(1, n_atoms // 3))
        
    def cleanup(self):
        """
        Mock cleanup method for trajectory resources.
        
        This is a no-op method that mimics the cleanup behavior of real
        trajectory objects. In actual trajectories, this would close files,
        release memory, and clean up temporary resources.
        """
        pass


class MockMDTraj:
    """Mock MDTraj trajectory for testing DaskMDTrajectory.from_mdtraj()."""
    def __init__(self, n_frames=10, n_atoms=2):
        """
        Initialize MockMDTraj with MDTraj-compatible interface.
        
        Parameters
        ----------
        n_frames : int, default=10
            Number of frames in the mock trajectory.
        n_atoms : int, default=2
            Number of atoms in the mock trajectory.
            
        Notes
        -----
        Creates a mock MDTraj trajectory with real MDTraj.Topology object
        for compatibility with DaskMDTrajectory.from_mdtraj() method.
        Uses deterministic random data (seed=42) for reproducible testing.
        """
        self.n_frames = n_frames
        self.n_atoms = n_atoms
        
        # Create fake coordinates and time
        np.random.seed(42)
        self.xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
        self.time = np.arange(n_frames, dtype=np.float32)
        
        # Mock topology
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        topology.add_atom("CA", element=md.element.carbon, residue=residue)
        if n_atoms > 1:
            topology.add_atom("CB", element=md.element.carbon, residue=residue)
        if n_atoms > 2:
            topology.add_atom("N", element=md.element.nitrogen, residue=residue)
        self.topology = topology
        
        # No unitcell data by default
        self.unitcell_vectors = None
        self.unitcell_lengths = None
        self.unitcell_angles = None


class MockPipelineData:
    """Mock PipelineData with nested structure."""
    def __init__(self):
        """
        Initialize MockPipelineData with complex nested structure.
        
        Creates nested data structures mimicking real PipelineData including
        trajectory_data, feature_data, cluster_data, data_selector_data,
        and mixed_analysis_data with realistic shapes and types for testing
        DataUtils save/load functionality with complex nested objects.
        """
        # Nested dict structure like real PipelineData
        self.trajectory_data = {
            "trajectories": [],
            "names": ["system_A_traj1", "system_B_traj1", "control_traj1"],
            "tags": {"system_A": [0], "system_B": [1], "control": [2]}
        }
        
        # Feature data nested dict
        self.feature_data = {
            0: {"distances": np.random.rand(1000, 45)},
            1: {"distances": np.random.rand(1200, 45)},
            2: {"angles": np.random.rand(800, 20)}
        }
        
        # Cluster data nested dict of dicts  
        self.cluster_data = {
            "dbscan_eps05": {
                "labels": np.array([0, 1, 1, 0, 2, 2]),
                "centers": np.random.rand(3, 45),
                "parameters": {"eps": 0.5, "min_samples": 5}
            },
            "kmeans_k3": {
                "labels": np.array([0, 1, 2, 0, 1, 2]),
                "centers": np.random.rand(3, 45),
                "parameters": {"n_clusters": 3, "random_state": 42}
            }
        }
        
        # Data selector data
        self.data_selector_data = {
            "folded_frames": {
                "trajectory_frames": {0: [10, 20, 30], 1: [15, 25, 35]},
                "metadata": {"criteria": "rmsd < 0.3"}
            }
        }
        
        # Mixed list structures
        self.mixed_analysis_data = [
            {"type": "pca", "explained_variance": 0.85},
            {"type": "lda", "n_components": 2},
            np.array([1.2, 2.3, 3.4, 4.5])
        ]


@pytest.fixture
def temp_dir():
    """
    Provide temporary directory for test files.
    
    Yields
    ------
    str
        Path to a temporary directory that will be automatically
        cleaned up after the test completes. Used for creating
        test files without affecting the filesystem.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_object():
    """
    Provide SimpleTestObject with basic Python data types.
    
    Returns
    -------
    SimpleTestObject
        Test object containing string, int, float, list, dict, None,
        and bool attributes for testing DataUtils save/load functionality
        with standard Python types.
    """
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
    """
    Provide SimpleTestObject with various numpy array types.
    
    Returns
    -------
    SimpleTestObject
        Test object containing small_array, large_array, int_array,
        and complex_array attributes for testing DataUtils save/load
        functionality with different numpy array types and dtypes.
    """
    np.random.seed(42)
    return SimpleTestObject(
        small_array=np.array([1, 2, 3, 4, 5]),
        large_array=np.random.random((10, 5)),
        int_array=np.array([[1, 2], [3, 4]], dtype=np.int32),
        complex_array=np.array([1+2j, 3+4j])
    )


@pytest.fixture
def memmap_object(temp_dir):
    """
    Provide MemmapTestObject with memory-mapped arrays.
    
    Parameters
    ----------
    temp_dir : str
        Temporary directory path from temp_dir fixture.
        
    Returns
    -------
    MemmapTestObject
        Test object with use_memmap=True containing data1 and data2
        memory-mapped arrays, plus data1_path and data2_path attributes
        for testing DataUtils memmap save/load functionality.
    """
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
        """
        Test that saving creates the output file.
        
        Validates that DataUtils.save_object() creates a valid .npy file
        in the specified directory.
        """
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
        """
        Test that memmaps without filename raise ValueError.
        
        Validates that memory-mapped arrays without filename attribute
        cause a ValueError with appropriate message when saving.
        """
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
        """
        Test that loading preserves all simple data types.
        
        Validates that all basic Python types (str, int, float,
        list, dict, bool, None) are correctly loaded and restored.
        """
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
        """
        Test that loading preserves numpy arrays correctly.
        
        Validates that different numpy array types (small, large, int, complex)
        are restored with correct values and data types.
        """
        save_path = os.path.join(temp_dir, "arrays.npy")
        DataUtils.save_object(array_object, save_path)
        
        loaded_obj = SimpleTestObject()
        DataUtils.load_object(loaded_obj, save_path)
        
        assert np.array_equal(loaded_obj.small_array, array_object.small_array)
        assert np.allclose(loaded_obj.large_array, array_object.large_array)
        assert np.array_equal(loaded_obj.int_array, array_object.int_array)
        assert np.array_equal(loaded_obj.complex_array, array_object.complex_array)
    
    def test_load_memmap_object_preserves_memmaps(self, memmap_object, temp_dir):
        """
        Test that loading preserves memmaps as memmaps with correct data.
        
        Validates that memory-mapped arrays after loading continue to exist as
        np.memmap objects and contain identical data.
        """
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
        """
        Test that memmaps with missing files are set to None.
        
        Validates that missing memmap files are gracefully handled as None
        without causing the entire loading process to fail.
        """
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
        
        save_path = os.path.join(temp_dir, "missing.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(fake_memmap_info, f)
        
        loaded_obj = MemmapTestObject(use_memmap=True)
        DataUtils.load_object(loaded_obj, save_path)
        
        assert loaded_obj.missing_data is None
    
    def test_load_memmap_with_fallback_path(self, temp_dir):
        """
        Test that memmaps can use fallback paths.
        
        Validates that memmaps can use alternative _path attributes
        for restoration when original paths are missing.
        """
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
        
        save_path = os.path.join(temp_dir, "fallback_test.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(memmap_info, f)
        
        # Load with fallback path
        loaded_obj = MemmapTestObject(use_memmap=True)
        loaded_obj.test_data_path = fallback_file
        DataUtils.load_object(loaded_obj, save_path)
        
        assert isinstance(loaded_obj.test_data, np.memmap)
        assert np.allclose(loaded_obj.test_data, test_data.astype(np.float32))
    
    def test_load_nested_object_structure(self, temp_dir):
        """
        Test that loading preserves nested object structures.
        
        Validates that complex nested objects with inner objects,
        nested dicts and nested lists are correctly reconstructed.
        """
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
        """
        Test that loading non-existent file raises FileNotFoundError.
        
        Validates that attempting to load a non-existent file
        causes an appropriate FileNotFoundError.
        """
        obj = SimpleTestObject()
        nonexistent_path = os.path.join(temp_dir, "nonexistent.npy")
        
        with pytest.raises(FileNotFoundError):
            DataUtils.load_object(obj, nonexistent_path)
    
    def test_load_corrupted_file_raises_error(self, temp_dir):
        """
        Test that loading corrupted files raises appropriate error.
        
        Validates that corrupted or invalid .npy files when loading
        cause appropriate errors (ValueError, OSError, UnpicklingError).
        """
        obj = SimpleTestObject()
        corrupted_path = os.path.join(temp_dir, "corrupted.npy")
        
        # Create corrupted file
        with open(corrupted_path, 'wb') as f:
            f.write(b"This is not a valid numpy file")
        
        with pytest.raises((ValueError, OSError, pickle.UnpicklingError)):
            DataUtils.load_object(obj, corrupted_path)
    
    def test_load_invalid_data_format_raises_error(self, temp_dir):
        """
        Test that loading file with wrong format raises error.
        
        Validates that files with invalid pickle data when loading
        cause corresponding unpickling errors.
        """
        obj = SimpleTestObject()
        invalid_path = os.path.join(temp_dir, "invalid.pkl")
        
        # Save invalid data that causes unpickling issues
        with open(invalid_path, 'wb') as f:
            f.write(b"invalid pickle data")
        
        with pytest.raises((ValueError, OSError, pickle.UnpicklingError)):
            DataUtils.load_object(obj, invalid_path)
    
    def test_load_memmap_creation_fails_gracefully(self, temp_dir):
        """
        Test that memmap creation failure is handled gracefully.
        
        Validates that invalid memmap parameters (invalid dtype) do not
        cause crashes but set the attribute to None.
        """
        obj = MemmapTestObject(use_memmap=True)
        invalid_memmap_path = os.path.join(temp_dir, "invalid_memmap.pkl")
        
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
        
        with open(invalid_memmap_path, 'wb') as f:
            pickle.dump(invalid_memmap_info, f)
        
        # Should not crash, but set attribute to None
        DataUtils.load_object(obj, invalid_memmap_path)
        assert obj.test_data is None
    
    def test_load_overwrites_existing_attributes(self, temp_dir):
        """
        Test that loading overwrites existing object attributes.
        
        Validates that when loading into existing objects, existing
        attributes are correctly overwritten and new ones added.
        """
        obj_with_existing = SimpleTestObject(existing_attr="old_value", new_attr="will_be_overwritten")
        
        # Save different data
        save_data = {"existing_attr": "new_value", "additional_attr": "added"}
        save_path = os.path.join(temp_dir, "overwrite.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        DataUtils.load_object(obj_with_existing, save_path)
        
        assert obj_with_existing.existing_attr == "new_value"
        assert obj_with_existing.additional_attr == "added"
        # Note: new_attr will still exist as it's not removed, only added to


class TestDataUtilsComplexObjects:
    """Test DataUtils with PipelineData-like complex nested structures."""
    
    def test_pipeline_data_like_nested_structure(self, temp_dir):
        """
        Test DataUtils with complex nested structure like PipelineData.
        
        Validates that pipeline-like nested structures with
        trajectory_data, feature_data, cluster_data are correctly saved/loaded.
        """
        
        # Create complex object
        original = MockPipelineData()
        
        # Save
        save_path = os.path.join(temp_dir, "pipeline_data_test.npy")
        DataUtils.save_object(original, save_path)
        assert os.path.exists(save_path)
        
        # Load into new object
        loaded = MockPipelineData()
        # Clear to test restoration
        loaded.trajectory_data = None
        loaded.feature_data = None
        loaded.cluster_data = None
        loaded.data_selector_data = None
        loaded.mixed_analysis_data = None
        
        DataUtils.load_object(loaded, save_path)
        
        # Verify trajectory_data exactly matches original
        assert loaded.trajectory_data["names"] == original.trajectory_data["names"]
        assert loaded.trajectory_data["tags"] == original.trajectory_data["tags"]
        
        # Verify feature_data arrays are identical
        assert np.array_equal(loaded.feature_data[0]["distances"], original.feature_data[0]["distances"])
        assert np.array_equal(loaded.feature_data[1]["distances"], original.feature_data[1]["distances"])
        assert np.array_equal(loaded.feature_data[2]["angles"], original.feature_data[2]["angles"])
        
        # Verify cluster_data exactly matches original
        assert np.array_equal(loaded.cluster_data["dbscan_eps05"]["labels"], original.cluster_data["dbscan_eps05"]["labels"])
        assert loaded.cluster_data["dbscan_eps05"]["parameters"] == original.cluster_data["dbscan_eps05"]["parameters"]
        assert np.array_equal(loaded.cluster_data["kmeans_k3"]["centers"], original.cluster_data["kmeans_k3"]["centers"])
        
        # Verify data_selector_data exactly matches original
        assert loaded.data_selector_data == original.data_selector_data
        
        # Verify mixed analysis data exactly matches original
        assert loaded.mixed_analysis_data[0] == original.mixed_analysis_data[0]
        assert loaded.mixed_analysis_data[1] == original.mixed_analysis_data[1]
        assert np.array_equal(loaded.mixed_analysis_data[2], original.mixed_analysis_data[2])
            
    def test_dask_trajectory_save_load(self, temp_dir):
        """
        Test saving and loading of DaskMDTrajectory objects.
        
        Validates that DaskMDTrajectory with zarr cache and metadata
        is exactly reconstructed with identical xyz/time data.
        """
        
        # Create mock MDTraj trajectory
        mock_traj = MockMDTraj(n_frames=10, n_atoms=2)
        
        # Create container with DaskMDTrajectory from mock traj (zarr cache in temp dir)
        test_zarr_cache = os.path.join(temp_dir, "test_zarr_cache")
        original = TrajectoryContainer()
        dask_traj = DaskMDTrajectory.from_mdtraj(mock_traj, 
                                                 zarr_cache_path=test_zarr_cache,
                                                 chunk_size=10, n_workers=1)
        original.trajectories.append(dask_traj)
        original.metadata = {
            "n_trajectories": 1,
            "total_frames": dask_traj.n_frames,
            "total_atoms": dask_traj.n_atoms
        }
        
        # Create deterministic analysis results
        np.random.seed(42)
        original.analysis_results = {
            "rmsd": np.random.rand(dask_traj.n_frames),
            "rg": np.random.rand(dask_traj.n_frames)
        }
        
        # Save
        save_path = os.path.join(temp_dir, "dask_trajectory_test.npy")  
        DataUtils.save_object(original, save_path)
        assert os.path.exists(save_path)
        
        # Load
        loaded = TrajectoryContainer()
        loaded.trajectories = None
        loaded.metadata = None
        loaded.analysis_results = None
        
        DataUtils.load_object(loaded, save_path)
        
        # Test: DaskMDTrajectory must be restored exactly as it was
        assert len(loaded.trajectories) == 1
        restored_traj = loaded.trajectories[0]
        original_traj = original.trajectories[0]
        
        # Must be DaskMDTrajectory - no fallbacks accepted
        assert isinstance(restored_traj, DaskMDTrajectory)
        
        # Must have identical properties
        assert restored_traj.n_frames == original_traj.n_frames
        assert restored_traj.n_atoms == original_traj.n_atoms
        assert restored_traj.trajectory_file == original_traj.trajectory_file
        assert restored_traj.topology_file == original_traj.topology_file
        assert restored_traj.chunk_size == original_traj.chunk_size
        assert restored_traj.zarr_cache_path == original_traj.zarr_cache_path
        
        # Test: First frame coordinates must be identical
        original_xyz = original_traj.xyz[0]
        restored_xyz = restored_traj.xyz[0]
        assert np.allclose(original_xyz, restored_xyz, atol=1e-10)
        
        # Test: Random frame coordinates must be identical
        test_frame = min(5, original_traj.n_frames - 1)
        original_frame_xyz = original_traj.xyz[test_frame]
        restored_frame_xyz = restored_traj.xyz[test_frame]
        assert np.allclose(original_frame_xyz, restored_frame_xyz, atol=1e-10)
        
        # Test: Time data must be identical
        assert np.allclose(original_traj.time, restored_traj.time, atol=1e-10)
        
        # Verify exact metadata reconstruction
        assert loaded.metadata["n_trajectories"] == original.metadata["n_trajectories"]
        assert loaded.metadata["total_frames"] == original.metadata["total_frames"]
        assert loaded.metadata["total_atoms"] == original.metadata["total_atoms"]
        
        # Verify exact analysis results reconstruction
        assert np.array_equal(loaded.analysis_results["rmsd"], original.analysis_results["rmsd"])
        assert np.array_equal(loaded.analysis_results["rg"], original.analysis_results["rg"])
        
        # Verify analysis results have correct shapes and values
        expected_frames = original.trajectories[0].n_frames
        assert loaded.analysis_results["rmsd"].shape == (expected_frames,)
        assert loaded.analysis_results["rg"].shape == (expected_frames,)
        
        # Verify analysis data types
        assert loaded.analysis_results["rmsd"].dtype == np.float64
        assert loaded.analysis_results["rg"].dtype == np.float64
        
        # Cleanup
        original_traj.cleanup()
        restored_traj.cleanup()
    
    def test_from_mdtraj_equivalence_with_direct_creation(self, temp_dir):
        """
        Test that from_mdtraj delivers identical result to direct creation.
        
        Validates that DaskMDTrajectory.from_mdtraj() and direct file creation
        produce equivalent trajectory objects with identical coordinates.
        """
        
        # Create real MDTraj trajectory and save to files
        mock_traj = MockMDTraj(n_frames=15, n_atoms=3)
        real_traj = md.Trajectory(mock_traj.xyz, mock_traj.topology, time=mock_traj.time)
        
        traj_file = os.path.join(temp_dir, "test_traj.xtc")
        top_file = os.path.join(temp_dir, "test_top.pdb")
        real_traj.save_xtc(traj_file)
        real_traj.save_pdb(top_file)
        
        # Method 1: Create DaskMDTrajectory directly from files
        zarr_cache_1 = os.path.join(temp_dir, "cache_direct")
        dask_traj_direct = DaskMDTrajectory(traj_file, top_file, 
                                           zarr_cache_path=zarr_cache_1,
                                           chunk_size=5, n_workers=1)
        
        # Method 2: Create DaskMDTrajectory via from_mdtraj
        zarr_cache_2 = os.path.join(temp_dir, "cache_from_mdtraj")
        dask_traj_from_md = DaskMDTrajectory.from_mdtraj(real_traj,
                                                        zarr_cache_path=zarr_cache_2,
                                                        chunk_size=5, n_workers=1)
        
        # Test equivalence - basic properties
        assert dask_traj_direct.n_frames == dask_traj_from_md.n_frames
        assert dask_traj_direct.n_atoms == dask_traj_from_md.n_atoms
        assert dask_traj_direct.chunk_size == dask_traj_from_md.chunk_size
        
        # Test equivalence - trajectory data
        direct_xyz = dask_traj_direct.xyz
        from_md_xyz = dask_traj_from_md.xyz
        assert np.allclose(direct_xyz, from_md_xyz, atol=1e-6)
        
        direct_time = dask_traj_direct.time
        from_md_time = dask_traj_from_md.time
        assert np.allclose(direct_time, from_md_time, atol=1e-6)
        
        # Test equivalence - topology comparison by atom count
        assert dask_traj_direct.topology.n_atoms == dask_traj_from_md.topology.n_atoms
        assert dask_traj_direct.topology.n_residues == dask_traj_from_md.topology.n_residues
        
        # Test equivalence - slice operations produce same results
        direct_slice = dask_traj_direct.slice(slice(0, 5))
        from_md_slice = dask_traj_from_md.slice(slice(0, 5))
        assert np.allclose(direct_slice.xyz, from_md_slice.xyz, atol=1e-6)
        assert np.allclose(direct_slice.time, from_md_slice.time, atol=1e-6)
        
        # Cleanup
        dask_traj_direct.cleanup()
        dask_traj_from_md.cleanup()
    
    def test_deeply_nested_mixed_types(self, temp_dir):
        """
        Test with deeply nested mixed-type structures.
        
        Validates that 4-level nested structures with numpy arrays,
        dicts, lists and mixed types are correctly saved/loaded.
        """
        
        # Create and save
        original = ComplexAnalysis()
        save_path = os.path.join(temp_dir, "complex_nested.npy")
        DataUtils.save_object(original, save_path)
        
        # Load and verify
        loaded = ComplexAnalysis()
        loaded.experiment_data = None
        loaded.time_series_data = None
        DataUtils.load_object(loaded, save_path)
        
        # Verify 4-level nested access
        assert loaded.experiment_data["systems"]["system_A"]["replicates"]["rep1"]["score"] == 0.85
        assert loaded.experiment_data["systems"]["system_B"]["metadata"]["temperature"] == 310
        assert np.array_equal(
            loaded.experiment_data["systems"]["system_A"]["replicates"]["rep2"]["frames"], 
            np.array([4, 5, 6])
        )
        
        # Verify analysis methods arrays are identical
        assert loaded.experiment_data["analysis_methods"][0]["name"] == original.experiment_data["analysis_methods"][0]["name"]
        assert np.array_equal(loaded.experiment_data["analysis_methods"][0]["components"], original.experiment_data["analysis_methods"][0]["components"])
        assert loaded.experiment_data["analysis_methods"][1]["name"] == original.experiment_data["analysis_methods"][1]["name"]
        assert np.array_equal(loaded.experiment_data["analysis_methods"][1]["embedding"], original.experiment_data["analysis_methods"][1]["embedding"])
        
        # Verify time series arrays are identical
        assert loaded.time_series_data[0]["observable"] == original.time_series_data[0]["observable"]
        assert np.array_equal(loaded.time_series_data[0]["time"], original.time_series_data[0]["time"])
        assert np.array_equal(loaded.time_series_data[0]["values"], original.time_series_data[0]["values"])
        assert loaded.time_series_data[1]["observable"] == original.time_series_data[1]["observable"]
        assert np.array_equal(loaded.time_series_data[1]["time"], original.time_series_data[1]["time"])
        assert np.array_equal(loaded.time_series_data[1]["values"], original.time_series_data[1]["values"])
        
        # Deeply nested mixed types successfully saved and loaded


class TestDataUtilsIntegration:
    """Integration tests for complete save/load workflows."""
    
    def test_round_trip_preserves_all_data(self, simple_object, temp_dir):
        """
        Test that complete save/load cycle preserves all data perfectly.
        
        Validates that round-trip (save → load) preserves every attribute value
        exactly without data loss or changes.
        """
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
        """
        Test that multiple save/load cycles do not corrupt data.
        
        Validates that 3 consecutive save/load cycles cause no
        data corruption or drift.
        """
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
        """
        Test that loaded objects are independent copies.
        
        Validates that loaded objects do not exist as references
        but as independent copies with separate memory space.
        """
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

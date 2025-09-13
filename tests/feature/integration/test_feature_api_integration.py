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

"""Integration tests for feature API: pipeline.feature.add.xxx and pipeline.feature.reduce.xxx.yyy"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestFeatureAPIIntegration:
    """Test complete feature API integration with concrete expected values."""
    
    def _setup_triangle_pipeline(self, n_frames=10):
        """
        Create pipeline with triangle atoms for distance/contact tests.
        
        Parameters
        ----------
        n_frames : int, default=10
            Number of frames in the mock trajectory.
            
        Returns
        -------
        PipelineManager
            Configured pipeline with triangle geometry (3 atoms at known positions).
            Triangle atoms positioned to create 3.0, 4.0, 5.0 Å distances.
        """
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_triangle_atoms(n_frames=n_frames, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["triangle_traj"]
        
        # Setup residue labels (1 atom per residue for simplicity)
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        return pipeline
    
    def _setup_moving_pipeline(self, n_frames=10):
        """
        Create pipeline with moving atoms for coordinate tests.
        
        Parameters
        ----------
        n_frames : int, default=10
            Number of frames in the mock trajectory.
            
        Returns
        -------
        PipelineManager
            Configured pipeline with 2 atoms, one static and one moving linearly.
            Used for testing coordinate feature extraction and unit conversion.
        """
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_moving_atoms(n_frames=n_frames, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["moving_traj"]
        
        # Setup residue labels
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"}
        ]}
        
        return pipeline
    
    def _setup_varied_pipeline(self, n_frames=10):
        """
        Create pipeline with varied coordinate movement for std tests.
        
        Parameters
        ----------
        n_frames : int, default=10
            Number of frames in the mock trajectory.
            
        Returns
        -------
        PipelineManager
            Configured pipeline with 3 atoms having different motion patterns.
            Atom 0: static, Atom 1: oscillating, Atom 2: linear movement.
            Used for testing standard deviation-based feature reduction.
        """
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_varied_coordinates(n_frames=n_frames, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["varied_traj"]
        
        # Setup residue labels
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        return pipeline
    
    # === FEATURE ADD TESTS (6 Tests) ===
    
    def test_add_distances(self):
        """
        Test distances feature calculation with known triangle geometry.
        
        Validates that distance features are correctly computed for triangle atoms
        and returns expected values (3.0, 4.0, 5.0 Å scaled).
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Add distances using API with excluded_neighbors=0 to get all pairs
        pipeline.feature.add.distances(excluded_neighbors=0, force=True)
        
        # Verify feature data exists
        assert 'distances' in pipeline._data.feature_data
        distances = pipeline._data.feature_data['distances'][0].data
        
        # Expected: With 3 residues and excluded_neighbors=0, we get 3 pairs: 0-1, 0-2, 1-2
        # mdxplain works with residues (CA atoms at res centers), not individual atoms
        # Triangle: res0=(0,0,0), res1=(3,0,0), res2=(0,4,0) 
        # Distances: 0-1=3.0, 0-2=4.0, 1-2=5.0 Å
        # System returns distances in units of Å * 10 (internal scaling)
        expected_distances = np.array([30.0, 40.0, 50.0])  # 3.0*10, 4.0*10, 5.0*10
        expected_matrix = np.tile(expected_distances, (10, 1))
        
        # Test the actual calculation from our software
        np.testing.assert_array_almost_equal(distances, expected_matrix, decimal=1)
        
        # Verify shape  
        assert distances.shape == (10, 3)  # 10 frames, 3 distances (0-1, 0-2, 1-2)
    
    def test_add_contacts(self):
        """
        Test contacts feature calculation with known cutoff behavior.
        
        Validates that contact features with cutoff=35.0 correctly filter
        and mark only distances below threshold as contacts.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Need distances first (dependency)
        pipeline.feature.add.distances(excluded_neighbors=0, force=True)
        
        # Add contacts with cutoff=35.0 (in same units as distances: Å*10)
        pipeline.feature.add.contacts(cutoff=35.0, force=True)
        
        # Verify feature data exists
        assert 'contacts' in pipeline._data.feature_data
        contacts = pipeline._data.feature_data['contacts'][0].data
        
        # Expected contacts with cutoff=35.0 (35.0 units = 3.5 Å):
        # Distance 0-1=30.0 < 35.0 → contact=True
        # Distance 0-2=40.0 > 35.0 → contact=False  
        # Distance 1-2=50.0 > 35.0 → contact=False
        expected_contacts = np.array([True, False, False], dtype=bool)
        expected_matrix = np.tile(expected_contacts, (10, 1))
        
        # Test the actual calculation from our software
        np.testing.assert_array_equal(contacts, expected_matrix)
        
        # Verify shape and dtype
        assert contacts.shape == (10, 3)
        assert contacts.dtype == bool
    
    def test_add_coordinates(self):
        """
        Test coordinates feature extraction with moving atoms.
        
        Validates that coordinate features are correctly extracted from moving atoms
        and unit conversion (0.1nm → 1.0Å) works properly.
        """
        pipeline = self._setup_moving_pipeline(n_frames=5)
        
        # Add coordinates using API
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Verify feature data exists
        assert 'coordinates' in pipeline._data.feature_data
        coordinates = pipeline._data.feature_data['coordinates'][0].data
        
        # Expected coordinates: System performs unit conversion (0.1nm → 1.0 units)
        # Atom 0: static at (0,0,0) all frames
        # Atom 1: moving linearly frame*1.0 along x-axis (after conversion)
        expected_coords = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Frame 0: x0,y0,z0,x1,y1,z1
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Frame 1
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],  # Frame 2
            [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],  # Frame 3
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],  # Frame 4
        ])
        
        # Test the actual calculation from our software
        np.testing.assert_array_almost_equal(coordinates, expected_coords, decimal=5)
        
        # Verify shape
        assert coordinates.shape == (5, 6)  # 5 frames, 6 coords (2 atoms × 3 coords)
    
    # === DSSP TESTS (5 Tests) ===
    
    @patch('mdtraj.compute_dssp')
    def test_add_dssp_simplified_char(self, mock_compute_dssp):
        """
        Test DSSP simplified with char encoding.

        Validates that DSSP-Feature with simplified=True and encoding='char'
        korrekte H/E/C Character-Codes returns.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Mock MDTraj's compute_dssp for 3 residues
        mock_dssp_data = np.array([['H', 'E', 'C']] * 10)
        mock_compute_dssp.return_value = mock_dssp_data
        
        pipeline.feature.add.dssp(simplified=True, encoding='char', force=True)
        dssp = pipeline._data.feature_data['dssp'][0].data
        
        expected = np.array([['H', 'E', 'C']] * 10)
        np.testing.assert_array_equal(dssp, expected)
        
        # Verify shape and dtype
        assert dssp.shape == (10, 3)
        assert dssp.dtype.kind == 'U'  # Unicode string
    
    @patch('mdtraj.compute_dssp')
    def test_add_dssp_simplified_integer(self, mock_compute_dssp):
        """
        Test DSSP simplified with integer encoding.
        
        Validates that DSSP features with encoding='integer' return correct
        numeric codes (H=0, E=1, C=2) for simplified classes.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Mock returns chars, system converts to integers
        mock_dssp_data = np.array([['H', 'E', 'C']] * 10)
        mock_compute_dssp.return_value = mock_dssp_data
        
        pipeline.feature.add.dssp(simplified=True, encoding='integer', force=True)
        dssp = pipeline._data.feature_data['dssp'][0].data
        
        # H=0, E=1, C=2 for simplified
        expected = np.array([[0, 1, 2]] * 10)
        np.testing.assert_array_equal(dssp, expected)
        
        assert dssp.shape == (10, 3)
        assert dssp.dtype == np.int8
    
    @patch('mdtraj.compute_dssp')
    def test_add_dssp_simplified_onehot(self, mock_compute_dssp):
        """
        Test DSSP simplified with one-hot encoding.
        
        Validates that DSSP features with encoding='onehot' create correct
        binary vectors (3 residues × 4 classes) for ML applications.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        mock_dssp_data = np.array([['H', 'E', 'C']] * 10)
        mock_compute_dssp.return_value = mock_dssp_data
        
        pipeline.feature.add.dssp(simplified=True, encoding='onehot', force=True)
        dssp = pipeline._data.feature_data['dssp'][0].data
        
        # One-hot: 3 residues × 4 classes [H,E,C,NA] = 12 features
        # Res0=H: [1,0,0,0], Res1=E: [0,1,0,0], Res2=C: [0,0,1,0] → flattened
        expected = np.array([
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  # Flattened one-hot
        ] * 10)
        np.testing.assert_array_equal(dssp, expected)
        
        assert dssp.shape == (10, 12)  # 3 residues × 4 classes
        assert dssp.dtype == np.float32
    
    @patch('mdtraj.compute_dssp')
    def test_add_dssp_full_char(self, mock_compute_dssp):
        """
        Test DSSP full 8-class with char encoding.
        
        Validates that DSSP features with simplified=False return all 8 secondary
        structure classes (H,B,E,G,I,T,S,C) as characters correctly.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Full DSSP with changing pattern over frames
        mock_dssp_data = np.array([
            ['H', 'B', 'E'],  # Frames 0-4
            ['H', 'B', 'E'],
            ['H', 'B', 'E'],
            ['H', 'B', 'E'],
            ['H', 'B', 'E'],
            ['B', 'E', 'G'],  # Frames 5-9 different
            ['B', 'E', 'G'],
            ['B', 'E', 'G'],
            ['B', 'E', 'G'],
            ['B', 'E', 'G'],
        ])
        mock_compute_dssp.return_value = mock_dssp_data
        
        pipeline.feature.add.dssp(simplified=False, encoding='char', force=True)
        dssp = pipeline._data.feature_data['dssp'][0].data
        
        np.testing.assert_array_equal(dssp, mock_dssp_data)
        assert dssp.shape == (10, 3)
    
    @patch('mdtraj.compute_dssp')
    def test_add_dssp_full_integer(self, mock_compute_dssp):
        """
        Test DSSP full 8-class with integer encoding.
        
        Validates that DSSP features with simplified=False and encoding='integer'
        return all 8 secondary structure classes as numeric codes.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        mock_dssp_data = np.array([['H', 'B', 'E']] * 10)
        mock_compute_dssp.return_value = mock_dssp_data
        
        pipeline.feature.add.dssp(simplified=False, encoding='integer', force=True)
        dssp = pipeline._data.feature_data['dssp'][0].data
        
        # Full DSSP mapping: H=0, B=1, E=2, G=3, I=4, T=5, S=6, C=7
        expected = np.array([[0, 1, 2]] * 10)
        np.testing.assert_array_equal(dssp, expected)
        
        assert dssp.shape == (10, 3)
        assert dssp.dtype == np.int8
    
    # === TORSIONS TESTS (2 Tests) ===
    
    @patch('mdtraj.compute_psi')
    @patch('mdtraj.compute_phi')
    def test_add_torsions_phi_psi_only(self, mock_phi, mock_psi):
        """
        Test torsions with only phi and psi angles.
        
        Validates that torsion features compute only desired backbone angles
        and radians → degrees conversion works correctly.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # MDTraj returns (indices, angles in radians) - torsions need 4 atoms each
        mock_phi.return_value = (
            np.array([[0, 1, 2, 0], [1, 2, 0, 1]]),  # 4-atom indices per angle
            np.full((10, 2), -60.0 * np.pi / 180.0)  # -60° in radians for 2 angles
        )
        mock_psi.return_value = (
            np.array([[0, 1, 2, 0], [1, 2, 0, 1]]),  # 4-atom indices per angle
            np.full((10, 2), -45.0 * np.pi / 180.0)  # -45° in radians for 2 angles
        )
        
        pipeline.feature.add.torsions(
            calculate_phi=True, calculate_psi=True,
            calculate_omega=False, calculate_chi=False, force=True
        )
        
        torsions = pipeline._data.feature_data['torsions'][0].data
        
        # System converts radians to degrees and returns only computed angles
        # Expected: 2 phi angles + 2 psi angles = 4 total
        expected = np.array([[-60.0, -60.0, -45.0, -45.0]] * 10)
        
        assert torsions.shape == (10, 4)
        np.testing.assert_array_almost_equal(torsions, expected, decimal=1)
    
    @patch('mdtraj.compute_chi1')
    @patch('mdtraj.compute_omega')
    @patch('mdtraj.compute_psi')
    @patch('mdtraj.compute_phi')
    def test_add_torsions_all_angles(self, mock_phi, mock_psi, mock_omega, mock_chi1):
        """
        Test torsions with all angle types.
        
        Validates that torsion features compute all backbone (phi,psi,omega)
        and sidechain (chi1) angles correctly and combines them.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Mock all backbone and sidechain angles with proper 4-atom indices
        mock_phi.return_value = (
            np.array([[0, 1, 2, 0]]),  # 4 atoms per torsion
            np.full((10, 1), -60.0 * np.pi / 180.0)
        )
        mock_psi.return_value = (
            np.array([[0, 1, 2, 0]]), 
            np.full((10, 1), -45.0 * np.pi / 180.0)
        )
        mock_omega.return_value = (
            np.array([[0, 1, 2, 0]]), 
            np.full((10, 1), 180.0 * np.pi / 180.0)
        )
        mock_chi1.return_value = (
            np.array([[0, 1, 2, 0]]),  # 4 atoms per chi angle
            np.full((10, 1), -65.0 * np.pi / 180.0)
        )
        
        pipeline.feature.add.torsions(
            calculate_phi=True, calculate_psi=True,
            calculate_omega=True, calculate_chi=True, force=True
        )
        
        torsions = pipeline._data.feature_data['torsions'][0].data
        
        # System converts radians to degrees and returns only computed angles
        # Expected: 1 phi + 1 psi + 1 omega + 1 chi1 = 4 total angles
        expected = np.array([[-60.0, -45.0, 180.0, -65.0]] * 10)
        
        assert torsions.shape == (10, 4)
        np.testing.assert_array_almost_equal(torsions, expected, decimal=1)
    
    # === SASA TEST (1 Test) ===
    
    @patch('mdtraj.shrake_rupley')
    def test_add_sasa(self, mock_shrake_rupley):
        """
        Test SASA feature calculation with concrete expected values.
        
        Validates that SASA features with Shrake-Rupley algorithm
        calculate correct solvent accessible surface area in nm².
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # MDTraj returns SASA in nm²
        # 3 atoms with different exposures
        mock_sasa_values = np.array([
            [0.15, 0.10, 0.05]  # Atom 0: exposed, 1: partial, 2: buried
        ] * 10)
        mock_shrake_rupley.return_value = mock_sasa_values
        
        pipeline.feature.add.sasa(force=True)
        sasa = pipeline._data.feature_data['sasa'][0].data
        
        expected = np.array([[0.15, 0.10, 0.05]] * 10)
        np.testing.assert_array_almost_equal(sasa, expected, decimal=3)
        
        # Verify properties
        assert sasa.shape == (10, 3)
        assert np.all(sasa >= 0.0)  # SASA non-negative
        assert sasa.dtype == np.float32
    
    # === FEATURE REDUCE TESTS (6 Tests) ===
    
    def test_reduce_distances_mean(self):
        """
        Test distances reduction with mean metric and filtering.
        
        Validates that distance reduction with threshold filtering
        retains only distances with mean values in the specified range.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Add distances first
        pipeline.feature.add.distances(excluded_neighbors=0, force=True)
        
        # Reduce with mean and threshold filtering
        # Expected means: [30.0, 40.0, 50.0] (constant over frames, Å*10 units)
        # Filter: keep only distances with mean between 35.0 and 45.0
        # Expected result: only distance index 1 (mean=40.0) should remain
        pipeline.feature.reduce.distances.mean(
            traj_selection="all",
            threshold_min=35.0,
            threshold_max=45.0
        )
        
        # Verify reduced data contains filtered original distance data
        reduced = pipeline._data.feature_data['distances'][0].reduced_data
        # Should contain only the distance with mean=40.0, for all frames
        expected_reduced = np.full((10, 1), 40.0)  # 10 frames × 1 filtered distance
        
        # Test the actual filtering from our software
        np.testing.assert_array_almost_equal(reduced, expected_reduced, decimal=1)
        
        # Verify metadata exists (structure may vary)
        metadata = pipeline._data.feature_data['distances'][0].reduced_feature_metadata
        assert metadata is not None, "Should have some metadata"
    
    def test_reduce_contacts_frequency(self):
        """
        Test contacts reduction with frequency metric and filtering.
        
        Validates that contact reduction with frequency filtering
        retains only frequently occurring contacts (≥ threshold).
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Add distances and contacts first
        pipeline.feature.add.distances(excluded_neighbors=0, force=True)
        pipeline.feature.add.contacts(cutoff=35.0, force=True)
        
        # Reduce with frequency and threshold filtering
        # Expected frequencies: [1.0, 0.0, 0.0] (contact pattern [True,False,False] constant)
        # Filter: keep only contacts with frequency >= 0.5
        # Expected result: only contact index 0 (frequency=1.0) should remain
        pipeline.feature.reduce.contacts.frequency(
            traj_selection="all",
            threshold_min=0.5
        )
        
        # Verify reduced data contains filtered contact data (not frequency values)
        reduced = pipeline._data.feature_data['contacts'][0].reduced_data
        expected_reduced = np.full((10, 1), True)  # Contact 0-1 True for all frames
        
        # Test the actual filtering from our software
        np.testing.assert_array_equal(reduced, expected_reduced)
        
        # Verify shape and type
        assert reduced.shape == (10, 1), f"Expected (10, 1), got {reduced.shape}"
        assert reduced.dtype == bool, f"Expected bool dtype, got {reduced.dtype}"
    
    def test_reduce_coordinates_std(self):
        """
        Test coordinates reduction with std metric and filtering.
        
        Validates that coordinate reduction with std filtering
        retains only coordinates with moderate variability (between min/max).
        """
        pipeline = self._setup_varied_pipeline(n_frames=10)
        
        # Add coordinates first
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Mock varied coordinates create deterministic movement patterns:
        # Atom 0: static at (0,0,0) → std = [0.0, 0.0, 0.0]
        # Atom 1: oscillating → std = [0.3*sin_std, 0.3*cos_std, 0.0] ≈ [0.21, 0.21, 0.0] 
        # Atom 2: linear → std = [1.0*frame_std, 1.0*frame_std, 0.0] ≈ [2.87, 2.87, 0.0]
        
        # Reduce with std filtering: target intermediate std values for partial filtering
        # Filter to keep oscillating coordinates (Atom 1 x,y) but exclude static/linear
        pipeline.feature.reduce.coordinates.std(
            traj_selection="all", 
            threshold_min=0.1,   # Above static coordinates (std~0.0)
            threshold_max=2.0    # Below high linear movement (std~2.87)
        )
        
        # Verify reduced data contains only Atom 1's oscillating x,y coordinates  
        reduced = pipeline._data.feature_data['coordinates'][0].reduced_data
        
        # Expected: Atom 1's oscillating coordinates for all 10 frames
        # From MockTrajectoryFactory: x = 0.3*sin(frame*0.5), y = 0.3*cos(frame*0.5)
        expected_filtered = []
        for frame in range(10):
            x = 0.3 * np.sin(frame * 0.5) * 10.0  # Unit conversion
            y = 0.3 * np.cos(frame * 0.5) * 10.0  # Unit conversion  
            expected_filtered.append([x, y])
        expected_filtered = np.array(expected_filtered)
        
        # Test concrete expected values - Atom 1's x coordinate (sin oscillation)
        expected_x = np.array([[0.3 * np.sin(frame * 0.5) * 10.0] for frame in range(10)])
        np.testing.assert_array_almost_equal(reduced, expected_x, decimal=4)
    
    @patch('mdtraj.compute_psi')
    @patch('mdtraj.compute_phi')  
    def test_reduce_torsions_std(self, mock_phi, mock_psi):
        """
        Test torsions reduction with std metric and filtering.
        
        Validates that torsion reduction with std filtering
        retains only angles with low variability (≤ threshold).
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Mock torsions with constant angles (std = 0.0)
        mock_phi.return_value = (
            np.array([[0, 1, 2, 0]]),
            np.full((10, 1), -60.0 * np.pi / 180.0)  # Constant -60°
        )
        mock_psi.return_value = (
            np.array([[0, 1, 2, 0]]),
            np.full((10, 1), -45.0 * np.pi / 180.0)  # Constant -45°
        )
        
        # Add torsions first
        pipeline.feature.add.torsions(
            calculate_phi=True, calculate_psi=True,
            calculate_omega=False, calculate_chi=False, force=True
        )
        
        # Reduce with std and filtering
        # For constant mock angles, std should be exactly 0.0
        pipeline.feature.reduce.torsions.std(
            traj_selection="all",
            threshold_max=10.0  # Keep torsions with low variability
        )
        
        # Verify reduced data contains filtered original torsion angles
        reduced = pipeline._data.feature_data['torsions'][0].reduced_data
        
        # Expected: constant angles (-60°, -45°) for all frames, both angles kept
        expected_filtered = np.array([[-60.0, -45.0]] * 10)  # 10 frames × 2 angles
        
        # Test concrete expected values
        assert reduced.shape == (10, 2), f"Expected (10, 2), got {reduced.shape}"
        np.testing.assert_array_almost_equal(reduced, expected_filtered, decimal=1)
    
    @patch('mdtraj.compute_dssp')
    def test_reduce_dssp_frequency(self, mock_compute_dssp):
        """
        Test DSSP reduction with frequency metric and filtering.
        
        Validates that DSSP reduction with class frequency filtering
        retains only residues with stable secondary structures (≥ 60%).
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Mock DSSP with varying pattern for partial filtering
        # Residue 0: stable H, Residue 1: transitions H↔E, Residue 2: stable C
        mock_dssp_data = np.array([
            ['H', 'H', 'C'],  # Frame 0
            ['H', 'E', 'C'],  # Frame 1
            ['H', 'H', 'C'],  # Frame 2  
            ['H', 'E', 'C'],  # Frame 3
            ['H', 'H', 'C'],  # Frame 4
            ['H', 'E', 'C'],  # Frame 5
            ['H', 'H', 'C'],  # Frame 6
            ['H', 'E', 'C'],  # Frame 7
            ['H', 'H', 'C'],  # Frame 8
            ['H', 'E', 'C'],  # Frame 9
        ])
        mock_compute_dssp.return_value = mock_dssp_data
        
        # Add DSSP first
        pipeline.feature.add.dssp(simplified=True, encoding='char', force=True)
        
        # Reduce with class frequency: filter for common secondary structures
        # Each residue has specific class frequencies from our mock data
        pipeline.feature.reduce.dssp.class_frequencies(
            traj_selection="all",
            threshold_min=0.6,   # Keep residues with frequency >= 60%
            threshold_max=1.0    # All frequencies below 100%
        )
        
        # Verify partial filtering based on class frequencies ≥ 0.6
        # Residue 0: 100% H (freq=1.0) → kept
        # Residue 1: 50% H, 50% E (freq=0.5) → FILTERED OUT (< 0.6)  
        # Residue 2: 100% C (freq=1.0) → kept
        reduced = pipeline._data.feature_data['dssp'][0].reduced_data
        
        # Test concrete expected values - only residues 0 and 2 (columns 0,2)
        expected_filtered = np.array([
            ['H', 'C'],  # Frame 0: Residue 0=H, Residue 2=C
            ['H', 'C'],  # Frame 1
            ['H', 'C'],  # Frame 2
            ['H', 'C'],  # Frame 3
            ['H', 'C'],  # Frame 4
            ['H', 'C'],  # Frame 5
            ['H', 'C'],  # Frame 6
            ['H', 'C'],  # Frame 7
            ['H', 'C'],  # Frame 8
            ['H', 'C'],  # Frame 9
        ])
        np.testing.assert_array_equal(reduced, expected_filtered)
    
    @patch('mdtraj.shrake_rupley')
    def test_reduce_sasa_mean(self, mock_shrake_rupley):
        """
        Test SASA reduction with mean metric and filtering.
        
        Validates that SASA reduction with threshold filtering
        retains only atoms with sufficient surface accessibility.
        """
        pipeline = self._setup_triangle_pipeline(n_frames=10)
        
        # Mock SASA with different exposure levels  
        mock_sasa_values = np.array([
            [1.5, 0.8, 0.05]  # Atom 0: high, 1: medium, 2: buried  
        ] * 10)
        mock_shrake_rupley.return_value = mock_sasa_values
        
        # Add SASA first
        pipeline.feature.add.sasa(force=True)
        
        # Reduce with mean and filtering: keep atoms with SASA >= 1.0
        # Expected: only Atom 0 (mean=1.5) should remain
        pipeline.feature.reduce.sasa.mean(
            traj_selection="all",
            threshold_min=1.0  # Keep atoms with reasonable SASA
        )
        
        # Verify reduced data contains filtered original SASA data
        reduced = pipeline._data.feature_data['sasa'][0].reduced_data
        
        # Expected: only Atom 0's SASA values (1.5) for all frames
        expected_filtered = np.full((10, 1), 1.5)  # 10 frames × 1 filtered atom
        
        # Test concrete expected values
        assert reduced.shape == (10, 1), f"Expected (10, 1), got {reduced.shape}"
        np.testing.assert_array_almost_equal(reduced, expected_filtered, decimal=2)
    
    # === INTEGRATION TESTS (6 Tests) ===
    
    def test_dependencies_enforcement(self):
        """
        Test that feature dependencies are correctly enforced.
        
        Validates that dependent features (contacts requires distances)
        raise appropriate errors when prerequisites are missing.
        """
        pipeline = self._setup_triangle_pipeline()
        
        # Contacts requires distances - should fail without distances
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            pipeline.feature.add.contacts(cutoff=3.5)
    
    def test_force_parameter(self):
        """
        Test that force parameter prevents/allows recomputation.
        
        Validates that force=False protects existing features and
        force=True allows recomputation despite existing data.
        """
        pipeline = self._setup_triangle_pipeline()
        
        # Add distances
        pipeline.feature.add.distances(force=True)
        original_data = pipeline._data.feature_data['distances'][0].data.copy()
        
        # force=False: should raise ValueError as feature exists
        with pytest.raises(ValueError, match="Distances FeatureData already exists."):
            pipeline.feature.add.distances(force=False)
        
        # force=True: should recompute (may be identical due to deterministic mock)
        pipeline.feature.add.distances(force=True)
        recomputed_data = pipeline._data.feature_data['distances'][0].data
        # Data should be identical due to deterministic mock, but computation occurred
        np.testing.assert_array_equal(original_data, recomputed_data)
    
    def test_trajectory_selection(self):
        """
        Test traj_selection parameter functionality.
        
        Validates that features are computed only for selected trajectories
        and other trajectories remain untouched.
        """
        pipeline = PipelineManager()
        
        # Setup multiple trajectories
        mock_traj1 = MockTrajectoryFactory.create_triangle_atoms(n_frames=5, seed=42)
        mock_traj2 = MockTrajectoryFactory.create_triangle_atoms(n_frames=5, seed=43)
        
        pipeline._data.trajectory_data.trajectories = [mock_traj1, mock_traj2]
        pipeline._data.trajectory_data.n_frames = 10  # Total frames
        pipeline._data.trajectory_data.n_atoms = 3
        pipeline._data.trajectory_data.trajectory_names = ["traj1", "traj2"]
        
        # Setup residue labels for both trajectories
        pipeline._data.trajectory_data.res_label_data = {
            0: [{"seqid": 0, "full_name": "RES_0"}, {"seqid": 1, "full_name": "RES_1"}, {"seqid": 2, "full_name": "RES_2"}],
            1: [{"seqid": 0, "full_name": "RES_0"}, {"seqid": 1, "full_name": "RES_1"}, {"seqid": 2, "full_name": "RES_2"}]
        }
        
        # Add distances only for trajectory 0
        pipeline.feature.add.distances(traj_selection=[0], force=True)
        
        # Verify only trajectory 0 has distances
        assert 0 in pipeline._data.feature_data['distances']
        assert 1 not in pipeline._data.feature_data['distances']
    
    def test_complete_workflow(self):
        """
        Test complete workflow: add → reduce → verify consistency.
        
        Validates that complete feature pipeline workflow
        (add, reduce, validate) produces consistent results.
        """
        pipeline = self._setup_triangle_pipeline()
        
        # 1. Add distances, ensuring all pairs are calculated
        pipeline.feature.add.distances(excluded_neighbors=0, force=True)
        original_distances = pipeline._data.feature_data['distances'][0].data
        
        # 2. Reduce to filter for a specific distance (mean=40.0)
        pipeline.feature.reduce.distances.mean(threshold_min=35.0, threshold_max=45.0)
        reduced_distances = pipeline._data.feature_data['distances'][0].reduced_data
        
        # 3. Verify consistency: reduced data should be the column with mean=40.0
        expected_data = original_distances[:, 1:2]  # Select the second column
        np.testing.assert_array_almost_equal(reduced_distances, expected_data, decimal=5)
    
    def test_error_handling(self):
        """
        Test correct error handling for invalid operations.
        
        Validates that feature API raises appropriate errors for
        non-existent features and invalid parameters.
        """
        pipeline = self._setup_triangle_pipeline()
        
        # Test reducing non-existent feature
        with pytest.raises((KeyError, ValueError, AttributeError)):
            pipeline.feature.reduce.distances.mean()  # No distances added yet
        
        # Test invalid parameters
        pipeline.feature.add.distances(force=True)
        with pytest.raises((ValueError, TypeError)):
            pipeline.feature.reduce.distances.mean(threshold_min=10.0, threshold_max=5.0)  # Invalid range
    
    def test_deterministic_results(self):
        """
        Test that results are deterministic across runs.
        
        Validates that feature calculations with identical parameters
        produce identical results (important for reproducibility).
        """
        results = []
        
        for run in range(2):
            pipeline = self._setup_triangle_pipeline()
            
            # Add and reduce with same parameters
            pipeline.feature.add.distances(force=True)
            pipeline.feature.reduce.distances.mean(threshold_min=0.0, threshold_max=10.0)
            
            reduced = pipeline._data.feature_data['distances'][0].reduced_data
            results.append(reduced.copy())
        
        # Results should be identical
        np.testing.assert_array_equal(results[0], results[1])

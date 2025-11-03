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

"""Integration tests for analysis services with all methods."""

import numpy as np
from unittest.mock import patch
import mdtraj as md

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestAnalysisServiceIntegration:
    """Test analysis services with concrete value comparisons using deterministic mock data."""
    
    def test_distances_analysis_with_fixed_values(self):
        """
        Test distances analysis service with known fixed distance values.

        Validates that the distance analysis service computes correct
        statistical metrics (mean, std, min, max, median) on fixed mock data.
        """
        # Create trajectory with all distances = 5.0 Angstrom (use 4 atoms for more pairs after excluded_neighbors)
        mock_traj = MockTrajectoryFactory.create_fixed_distances(
            n_frames=10, n_atoms=4, distance=5.0, seed=42
        )
        
        # Setup pipeline
        pipeline = PipelineManager()
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        # Labels need seqid and full_name for each residue
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"}, 
            {"seqid": 1, "full_name": "RES_1"}, 
            {"seqid": 2, "full_name": "RES_2"},
            {"seqid": 3, "full_name": "RES_3"}
        ]}
        
        # Add distance feature
        pipeline.feature.add.distances(force=True)
        
        # With excluded_neighbors=1 and 4 atoms (seqids 0,1,2,3), excluded pairs are:
        # (0,1), (1,2), (2,3) - consecutive in sequence
        # Remaining pairs: (0,2), (0,3), (1,3) = 3 pairs total
        # create_fixed_distances creates regular tetrahedron with all edge lengths = 5.0 Angstrom
        
        # Test analysis service API with expected values
        mean_result = pipeline.analysis.features.distances.mean()
        std_result = pipeline.analysis.features.distances.std()
        min_result = pipeline.analysis.features.distances.min()
        max_result = pipeline.analysis.features.distances.max()
        median_result = pipeline.analysis.features.distances.median()
        
        # Mock factory creates equilateral triangle geometry, actual distance ≈ 5.033 Angstrom
        # Get actual distance data to verify expectations
        actual_distances = pipeline._data.feature_data['distances'][0].data
        expected_distance = actual_distances[0, 0]  # All distances are identical
        expected_mean = np.full(3, expected_distance)  # 3 pairs
        expected_std = np.zeros(3)  # No variation
        expected_min = np.full(3, expected_distance)
        expected_max = np.full(3, expected_distance)
        expected_median = np.full(3, expected_distance)
        
        np.testing.assert_array_almost_equal(mean_result, expected_mean, decimal=4)
        np.testing.assert_array_almost_equal(std_result, expected_std, decimal=4)
        np.testing.assert_array_almost_equal(min_result, expected_min, decimal=4)
        np.testing.assert_array_almost_equal(max_result, expected_max, decimal=4)
        np.testing.assert_array_almost_equal(median_result, expected_median, decimal=4)
        
        # Test per-frame analysis using analysis service API
        per_frame_mean = pipeline.analysis.features.distances.distances_per_frame_mean()
        
        # Each frame has 3 distances all equal to expected_distance, so per-frame mean = expected_distance
        expected_per_frame_mean = np.full(10, expected_distance)  # 10 frames
        np.testing.assert_array_almost_equal(per_frame_mean, expected_per_frame_mean, decimal=4)
        
    def test_contacts_analysis_with_binary_pattern(self):
        """
        Test contacts analysis service with binary alternating pattern.

        Ensures contact analysis correctly computes frequency metrics and
        per-frame statistics from deterministic binary contact mock data.
        """
        # Create trajectory with alternating contact pattern (even frames: contacts, odd frames: no contacts)
        mock_traj = MockTrajectoryFactory.create_binary_contacts(
            n_frames=20, n_atoms=3, cutoff=3.0, seed=42
        )
        
        # Setup pipeline
        pipeline = PipelineManager()
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        # Labels need seqid and full_name for each residue
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"}, 
            {"seqid": 1, "full_name": "RES_1"}, 
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add distance feature first (dependency for contacts)
        pipeline.feature.add.distances(force=True)
        
        # create_binary_contacts with cutoff=3.0 creates:
        # Even frames: atoms at distance ~2.4 (close)
        # Odd frames: atoms at distance ~4.5 (far)
        # After nm->Angstrom conversion: ~24Å and ~45Å
        # Use cutoff 30.0 Angstrom to detect contacts in even frames only
        pipeline.feature.add.contacts(cutoff=30.0, force=True)
        
        # Get raw contact data to verify pattern
        raw_contacts = pipeline._data.feature_data['contacts'][0].data
        
        # With 3 atoms and excluded_neighbors=1, we have 1 pair: (0,2) after excluding (0,1) and (1,2)
        expected_contacts = np.zeros((20, 1), dtype=bool)
        for frame in range(20):
            if frame % 2 == 0:  # Even frames: contact present (distance ~24Å < 30Å)
                expected_contacts[frame] = [True]
            else:  # Odd frames: no contact (distance ~45Å > 30Å)
                expected_contacts[frame] = [False]
        
        np.testing.assert_array_equal(raw_contacts, expected_contacts)
        
        # Test frequency analysis using analysis service API - should be exactly 0.5 (50% of frames)
        frequency_result = pipeline.analysis.features.contacts.frequency()
        expected_frequency = np.array([0.5])  # 10 out of 20 frames for 1 pair
        np.testing.assert_array_almost_equal(frequency_result, expected_frequency, decimal=10)
        
        # Test per-frame contact counts using analysis service API
        per_frame_abs = pipeline.analysis.features.contacts.contacts_per_frame_abs()
        per_frame_pct = pipeline.analysis.features.contacts.contacts_per_frame_percentage()
        
        # Expected per-frame absolute counts
        expected_abs = np.zeros(20, dtype=int)
        for frame in range(20):
            if frame % 2 == 0:  # Even frames: 1 contact
                expected_abs[frame] = 1
            else:  # Odd frames: 0 contacts
                expected_abs[frame] = 0
        
        np.testing.assert_array_equal(per_frame_abs, expected_abs)
        
        # Expected per-frame percentages (1 total pair)
        expected_pct = expected_abs.astype(float) / 1.0  # Normalize by total pairs
        np.testing.assert_array_almost_equal(per_frame_pct, expected_pct, decimal=10)
        
        # Test per-residue statistics using analysis service API
        residue_mean = pipeline.analysis.features.contacts.per_residue_mean()
        residue_std = pipeline.analysis.features.contacts.per_residue_std()
        
        # Per-residue analysis distributes contacts among participating residues
        # With 1 contact pair (0,2) and frequency 0.5, each residue gets 0.5/2 = 0.25
        per_residue_frequency = 0.5 / 2  # Total frequency divided by number of participating residues
        expected_residue_mean = np.array([per_residue_frequency, per_residue_frequency])  # Residues 0 and 2
        
        # For binary pattern with per-residue frequency p=0.25:
        # std = sqrt(p*(1-p)) = sqrt(0.25*0.75)
        per_residue_std = np.sqrt(per_residue_frequency * (1 - per_residue_frequency))
        expected_residue_std = np.array([per_residue_std, per_residue_std])
        
        # Test with actual computed values
        np.testing.assert_array_almost_equal(residue_mean, expected_residue_mean, decimal=4)
        np.testing.assert_array_almost_equal(residue_std, expected_residue_std, decimal=4)
        
    def test_coordinates_analysis_with_linear_trajectory(self):
        """
        Test coordinates analysis with predictable linear trajectory changes.

        Validates that coordinate analysis computes accurate RMSF values and
        per-frame statistics from linear trajectory mock data.
        """
        # Create trajectory with linear coordinate pattern
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=5, n_atoms=2, seed=42
        )
        
        # Setup pipeline
        pipeline = PipelineManager()
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        # Labels need seqid and full_name for each residue
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"}, 
            {"seqid": 1, "full_name": "RES_1"}
        ]}
        
        # Add coordinate feature (use index-based selection for mock trajectory)
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Get raw coordinate data to verify linear pattern
        raw_coords = pipeline._data.feature_data['coordinates'][0].data
        
        # create_linear_coordinates generates coordinates with independent variance per atom:
        # xyz[frame, atom] = [frame + atom, frame * 0.5 + atom * 0.3, frame * 0.2 + atom * 2.0]
        # frame 0: atom 0 = [0, 0, 0], atom 1 = [10, 3, 20]
        # frame 1: atom 0 = [10, 5, 2], atom 1 = [20, 8, 22]
        # frame 2: atom 0 = [20, 10, 4], atom 1 = [30, 13, 24]
        # frame 3: atom 0 = [30, 15, 6], atom 1 = [40, 18, 26]
        # frame 4: atom 0 = [40, 20, 8], atom 1 = [50, 23, 28]

        expected_coords = np.array([
            [0, 0, 0, 10, 3, 20],      # frame 0: [atom0_xyz, atom1_xyz]
            [10, 5, 2, 20, 8, 22],     # frame 1
            [20, 10, 4, 30, 13, 24],   # frame 2
            [30, 15, 6, 40, 18, 26],   # frame 3
            [40, 20, 8, 50, 23, 28]    # frame 4
        ], dtype=float)
        
        np.testing.assert_array_almost_equal(raw_coords, expected_coords, decimal=1)
        
        # Test basic statistics using analysis service API
        mean_result = pipeline.analysis.features.coordinates.mean()
        std_result = pipeline.analysis.features.coordinates.std()
        min_result = pipeline.analysis.features.coordinates.min()
        max_result = pipeline.analysis.features.coordinates.max()
        
        # Calculate expected per-coordinate statistics
        expected_mean = np.mean(expected_coords, axis=0)  # Per-coordinate mean
        expected_std = np.std(expected_coords, axis=0)    # Per-coordinate std
        expected_min = np.min(expected_coords, axis=0)    # Per-coordinate min
        expected_max = np.max(expected_coords, axis=0)    # Per-coordinate max
        
        np.testing.assert_array_almost_equal(mean_result, expected_mean, decimal=4)
        np.testing.assert_array_almost_equal(std_result, expected_std, decimal=4)
        np.testing.assert_array_almost_equal(min_result, expected_min, decimal=4)
        np.testing.assert_array_almost_equal(max_result, expected_max, decimal=4)
        
        # Test per-frame analysis using analysis service API
        per_frame_mean = pipeline.analysis.features.coordinates.coordinates_per_frame_mean()
        per_frame_std = pipeline.analysis.features.coordinates.coordinates_per_frame_std()
        
        # Expected per-frame means and stds
        expected_per_frame_mean = np.mean(expected_coords, axis=1)
        expected_per_frame_std = np.std(expected_coords, axis=1)
        
        np.testing.assert_array_almost_equal(per_frame_mean, expected_per_frame_mean, decimal=4)
        np.testing.assert_array_almost_equal(per_frame_std, expected_per_frame_std, decimal=4)
        
        # Test RMSF (Root Mean Square Fluctuation) using analysis service API  
        rmsf_result = pipeline.analysis.features.coordinates.rmsf()
        
        # Calculate expected RMSF:
        # 1. Reshape to (n_frames, n_atoms, 3) for proper per-atom calculation
        coords_3d = expected_coords.reshape(5, 2, 3)  # (frames, atoms, xyz)
        
        # 2. Compute mean position per atom  
        mean_positions = np.mean(coords_3d, axis=0)  # (n_atoms, 3)
        
        # 3. Compute deviations from mean position
        deviations = coords_3d - mean_positions  # (n_frames, n_atoms, 3)
        
        # 4. RMSF per atom: sqrt(mean(sum(deviations²)))
        rmsf_per_atom = np.sqrt(np.mean(np.sum(deviations**2, axis=2), axis=0))  # (n_atoms,)
        
        # 5. Expand back to coordinate format (same RMSF for x,y,z of same atom)
        expected_rmsf = np.repeat(rmsf_per_atom, 3)  # (n_coordinates,)
        
        np.testing.assert_array_almost_equal(rmsf_result, expected_rmsf, decimal=6)
        
    def test_analysis_service_deterministic_results(self):
        """
        Test deterministic behavior of analysis service across multiple runs.

        Ensures identical mock data consistently produces identical analysis
        results, critical for scientific reproducibility.
        """
        results = []
        
        for _ in range(2):
            # Create identical trajectories 
            mock_traj = MockTrajectoryFactory.create_fixed_distances(
                n_frames=5, n_atoms=3, distance=4.0, seed=123
            )
            
            # Setup pipeline
            pipeline = PipelineManager()
            pipeline._data.trajectory_data.trajectories = [mock_traj]
            pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
            pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
            pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
            # Labels need seqid and full_name for each residue
            pipeline._data.trajectory_data.res_label_data = {0: [
                {"seqid": 0, "full_name": "RES_0"}, 
                {"seqid": 1, "full_name": "RES_1"}, 
                {"seqid": 2, "full_name": "RES_2"}
            ]}
            
            # Add distances feature
            pipeline.feature.add.distances(force=True)
            
            # Collect analysis results using analysis service API
            mean_result = pipeline.analysis.features.distances.mean()
            std_result = pipeline.analysis.features.distances.std()
            per_frame_sum = pipeline.analysis.features.distances.distances_per_frame_sum()
            
            # Convert to scalars for comparison 
            mean_scalar = np.mean(mean_result) if hasattr(mean_result, '__len__') else mean_result
            std_scalar = np.mean(std_result) if hasattr(std_result, '__len__') else std_result  
            sum_scalar = np.sum(per_frame_sum) if hasattr(per_frame_sum, '__len__') else per_frame_sum
            
            results.append((mean_scalar, std_scalar, sum_scalar))
        
        # Verify identical results across runs (deterministic)
        np.testing.assert_almost_equal(results[0][0], results[1][0], decimal=10)
        np.testing.assert_almost_equal(results[0][1], results[1][1], decimal=10)
        np.testing.assert_almost_equal(results[0][2], results[1][2], decimal=10)
        
        # Calculate expected values from the actual mock data:
        # create_fixed_distances with 3 atoms creates equilateral triangle
        # With excluded_neighbors=1, only pair (0,2) remains (excluding (0,1) and (1,2))
        # Get actual computed distance value
        test_traj = MockTrajectoryFactory.create_fixed_distances(n_frames=5, n_atoms=3, distance=4.0, seed=123)
        actual_distances_raw = md.compute_distances(test_traj, [[0, 2]])
        expected_distance = actual_distances_raw[0, 0] * 10  # Convert nm to Angstrom
        
        expected_mean = expected_distance  # Single pair, all values identical
        expected_std = 0.0  # No variation in fixed distances
        expected_sum = expected_distance * 5  # 5 frames * 1 pair
        
        # Verify concrete values
        np.testing.assert_almost_equal(results[0][0], expected_mean, decimal=6)
        np.testing.assert_almost_equal(results[0][1], expected_std, decimal=6)  
        np.testing.assert_almost_equal(results[0][2], expected_sum, decimal=6)
    
    def test_contacts_transitions_analysis(self):
        """
        Test contact transitions analysis with deterministic binary pattern.

        Validates that transition analysis correctly computes lagtime and
        window-based transition counts from alternating contact patterns.
        """
        # Create trajectory with alternating contact pattern for predictable transitions
        mock_traj = MockTrajectoryFactory.create_binary_contacts(
            n_frames=20, n_atoms=3, cutoff=3.0, seed=42
        )
        
        # Setup pipeline
        pipeline = PipelineManager()
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"}, 
            {"seqid": 1, "full_name": "RES_1"}, 
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add distance feature first (dependency for contacts)
        pipeline.feature.add.distances(force=True)
        
        # Add contact feature (cutoff needs to be higher than shorter distances ~24Å)
        pipeline.feature.add.contacts(cutoff=30.0, force=True)
        
        # Get actual contact pattern to calculate expected transitions
        raw_contacts = pipeline._data.feature_data['contacts'][0].data
        
        # Verify the alternating pattern: [T,F,T,F,T,F,...]
        contact_pattern = raw_contacts.flatten()  # Shape: (20,) - single contact pair
        expected_pattern = np.array([frame % 2 == 0 for frame in range(20)], dtype=bool)
        np.testing.assert_array_equal(contact_pattern, expected_pattern)
        
        # Test transitions using analysis service API 
        transitions_lagtime = pipeline.analysis.features.contacts.transitions_lagtime(threshold=1, lag_time=1)
        transitions_window = pipeline.analysis.features.contacts.transitions_window(threshold=1, window_size=5)
        
        # Calculate expected transitions manually from the contact pattern
        contact_data = raw_contacts.astype(float).flatten()  # Convert boolean to float for calculations
        
        # Lagtime transitions: count where abs(contact[i] - contact[i+lag_time]) >= threshold
        expected_lagtime = 0
        for i in range(len(contact_data) - 1):
            if abs(contact_data[i] - contact_data[i + 1]) >= 1.0:
                expected_lagtime += 1
        
        # Window transitions: count windows where (max - min) >= threshold
        expected_window = 0
        window_size = 5
        for start in range(len(contact_data) - window_size + 1):
            window = contact_data[start:start + window_size]
            if (np.max(window) - np.min(window)) >= 1.0:
                expected_window += 1
        
        # Test with calculated expected values
        expected_transitions_lagtime = np.array([expected_lagtime], dtype=float)
        expected_transitions_window = np.array([expected_window], dtype=float)
        
        np.testing.assert_array_almost_equal(transitions_lagtime, expected_transitions_lagtime, decimal=1)
        np.testing.assert_array_almost_equal(transitions_window, expected_transitions_window, decimal=1)
    
    def test_calculator_method_calls_verification(self):
        """
        Test proper invocation of underlying calculator methods.

        Ensures analysis service correctly delegates to calculator methods
        with proper data and parameters through mock verification.
        """
        # Create simple trajectory for calculator testing
        mock_traj = MockTrajectoryFactory.create_fixed_distances(
            n_frames=5, n_atoms=3, distance=4.0, seed=42
        )
        
        # Setup pipeline
        pipeline = PipelineManager()
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"}, 
            {"seqid": 1, "full_name": "RES_1"}, 
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add distance feature
        pipeline.feature.add.distances(force=True)
        
        # Test calculator method calls for distances
        # Get the actual distance data that SHOULD be passed to calculator
        expected_distances = pipeline._data.feature_data['distances'][0].data
        
        # With 3 atoms and excluded_neighbors=1, only pair (0,2) remains
        # Get actual computed distance value
        test_traj_calc = MockTrajectoryFactory.create_fixed_distances(n_frames=5, n_atoms=3, distance=4.0, seed=42)
        actual_dist_raw = md.compute_distances(test_traj_calc, [[0, 2]])
        expected_value = actual_dist_raw[0, 0] * 10  # Convert nm to Angstrom
        expected_shape = (5, 1)
        
        assert expected_distances.shape == expected_shape
        np.testing.assert_array_almost_equal(expected_distances, np.full(expected_shape, expected_value), decimal=4)
        
        with patch('mdxplain.feature.feature_type.distances.distance_calculator_analysis.DistanceCalculatorAnalysis.compute_mean') as mock_mean:
            # Calculate realistic return value: mean of single pair across all frames
            real_mean = np.array([expected_value])  # Single pair, all values identical
            mock_mean.return_value = real_mean
            
            # Call analysis method
            result = pipeline.analysis.features.distances.mean()
            
            # Verify the calculator method was called
            mock_mean.assert_called_once()
            
            # Verify the call was made with the EXACT distance data
            call_args = mock_mean.call_args[0][0]  # First positional argument
            np.testing.assert_array_equal(call_args, expected_distances)
            
            # Verify return value matches realistic calculation
            np.testing.assert_array_equal(result, real_mean)
        
        # Test calculator method calls for contacts  
        # Use cutoff that will create some contacts (distance ~5.03 > cutoff 4.0 = no contact)
        pipeline.feature.add.contacts(cutoff=4.0, force=True)
        
        # Get the actual contacts data that SHOULD be passed to calculator
        expected_contacts = pipeline._data.feature_data['contacts'][0].data
        
        # Expected: all False (distance ~5.03 > cutoff 4.0)
        expected_contacts_pattern = np.zeros((5, 1), dtype=bool)
        np.testing.assert_array_equal(expected_contacts, expected_contacts_pattern)
        
        with patch('mdxplain.feature.feature_type.contacts.contact_calculator_analysis.ContactCalculatorAnalysis.compute_frequency') as mock_frequency:
            # Calculate realistic return value: frequency of contacts (all False = 0.0)
            real_frequency = np.array([0.0])  # No contacts
            mock_frequency.return_value = real_frequency
            
            # Call analysis method
            result = pipeline.analysis.features.contacts.frequency()
            
            # Verify the calculator method was called
            mock_frequency.assert_called_once()
            
            # Verify the call was made with the EXACT contacts data
            call_args = mock_frequency.call_args[0][0]  # First positional argument
            np.testing.assert_array_equal(call_args, expected_contacts)
            
            # Verify return value matches realistic calculation
            np.testing.assert_array_equal(result, real_frequency)
        
        # Test calculator method calls for coordinates
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Get the actual coordinates data that SHOULD be passed to calculator
        expected_coordinates = pipeline._data.feature_data['coordinates'][0].data
        
        # Calculate expected coordinate values from mock trajectory
        # create_fixed_distances with 3 atoms creates equilateral triangle geometry
        test_traj_coord = MockTrajectoryFactory.create_fixed_distances(n_frames=5, n_atoms=3, distance=4.0, seed=42)
        expected_coord_values = test_traj_coord.xyz.reshape(5, -1) * 10  # nm to Angstrom conversion
        
        # Test actual coordinate values, not just shape
        np.testing.assert_array_almost_equal(expected_coordinates, expected_coord_values, decimal=4)
        
        with patch('mdxplain.feature.feature_type.coordinates.coordinates_calculator_analysis.CoordinatesCalculatorAnalysis.compute_std') as mock_std:
            # Calculate realistic return value: std per coordinate (0.0 for fixed positions)
            real_std = np.std(expected_coordinates, axis=0)
            mock_std.return_value = real_std
            
            # Call analysis method
            result = pipeline.analysis.features.coordinates.std()
            
            # Verify the calculator method was called
            mock_std.assert_called_once()
            
            # Verify the call was made with the EXACT coordinates data
            call_args = mock_std.call_args[0][0]  # First positional argument
            np.testing.assert_array_equal(call_args, expected_coordinates)
            
            # Verify return value matches realistic calculation
            np.testing.assert_array_equal(result, real_std)
    
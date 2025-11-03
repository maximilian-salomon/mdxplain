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

"""Integration tests for FeatureSelectorAddService property-based API."""

import numpy as np
import pytest
import mdtraj as md
from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances
from mdxplain.feature.feature_type.contacts.contacts import Contacts


class TestFeatureSelectorAddService:
    """Test new property-based add service API for feature selection."""

    def setup_method(self):
        """
        Create synthetic trajectory with controlled properties.

        Creates a simple synthetic trajectory with 7 residues and 20 frames.
        Identical setup to TestSelectionStrings for consistency.
        """
        # Create topology: ALA(1), GLY(2), VAL(3), ALA(4), GLY(5), LEU(6), PHE(7)
        topology = md.Topology()
        chain = topology.add_chain()

        residue_names = ["ALA", "GLY", "VAL", "ALA", "GLY", "LEU", "PHE"]
        residues = []

        for name in residue_names:
            residues.append(topology.add_residue(name, chain))

        # Add CA atoms
        for residue in residues:
            topology.add_atom("CA", md.element.carbon, residue)

        # Create coordinates (7 atoms, 20 frames)
        coordinates = []
        for frame in range(20):
            frame_coords = []
            for atom_idx in range(7):
                x = atom_idx * 1.0
                y = frame * 0.1
                z = 0.0
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)

        xyz = np.array(coordinates)
        self.test_traj = md.Trajectory(xyz, topology)

        # Setup pipeline
        self.pipeline = PipelineManager()
        self.pipeline.data.trajectory_data.trajectories = [self.test_traj]
        self.pipeline.data.trajectory_data.trajectory_names = ["synthetic"]

        # Create residue metadata
        residue_metadata = []
        for i, res in enumerate(self.test_traj.topology.residues):
            residue_metadata.append({
                "resid": res.resSeq + 1,
                "seqid": res.index + 1,
                "resname": res.name,
                "aaa_code": res.name,
                "a_code": res.name[0] if res.name else "X",
                "consensus": None,
                "full_name": f"{res.name}{res.index + 1}",
                "index": res.index
            })

        self.pipeline.data.trajectory_data.res_label_data = {0: residue_metadata}

        # Add features: distances with excluded_neighbors=0 (all pairs)
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))

    def _get_selected_indices(self, selector_name: str):
        """
        Execute feature selection and retrieve selected feature indices.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector to execute and retrieve results from.

        Returns
        -------
        list of int
            List of selected feature indices from trajectory 0 for the distances
            feature type.
        """
        self.pipeline.feature_selector.select(selector_name, reference_traj=0)
        return self.pipeline.data.selected_feature_data[selector_name].get_results("distances")["trajectory_indices"][0]["indices"]

    def test_add_distances_selection(self):
        """
        Test add.distances() API for basic selection.

        Validates that add.distances() with ALA selection produces
        the same results as traditional add() method for consistency.
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances("test", "res ALA")
        indices = self._get_selected_indices("test")

        # Calculate expected ALA pairs (residues 0,3)
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:  # ALA residues
                    expected.append(pair_idx)
                pair_idx += 1

        assert sorted(indices) == sorted(expected)

    def test_add_distances_with_parameters(self):
        """
        Test add_to().distances() with additional parameters.

        Validates that all parameters (use_reduced, require_all_partners, etc.)
        are correctly passed through the service API.
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances(
            "test",
            "res ALA",
            use_reduced=False,
            require_all_partners=True,
            common_denominator=True
        )
        indices = self._get_selected_indices("test")

        # With require_all_partners=True, only pairs where BOTH partners are ALA
        # ALA residues: 0, 3
        # Only pair (0,3) has both partners as ALA
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] and j in [0, 3]:  # Both partners must be ALA
                    expected.append(pair_idx)
                pair_idx += 1

        assert sorted(indices) == sorted(expected)

    def test_add_contacts_selection(self):
        """
        Test add_to().contacts() API for contacts feature.

        Validates that add_to().contacts() works correctly for contacts
        feature type and produces expected results.
        """
        # Add contacts feature first
        self.pipeline.feature.add_feature(Contacts(cutoff=4.0))

        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.contacts("test", "res GLY")
        self.pipeline.feature_selector.select("test", reference_traj=0)

        # Verify contacts results exist
        results = self.pipeline.data.selected_feature_data["test"].get_results("contacts")
        assert "trajectory_indices" in results
        assert 0 in results["trajectory_indices"]

        # GLY residues are at indices 1,4 - should have pairs containing these
        indices = results["trajectory_indices"][0]["indices"]

        # Calculate expected GLY pairs
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [1, 4] or j in [1, 4]:  # GLY residues
                    expected.append(pair_idx)
                pair_idx += 1

        assert sorted(indices) == sorted(expected)

    def test_add_multiple_features(self):
        """
        Test multiple feature types via add_to() service.

        Validates that add_to() can be used to add selections for multiple
        feature types to the same selector configuration.
        """
        # Add contacts feature
        self.pipeline.feature.add_feature(Contacts(cutoff=4.0))

        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances("test", "res ALA")
        self.pipeline.feature_selector.add.contacts("test", "res GLY")
        self.pipeline.feature_selector.select("test", reference_traj=0)

        # Get explicit results
        results = self.pipeline.data.selected_feature_data["test"].get_all_results()

        distance_indices = results["distances"]["trajectory_indices"][0]["indices"]
        contact_indices = results["contacts"]["trajectory_indices"][0]["indices"]

        # Calculate expected ALA distance pairs (residues 0,3)
        expected_distances = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:  # ALA residues
                    expected_distances.append(pair_idx)
                pair_idx += 1

        # Calculate expected GLY contact pairs (residues 1,4)
        expected_contacts = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [1, 4] or j in [1, 4]:  # GLY residues
                    expected_contacts.append(pair_idx)
                pair_idx += 1

        # Verify exact indices
        assert sorted(distance_indices) == sorted(expected_distances), \
            f"Distance indices mismatch: expected {expected_distances}, got {distance_indices}"
        assert sorted(contact_indices) == sorted(expected_contacts), \
            f"Contact indices mismatch: expected {expected_contacts}, got {contact_indices}"

    def test_add_coordinates_selection(self):
        """
        Test add_to().coordinates() API for coordinates feature.

        Validates that coordinates feature selections work correctly
        through the add_to() service API. Coordinates are non-pairwise features.
        """
        # Import here to avoid circular imports
        from mdxplain.feature.feature_type.coordinates.coordinates import Coordinates

        # Add coordinates feature with correct parameters
        self.pipeline.feature.add_feature(Coordinates(selection="name CA"))

        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.coordinates("test", "all")
        self.pipeline.feature_selector.select("test", reference_traj=0)

        # Get explicit results
        results = self.pipeline.data.selected_feature_data["test"].get_results("coordinates")
        indices = results["trajectory_indices"][0]["indices"]

        # For coordinates with "all" selection and "name CA", should get all 7 CA atoms
        # Each atom contributes x, y, z coordinates = 21 total features
        # With selection="all", we select all 21 coordinate features
        expected_indices = list(range(21))  # All coordinate features (7 atoms × 3 dimensions)

        assert sorted(indices) == sorted(expected_indices), \
            f"Coordinates indices mismatch: expected {expected_indices}, got {indices}"

    def test_add_to_nonexistent_selector_raises_error(self):
        """
        Test add.distances() with nonexistent selector raises ValueError.

        Validates that attempting to use add.distances() on a selector that doesn't
        exist raises appropriate error with helpful message.
        """
        with pytest.raises(ValueError, match="Feature selector 'nonexistent' does not exist"):
            self.pipeline.feature_selector.add.distances("nonexistent", "res ALA")

    def test_add_api_vs_traditional_add_consistency(self):
        """
        Test add.distances() produces same results as traditional add() method.

        Validates that both API styles produce identical results for
        equivalent selections, ensuring backward compatibility.
        """
        # Create two selectors with equivalent selections
        self.pipeline.feature_selector.create("traditional")
        self.pipeline.feature_selector.add_selection("traditional", "distances", "res ALA and res GLY")

        self.pipeline.feature_selector.create("service")
        self.pipeline.feature_selector.add.distances("service", "res ALA and res GLY")

        # Select both
        self.pipeline.feature_selector.select("traditional", reference_traj=0)
        self.pipeline.feature_selector.select("service", reference_traj=0)

        # Get results
        traditional_indices = self.pipeline.data.selected_feature_data["traditional"].get_results("distances")["trajectory_indices"][0]["indices"]
        service_indices = self.pipeline.data.selected_feature_data["service"].get_results("distances")["trajectory_indices"][0]["indices"]

        # Results should be identical
        assert sorted(traditional_indices) == sorted(service_indices)

    def test_add_chaining_multiple_selections(self):
        """
        Test chaining multiple add_to() calls for same feature type.

        Validates that multiple add_to().distances() calls on the same
        selector combine selections using UNION logic.
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances("test", "res ALA")
        self.pipeline.feature_selector.add.distances("test", "res GLY")
        indices = self._get_selected_indices("test")

        # Should be union of ALA and GLY selections
        # ALA: indices 0,3; GLY: indices 1,4
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 1, 3, 4] or j in [0, 1, 3, 4]:  # ALA or GLY
                    expected.append(pair_idx)
                pair_idx += 1

        assert sorted(indices) == sorted(expected)
        # Ensure no duplicates
        assert len(indices) == len(set(indices))

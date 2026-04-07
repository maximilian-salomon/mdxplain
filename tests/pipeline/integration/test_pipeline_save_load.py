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

"""Integration tests for pipeline save and load functionality."""

import pickle
import tarfile
import hashlib
import pytest
import numpy as np
import mdtraj as md
from pathlib import Path
import re
import tempfile
import shutil

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances import Distances
from mdxplain.feature.feature_type.contacts import Contacts
from mdxplain.clustering.cluster_type.dbscan import DBSCAN
from mdxplain.decomposition.decomposition_type.pca import PCA
from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory
from mdxplain.utils.archive_utils import ArchiveUtils
from mdxplain.utils.cleanup_utils import CleanupUtils
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestPipelineSaveLoad:
    """Test pipeline save and load functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def _build_res_labels(self, n_atoms: int) -> list:
        """
        Build residue labels for mock trajectories.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the mock trajectory.

        Returns
        -------
        list
            List of residue label dictionaries.
        """
        return [
            {"seqid": idx, "full_name": f"RES_{idx}"} for idx in range(n_atoms)
        ]

    def _assign_mock_trajectory(
        self, pipeline: PipelineManager, mock_traj, name: str
    ) -> None:
        """
        Assign mock trajectory data to the pipeline.

        Parameters
        ----------
        pipeline : PipelineManager
            Pipeline to update.
        mock_traj : object
            Mock trajectory with xyz data and topology info.
        name : str
            Trajectory name to set.

        Returns
        -------
        None
        """
        traj_data = pipeline._data.trajectory_data
        traj_data.trajectories = [mock_traj]
        traj_data.trajectory_names = [name]
        traj_data.n_frames = mock_traj.n_frames
        traj_data.n_atoms = mock_traj.n_atoms
        traj_data.res_label_data = {0: self._build_res_labels(mock_traj.n_atoms)}

    def _build_memmap_pipeline(
        self, temp_dir: Path, cache_root: Path = None
    ) -> PipelineManager:
        """
        Build a pipeline with memmap-backed feature data.

        Parameters
        ----------
        temp_dir : Path
            Temporary directory for cache files.

        Returns
        -------
        PipelineManager
            Pipeline configured for memmap-backed feature computation.
        """
        cache_dir = cache_root if cache_root is not None else (temp_dir / "cache")
        pipeline = PipelineManager(use_memmap=True, cache_dir=str(cache_dir))
        mock_traj = MockTrajectoryFactory.create_simple(
            n_frames=120, n_atoms=25, seed=42
        )
        self._assign_mock_trajectory(pipeline, mock_traj, "mock_trajectory")
        pipeline.feature.add_feature(Distances(), force=True)
        return pipeline

    def _build_non_memmap_pipeline(
        self, temp_dir: Path, cache_root: Path = None
    ) -> PipelineManager:
        """
        Build a pipeline with in-memory (non-memmap) feature data.
        """
        cache_dir = (
            cache_root if cache_root is not None else (temp_dir / "cache_no_memmap")
        )
        pipeline = PipelineManager(use_memmap=False, cache_dir=str(cache_dir))
        mock_traj = MockTrajectoryFactory.create_simple(
            n_frames=120, n_atoms=25, seed=42
        )
        self._assign_mock_trajectory(pipeline, mock_traj, "mock_trajectory")
        pipeline.feature.add_feature(Distances(), force=True)
        return pipeline

    def _create_mdtraj_input_files(
        self, temp_dir: Path, stem: str, *, n_frames: int, n_atoms: int, seed: int
    ) -> tuple[str, str, np.ndarray]:
        """
        Create small MDTraj input files (xtc/pdb) for DaskMDTrajectory tests.
        """
        rng = np.random.default_rng(seed)
        xyz = rng.random((n_frames, n_atoms, 3), dtype=np.float32)
        topology = md.Topology()
        chain = topology.add_chain()
        for atom_idx in range(n_atoms):
            residue = topology.add_residue("ALA", chain)
            topology.add_atom(f"C{atom_idx}", md.element.carbon, residue)
        traj = md.Trajectory(xyz, topology)
        traj_file = temp_dir / f"{stem}.xtc"
        top_file = temp_dir / f"{stem}.pdb"
        traj.save_xtc(str(traj_file))
        traj[0].save_pdb(str(top_file))
        return str(traj_file), str(top_file), xyz

    def _build_memmap_pipeline_with_dask_trajectories(
        self,
        temp_dir: Path,
        cache_root: Path,
        n_trajectories: int = 1,
    ) -> tuple[PipelineManager, list[np.ndarray]]:
        """
        Build memmap pipeline with one or more DaskMDTrajectory inputs.
        """
        pipeline = PipelineManager(use_memmap=True, cache_dir=str(cache_root))
        runtime_cache = Path(pipeline.get_config()["cache_dir"])

        trajectories = []
        trajectory_names = []
        res_label_data = {}
        expected_xyz_blocks = []
        total_frames = 0
        n_atoms = 6

        for traj_idx in range(n_trajectories):
            traj_file, top_file, xyz = self._create_mdtraj_input_files(
                temp_dir,
                f"dask_input_{traj_idx}",
                n_frames=12 + traj_idx,
                n_atoms=n_atoms,
                seed=100 + traj_idx,
            )
            zarr_cache_path = runtime_cache / f"traj_{traj_idx}.dask.zarr"
            dask_traj = DaskMDTrajectory(
                traj_file,
                top_file,
                zarr_cache_path=str(zarr_cache_path),
                chunk_size=4,
            )
            trajectories.append(dask_traj)
            trajectory_names.append(f"dask_traj_{traj_idx}")
            res_label_data[traj_idx] = self._build_res_labels(n_atoms)
            expected_xyz_blocks.append(np.array(dask_traj.xyz))
            total_frames += dask_traj.n_frames

        traj_data = pipeline._data.trajectory_data
        traj_data.trajectories = trajectories
        traj_data.trajectory_names = trajectory_names
        traj_data.n_frames = total_frames
        traj_data.n_atoms = n_atoms
        traj_data.res_label_data = res_label_data

        pipeline.feature.add_feature(Distances(), force=True)
        return pipeline, expected_xyz_blocks

    def _build_memmap_pipeline_with_dask_full_analysis(
        self,
        temp_dir: Path,
        cache_root: Path,
        n_trajectories: int = 2,
    ) -> tuple[PipelineManager, list[np.ndarray]]:
        """
        Build a memmap pipeline with Dask trajectories and downstream analysis.

        The pipeline includes distances, feature selection, decomposition, and
        clustering so that archive reload validation covers all memmap-backed
        analysis layers.
        """
        pipeline, expected_xyz_blocks = self._build_memmap_pipeline_with_dask_trajectories(
            temp_dir=temp_dir,
            cache_root=cache_root,
            n_trajectories=n_trajectories,
        )
        pipeline.feature_selector.create("all_distances")
        pipeline.feature_selector.add_selection(
            "all_distances",
            "distances",
            "all",
            use_reduced=False,
        )
        pipeline.feature_selector.select("all_distances")
        pipeline.decomposition.add_decomposition(
            "all_distances",
            PCA(n_components=2),
            decomposition_name="all_distances_pca",
        )
        pipeline.clustering.add_clustering(
            "all_distances",
            DBSCAN(eps=1.5, min_samples=2),
            use_decomposed=False,
            cluster_name="all_distances_cluster",
        )
        return pipeline, expected_xyz_blocks

    def _collect_memmaps_recursive(self, root: object) -> list[np.memmap]:
        """
        Collect all numpy.memmap instances reachable from a root object graph.
        """
        memmaps = []
        seen = set()
        stack = [root]
        while stack:
            value = stack.pop()
            if value is None:
                continue
            if isinstance(value, np.memmap):
                memmaps.append(value)
                continue
            if isinstance(value, (str, bytes, int, float, bool, np.generic)):
                continue
            if isinstance(value, np.ndarray):
                continue

            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)

            if isinstance(value, dict):
                stack.extend(value.values())
                continue
            if isinstance(value, (list, tuple, set)):
                stack.extend(value)
                continue
            if hasattr(value, "__dict__"):
                stack.extend(vars(value).values())
        return memmaps

    def _list_relative_files(self, root: Path) -> set:
        """Return all files below root as normalized relative paths."""
        if not root.exists():
            return set()
        return {
            str(path.relative_to(root)).replace("\\", "/")
            for path in root.rglob("*")
            if path.is_file()
        }

    def _list_archive_cache_files(self, archive_path: str) -> set:
        """Return archived file paths stored below cache/ (relative to cache root)."""
        files = set()
        with tarfile.open(archive_path, "r:*") as tar:
            for name in tar.getnames():
                normalized = name.replace("\\", "/")
                if not normalized.startswith("cache/"):
                    continue
                rel = normalized[len("cache/") :]
                if rel and not rel.endswith("/"):
                    files.add(rel)
        return files

    def _list_archive_file_members(self, archive_path: str) -> list:
        """Return normalized sorted archive file member names."""
        members = []
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    members.append(member.name.replace("\\", "/"))
        return sorted(members)

    def _archive_file_hashes(self, archive_path: str, prefix: str = "") -> dict:
        """Return sha256 per archived file member (optionally filtered by prefix)."""
        hashes = {}
        normalized_prefix = prefix.replace("\\", "/")
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                normalized_name = member.name.replace("\\", "/")
                if normalized_prefix and not normalized_name.startswith(normalized_prefix):
                    continue
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    continue
                hashes[normalized_name] = hashlib.sha256(file_obj.read()).hexdigest()
        return dict(sorted(hashes.items()))

    def _list_scoped_cache_dirs(self, cache_root: Path) -> list:
        """Return scoped cache subdirectories under a cache root."""
        if not cache_root.exists():
            return []
        return sorted(
            [path for path in cache_root.iterdir() if path.is_dir()],
            key=lambda p: p.name,
        )

    def test_save_load_empty_pipeline(self, temp_dir):
        """
        Test that empty pipeline save/load preserves structure.

        Validates that empty PipelineManager is correctly saved and loaded
        with all standard managers (trajectory, feature, clustering, decomposition).
        """
        # Create empty pipeline
        pipeline = PipelineManager()
        
        # Save pipeline
        save_path = temp_dir / "empty_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Verify file exists
        assert save_path.exists()
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify basic structure
        assert loaded is not None
        assert hasattr(loaded, 'trajectory')
        assert hasattr(loaded, 'feature')
        assert hasattr(loaded, 'clustering')
        assert hasattr(loaded, 'decomposition')
        
    def test_save_load_pipeline_with_trajectory(self, temp_dir):
        """
        Test that pipeline with trajectory data saves and loads correctly.

        Validates that pipeline with mock trajectory (n_frames, n_atoms, xyz)
        is identically reconstructed after save/load cycle.
        """
        # Create pipeline with mock trajectory
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=50, n_atoms=20, seed=42)
        
        # Manually set trajectory data (bypassing file loading)
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        
        # Save pipeline
        save_path = temp_dir / "trajectory_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify trajectory data
        assert loaded._data.trajectory_data.n_frames == 50
        assert loaded._data.trajectory_data.n_atoms == 20
        assert len(loaded._data.trajectory_data.trajectories) == 1
        
        # Verify coordinates are identical
        original_xyz = pipeline._data.trajectory_data.trajectories[0].xyz
        loaded_xyz = loaded._data.trajectory_data.trajectories[0].xyz
        np.testing.assert_array_equal(original_xyz, loaded_xyz)
        
    def test_save_load_pipeline_with_features(self, temp_dir):
        """
        Test that pipeline with feature data preserves arrays and metadata.

        Validates that pipeline with distances features (data + metadata)
        is correctly saved and loaded with identical feature arrays.
        """
        # Create pipeline with trajectory and features
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=30, n_atoms=10, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distance feature - minimal for save/load testing
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Save original data for comparison
        original_distances = pipeline._data.feature_data["distances"][0].data.copy()
        
        # Save pipeline
        save_path = temp_dir / "features_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify feature data exists
        assert "distances" in loaded._data.feature_data
        
        # Verify distances data
        np.testing.assert_array_equal(loaded._data.feature_data["distances"][0].data, original_distances)
        assert loaded._data.feature_data["distances"][0].data.shape == (30, 36)  # All distance pairs for 10 atoms
        
        # Verify feature metadata
        assert loaded._data.feature_data["distances"][0].feature_metadata is not None
        
    def test_save_load_pipeline_with_clustering(self, temp_dir):
        """
        Test that pipeline with clustering results preserves assignments.

        Validates that DBSCAN clustering (labels + metadata) is correctly
        saved and loaded with identical cluster assignments.
        """
        # Create pipeline with features and clustering
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_two_state(n_atoms=20, n_frames=50, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distances and create selection
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Create feature selection
        pipeline.feature_selector.create("test_selection")
        pipeline.feature_selector.add_selection( "test_selection", "distances", "all")
        pipeline.feature_selector.select("test_selection")
        
        # Add clustering
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        pipeline.clustering.add_clustering("test_selection", dbscan, use_decomposed=False, cluster_name="test_cluster")
        
        # Save original clustering results
        original_labels = pipeline._data.cluster_data['test_cluster'].labels.copy()
        original_metadata = pipeline._data.cluster_data['test_cluster'].metadata.copy()
        
        # Save pipeline
        save_path = temp_dir / "clustering_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify clustering data
        assert 'test_cluster' in loaded._data.cluster_data
        np.testing.assert_array_equal(loaded._data.cluster_data['test_cluster'].labels, original_labels)
        
        # Verify clustering metadata
        assert loaded._data.cluster_data['test_cluster'].metadata['algorithm'] == original_metadata['algorithm']
        assert loaded._data.cluster_data['test_cluster'].metadata['n_clusters'] == original_metadata['n_clusters']
        assert loaded._data.cluster_data['test_cluster'].metadata['n_noise'] == original_metadata['n_noise']
        
    def test_save_load_pipeline_with_decomposition(self, temp_dir):
        """
        Test that pipeline with decomposition results preserves components.

        Validates that PCA decomposition (components + explained variance)
        is correctly saved and loaded with identical principal components.
        """
        # Create pipeline with features and decomposition
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=40, n_atoms=15, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distances
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Create feature selection
        pipeline.feature_selector.create("decomp_selection")
        pipeline.feature_selector.add_selection( "decomp_selection", "distances", "all")
        pipeline.feature_selector.select("decomp_selection")
        
        # Add PCA decomposition
        pca = PCA(n_components=2)
        pipeline.decomposition.add_decomposition("decomp_selection", pca, decomposition_name="test_pca")
        
        # Save original decomposition results
        original_components = pipeline._data.decomposition_data['test_pca'].data.copy()
        original_metadata = pipeline._data.decomposition_data['test_pca'].metadata.copy()
        
        # Save pipeline
        save_path = temp_dir / "decomposition_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify decomposition data
        assert 'test_pca' in loaded._data.decomposition_data
        np.testing.assert_array_equal(loaded._data.decomposition_data['test_pca'].data, original_components)
        assert loaded._data.decomposition_data['test_pca'].data.shape == (40, 2)
        
        # Verify decomposition metadata
        assert loaded._data.decomposition_data['test_pca'].metadata['method'] == original_metadata['method']
        assert loaded._data.decomposition_data['test_pca'].metadata['hyperparameters']['n_components'] == original_metadata['hyperparameters']['n_components']
        assert 'explained_variance_ratio' in loaded._data.decomposition_data['test_pca'].metadata
        
    def test_save_load_complete_pipeline(self, temp_dir):
        """
        Test that complete pipeline saves and loads all components correctly.

        Validates that complete pipeline (trajectory, features, clustering,
        decomposition, feature selectors) is correctly reconstructed.
        """
        # Create complete pipeline
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_atoms=10, n_frames=25, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add multiple features
        distances = Distances()
        contacts = Contacts(cutoff=5.0)
        pipeline.feature.add_feature(distances, force=True)
        pipeline.feature.add_feature(contacts, force=True)
        
        # Reduce some features
        pipeline.feature.reduce_data(Distances(), "mean", threshold_min=0.0)
        
        # Create feature selections
        pipeline.feature_selector.create("all_features")
        pipeline.feature_selector.add_selection( "all_features", "distances", "all", use_reduced=True)
        pipeline.feature_selector.add_selection( "all_features", "contacts", "all", use_reduced=False)
        pipeline.feature_selector.select("all_features")
        
        # Add clustering and decomposition
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        pca = PCA(n_components=3)
        pipeline.clustering.add_clustering("all_features", dbscan, use_decomposed=False, cluster_name="complete_cluster")
        pipeline.decomposition.add_decomposition("all_features", pca, decomposition_name="complete_pca")
        
        # Save all original data for comparison
        original_dist_data = pipeline._data.feature_data['distances'][0].data.copy()
        original_cont_data = pipeline._data.feature_data['contacts'][0].data.copy()
        original_reduced_dist = pipeline._data.feature_data['distances'][0].reduced_data.copy()
        original_cluster_labels = pipeline._data.cluster_data['complete_cluster'].labels.copy()
        original_pca_components = pipeline._data.decomposition_data['complete_pca'].data.copy()
        original_pca_metadata = pipeline._data.decomposition_data['complete_pca'].metadata.copy()
        
        # Save pipeline
        save_path = temp_dir / "complete_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Verify file exists and has reasonable size
        assert save_path.exists()
        assert save_path.stat().st_size > 1000  # Should be substantial
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify trajectory
        assert loaded._data.trajectory_data.n_frames == 25
        assert loaded._data.trajectory_data.n_atoms == 10
        
        # Verify features
        np.testing.assert_array_equal(loaded._data.feature_data['distances'][0].data, original_dist_data)
        np.testing.assert_array_equal(loaded._data.feature_data['contacts'][0].data, original_cont_data)
        np.testing.assert_array_equal(loaded._data.feature_data['distances'][0].reduced_data, original_reduced_dist)
        
        # Verify clustering
        np.testing.assert_array_equal(loaded._data.cluster_data['complete_cluster'].labels, original_cluster_labels)
        assert loaded._data.cluster_data['complete_cluster'].metadata['algorithm'] == 'dbscan'
        
        # Verify decomposition
        np.testing.assert_array_equal(loaded._data.decomposition_data['complete_pca'].data, original_pca_components)
        assert loaded._data.decomposition_data['complete_pca'].metadata['method'] == original_pca_metadata['method']
        
        # Verify feature selectors exist
        assert "all_features" in loaded._data.selected_feature_data

    def test_save_load_pipeline_with_memmap(self, temp_dir):
        """
        Verify memmap-backed feature data survives save/load.
        """
        pipeline = self._build_memmap_pipeline(temp_dir)
        loaded = None
        try:
            feature_data = pipeline._data.feature_data["distances"][0]
            data_path = Path(feature_data.cache_path)
            assert data_path.exists()
            assert isinstance(feature_data.data, np.memmap)
            save_path = temp_dir / "memmap_pipeline.pkl"
            pipeline.save_to_single_file(str(save_path))
            with open(save_path, "rb") as handle:
                saved = pickle.load(handle)
            saved_feature = saved["feature_data"]["distances"][0]
            assert isinstance(saved_feature.data, dict)
            assert saved_feature.data.get("_is_memmap") is True
            assert Path(saved_feature.data["original_path"]).resolve() == data_path.resolve()
            loaded = PipelineManager.load_from_single_file(
                str(save_path), cache_dir=str(temp_dir / "cache")
            )
            loaded_feature = loaded._data.feature_data["distances"][0]
            assert isinstance(loaded_feature.data, np.memmap)
            assert Path(loaded_feature.data.filename).resolve() == data_path.resolve()
            np.testing.assert_array_equal(
                loaded_feature.data, np.array(feature_data.data)
            )
        finally:
            if loaded is not None:
                loaded.close()
            pipeline.close()

    def test_save_close_load_pipeline_with_memmap(self, temp_dir):
        """
        Verify save -> close -> load -> use works for memmap-backed data.
        """
        pipeline = self._build_memmap_pipeline(temp_dir)
        feature_data = pipeline._data.feature_data["distances"][0]
        expected = np.array(feature_data.data)
        data_path = Path(feature_data.cache_path)
        save_path = temp_dir / "memmap_pipeline_close.pkl"
        loaded = None

        pipeline.save_to_single_file(str(save_path))
        pipeline.close()

        # close() must release handles but keep cache files for later loading.
        assert data_path.exists()

        try:
            loaded = PipelineManager.load_from_single_file(
                str(save_path), cache_dir=str(temp_dir / "cache")
            )
            loaded_feature = loaded._data.feature_data["distances"][0]
            assert isinstance(loaded_feature.data, np.memmap)
            np.testing.assert_array_equal(loaded_feature.data, expected)

            # Loaded pipeline remains usable.
            loaded_mean = loaded_feature.analysis.compute_mean()
            np.testing.assert_allclose(loaded_mean, expected.mean(axis=0))
        finally:
            if loaded is not None:
                loaded.close()
        
    def test_save_load_preserves_bound_methods(self, temp_dir):
        """
        Test that save/load preserves bound analysis method functionality.

        Validates that feature.analysis methods still work after save/load
        and return identical results.
        """
        # Create pipeline with feature
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=20, n_atoms=8, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distance feature
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Test analysis method before save
        original_mean = pipeline._data.feature_data['distances'][0].analysis.compute_mean()
        
        # Save and load pipeline
        save_path = temp_dir / "bound_methods_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Test analysis method after load
        loaded_mean = loaded._data.feature_data['distances'][0].analysis.compute_mean()
        
        # Verify methods work and give same results
        np.testing.assert_almost_equal(loaded_mean, original_mean, decimal=10)
        
        # Verify other analysis methods are available
        assert hasattr(loaded._data.feature_data['distances'][0].analysis, 'compute_std')
        assert hasattr(loaded._data.feature_data['distances'][0].analysis, 'compute_min')
        assert hasattr(loaded._data.feature_data['distances'][0].analysis, 'compute_max')
        
        # Test another method
        original_std = pipeline._data.feature_data['distances'][0].analysis.compute_std()
        loaded_std = loaded._data.feature_data['distances'][0].analysis.compute_std()
        np.testing.assert_almost_equal(loaded_std, original_std, decimal=10)
        
    def test_save_load_error_handling(self, temp_dir):
        """
        Test error handling in save/load operations.

        Validates that non-existent files raise FileNotFoundError and
        invalid paths raise OSError/PermissionError.
        """
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            PipelineManager.load_from_single_file(str(temp_dir / "nonexistent.pkl"))
            
        # Test saving to invalid path
        pipeline = PipelineManager()
        invalid_path = "/invalid/path/pipeline.pkl"
        
        with pytest.raises((OSError, PermissionError)):
            pipeline.save_to_single_file(invalid_path)

    def test_load_single_file_warns_if_memmap_cache_missing(self, temp_dir):
        """
        Missing memmap cache files should emit warning but not fail single-file load.
        """
        pipeline = self._build_memmap_pipeline(temp_dir)
        save_path = temp_dir / "missing_memmap_cache.pkl"
        runtime_cache = Path(pipeline.get_config()["cache_dir"])
        pipeline.save_to_single_file(str(save_path))
        pipeline.close()

        shutil.rmtree(runtime_cache)

        with pytest.warns(RuntimeWarning, match="Memmap restore skipped"):
            loaded = PipelineManager.load_from_single_file(str(save_path))
        try:
            loaded_feature = loaded._data.feature_data["distances"][0]
            assert loaded_feature.data is None
        finally:
            loaded.close()

    def test_load_from_single_file_corrupted_pickle_raises(self, temp_dir):
        """
        Corrupted single-file payload should fail fast during load.
        """
        broken_path = temp_dir / "corrupted_pipeline.pkl"
        broken_path.write_bytes(b"not-a-valid-pickle-payload")
        with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
            PipelineManager.load_from_single_file(str(broken_path))

    def test_load_from_archive_corrupted_archive_raises(self, temp_dir):
        """
        Corrupted archive payload should raise tar extraction/parse errors.
        """
        broken_archive = temp_dir / "corrupted_analysis.tar.gz"
        broken_archive.write_bytes(b"not-a-valid-tar-archive")
        with pytest.raises((tarfile.ReadError, EOFError, OSError)):
            PipelineManager.load_from_archive(
                str(broken_archive),
                verify=False,
            )

    def test_load_from_archive_missing_pipeline_pkl_raises(self, temp_dir):
        """
        Valid tar archive without pipeline.pkl should raise a clear error.
        """
        source_cache = temp_dir / "cache_payload"
        source_cache.mkdir(parents=True, exist_ok=True)
        (source_cache / "dummy.dat").write_bytes(b"dummy-cache-content")

        broken_archive = temp_dir / "missing_pipeline.tar.gz"
        with tarfile.open(broken_archive, "w:gz") as tar:
            tar.add(source_cache, arcname="cache")

        with pytest.raises(FileNotFoundError, match="pipeline.pkl not found"):
            PipelineManager.load_from_archive(
                str(broken_archive),
                cache_dir=str(temp_dir / "restore_cache"),
                verify=False,
            )

    def test_create_sharable_archive_with_sha_writes_sidecar(self, temp_dir):
        """
        Archive creation should optionally emit a SHA256 sidecar file.
        """
        pipeline = self._build_non_memmap_pipeline(temp_dir)
        try:
            archive_path = pipeline.create_sharable_archive(
                str(temp_dir / "with_sha"),
                compression="gz",
                sha=True,
            )
        finally:
            pipeline.close()

        sha_path = Path(ArchiveUtils.get_sha256_file_path(archive_path))
        assert sha_path.exists()
        actual_sha = ArchiveUtils.parse_sha256_text(
            sha_path.read_text(encoding="utf-8")
        )
        assert actual_sha == ArchiveUtils.compute_sha256(archive_path)

    def test_load_from_archive_local_verify_true_accepts_sha_file(self, temp_dir):
        """
        Local archive loads should support explicit SHA256 verification.
        """
        pipeline = self._build_non_memmap_pipeline(temp_dir)
        expected_mean = (
            pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
        )
        try:
            archive_path = pipeline.create_sharable_archive(
                str(temp_dir / "verified_local"),
                compression="gz",
                sha=True,
            )
        finally:
            pipeline.close()

        sha_path = ArchiveUtils.get_sha256_file_path(archive_path)
        loaded = PipelineManager.load_from_archive(
            archive_path,
            cache_dir=str(temp_dir / "verified_local_cache"),
            verify=True,
            sha=sha_path,
        )
        try:
            loaded_feature = loaded._data.feature_data["distances"][0]
            np.testing.assert_array_equal(
                loaded_feature.analysis.compute_mean(),
                expected_mean,
            )
        finally:
            loaded.close()

    def test_load_from_archive_remote_reuses_existing_download(self, temp_dir):
        """
        Remote archive loads should reuse existing downloads when allowed.
        """
        pipeline = self._build_non_memmap_pipeline(temp_dir)
        try:
            archive_path = pipeline.create_sharable_archive(
                str(temp_dir / "remote_reuse_source"),
                compression="gz",
            )
        finally:
            pipeline.close()

        download_target = temp_dir / "downloads" / "archive.tar.gz"
        download_target.parent.mkdir(parents=True, exist_ok=True)
        download_target.write_bytes(Path(archive_path).read_bytes())

        with pytest.warns(RuntimeWarning, match="Reusing the existing file"):
            loaded = PipelineManager.load_from_archive(
                str(download_target),
                cache_dir=str(temp_dir / "remote_reuse_cache"),
                verify=False,
                download_url=Path(archive_path).resolve().as_uri(),
                overwrite=False,
            )
        try:
            assert Path(loaded.get_config()["cache_dir"]).exists()
            assert download_target.read_bytes() == Path(archive_path).read_bytes()
        finally:
            loaded.close()

    def test_load_from_archive_default_verify_requires_sha(self, temp_dir):
        """
        Default archive verification should require SHA input.
        """
        pipeline = self._build_non_memmap_pipeline(temp_dir)
        try:
            archive_path = pipeline.create_sharable_archive(
                str(temp_dir / "remote_verify_required"),
                compression="gz",
            )
        finally:
            pipeline.close()

        with pytest.raises(ValueError, match="Archive verification requires sha"):
            PipelineManager.load_from_archive(
                str(temp_dir / "remote_verify_required.tar.gz"),
                cache_dir=str(temp_dir / "remote_verify_required_cache"),
                download_url=Path(archive_path).resolve().as_uri(),
            )

    def test_close_is_idempotent_and_keeps_cache_file(self, temp_dir):
        """
        close() should be idempotent and should not delete cache files.
        """
        pipeline = self._build_memmap_pipeline(temp_dir)
        feature_data = pipeline._data.feature_data["distances"][0]
        cache_file = Path(feature_data.cache_path)
        assert cache_file.exists()

        pipeline.close()
        pipeline.close()

        assert cache_file.exists()

    def test_two_pipelines_same_base_cache_close_one_not_break_other(self, temp_dir):
        """
        Closing one pipeline should not invalidate another pipeline with same base cache.
        """
        base_cache = temp_dir / "shared_cache"
        p1 = PipelineManager(use_memmap=True, cache_dir=str(base_cache))
        p2 = PipelineManager(use_memmap=True, cache_dir=str(base_cache))

        try:
            t1 = MockTrajectoryFactory.create_simple(n_frames=20, n_atoms=8, seed=1)
            t2 = MockTrajectoryFactory.create_simple(n_frames=20, n_atoms=8, seed=2)
            self._assign_mock_trajectory(p1, t1, "traj_1")
            self._assign_mock_trajectory(p2, t2, "traj_2")
            p1.feature.add_feature(Distances(), force=True)
            p2.feature.add_feature(Distances(), force=True)
            expected_mean2 = (
                p2._data.feature_data["distances"][0].analysis.compute_mean().copy()
            )

            p1.close()

            # Pipeline 2 remains usable after pipeline 1 has been closed.
            mean2 = p2._data.feature_data["distances"][0].analysis.compute_mean()
            np.testing.assert_array_equal(mean2, expected_mean2)
        finally:
            p2.close()

    def test_archive_roundtrip_with_memmap_pipeline_is_usable(self, temp_dir):
        """
        create_sharable_archive/load_from_archive should preserve memmap usability.
        """
        pipeline = self._build_memmap_pipeline(temp_dir)
        loaded = None
        try:
            expected_data = np.array(pipeline._data.feature_data["distances"][0].data)
            expected_mean = (
                pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
            )
            archive_path = pipeline.create_sharable_archive(
                str(temp_dir / "analysis_memmap"),
                compression="gz",
            )
        finally:
            pipeline.close()

        try:
            loaded = PipelineManager.load_from_archive(
                archive_path,
                cache_dir=str(temp_dir / "restored_cache"),
                verify=False,
            )
            loaded_feature = loaded._data.feature_data["distances"][0]
            assert isinstance(loaded_feature.data, np.memmap)
            np.testing.assert_array_equal(np.array(loaded_feature.data), expected_data)
            mean = loaded_feature.analysis.compute_mean()
            np.testing.assert_array_equal(mean, expected_mean)
        finally:
            if loaded is not None:
                loaded.close()

    def test_archive_roundtrip_rebinds_zarr_and_memmaps_single_trajectory(self, temp_dir):
        """
        Three archive save/load cycles should keep single-trajectory zarr/memmaps rebound.
        """
        cache_root = temp_dir / "zarr_memmap_single_cache"
        pipeline, expected_xyz_blocks = self._build_memmap_pipeline_with_dask_trajectories(
            temp_dir,
            cache_root=cache_root,
            n_trajectories=1,
        )
        try:
            expected_mean = (
                pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
            )
            source_runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
            source_zarr_path = Path(
                pipeline._data.trajectory_data.trajectories[0].zarr_cache_path
            ).resolve()
            source_memmap_path = Path(
                pipeline._data.feature_data["distances"][0].cache_path
            ).resolve()
            archive_path = pipeline.create_sharable_archive(
                str(temp_dir / "analysis_zarr_memmap_single"),
                compression="gz",
            )
        finally:
            pipeline.close()

        runtime_caches = [source_runtime_cache]
        for cycle in range(1, 4):
            loaded = PipelineManager.load_from_archive(
                archive_path,
                cache_dir=str(temp_dir / "restored_zarr_memmap_single"),
                verify=False,
            )
            try:
                runtime_cache = Path(loaded.get_config()["cache_dir"]).resolve()
                runtime_caches.append(runtime_cache)
                assert runtime_cache != source_runtime_cache

                loaded_traj = loaded._data.trajectory_data.trajectories[0]
                assert isinstance(loaded_traj, DaskMDTrajectory)
                loaded_zarr_path = Path(loaded_traj.zarr_cache_path).resolve()
                assert loaded_zarr_path.parent == runtime_cache
                assert loaded_zarr_path != source_zarr_path
                np.testing.assert_allclose(
                    np.array(loaded_traj.xyz),
                    expected_xyz_blocks[0],
                )

                loaded_feature = loaded._data.feature_data["distances"][0]
                assert isinstance(loaded_feature.data, np.memmap)
                loaded_memmap_path = Path(loaded_feature.cache_path).resolve()
                assert loaded_memmap_path.parent == runtime_cache
                assert loaded_memmap_path != source_memmap_path
                np.testing.assert_array_equal(
                    loaded_feature.analysis.compute_mean(),
                    expected_mean,
                )

                archive_path = loaded.create_sharable_archive(
                    str(temp_dir / f"analysis_zarr_memmap_single_cycle_{cycle}"),
                    compression="gz",
                )
            finally:
                loaded.close()

        assert len(set(runtime_caches)) == 4

    def test_archive_roundtrip_rebinds_zarr_and_memmaps_all_trajectories(self, temp_dir):
        """
        Three archive save/load cycles should keep all trajectory zarr/memmaps rebound.
        """
        cache_root = temp_dir / "zarr_memmap_multi_cache"
        pipeline, expected_xyz_blocks = self._build_memmap_pipeline_with_dask_trajectories(
            temp_dir,
            cache_root=cache_root,
            n_trajectories=2,
        )
        try:
            expected_means = {
                traj_idx: feature.analysis.compute_mean().copy()
                for traj_idx, feature in pipeline._data.feature_data["distances"].items()
            }
            source_runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
            archive_path = pipeline.create_sharable_archive(
                str(temp_dir / "analysis_zarr_memmap_multi"),
                compression="gz",
            )
        finally:
            pipeline.close()

        runtime_caches = [source_runtime_cache]
        for cycle in range(1, 4):
            loaded = PipelineManager.load_from_archive(
                archive_path,
                cache_dir=str(temp_dir / "restored_zarr_memmap_multi"),
                verify=False,
            )
            try:
                runtime_cache = Path(loaded.get_config()["cache_dir"]).resolve()
                runtime_caches.append(runtime_cache)
                assert runtime_cache != source_runtime_cache

                for traj_idx, loaded_traj in enumerate(
                    loaded._data.trajectory_data.trajectories
                ):
                    assert isinstance(loaded_traj, DaskMDTrajectory)
                    zarr_path = Path(loaded_traj.zarr_cache_path).resolve()
                    assert zarr_path.parent == runtime_cache
                    np.testing.assert_allclose(
                        np.array(loaded_traj.xyz),
                        expected_xyz_blocks[traj_idx],
                    )

                for traj_idx, loaded_feature in loaded._data.feature_data["distances"].items():
                    assert isinstance(loaded_feature.data, np.memmap)
                    memmap_path = Path(loaded_feature.cache_path).resolve()
                    assert memmap_path.parent == runtime_cache
                    np.testing.assert_array_equal(
                        loaded_feature.analysis.compute_mean(),
                        expected_means[traj_idx],
                    )

                archive_path = loaded.create_sharable_archive(
                    str(temp_dir / f"analysis_zarr_memmap_multi_cycle_{cycle}"),
                    compression="gz",
                )
            finally:
                loaded.close()

        assert len(set(runtime_caches)) == 4

    def test_archive_roundtrip_three_cycles_rebinds_all_dask_and_memmaps(self, temp_dir):
        """
        Three archive save/load cycles should keep all Dask/zarr and memmaps usable.
        """
        cache_root = temp_dir / "archive_cycle_dask_memmap_cache"
        pipeline, expected_xyz_blocks = self._build_memmap_pipeline_with_dask_full_analysis(
            temp_dir=temp_dir,
            cache_root=cache_root,
            n_trajectories=2,
        )

        baseline_feature_means = {
            traj_idx: feature.analysis.compute_mean().copy()
            for traj_idx, feature in pipeline._data.feature_data["distances"].items()
        }
        baseline_decomposition = np.array(
            pipeline._data.decomposition_data["all_distances_pca"].data
        )
        baseline_cluster_labels = np.array(
            pipeline._data.cluster_data["all_distances_cluster"].labels
        )

        archive_paths = [
            temp_dir / "archive_cycle_dask_memmap_0",
            temp_dir / "archive_cycle_dask_memmap_1",
            temp_dir / "archive_cycle_dask_memmap_2",
            temp_dir / "archive_cycle_dask_memmap_3",
        ]
        archive_path = pipeline.create_sharable_archive(
            str(archive_paths[0]),
            compression="gz",
        )
        expected_archive_members = self._list_archive_file_members(archive_path)
        expected_cache_hashes = self._archive_file_hashes(
            archive_path, prefix="cache/"
        )
        assert len(expected_cache_hashes) > 0
        initial_runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
        pipeline.close()

        runtime_caches = [initial_runtime_cache]
        for cycle in range(1, 4):
            current_members = self._list_archive_file_members(archive_path)
            current_cache_hashes = self._archive_file_hashes(
                archive_path, prefix="cache/"
            )
            assert current_members == expected_archive_members
            assert current_cache_hashes == expected_cache_hashes

            loaded = PipelineManager.load_from_archive(
                archive_path,
                cache_dir=str(cache_root),
                verify=False,
            )
            try:
                runtime_cache = Path(loaded.get_config()["cache_dir"]).resolve()
                runtime_caches.append(runtime_cache)

                for traj_idx, loaded_traj in enumerate(
                    loaded._data.trajectory_data.trajectories
                ):
                    assert isinstance(loaded_traj, DaskMDTrajectory)
                    zarr_path = Path(loaded_traj.zarr_cache_path).resolve()
                    assert zarr_path.parent == runtime_cache
                    np.testing.assert_allclose(
                        np.array(loaded_traj.xyz),
                        expected_xyz_blocks[traj_idx],
                    )

                for traj_idx, loaded_feature in loaded._data.feature_data["distances"].items():
                    assert isinstance(loaded_feature.data, np.memmap)
                    feature_path = Path(loaded_feature.cache_path).resolve()
                    assert runtime_cache in feature_path.parents
                    np.testing.assert_array_equal(
                        loaded_feature.analysis.compute_mean(),
                        baseline_feature_means[traj_idx],
                    )

                loaded_decomposition = loaded._data.decomposition_data["all_distances_pca"]
                assert isinstance(loaded_decomposition.data, np.memmap)
                decomposition_path = Path(loaded_decomposition.cache_path).resolve()
                assert runtime_cache in decomposition_path.parents
                np.testing.assert_array_equal(
                    np.array(loaded_decomposition.data),
                    baseline_decomposition,
                )

                loaded_cluster = loaded._data.cluster_data["all_distances_cluster"]
                assert isinstance(loaded_cluster.labels, np.memmap)
                labels_path = Path(loaded_cluster.labels.filename).resolve()
                assert runtime_cache in labels_path.parents
                np.testing.assert_array_equal(
                    np.array(loaded_cluster.labels),
                    baseline_cluster_labels,
                )

                all_memmaps = self._collect_memmaps_recursive(loaded._data)
                assert len(all_memmaps) > 0
                for memmap_array in all_memmaps:
                    memmap_path = Path(memmap_array.filename).resolve()
                    assert runtime_cache in memmap_path.parents
                    sample = np.asarray(memmap_array).reshape(-1)[:1]
                    assert sample.size == 1

                archive_path = loaded.create_sharable_archive(
                    str(archive_paths[cycle]),
                    compression="gz",
                )
                created_members = self._list_archive_file_members(archive_path)
                created_cache_hashes = self._archive_file_hashes(
                    archive_path, prefix="cache/"
                )
                assert created_members == expected_archive_members
                assert created_cache_hashes == expected_cache_hashes
            finally:
                loaded.close()

        assert len(set(runtime_caches)) == 4
        final_scoped_dirs = self._list_scoped_cache_dirs(cache_root)
        assert {path.resolve() for path in final_scoped_dirs} == set(runtime_caches)

    def test_archive_roundtrip_ten_cycles_releases_handles_and_allows_cache_delete(
        self, temp_dir
    ):
        """
        Repeated archive load/save cycles should not leak handles.

        Each loaded runtime cache must be deletable immediately after close().
        """
        cache_root = temp_dir / "archive_cycle_stress_cache"
        pipeline, expected_xyz_blocks = self._build_memmap_pipeline_with_dask_trajectories(
            temp_dir=temp_dir,
            cache_root=cache_root,
            n_trajectories=1,
        )
        baseline_mean = (
            pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
        )
        source_runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
        archive_path = pipeline.create_sharable_archive(
            str(temp_dir / "archive_cycle_stress_0"),
            compression="gz",
        )
        pipeline.close()

        removed_runtime_dirs = []
        for cycle in range(1, 11):
            loaded = PipelineManager.load_from_archive(
                archive_path,
                cache_dir=str(cache_root),
                verify=False,
            )
            runtime_cache = Path(loaded.get_config()["cache_dir"]).resolve()
            try:
                loaded_traj = loaded._data.trajectory_data.trajectories[0]
                assert isinstance(loaded_traj, DaskMDTrajectory)
                np.testing.assert_allclose(np.array(loaded_traj.xyz), expected_xyz_blocks[0])

                loaded_feature = loaded._data.feature_data["distances"][0]
                assert isinstance(loaded_feature.data, np.memmap)
                np.testing.assert_array_equal(
                    loaded_feature.analysis.compute_mean(),
                    baseline_mean,
                )
            finally:
                loaded.close()

            assert CleanupUtils.remove_tree(
                runtime_cache,
                missing_ok=False,
                purpose="stress runtime cache directory",
            )
            assert not runtime_cache.exists()
            removed_runtime_dirs.append(runtime_cache)

        assert len(set(removed_runtime_dirs)) == 10
        assert cache_root.exists()
        assert list(cache_root.iterdir()) == [source_runtime_cache]
        assert CleanupUtils.remove_tree(
            source_runtime_cache,
            missing_ok=False,
            purpose="source runtime cache directory",
        )
        assert list(cache_root.iterdir()) == []

    def test_archive_roundtrip_after_close_still_loads(self, temp_dir):
        """
        Saving archive, closing source, then loading must still work.
        """
        pipeline = self._build_memmap_pipeline(temp_dir)
        loaded = None
        archive_path = pipeline.create_sharable_archive(
            str(temp_dir / "analysis_after_close"),
            compression="bz2",
        )
        pipeline.close()

        try:
            loaded = PipelineManager.load_from_archive(
                archive_path,
                cache_dir=str(temp_dir / "archive_cache"),
                verify=False,
            )
            feature = loaded._data.feature_data["distances"][0]
            assert isinstance(feature.data, np.memmap)
            _ = feature.analysis.compute_std()
        finally:
            if loaded is not None:
                loaded.close()

    def test_archive_load_creates_fresh_scoped_cache_each_time_memmap(self, temp_dir):
        """
        Repeated archive load should create a fresh scoped cache dir per load (memmap=True).

        Expected root layout after two loads:

            <cache_root>/
                cache_<uuid0>_<timestamp0>/
                    ... initial pipeline cache content ...
                cache_<uuid1>_<timestamp1>/
                    ... files extracted from archive cycle 0 ...
                cache_<uuid2>_<timestamp2>/
                    ... files extracted from archive cycle 1 ...
        """
        cache_root = temp_dir / "archive_cycle_memmap_cache"
        pipeline = self._build_memmap_pipeline(temp_dir, cache_root=cache_root)
        initial_runtime_cache = Path(pipeline.get_config()["cache_dir"])
        baseline_mean = (
            pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
        )

        archive_path_cycle0 = pipeline.create_sharable_archive(
            str(temp_dir / "archive_cycle_memmap_0"),
            compression="gz",
        )
        expected_files_cycle0 = sorted(self._list_archive_cache_files(archive_path_cycle0))
        actual_files_initial = sorted(self._list_relative_files(initial_runtime_cache))
        assert actual_files_initial == expected_files_cycle0
        pipeline.close()

        loaded_cycle1 = PipelineManager.load_from_archive(
            archive_path_cycle0,
            cache_dir=str(cache_root),
            verify=False,
        )
        try:
            runtime_cache_cycle1 = Path(loaded_cycle1.get_config()["cache_dir"])
            assert runtime_cache_cycle1.parent == cache_root.resolve()
            assert re.fullmatch(
                r"cache_[0-9a-f]{32}_\d{8}_\d{6}", runtime_cache_cycle1.name
            )
            expected_files_cycle1 = sorted(
                self._list_archive_cache_files(archive_path_cycle0)
            )
            actual_files_cycle1 = sorted(self._list_relative_files(runtime_cache_cycle1))
            assert actual_files_cycle1 == expected_files_cycle1

            feature_cycle1 = loaded_cycle1._data.feature_data["distances"][0]
            assert isinstance(feature_cycle1.data, np.memmap)
            assert (
                Path(feature_cycle1.cache_path).resolve().parent
                == runtime_cache_cycle1.resolve()
            )
            np.testing.assert_array_equal(
                feature_cycle1.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle1.analysis, "compute_std")

            archive_path_cycle1 = loaded_cycle1.create_sharable_archive(
                str(temp_dir / "archive_cycle_memmap_1"),
                compression="gz",
            )
        finally:
            loaded_cycle1.close()

        loaded_cycle2 = PipelineManager.load_from_archive(
            archive_path_cycle1,
            cache_dir=str(cache_root),
            verify=False,
        )
        try:
            runtime_cache_cycle2 = Path(loaded_cycle2.get_config()["cache_dir"])
            assert runtime_cache_cycle2.parent == cache_root.resolve()
            assert re.fullmatch(
                r"cache_[0-9a-f]{32}_\d{8}_\d{6}", runtime_cache_cycle2.name
            )
            expected_files_cycle2 = sorted(
                self._list_archive_cache_files(archive_path_cycle1)
            )
            actual_files_cycle2 = sorted(self._list_relative_files(runtime_cache_cycle2))
            assert actual_files_cycle2 == expected_files_cycle2

            feature_cycle2 = loaded_cycle2._data.feature_data["distances"][0]
            assert isinstance(feature_cycle2.data, np.memmap)
            assert (
                Path(feature_cycle2.cache_path).resolve().parent
                == runtime_cache_cycle2.resolve()
            )
            np.testing.assert_array_equal(
                feature_cycle2.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle2.analysis, "compute_std")
        finally:
            loaded_cycle2.close()

        assert runtime_cache_cycle1 != runtime_cache_cycle2

        final_scoped_dirs = self._list_scoped_cache_dirs(cache_root)
        assert len(final_scoped_dirs) == 3
        assert set(final_scoped_dirs) == {
            initial_runtime_cache,
            runtime_cache_cycle1,
            runtime_cache_cycle2,
        }

        expected_tree = {
            "archive_cycle_memmap_cache": {
                initial_runtime_cache.name: expected_files_cycle0,
                runtime_cache_cycle1.name: expected_files_cycle1,
                runtime_cache_cycle2.name: expected_files_cycle2,
            }
        }
        actual_tree = {
            "archive_cycle_memmap_cache": {
                initial_runtime_cache.name: actual_files_initial,
                runtime_cache_cycle1.name: actual_files_cycle1,
                runtime_cache_cycle2.name: actual_files_cycle2,
            }
        }
        assert actual_tree == expected_tree

    def test_archive_load_creates_fresh_scoped_cache_each_time_no_memmap(self, temp_dir):
        """
        Repeated archive load should create a fresh scoped cache dir per load (memmap=False).

        Expected root layout after two loads:

            <cache_root>/
                cache_<uuid0>_<timestamp0>/
                    ... initial pipeline cache content ...
                cache_<uuid1>_<timestamp1>/
                    ... files extracted from archive cycle 0 ...
                cache_<uuid2>_<timestamp2>/
                    ... files extracted from archive cycle 1 ...
        """
        cache_root = temp_dir / "archive_cycle_nomem_cache"
        pipeline = self._build_non_memmap_pipeline(temp_dir, cache_root=cache_root)
        initial_runtime_cache = Path(pipeline.get_config()["cache_dir"])
        baseline_mean = (
            pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
        )

        archive_path_cycle0 = pipeline.create_sharable_archive(
            str(temp_dir / "archive_cycle_nomem_0"),
            compression="gz",
        )
        expected_files_cycle0 = sorted(self._list_archive_cache_files(archive_path_cycle0))
        actual_files_initial = sorted(self._list_relative_files(initial_runtime_cache))
        assert actual_files_initial == expected_files_cycle0
        pipeline.close()

        loaded_cycle1 = PipelineManager.load_from_archive(
            archive_path_cycle0,
            cache_dir=str(cache_root),
            verify=False,
        )
        try:
            runtime_cache_cycle1 = Path(loaded_cycle1.get_config()["cache_dir"])
            assert runtime_cache_cycle1.parent == cache_root.resolve()
            assert re.fullmatch(
                r"cache_[0-9a-f]{32}_\d{8}_\d{6}", runtime_cache_cycle1.name
            )
            expected_files_cycle1 = sorted(
                self._list_archive_cache_files(archive_path_cycle0)
            )
            actual_files_cycle1 = sorted(self._list_relative_files(runtime_cache_cycle1))
            assert actual_files_cycle1 == expected_files_cycle1

            feature_cycle1 = loaded_cycle1._data.feature_data["distances"][0]
            assert not isinstance(feature_cycle1.data, np.memmap)
            np.testing.assert_array_equal(
                feature_cycle1.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle1.analysis, "compute_std")

            archive_path_cycle1 = loaded_cycle1.create_sharable_archive(
                str(temp_dir / "archive_cycle_nomem_1"),
                compression="gz",
            )
        finally:
            loaded_cycle1.close()

        loaded_cycle2 = PipelineManager.load_from_archive(
            archive_path_cycle1,
            cache_dir=str(cache_root),
            verify=False,
        )
        try:
            runtime_cache_cycle2 = Path(loaded_cycle2.get_config()["cache_dir"])
            assert runtime_cache_cycle2.parent == cache_root.resolve()
            assert re.fullmatch(
                r"cache_[0-9a-f]{32}_\d{8}_\d{6}", runtime_cache_cycle2.name
            )
            expected_files_cycle2 = sorted(
                self._list_archive_cache_files(archive_path_cycle1)
            )
            actual_files_cycle2 = sorted(self._list_relative_files(runtime_cache_cycle2))
            assert actual_files_cycle2 == expected_files_cycle2

            feature_cycle2 = loaded_cycle2._data.feature_data["distances"][0]
            assert not isinstance(feature_cycle2.data, np.memmap)
            np.testing.assert_array_equal(
                feature_cycle2.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle2.analysis, "compute_std")
        finally:
            loaded_cycle2.close()

        assert runtime_cache_cycle1 != runtime_cache_cycle2

        final_scoped_dirs = self._list_scoped_cache_dirs(cache_root)
        assert len(final_scoped_dirs) == 3
        assert set(final_scoped_dirs) == {
            initial_runtime_cache,
            runtime_cache_cycle1,
            runtime_cache_cycle2,
        }

        expected_tree = {
            "archive_cycle_nomem_cache": {
                initial_runtime_cache.name: expected_files_cycle0,
                runtime_cache_cycle1.name: expected_files_cycle1,
                runtime_cache_cycle2.name: expected_files_cycle2,
            }
        }
        actual_tree = {
            "archive_cycle_nomem_cache": {
                initial_runtime_cache.name: actual_files_initial,
                runtime_cache_cycle1.name: actual_files_cycle1,
                runtime_cache_cycle2.name: actual_files_cycle2,
            }
        }
        assert actual_tree == expected_tree

    def test_single_file_roundtrip_three_iterations_memmap(self, temp_dir):
        """
        save/load/save/load/save with memmap should preserve bindings/means
        and keep using the original scoped cache directory.

        Expected layout:

            <original_cache_root>/
                cache_<uuid0>_<timestamp0>/
                    distances_mock_trajectory.dat
        """
        pipeline = self._build_memmap_pipeline(temp_dir)
        original_runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
        original_cache_root = original_runtime_cache.parent
        baseline_mean = (
            pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
        )
        expected_files_original = sorted(self._list_relative_files(original_runtime_cache))
        save_path_0 = temp_dir / "single_memmap_0.pkl"
        save_path_1 = temp_dir / "single_memmap_1.pkl"
        save_path_2 = temp_dir / "single_memmap_2.pkl"
        save_path_3 = temp_dir / "single_memmap_3.pkl"
        cache_root_override = temp_dir / "single_memmap_cache_override"
        pipeline.save_to_single_file(str(save_path_0))
        pipeline.close()

        loaded_cycle1 = PipelineManager.load_from_single_file(
            str(save_path_0),
            cache_dir=str(cache_root_override),
        )
        try:
            runtime_cache_cycle1 = Path(loaded_cycle1.get_config()["cache_dir"]).resolve()
            assert runtime_cache_cycle1 == original_runtime_cache
            feature_cycle1 = loaded_cycle1._data.feature_data["distances"][0]
            assert isinstance(feature_cycle1.data, np.memmap)
            assert (
                Path(feature_cycle1.cache_path).resolve().parent
                == runtime_cache_cycle1.resolve()
            )
            actual_files_cycle1 = sorted(self._list_relative_files(runtime_cache_cycle1))
            np.testing.assert_array_equal(
                feature_cycle1.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle1.analysis, "compute_std")
            loaded_cycle1.save_to_single_file(str(save_path_1))
        finally:
            loaded_cycle1.close()

        loaded_cycle2 = PipelineManager.load_from_single_file(
            str(save_path_1),
            cache_dir=str(cache_root_override),
        )
        try:
            runtime_cache_cycle2 = Path(loaded_cycle2.get_config()["cache_dir"]).resolve()
            assert runtime_cache_cycle2 == original_runtime_cache
            feature_cycle2 = loaded_cycle2._data.feature_data["distances"][0]
            assert isinstance(feature_cycle2.data, np.memmap)
            assert (
                Path(feature_cycle2.cache_path).resolve().parent
                == runtime_cache_cycle2.resolve()
            )
            actual_files_cycle2 = sorted(self._list_relative_files(runtime_cache_cycle2))
            np.testing.assert_array_equal(
                feature_cycle2.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle2.analysis, "compute_std")
            loaded_cycle2.save_to_single_file(str(save_path_2))
        finally:
            loaded_cycle2.close()

        loaded_cycle3 = PipelineManager.load_from_single_file(
            str(save_path_2),
            cache_dir=str(cache_root_override),
        )
        try:
            runtime_cache_cycle3 = Path(loaded_cycle3.get_config()["cache_dir"]).resolve()
            assert runtime_cache_cycle3 == original_runtime_cache
            feature_cycle3 = loaded_cycle3._data.feature_data["distances"][0]
            assert isinstance(feature_cycle3.data, np.memmap)
            assert (
                Path(feature_cycle3.cache_path).resolve().parent
                == runtime_cache_cycle3.resolve()
            )
            actual_files_cycle3 = sorted(self._list_relative_files(runtime_cache_cycle3))
            np.testing.assert_array_equal(
                feature_cycle3.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle3.analysis, "compute_std")
            loaded_cycle3.save_to_single_file(str(save_path_3))
        finally:
            loaded_cycle3.close()

        final_scoped_dirs_original_root = self._list_scoped_cache_dirs(original_cache_root)
        assert set(final_scoped_dirs_original_root) == {original_runtime_cache}
        assert not cache_root_override.exists()

        expected_tree = {
            original_cache_root.name: {
                original_runtime_cache.name: expected_files_original,
            }
        }
        actual_tree = {
            original_cache_root.name: {
                original_runtime_cache.name: actual_files_cycle3,
            }
        }
        assert actual_tree == expected_tree

    def test_single_file_roundtrip_three_iterations_no_memmap(self, temp_dir):
        """
        save/load/save/load/save with use_memmap=False should preserve bindings/means
        and keep using the original scoped cache directory.

        Expected layout:

            <original_cache_root>/
                cache_<uuid0>_<timestamp0>/
                    ... no memmap data files required ...
        """
        pipeline = self._build_non_memmap_pipeline(temp_dir)
        original_runtime_cache = Path(pipeline.get_config()["cache_dir"]).resolve()
        original_cache_root = original_runtime_cache.parent
        baseline_mean = (
            pipeline._data.feature_data["distances"][0].analysis.compute_mean().copy()
        )
        expected_files_original = sorted(self._list_relative_files(original_runtime_cache))
        save_path_0 = temp_dir / "single_nomem_0.pkl"
        save_path_1 = temp_dir / "single_nomem_1.pkl"
        save_path_2 = temp_dir / "single_nomem_2.pkl"
        save_path_3 = temp_dir / "single_nomem_3.pkl"
        cache_root_override = temp_dir / "single_nomem_cache_override"
        pipeline.save_to_single_file(str(save_path_0))
        pipeline.close()

        loaded_cycle1 = PipelineManager.load_from_single_file(
            str(save_path_0),
            cache_dir=str(cache_root_override),
        )
        try:
            runtime_cache_cycle1 = Path(loaded_cycle1.get_config()["cache_dir"]).resolve()
            assert runtime_cache_cycle1 == original_runtime_cache
            feature_cycle1 = loaded_cycle1._data.feature_data["distances"][0]
            assert not isinstance(feature_cycle1.data, np.memmap)
            actual_files_cycle1 = sorted(self._list_relative_files(runtime_cache_cycle1))
            np.testing.assert_array_equal(
                feature_cycle1.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle1.analysis, "compute_std")
            loaded_cycle1.save_to_single_file(str(save_path_1))
        finally:
            loaded_cycle1.close()

        loaded_cycle2 = PipelineManager.load_from_single_file(
            str(save_path_1),
            cache_dir=str(cache_root_override),
        )
        try:
            runtime_cache_cycle2 = Path(loaded_cycle2.get_config()["cache_dir"]).resolve()
            assert runtime_cache_cycle2 == original_runtime_cache
            feature_cycle2 = loaded_cycle2._data.feature_data["distances"][0]
            assert not isinstance(feature_cycle2.data, np.memmap)
            actual_files_cycle2 = sorted(self._list_relative_files(runtime_cache_cycle2))
            np.testing.assert_array_equal(
                feature_cycle2.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle2.analysis, "compute_std")
            loaded_cycle2.save_to_single_file(str(save_path_2))
        finally:
            loaded_cycle2.close()

        loaded_cycle3 = PipelineManager.load_from_single_file(
            str(save_path_2),
            cache_dir=str(cache_root_override),
        )
        try:
            runtime_cache_cycle3 = Path(loaded_cycle3.get_config()["cache_dir"]).resolve()
            assert runtime_cache_cycle3 == original_runtime_cache
            feature_cycle3 = loaded_cycle3._data.feature_data["distances"][0]
            assert not isinstance(feature_cycle3.data, np.memmap)
            actual_files_cycle3 = sorted(self._list_relative_files(runtime_cache_cycle3))
            np.testing.assert_array_equal(
                feature_cycle3.analysis.compute_mean(), baseline_mean
            )
            assert hasattr(feature_cycle3.analysis, "compute_std")
            loaded_cycle3.save_to_single_file(str(save_path_3))
        finally:
            loaded_cycle3.close()

        final_scoped_dirs_original_root = self._list_scoped_cache_dirs(original_cache_root)
        assert set(final_scoped_dirs_original_root) == {original_runtime_cache}
        assert not cache_root_override.exists()

        expected_tree = {
            original_cache_root.name: {
                original_runtime_cache.name: expected_files_original,
            }
        }
        actual_tree = {
            original_cache_root.name: {
                original_runtime_cache.name: actual_files_cycle3,
            }
        }
        assert actual_tree == expected_tree

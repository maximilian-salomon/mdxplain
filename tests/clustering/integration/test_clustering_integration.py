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

"""Integration tests for clustering algorithms, verifying internal and external calls."""

import numpy as np
from unittest.mock import patch, MagicMock, ANY
from unittest.mock import PropertyMock

from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestClusteringIntegration:
    """
    Test clustering integration with the pipeline.
    This class contains two sets of tests for each clustering method:
    1. Internal Path Tests: Mock the internal `_perform_*` method in the calculator
       to ensure the correct logic path is taken based on the `method` parameter.
    2. External Lib Tests: Mock the external clustering library (e.g., Sklearn)
       to ensure it's called with the correct parameters from our calculator.
    """

    def _setup_pipeline(self, n_frames=20, n_atoms=3):
        """
        Set up a standard pipeline for clustering integration tests.
        
        Parameters
        ----------
        n_frames : int, default=20
            Number of frames in the mock trajectory.
        n_atoms : int, default=3
            Number of atoms in the mock trajectory.
            
        Returns
        -------
        PipelineManager
            Configured pipeline with two-state mock trajectory, coordinates feature,
            and completed feature selection ready for clustering operations.
        """
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_two_state(n_frames=n_frames, n_atoms=n_atoms, seed=42)
        
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_traj"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline.trajectory.add_labels(traj_selection="all")

        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("input_selection")
        pipeline.feature_selector.add("input_selection", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("input_selection")
        
        return pipeline

    # === 1. DBSCAN: Internal Path Tests ===

    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.DBSCANCalculator._perform_standard_clustering')
    def test_internal_dbscan_standard_path(self, mock_perform_standard):
        """
        Test DBSCAN standard clustering path without sampling.

        Verifies that _perform_standard_clustering is called and
        labels are correctly stored in pipeline._data.cluster_data.
        """
        pipeline = self._setup_pipeline()
        mock_labels = np.arange(20) % 2
        mock_perform_standard.return_value = (mock_labels, MagicMock())
        
        pipeline.clustering.add.dbscan(
            selection_name="input_selection", eps=0.5, min_samples=5, use_decomposed=False
        )

        mock_perform_standard.assert_called_once()
        cluster_data = list(pipeline._data.cluster_data.values())[0]
        np.testing.assert_array_equal(cluster_data.labels, mock_labels)
        assert cluster_data.metadata['hyperparameters']['eps'] == 0.5

    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.DBSCANCalculator._perform_precomputed_clustering')
    def test_internal_dbscan_precomputed_path(self, mock_perform_precomputed):
        """
        Test DBSCAN precomputed clustering path with precomputed distances.

        Validates that _perform_precomputed_clustering is invoked when
        metric='precomputed' and all parameters are correctly transferred.
        """
        pipeline = self._setup_pipeline()
        mock_labels = np.arange(20) % 3
        mock_perform_precomputed.return_value = (mock_labels, MagicMock())
        
        pipeline.clustering.add.dbscan(
            selection_name="input_selection", eps=0.7, min_samples=3, method="precomputed", force=True, use_decomposed=False
        )

        mock_perform_precomputed.assert_called_once()
        cluster_data = list(pipeline._data.cluster_data.values())[0]
        np.testing.assert_array_equal(cluster_data.labels, mock_labels)
        assert cluster_data.metadata['hyperparameters']['method'] == 'precomputed'

    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.DBSCANCalculator._perform_knn_sampling')
    def test_internal_dbscan_knn_sampling_path(self, mock_perform_knn):
        """
        Test DBSCAN knn_sampling clustering path with data reduction.

        Validates that _perform_knn_sampling is called for performance
        optimization on large datasets.
        """
        pipeline = self._setup_pipeline(n_frames=100)
        mock_labels = np.zeros(100)
        mock_perform_knn.return_value = (mock_labels, MagicMock())
        
        pipeline.clustering.add.dbscan(
            selection_name="input_selection", eps=0.5, method="knn_sampling", sample_fraction=0.2, force=True, use_decomposed=False
        )

        mock_perform_knn.assert_called_once()
        cluster_data = list(pipeline._data.cluster_data.values())[0]
        np.testing.assert_array_equal(cluster_data.labels, mock_labels)
        assert cluster_data.metadata['hyperparameters']['method'] == 'knn_sampling'

    # === 2. DBSCAN: External Lib Tests ===

    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.SklearnDBSCAN')
    def test_external_dbscan_standard_call(self, mock_sklearn_dbscan):
        """
        Test external SklearnDBSCAN calls with standard method.

        Validates that sklearn DBSCAN is called with correct parameters
        and mock labels are propagated back into the system.
        """
        pipeline = self._setup_pipeline()
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.arange(20) % 2
        mock_sklearn_dbscan.return_value = mock_instance

        pipeline.clustering.add.dbscan(
            selection_name="input_selection", eps=0.5, min_samples=5, use_decomposed=False
        )

        mock_sklearn_dbscan.assert_called_once_with(eps=0.5, min_samples=5)
        mock_instance.fit_predict.assert_called_once_with(ANY)

    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.SklearnDBSCAN')
    def test_external_dbscan_precomputed_call(self, mock_sklearn_dbscan):
        """
        Test external SklearnDBSCAN calls with precomputed distances.

        Validates that sklearn DBSCAN is called with metric='precomputed'
        parameter and distance matrix is correctly passed.
        """
        pipeline = self._setup_pipeline()
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.arange(20) % 3
        mock_sklearn_dbscan.return_value = mock_instance

        pipeline.clustering.add.dbscan(
            "input_selection", eps=0.7, min_samples=3, method="precomputed", force=True, use_decomposed=False
        )

        mock_sklearn_dbscan.assert_called_once_with(eps=0.7, min_samples=3, metric='precomputed', n_jobs=-1)
        mock_instance.fit_predict.assert_called_once_with(ANY)

    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.DBSCANCalculator._calculate_sample_size')
    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.np.random.choice')
    @patch('mdxplain.clustering.cluster_type.interfaces.calculator_base.KNeighborsClassifier')
    @patch('mdxplain.clustering.cluster_type.dbscan.dbscan_calculator.SklearnDBSCAN')
    def test_external_dbscan_knn_sampling_call(self, mock_sklearn_dbscan, mock_knn, mock_random_choice, mock_calculate_sample_size):
        """
        Test external SklearnDBSCAN + KNN calls with knn_sampling method.

        Validates that sklearn DBSCAN on sample and KNN for prediction
        on remaining data are correctly invoked.
        """
        pipeline = self._setup_pipeline(n_frames=100)  # Small test dataset
        
        # Mock sample size calculation to force real sampling with small dataset
        mock_calculate_sample_size.return_value = 20  # Force 20% sampling
        mock_sample_indices = np.arange(20)  # 20% of 100 frames
        mock_random_choice.return_value = mock_sample_indices

        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.fit_predict.return_value = np.arange(20) % 2
        mock_sklearn_dbscan.return_value = mock_dbscan_instance

        mock_knn_instance = MagicMock()
        mock_knn_instance.predict.return_value = np.arange(80) % 2  # Predict remaining 80 frames
        mock_knn.return_value = mock_knn_instance

        pipeline.clustering.add.dbscan(
            "input_selection", eps=0.5, method="knn_sampling", sample_fraction=0.2, knn_neighbors=10, force=True, use_decomposed=False
        )

        assert mock_random_choice.call_count >= 1  # May be called multiple times in sampling process
        # Check that at least one call had the expected parameters
        calls = mock_random_choice.call_args_list
        expected_call_found = any(call.args == (100,) and call.kwargs.get('size') == 20 and call.kwargs.get('replace') == False for call in calls)
        assert expected_call_found, f"Expected call with (100, size=20, replace=False) not found. Actual calls: {calls}"
        mock_sklearn_dbscan.assert_called_once_with(eps=0.5, min_samples=5)
        mock_dbscan_instance.fit_predict.assert_called_once()
        mock_knn.assert_called_once_with(n_neighbors=10)
        mock_knn_instance.fit.assert_called_once()
        mock_knn_instance.predict.assert_called()

    # === 3. HDBSCAN: Internal Path Tests ===

    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.HDBSCANCalculator._perform_standard_clustering')
    def test_internal_hdbscan_standard_path(self, mock_perform_standard):
        """
        Test HDBSCAN standard clustering path without Approximate Predict.
        Validates that _perform_standard_clustering is called for HDBSCAN
        and hierarchical clustering labels are correctly stored.
        """
        pipeline = self._setup_pipeline()
        mock_labels = np.arange(20) % 2
        mock_perform_standard.return_value = (mock_labels, MagicMock())
        
        pipeline.clustering.add.hdbscan(
            selection_name="input_selection", min_cluster_size=5, use_decomposed=False
        )

        mock_perform_standard.assert_called_once()
        cluster_data = list(pipeline._data.cluster_data.values())[0]
        np.testing.assert_array_equal(cluster_data.labels, mock_labels)
        assert cluster_data.metadata['hyperparameters']['min_cluster_size'] == 5

    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.HDBSCANCalculator._perform_approximate_predict')
    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.HDBSCANCalculator._perform_standard_clustering')
    def test_internal_hdbscan_approximate_predict_path(self, mock_perform_standard, mock_perform_approx):
        """
        Test HDBSCAN approximate_predict clustering path with sampling training.
        Validates that _perform_approximate_predict is called for
        fast predictions on large datasets with HDBSCAN.
        """
        pipeline = self._setup_pipeline(n_frames=200) # More than 100 to avoid fallback
        mock_labels = np.arange(200) % 2
        mock_model = MagicMock()
        mock_model.exemplars_ = [] 
        mock_perform_approx.return_value = (mock_labels, mock_model)
        
        pipeline.clustering.add.hdbscan(
            selection_name="input_selection",
            min_cluster_size=5,
            use_decomposed=False,
            method="approximate_predict",
            cluster_selection_method='leaf',
            force=True
        )
        
        mock_perform_approx.assert_called_once()
        mock_perform_standard.assert_not_called()

    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.HDBSCANCalculator._perform_knn_sampling')
    def test_internal_hdbscan_knn_sampling_path(self, mock_perform_knn):
        """
        Test HDBSCAN knn_sampling clustering path with data reduction.
        Validates that _perform_knn_sampling method is called for HDBSCAN
        and performance optimization occurs during hierarchical clustering.
        """
        pipeline = self._setup_pipeline(n_frames=100)
        mock_labels = np.zeros(100)
        mock_perform_knn.return_value = (mock_labels, MagicMock())
        
        pipeline.clustering.add.hdbscan(
            selection_name="input_selection",
            use_decomposed=False,
            method="knn_sampling",
            sample_fraction=0.2,
            force=True
        )

        mock_perform_knn.assert_called_once()
        cluster_data = list(pipeline._data.cluster_data.values())[0]
        np.testing.assert_array_equal(cluster_data.labels, mock_labels)
        assert cluster_data.metadata['hyperparameters']['method'] == 'knn_sampling'

    # === 4. HDBSCAN: External Lib Tests ===

    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.hdbscan.HDBSCAN')
    def test_external_hdbscan_standard_call(self, mock_hdbscan_lib):
        """
        Test external hdbscan.HDBSCAN calls with standard method.
        Validates that hdbscan library is called with correct parameters
        and returns hierarchical density-based cluster labels.
        """
        pipeline = self._setup_pipeline()
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.arange(20) % 2
        mock_hdbscan_lib.return_value = mock_instance

        pipeline.clustering.add.hdbscan(
            selection_name="input_selection", min_cluster_size=5, min_samples=3, use_decomposed=False
        )

        mock_hdbscan_lib.assert_called_once_with(
            min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.0, cluster_selection_method='eom'
        )
        mock_instance.fit_predict.assert_called_once_with(ANY)

    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.HDBSCANCalculator._calculate_sample_size')
    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.np.random.choice')
    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.hdbscan.approximate_predict')
    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.hdbscan.HDBSCAN')
    def test_external_hdbscan_approximate_predict_call(self, mock_hdbscan_class, mock_approximate_predict, mock_random_choice, mock_calculate_sample_size):
        """
        Test external hdbscan.approximate_predict calls with sampling.
        Validates that HDBSCAN is trained on sample data and
        approximate_predict is used for fast predictions on full dataset.
        """
        pipeline = self._setup_pipeline(n_frames=100)  # Small test dataset

        # Mock sample size calculation to force real sampling with small dataset
        mock_calculate_sample_size.return_value = 20  # Force 20% sampling
        mock_sample_indices = np.arange(20)  # 20% of 100 frames
        mock_random_choice.return_value = mock_sample_indices

        # Mock the HDBSCAN class correctly using PropertyMock
        from unittest.mock import PropertyMock
        
        mock_hdbscan_instance = MagicMock()
        expected_labels = np.array([0, 1] * 10)  # 20 labels for sampled frames
        
        # Use PropertyMock to ensure labels_ returns actual array
        type(mock_hdbscan_instance).labels_ = PropertyMock(return_value=expected_labels)
        
        # The class returns the instance directly (after fit() is called on it)
        mock_hdbscan_class.return_value = mock_hdbscan_instance

        mock_approximate_predict.return_value = (np.arange(100) % 2, np.ones(100))  # Predict all 100 frames

        pipeline.clustering.add.hdbscan(
            selection_name="input_selection",
            use_decomposed=False,
            method="approximate_predict",
            sample_fraction=0.2,
            force=True
        )

        mock_hdbscan_class.assert_called_once()
        assert mock_random_choice.call_count >= 1  # May be called multiple times in sampling process
        mock_approximate_predict.assert_called_once()

    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.HDBSCANCalculator._calculate_sample_size')
    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.np.random.choice')
    @patch('mdxplain.clustering.cluster_type.interfaces.calculator_base.KNeighborsClassifier')
    @patch('mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator.hdbscan.HDBSCAN')
    def test_external_hdbscan_knn_sampling_call(self, mock_hdbscan_class, mock_knn, mock_random_choice, mock_calculate_sample_size):
        """
        Test external HDBSCAN + KNN calls with knn_sampling method.
        Validates that HDBSCAN is applied to sample data and KNN is used
        for prediction on remaining data in hierarchical clustering workflow.
        """
        pipeline = self._setup_pipeline(n_frames=100)  # Small test dataset
        
        # Mock sample size calculation to force real sampling with small dataset
        mock_calculate_sample_size.return_value = 20  # Force 20% sampling
        mock_sample_indices = np.arange(20)  # 20% of 100 frames
        mock_random_choice.return_value = mock_sample_indices

        # Mock the HDBSCAN class correctly using PropertyMock
        mock_hdbscan_instance = MagicMock()
        expected_labels = np.array([0, 1] * 10)  # 20 labels for sampled frames
        
        # Use PropertyMock to ensure labels_ returns actual array
        type(mock_hdbscan_instance).labels_ = PropertyMock(return_value=expected_labels)
        # Also mock fit_predict to return the expected labels
        mock_hdbscan_instance.fit_predict.return_value = expected_labels
        
        # The class returns the instance directly (after fit() is called on it)
        mock_hdbscan_class.return_value = mock_hdbscan_instance

        mock_knn_instance = MagicMock()
        mock_knn_instance.predict.return_value = np.arange(80) % 2  # Predict remaining 80 frames
        mock_knn.return_value = mock_knn_instance

        pipeline.clustering.add.hdbscan(
            selection_name="input_selection",
            use_decomposed=False,
            method="knn_sampling",
            sample_fraction=0.2,
            force=True
        )

        mock_hdbscan_class.assert_called_once()
        assert mock_random_choice.call_count >= 1  # May be called multiple times in sampling process
        mock_knn.assert_called_once()
        mock_knn_instance.fit.assert_called_once()
        mock_knn_instance.predict.assert_called_once()

    # === 5. DPA: Internal Path Tests ===

    @patch('mdxplain.clustering.cluster_type.dpa.dpa_calculator.DPACalculator._perform_standard_clustering')
    def test_internal_dpa_standard_path(self, mock_perform_standard):
        """
        Test DPA standard clustering path without sampling.

        Validates that _perform_standard_clustering is called for DPA (Density Peak Advanced)
        and density-based cluster labels are correctly stored.
        """
        pipeline = self._setup_pipeline()
        mock_labels = np.arange(20) % 2
        mock_perform_standard.return_value = (mock_labels, MagicMock())
        
        pipeline.clustering.add.dpa(
            selection_name="input_selection",
            use_decomposed=False,
            force=True
        )

        mock_perform_standard.assert_called_once()
        cluster_data = list(pipeline._data.cluster_data.values())[0]
        np.testing.assert_array_equal(cluster_data.labels, mock_labels)
        assert cluster_data.metadata['hyperparameters']['method'] == 'standard'

    @patch('mdxplain.clustering.cluster_type.dpa.dpa_calculator.DPACalculator._perform_knn_sampling')
    def test_internal_dpa_knn_sampling_path(self, mock_perform_knn):
        """
        Test DPA knn_sampling clustering path with data reduction.
        Validates that _perform_knn_sampling is called for DPA
        and density peak clustering occurs with performance optimization.
        """
        pipeline = self._setup_pipeline(n_frames=100)
        mock_labels = np.zeros(100)
        mock_perform_knn.return_value = (mock_labels, MagicMock())
        
        pipeline.clustering.add.dpa(
            selection_name="input_selection",
            use_decomposed=False,
            method="knn_sampling",
            sample_fraction=0.2,
            force=True
        )

        mock_perform_knn.assert_called_once()
        cluster_data = list(pipeline._data.cluster_data.values())[0]
        np.testing.assert_array_equal(cluster_data.labels, mock_labels)
        assert cluster_data.metadata['hyperparameters']['method'] == 'knn_sampling'

    # === 6. DPA: External Lib Tests ===

    @patch('mdxplain.clustering.cluster_type.dpa.dpa_calculator.DensityPeakAdvanced')
    def test_external_dpa_standard_call(self, mock_dpa_class):
        """
        Test external DensityPeakAdvanced calls with standard method.
        Validates that DPA library is called correctly and
        density peak clustering works with halos and cluster centers.
        """
        pipeline = self._setup_pipeline()
        
        # Configure the mock so that after fit() the instance has the correct attributes
        mock_dpa_instance = MagicMock()
        expected_labels = np.arange(20) % 2
        expected_halos = np.zeros(20)
        
        # Use PropertyMock to ensure attributes return actual arrays
        type(mock_dpa_instance).labels_ = PropertyMock(return_value=expected_labels)
        type(mock_dpa_instance).halos_ = PropertyMock(return_value=expected_halos)
        
        # The class returns the instance directly (after fit() is called on it)
        mock_dpa_class.return_value = mock_dpa_instance

        pipeline.clustering.add.dpa(
            selection_name="input_selection",
            use_decomposed=False,
            force=True
        )
        
        mock_dpa_class.assert_called_once()
        mock_dpa_instance.fit.assert_called_once()

    @patch('mdxplain.clustering.cluster_type.dpa.dpa_calculator.DPACalculator._calculate_sample_size')
    @patch('mdxplain.clustering.cluster_type.dpa.dpa_calculator.DensityPeakAdvanced')
    @patch('mdxplain.clustering.cluster_type.interfaces.calculator_base.KNeighborsClassifier')
    @patch('mdxplain.clustering.cluster_type.dpa.dpa_calculator.np.random.choice')
    def test_external_dpa_knn_sampling_call(self, mock_random_choice, mock_knn_class, mock_dpa_class, mock_calculate_sample_size):
        """
        Test external DensityPeakAdvanced + KNN calls with knn_sampling.
        Validates that DPA is applied to sample data and KNN is used
        for prediction on remaining data in density peak clustering workflow.
        """
        pipeline = self._setup_pipeline(n_frames=100)  # Small test dataset

        # Mock sample size calculation to force real sampling with small dataset
        mock_calculate_sample_size.return_value = 20  # Force 20% sampling
        mock_sample_indices = np.arange(20)  # 20% of 100 frames
        mock_random_choice.return_value = mock_sample_indices

        # Mock the DensityPeakAdvanced class correctly
        from unittest.mock import PropertyMock
        
        # Configure the mock so that after fit() the instance has the correct attributes
        mock_dpa_instance = MagicMock()
        expected_labels = np.array([0, 1] * 10)  # 20 labels for sampled frames
        expected_halos = np.zeros(20)
        
        # Use PropertyMock to ensure attributes return actual arrays
        type(mock_dpa_instance).labels_ = PropertyMock(return_value=expected_labels)
        type(mock_dpa_instance).halos_ = PropertyMock(return_value=expected_halos)
        
        # The class returns the instance directly (after fit() is called on it)
        mock_dpa_class.return_value = mock_dpa_instance

        mock_knn_instance = MagicMock()
        mock_knn_instance.predict.return_value = np.arange(80) % 2  # Predict remaining 80 frames
        mock_knn_class.return_value = mock_knn_instance

        pipeline.clustering.add.dpa(
            selection_name="input_selection",
            use_decomposed=False,
            method="knn_sampling",
            force=True,
            sample_fraction=0.2
        )
        
        mock_dpa_class.assert_called_once()
        mock_dpa_instance.fit.assert_called_once()
        mock_knn_class.assert_called_once()
        mock_knn_instance.fit.assert_called_once()
        mock_knn_instance.predict.assert_called_once()

# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Gemini 3.0.
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

import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
from mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator import KernelPCACalculator

class TestKernelPCALandmarkSelection:
    """Test landmark selection in KernelPCA."""

    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.Nystroem')
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.IncrementalPCA')
    def test_nystrom_kmeans_selection_calls_base_method(self, mock_ipca, mock_nystroem):
        """Test that base class _select_landmarks_kmeans is used and Nystroem fitted with real points."""
        calculator = KernelPCACalculator(use_memmap=False)
        data = np.random.rand(100, 10)
        n_landmarks = 10
        
        # Mock the base class method _select_landmarks_kmeans
        # Note: We need to mock it on the instance or class
        with patch.object(calculator, '_select_landmarks_kmeans', return_value=np.arange(n_landmarks)) as mock_select:
            
            # Setup mocks
            mock_nystroem_instance = MagicMock()
            mock_nystroem_instance.transform.return_value = np.zeros((100, n_landmarks), dtype=np.float32)
            mock_nystroem.return_value = mock_nystroem_instance

            mock_ipca_instance = MagicMock()
            mock_ipca_instance.transform.return_value = np.zeros((100, 2), dtype=np.float32)
            mock_ipca.return_value = mock_ipca_instance

            # Call compute
            calculator.compute(
                data,
                n_components=2,
                use_nystrom=True,
                n_landmarks=n_landmarks,
                landmark_selection="kmeans",
                random_state=42
            )

            # Verify _select_landmarks_kmeans was called
            mock_select.assert_called_once()
            
            # Verify Nystroem was fit with the subset of data corresponding to indices 0..9
            # Since we mocked return_value=np.arange(n_landmarks), it should be data[:n_landmarks]
            # Use allclose because of float32 cast
            expected_landmarks = data[:n_landmarks].astype(np.float32)
            
            # Check arguments of fit
            args, _ = mock_nystroem_instance.fit.call_args
            np.testing.assert_array_almost_equal(args[0], expected_landmarks)

    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.Nystroem')
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.IncrementalPCA')
    def test_nystrom_random_selection_manual_sampling(self, mock_ipca, mock_nystroem):
        """Test that 'random' selection manually samples landmarks and fits Nystroem with them."""
        calculator = KernelPCACalculator(use_memmap=False)
        n_samples = 100
        n_landmarks = 10
        data = np.random.rand(n_samples, 5)
        
        # Setup mocks
        mock_nystroem_instance = MagicMock()
        mock_nystroem_instance.transform.return_value = np.zeros((n_samples, n_landmarks), dtype=np.float32)
        mock_nystroem.return_value = mock_nystroem_instance

        mock_ipca_instance = MagicMock()
        mock_ipca_instance.transform.return_value = np.zeros((n_samples, 2), dtype=np.float32)
        mock_ipca.return_value = mock_ipca_instance

        # Call compute
        calculator.compute(
            data,
            n_components=2,
            use_nystrom=True,
            n_landmarks=n_landmarks,
            landmark_selection="random",
            random_state=42
        )

        # Verify Nystroem fit was called
        mock_nystroem_instance.fit.assert_called_once()
        
        # Get the arguments passed to fit
        args, _ = mock_nystroem_instance.fit.call_args
        fitted_data = args[0]
        
        # Check that we passed a subset of size n_landmarks
        assert fitted_data.shape == (n_landmarks, 5)
        # Check dtype
        assert fitted_data.dtype == np.float32
        
        # Check that rows in fitted_data exist in original data (approximately, due to float32 cast)
        # This confirms we sampled real points
        # Because of float32 cast, we use allclose
        # Just check one point to be sure. 
        # Since we use random_state=42, the selection should be deterministic, but let's be robust.
        # We check if the first row of fitted_data is close to ANY row in data.
        found_match = False
        first_row = fitted_data[0]
        for row in data:
            if np.allclose(row.astype(np.float32), first_row):
                found_match = True
                break
        assert found_match, "Fitted landmarks should be subset of real data"

    def test_chunk_size_warning_and_adjustment(self):
        """Test that chunk_size is adjusted and warning printed if too small for IPCA."""
        # chunk_size=2, n_components=5 -> Should warn and adjust
        calculator = KernelPCACalculator(use_memmap=False, chunk_size=2)
        data = np.random.rand(20, 5)
        n_components = 5
        
        # Use simple mocks to bypass actual computation
        with patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.Nystroem') as mock_nystroem, \
             patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.IncrementalPCA') as mock_ipca, \
             patch('builtins.print') as mock_print:
            
            mock_nystroem_instance = MagicMock()
            # Dynamic return shape based on input
            mock_nystroem_instance.transform.side_effect = lambda x: np.zeros((x.shape[0], 5), dtype=np.float32)
            mock_nystroem.return_value = mock_nystroem_instance
            
            mock_ipca_instance = MagicMock()
            # Dynamic return shape based on input
            mock_ipca_instance.transform.side_effect = lambda x: np.zeros((x.shape[0], 5), dtype=np.float32)
            mock_ipca.return_value = mock_ipca_instance

            calculator.compute(
                data,
                n_components=n_components,
                use_nystrom=True,
                n_landmarks=5,
                landmark_selection="random"
            )
            
            # Check if warning was printed
            # We look for a call to print that contains "Warning" and "chunk_size"
            warning_printed = any("Warning" in str(args) and "chunk_size" in str(args) for args, _ in mock_print.call_args_list)
            assert warning_printed, "Should have printed a warning about chunk_size adjustment"

            # Check if IncrementalPCA was initialized with adjusted batch_size
            # The batch_size passed to IPCA should be at least n_components (5)
            # mock_ipca is the class constructor
            _, kwargs = mock_ipca.call_args
            assert kwargs['batch_size'] >= n_components

    def test_invalid_landmark_selection(self):
        """Test that a ValueError is raised for invalid landmark_selection values."""
        calculator = KernelPCACalculator(use_memmap=False)
        data = np.random.rand(100, 10)
        
        with pytest.raises(ValueError, match="Invalid landmark_selection"):
            calculator.compute(
                data,
                n_components=2,
                use_nystrom=True,
                landmark_selection="invalid_method"
            )

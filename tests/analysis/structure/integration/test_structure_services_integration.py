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

"""Complete integration tests for structure service hierarchy.

Tests all service paths with default/custom parameters and corner cases.
Verifies correct Calculator instantiation and method calls.
"""

from unittest.mock import Mock, patch
import numpy as np
import pytest
import mdtraj as md

from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.analysis.structure.services.rmsf_per_atom_service import RMSFPerAtomService
from mdxplain.analysis.structure.services.rmsd_mean_service import RMSDMeanService
from mdxplain.analysis.structure.services.rmsf_per_residue_service import RMSFPerResidueService


class TestStructureServicesIntegration:
    """Integration tests for structure analysis service hierarchy.

    Tests verify that service facades correctly instantiate calculators
    and pass parameters through all layers.
    """

    # ========== SETUP ==========

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment with mock pipeline.

        Creates a mock pipeline with necessary structure for testing
        all service paths.
        """
        self.pipeline = self._create_mock_pipeline()
        self._rmsf_patches = []
        self._rmsd_patches = []

        yield

        # Cleanup: Stop all patches
        for p in self._rmsf_patches:
            p.stop()
        for p in self._rmsd_patches:
            p.stop()

    def _create_mock_pipeline(self):
        """Create mock pipeline with required structure.

        Sets up a minimal pipeline with trajectory data and
        configuration needed for service testing.

        Parameters
        ----------
        None

        Returns
        -------
        Mock
            Mock pipeline object with analysis.structure service hierarchy
        """
        pipeline = PipelineManager()

        # Create mock trajectory data
        mock_data = Mock()
        mock_data.trajectory_data = Mock()
        mock_data.trajectory_data.trajectories = [
            self._create_mock_trajectory(i) for i in range(5)
        ]
        mock_data.trajectory_data.trajectory_names = [f"traj_{i}" for i in range(5)]

        # Configure get_trajectory_indices to return proper lists
        def mock_get_trajectory_indices(selection):
            if selection == 'all':
                return [0, 1, 2, 3, 4]
            elif isinstance(selection, list):
                return selection
            elif isinstance(selection, int):
                return [selection]
            else:
                return [0]  # Default fallback

        mock_data.trajectory_data.get_trajectory_indices = Mock(side_effect=mock_get_trajectory_indices)
        mock_data.chunk_size = 500   # NON-default (Default: 2000)
        mock_data.use_memmap = True  # NON-default (Default: False)

        # Attach to pipeline
        pipeline._data = mock_data

        return pipeline

    def _create_mock_trajectory(self, traj_index=0):
        """Create mock trajectory for testing with frame tracking.

        Parameters
        ----------
        traj_index : int, optional
            Index of this trajectory in the trajectory list

        Returns
        -------
        Mock
            Mock trajectory with required attributes and frame tracking
        """
        traj = Mock(spec=md.Trajectory)
        traj.n_frames = 100
        traj.n_atoms = 100
        traj.topology = Mock()
        traj.topology.n_residues = 25

        # Configure topology.select to return proper numpy arrays
        def mock_select(selection_string):
            if "CA" in selection_string:
                return np.array([0, 4, 8, 12], dtype=int)
            elif "CB" in selection_string:
                return np.array([1, 5, 9, 13], dtype=int)
            elif "INVALID" in selection_string:
                return np.array([], dtype=int)  # Empty array triggers validation error
            elif "INCONSISTENT" in selection_string:
                # Return different atom counts for different trajectories
                if traj_index == 0:
                    return np.array([0, 1, 2, 3, 4], dtype=int)  # 5 atoms
                else:
                    return np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)  # 7 atoms
            else:
                return np.array([0, 1, 2, 3], dtype=int)

        traj.topology.select = Mock(side_effect=mock_select)

        # Track frame access for verification
        def mock_getitem(frame_idx):
            frame = Mock()
            frame._source_traj_index = traj_index  # Track which trajectory
            frame._frame_index = frame_idx         # Track which frame
            return frame

        traj.__getitem__ = Mock(side_effect=mock_getitem)
        return traj

    def _get_standard_rmsf_result(self):
        """Get standard RMSF return values for testing.

        Returns
        -------
        list[np.ndarray]
            List of numpy arrays representing RMSF values
        """
        return [np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32) for _ in range(5)]

    def _get_standard_rmsd_result(self):
        """Get standard RMSD return values for testing.

        Returns
        -------
        list[np.ndarray]
            List of numpy arrays representing RMSD values
        """
        return [np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32) for _ in range(5)]

    def _resolve_atom_indices(self, atom_selection):
        """Resolve atom selection to indices for testing.

        Parameters
        ----------
        atom_selection : str or None
            Atom selection string

        Returns
        -------
        np.ndarray or None
            Mock atom indices or None for 'all'
        """
        if atom_selection is None or atom_selection == "all":
            return None
        elif "CA" in atom_selection:
            return np.array([0, 4, 8, 12], dtype=int)  # Mock CA indices
        elif "CB" in atom_selection:
            return np.array([1, 5, 9, 13], dtype=int)  # Mock CB indices
        else:
            return np.array([0, 1, 2, 3], dtype=int)  # Generic indices

    def _get_expected_trajectories(self, args):
        """Get expected trajectory objects based on args.

        Parameters
        ----------
        args : dict
            Service method arguments

        Returns
        -------
        list
            Expected trajectory objects
        """
        trajectories = self.pipeline._data.trajectory_data.trajectories
        traj_selection = args.get('traj_selection', 'all')
        if traj_selection == 'all':
            return trajectories
        elif isinstance(traj_selection, list):
            return [trajectories[i] for i in traj_selection]
        else:
            return [trajectories[traj_selection]]

    def _resolve_reference_index(self, args):
        """Resolve reference trajectory index for per-residue calculations.

        Maps the global reference trajectory index to local index within
        the selected trajectories using same logic as TrajectoryServiceHelper.

        Parameters
        ----------
        args : dict
            Service method arguments

        Returns
        -------
        int
            Local reference trajectory index within selected trajectories
        """
        reference_traj_selection = args.get('reference_traj_selection')
        if reference_traj_selection is None:
            return 0

        # Get selected trajectory indices
        traj_selection = args.get('traj_selection', 'all')
        if traj_selection == 'all':
            selected_indices = [0, 1, 2, 3, 4]
        elif isinstance(traj_selection, list):
            selected_indices = traj_selection
        elif isinstance(traj_selection, int):
            selected_indices = [traj_selection]
        else:
            selected_indices = [0]

        # Get global reference index
        if isinstance(reference_traj_selection, int):
            ref_global_idx = reference_traj_selection
        else:
            ref_global_idx = 0

        # Find position of reference in selected trajectories
        if ref_global_idx in selected_indices:
            return selected_indices.index(ref_global_idx)
        else:
            # If not in selection, assume it gets auto-included at start
            return 0

    # ========== SETUP HELPERS ==========

    def setup_rmsf_calculator_mock(self):
        """Setup RMSF Calculator mock with standard return values.

        Patches RMSFCalculator in all service modules and the helper's build_result_map
        to ensure proper list-to-dictionary conversion.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Mock, Mock]
            (mock_calculator_class, mock_calculator_instance)
        """
        # Patch RMSFCalculator in all service modules that use it
        patch1 = patch('mdxplain.analysis.structure.services.rmsf_per_atom_service.RMSFCalculator', autospec=False)
        patch2 = patch('mdxplain.analysis.structure.services.rmsf_per_residue_service.RMSFCalculator', autospec=False)

        # Patch the helper's build_result_map method to convert list to dict
        patch3 = patch('mdxplain.analysis.structure.helpers.trajectory_service_helper.TrajectoryServiceHelper.build_result_map')

        mock_calc1 = patch1.start()
        mock_calc2 = patch2.start()
        mock_helper = patch3.start()

        # Create shared mock instance
        mock_instance = Mock()
        mock_calc1.return_value = mock_instance
        mock_calc2.return_value = mock_instance

        # Create wrapper that delegates to whichever mock was actually called
        class CombinedMock:
            def assert_called_once(self):
                total_calls = mock_calc1.call_count + mock_calc2.call_count
                if total_calls != 1:
                    raise AssertionError(f"Expected calculator to be called once. Called {total_calls} times.")

            @property
            def call_args(self):
                if mock_calc1.call_count > 0:
                    return mock_calc1.call_args
                elif mock_calc2.call_count > 0:
                    return mock_calc2.call_args
                return None

        combined_mock = CombinedMock()

        # Configure calculator return values as dictionaries (final service format)
        mock_instance.calculate_per_atom.return_value = {
            f"traj_{i}": array for i, array in enumerate(self._get_standard_rmsf_result())
        }
        mock_instance.calculate_per_residue.return_value = {
            f"traj_{i}": array for i, array in enumerate(self._get_standard_rmsf_result())
        }

        # Configure helper to return the input as-is since calculator already returns dict
        def mock_build_result_map(trajectory_names, arrays, cross_trajectory):
            # Calculator now returns dict, so just return it unchanged
            return arrays

        mock_helper.side_effect = mock_build_result_map

        # Store patches for cleanup
        self._rmsf_patches = [patch1, patch2, patch3]

        return combined_mock, mock_instance

    def setup_rmsd_calculator_mock(self):
        """Setup RMSD Calculator mock with standard return values.

        Patches RMSDCalculator in all service modules (mean, median, mad) and the
        helper's _build_result_map to ensure proper list-to-dictionary conversion.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Mock, Mock]
            (mock_calculator_class, mock_calculator_instance)
        """
        # Create a shared Calculator mock that all services will use
        shared_calc_mock = Mock()
        shared_instance = Mock()
        shared_calc_mock.return_value = shared_instance

        # Patch RMSDCalculator in all service modules with the SAME mock object
        patch1 = patch('mdxplain.analysis.structure.services.rmsd_mean_service.RMSDCalculator', new=shared_calc_mock)
        patch3 = patch('mdxplain.analysis.structure.services.rmsd_median_service.RMSDCalculator', new=shared_calc_mock)
        patch5 = patch('mdxplain.analysis.structure.services.rmsd_mad_service.RMSDCalculator', new=shared_calc_mock)

        # Patch the _build_result_map methods for all services
        patch2 = patch('mdxplain.analysis.structure.services.rmsd_mean_service.RMSDMeanService._build_result_map')
        patch4 = patch('mdxplain.analysis.structure.services.rmsd_median_service.RMSDMedianService._build_result_map')
        patch6 = patch('mdxplain.analysis.structure.services.rmsd_mad_service.RMSDMadService._build_result_map')

        patch1.start()
        patch3.start()
        patch5.start()
        mock_helper = patch2.start()
        mock_helper_median = patch4.start()
        mock_helper_mad = patch6.start()

        # Use the shared mocks for test verification
        mock_calc = shared_calc_mock
        mock_instance = shared_instance

        # Configure calculator return values as dictionaries (final service format)
        mock_instance.rmsd_to_reference.return_value = {
            f"traj_{i}": array for i, array in enumerate(self._get_standard_rmsd_result())
        }
        mock_instance.frame_to_frame.return_value = {
            f"traj_{i}": array for i, array in enumerate(self._get_standard_rmsd_result())
        }
        mock_instance.window.return_value = {
            f"traj_{i}": array for i, array in enumerate(self._get_standard_rmsd_result())
        }

        # Configure helper to return the input as-is since calculator already returns dict
        def mock_build_result_map(indices, arrays):
            # Calculator now returns dict, so just return it unchanged
            return arrays

        mock_helper.side_effect = mock_build_result_map
        mock_helper_median.side_effect = mock_build_result_map
        mock_helper_mad.side_effect = mock_build_result_map

        # Store patches for cleanup
        self._rmsd_patches = [patch1, patch2, patch3, patch4, patch5, patch6]

        return mock_calc, mock_instance

    # ========== GET ARGS HELPERS ==========

    def get_default_args_rmsf_per_atom(self):
        """Get default arguments for RMSF per-atom calculations.

        Returns all required parameters with default values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with all required parameters for default test
        """
        return {
            'traj_selection': 'all',
            'atom_selection': 'all',
            'cross_trajectory': False
        }

    def get_custom_args_rmsf_per_atom(self):
        """Get custom arguments for RMSF per-atom calculations.

        Returns non-default values to test parameter propagation.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Custom parameter dictionary with traj_selection,
            atom_selection, and cross_trajectory
        """
        return {
            'traj_selection': [0, 2, 4],
            'atom_selection': "name CA",
            'cross_trajectory': True
        }

    def get_default_args_rmsf_per_residue(self):
        """Get default arguments for RMSF per-residue calculations.

        Returns all required parameters with default values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with all required parameters for default test
        """
        return {
            'traj_selection': 'all',
            'atom_selection': 'all',
            'cross_trajectory': False,
            'reference_traj_selection': None
        }

    def get_custom_args_rmsf_per_residue(self):
        """Get custom arguments for RMSF per-residue calculations.

        Returns non-default values to test parameter propagation.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Custom parameter dictionary with traj_selection, atom_selection,
            cross_trajectory, and reference_traj_selection
        """
        return {
            'traj_selection': [1, 3],
            'atom_selection': "name CB",
            'cross_trajectory': True,
            'reference_traj_selection': 1
        }

    def get_default_args_rmsd_to_reference(self):
        """Get default arguments for RMSD to-reference calculations.

        Returns all required parameters with default values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with all required parameters for default test
        """
        return {
            'reference_traj': 0,
            'reference_frame': 0,
            'traj_selection': 'all',
            'atom_selection': 'all'
        }

    def get_custom_args_rmsd_to_reference(self):
        """Get custom arguments for RMSD to-reference calculations.

        Returns non-default values to test parameter propagation.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Custom parameter dictionary with reference_traj, reference_frame,
            traj_selection, and atom_selection
        """
        return {
            'reference_traj': 2,
            'reference_frame': 5,
            'traj_selection': [1, 2, 3],
            'atom_selection': "name CA"
        }

    def get_default_args_rmsd_frame_to_frame(self):
        """Get default arguments for RMSD frame-to-frame calculations.

        Returns all required parameters with default values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with all required parameters for default test
        """
        return {
            'lag': 1,
            'traj_selection': 'all',
            'atom_selection': 'all'
        }

    def get_custom_args_rmsd_frame_to_frame(self):
        """Get custom arguments for RMSD frame-to-frame calculations.

        Returns non-default values to test parameter propagation.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Custom parameter dictionary with lag, traj_selection,
            and atom_selection
        """
        return {
            'lag': 3,
            'traj_selection': [0, 2],
            'atom_selection': "name CB"
        }

    def get_default_args_rmsd_window_frame_to_start(self):
        """Get default arguments for RMSD window frame-to-start calculations.

        Returns all required parameters with default values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with all required parameters for default test
        """
        return {
            'window_size': 10,
            'stride': 10,  # Default = window_size
            'traj_selection': 'all',
            'atom_selection': 'all'
        }

    def get_custom_args_rmsd_window_frame_to_start(self):
        """Get custom arguments for RMSD window frame-to-start calculations.

        Returns non-default values to test parameter propagation.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Custom parameter dictionary with window_size, stride,
            traj_selection, and atom_selection
        """
        return {
            'window_size': 20,
            'stride': 5,
            'traj_selection': [1, 4],
            'atom_selection': "name CA"
        }

    def get_default_args_rmsd_window_frame_to_frame(self):
        """Get default arguments for RMSD window frame-to-frame calculations.

        Returns all required parameters with default values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with all required parameters for default test
        """
        return {
            'window_size': 10,
            'stride': 10,  # Default = window_size
            'lag': 1,
            'traj_selection': 'all',
            'atom_selection': 'all'
        }

    def get_custom_args_rmsd_window_frame_to_frame(self):
        """Get custom arguments for RMSD window frame-to-frame calculations.

        Returns non-default values to test parameter propagation.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Custom parameter dictionary with window_size, stride, lag,
            traj_selection, and atom_selection
        """
        return {
            'window_size': 15,
            'stride': 3,
            'lag': 2,
            'traj_selection': [0, 3],
            'atom_selection': "name CB"
        }


    # ========== VERIFY HELPERS ==========

    def verify_rmsf_calculator_init(self, mock_calc, args):
        """Verify RMSF Calculator was initialized correctly.

        Checks that the Calculator constructor was called with
        expected trajectories based on args.

        Parameters
        ----------
        mock_calc : Mock
            The mocked RMSFCalculator class
        args : dict
            The arguments passed to the service method

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If initialization parameters don't match expected
        """
        mock_calc.assert_called_once()
        call_kwargs = mock_calc.call_args[1]

        # Verify trajectories
        expected_trajectories = self._get_expected_trajectories(args)
        assert call_kwargs['trajectories'] == expected_trajectories

        # Verify pipeline_data config (NOT from args!)
        assert call_kwargs['chunk_size'] == 500   # From mock_data
        assert call_kwargs['use_memmap'] == True  # From mock_data

    def verify_rmsd_calculator_init(self, mock_calc, args):
        """Verify RMSD Calculator was initialized correctly.

        Checks that the Calculator constructor was called with
        expected trajectories (uses positional args for RMSD).

        Parameters
        ----------
        mock_calc : Mock
            The mocked RMSDCalculator class
        args : dict
            The arguments passed to the service method

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If initialization parameters don't match expected
        """
        mock_calc.assert_called_once()
        call_args = mock_calc.call_args[0]  # Positional args for RMSD

        # Verify trajectories
        expected_trajectories = self._get_expected_trajectories(args)
        assert call_args[0] == expected_trajectories

        # Verify pipeline_data config (positions 1 and 2)
        assert call_args[1] == 500   # chunk_size from mock_data
        assert call_args[2] == True   # use_memmap from mock_data

    def verify_rmsf_per_atom_call(self, mock_instance, args, reference_mode, metric):
        """Verify calculate_per_atom was called correctly.

        Validates that calculate_per_atom received correct parameters
        from args plus method-specific reference_mode and metric.

        Parameters
        ----------
        mock_instance : Mock
            The mocked RMSFCalculator instance
        args : dict
            The arguments passed to the service method
        reference_mode : str
            Expected reference mode ('mean' or 'median')
        metric : str
            Expected metric ('mean', 'median', or 'mad')

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If method parameters don't match expected
        """
        mock_instance.calculate_per_atom.assert_called_once()
        call_kwargs = mock_instance.calculate_per_atom.call_args[1]

        # Verify args passed correctly
        assert call_kwargs['cross_trajectory'] == args['cross_trajectory']

        expected_atom_indices = self._resolve_atom_indices(args['atom_selection'])
        if expected_atom_indices is not None:
            np.testing.assert_array_equal(call_kwargs['atom_indices'], expected_atom_indices)
        else:
            assert call_kwargs['atom_indices'] is None

        # Verify method-specific parameters
        assert call_kwargs['reference_mode'] == reference_mode
        assert call_kwargs['metric'] == metric

    def verify_rmsf_per_residue_call(self, mock_instance, args, reference_mode, metric, aggregator):
        """Verify calculate_per_residue was called correctly.

        Validates that calculate_per_residue received correct parameters
        from args plus method-specific values.

        Parameters
        ----------
        mock_instance : Mock
            The mocked RMSFCalculator instance
        args : dict
            The arguments passed to the service method
        reference_mode : str
            Expected reference mode ('mean' or 'median')
        metric : str
            Expected metric ('mean', 'median', or 'mad')
        aggregator : str
            Expected residue aggregator ('mean', 'median', 'rms', 'rms_median')

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If method parameters don't match expected
        """
        mock_instance.calculate_per_residue.assert_called_once()
        call_kwargs = mock_instance.calculate_per_residue.call_args[1]

        # Verify args passed correctly
        assert call_kwargs['cross_trajectory'] == args['cross_trajectory']
        assert call_kwargs['reference_trajectory_index'] == self._resolve_reference_index(args)

        expected_atom_indices = self._resolve_atom_indices(args['atom_selection'])
        if expected_atom_indices is not None:
            np.testing.assert_array_equal(call_kwargs['atom_indices'], expected_atom_indices)
        else:
            assert call_kwargs['atom_indices'] is None

        # Verify method-specific parameters
        assert call_kwargs['reference_mode'] == reference_mode
        assert call_kwargs['metric'] == metric
        assert call_kwargs['residue_aggregator'] == aggregator

    def verify_rmsd_to_reference_call(self, mock_instance, args, metric):
        """Verify rmsd_to_reference was called correctly.

        Validates that rmsd_to_reference received correct parameters
        from args and method-specific metric.

        Parameters
        ----------
        mock_instance : Mock
            The mocked RMSDCalculator instance
        args : dict
            The arguments passed to the service method
        metric : str
            Expected metric ('mean', 'median', or 'mad')

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If method parameters don't match expected
        """
        mock_instance.rmsd_to_reference.assert_called_once()
        call_args = mock_instance.rmsd_to_reference.call_args[0]

        # Verify reference frame extraction - first arg is reference_frame_traj
        expected_traj_idx = args['reference_traj']
        expected_frame_idx = args['reference_frame']

        # Check that correct trajectory was accessed
        mock_traj = self.pipeline._data.trajectory_data.trajectories[expected_traj_idx]
        mock_traj.__getitem__.assert_called_with(expected_frame_idx)

        # Verify it's the frame from the correct trajectory
        reference_frame_traj = call_args[0]
        assert reference_frame_traj._source_traj_index == expected_traj_idx
        assert reference_frame_traj._frame_index == expected_frame_idx

        # Second arg: atom_indices
        expected_atom_indices = self._resolve_atom_indices(args['atom_selection'])
        if expected_atom_indices is not None:
            np.testing.assert_array_equal(call_args[1], expected_atom_indices)
        else:
            assert call_args[1] is None

        # Third arg: metric
        assert call_args[2] == metric

    def verify_rmsd_frame_to_frame_call(self, mock_instance, args, metric):
        """Verify frame_to_frame was called correctly.

        Validates that frame_to_frame received correct parameters
        from args and method-specific metric.

        Parameters
        ----------
        mock_instance : Mock
            The mocked RMSDCalculator instance
        args : dict
            The arguments passed to the service method
        metric : str
            Expected metric ('mean', 'median', or 'mad')

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If method parameters don't match expected
        """
        mock_instance.frame_to_frame.assert_called_once()
        call_args = mock_instance.frame_to_frame.call_args[0]

        # Verify positional arguments
        # First arg: atom_indices
        expected_atom_indices = self._resolve_atom_indices(args['atom_selection'])
        if expected_atom_indices is not None:
            np.testing.assert_array_equal(call_args[0], expected_atom_indices)
        else:
            assert call_args[0] is None

        # Second arg: lag
        assert call_args[1] == args['lag']

        # Third arg: metric
        assert call_args[2] == metric

    def verify_rmsd_window_call(self, mock_instance, args, metric, mode):
        """Verify window was called correctly.

        Validates that window received correct parameters
        from args and method-specific values.

        Parameters
        ----------
        mock_instance : Mock
            The mocked RMSDCalculator instance
        args : dict
            The arguments passed to the service method
        metric : str
            Expected metric ('mean', 'median', or 'mad')
        mode : str
            Expected window mode ('frame_to_start' or 'frame_to_frame')

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If method parameters don't match expected
        """
        mock_instance.window.assert_called_once()
        call_args = mock_instance.window.call_args[0]
        call_kwargs = mock_instance.window.call_args[1]

        # Verify positional arguments
        # Arg 0: atom_indices
        expected_atom_indices = self._resolve_atom_indices(args['atom_selection'])
        if expected_atom_indices is not None:
            np.testing.assert_array_equal(call_args[0], expected_atom_indices)
        else:
            assert call_args[0] is None

        # Arg 1: window_size (required)
        assert call_args[1] == args['window_size']

        # Arg 2: stride (defaults to window_size if not provided)
        assert call_args[2] == args['stride']

        # Arg 3: metric
        assert call_args[3] == metric

        # Verify keyword arguments
        # mode is passed as keyword argument
        assert call_kwargs['mode'] == mode

        # lag is passed as keyword argument only for frame_to_frame mode
        if mode == "frame_to_frame":
            assert call_kwargs['lag'] == args['lag']
        else:
            assert 'lag' not in call_kwargs  # frame_to_start has no lag

    def verify_shortcut_delegation(self, mock_instance, expected_method):
        """Verify shortcut method delegated to expected underlying method.

        Validates that shortcut properly delegates to full service path.

        Parameters
        ----------
        mock_instance : Mock
            The mocked calculator instance
        expected_method : str
            Expected underlying method name

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If delegation doesn't match expected method
        """
        method = getattr(mock_instance, expected_method)
        method.assert_called_once()

    # ========== RMSF PER-ATOM TESTS ==========

    def test_rmsf_mean_per_atom_to_mean_reference_default(self):
        """Test RMSF mean per-atom to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_atom
        called with reference_mode='mean' and metric='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_mean_per_atom_to_mean_reference_custom(self):
        """Test RMSF mean per-atom to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_atom
        called with custom atom selection and cross_trajectory=True.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_mean_per_atom_to_median_reference_default(self):
        """Test RMSF mean per-atom to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_atom
        called with reference_mode='median' and metric='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mean.per_atom.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="median", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_mean_per_atom_to_median_reference_custom(self):
        """Test RMSF mean per-atom to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_atom
        called with custom atom selection and cross_trajectory=True.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mean.per_atom.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="median", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_median_per_atom_to_mean_reference_default(self):
        """Test RMSF median per-atom to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_atom
        called with reference_mode='mean' and metric='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.median.per_atom.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="median")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_median_per_atom_to_mean_reference_custom(self):
        """Test RMSF median per-atom to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_atom
        called with custom atom selection and cross_trajectory=True.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.median.per_atom.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="median")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_median_per_atom_to_median_reference_default(self):
        """Test RMSF median per-atom to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_atom
        called with reference_mode='median' and metric='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.median.per_atom.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="median", metric="median")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_median_per_atom_to_median_reference_custom(self):
        """Test RMSF median per-atom to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_atom
        called with custom atom selection and cross_trajectory=True.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.median.per_atom.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="median", metric="median")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_mad_per_atom_to_mean_reference_default(self):
        """Test RMSF mad per-atom to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_atom
        called with reference_mode='mean' and metric='mad'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mad.per_atom.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mad")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_mad_per_atom_to_mean_reference_custom(self):
        """Test RMSF mad per-atom to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_atom
        called with custom atom selection and cross_trajectory=True.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mad.per_atom.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mad")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_mad_per_atom_to_median_reference_default(self):
        """Test RMSF mad per-atom to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_atom
        called with reference_mode='median' and metric='mad'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mad.per_atom.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="median", metric="mad")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_mad_per_atom_to_median_reference_custom(self):
        """Test RMSF mad per-atom to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_atom
        called with custom atom selection and cross_trajectory=True.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.mad.per_atom.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="median", metric="mad")
        assert result == mock_instance.calculate_per_atom.return_value

    # ========== RMSF PER-RESIDUE TESTS ==========

    # Mean metric, mean reference, 4 aggregators
    def test_rmsf_mean_per_residue_mean_to_mean_reference_default(self):
        """Test RMSF mean per-residue mean aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mean', and aggregator='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_mean_to_mean_reference_custom(self):
        """Test RMSF mean per-residue mean aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_mean_to_median_reference_default(self):
        """Test RMSF mean per-residue mean aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mean', and aggregator='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_mean_to_median_reference_custom(self):
        """Test RMSF mean per-residue mean aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mean metric, mean reference, median aggregator
    def test_rmsf_mean_per_residue_median_to_mean_reference_default(self):
        """Test RMSF mean per-residue median aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mean', and aggregator='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_median_to_mean_reference_custom(self):
        """Test RMSF mean per-residue median aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mean metric, median reference, median aggregator
    def test_rmsf_mean_per_residue_median_to_median_reference_default(self):
        """Test RMSF mean per-residue median aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mean', and aggregator='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_median_to_median_reference_custom(self):
        """Test RMSF mean per-residue median aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mean metric, mean reference, rms aggregator
    def test_rmsf_mean_per_residue_rms_to_mean_reference_default(self):
        """Test RMSF mean per-residue rms aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mean', and aggregator='rms'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_rms_to_mean_reference_custom(self):
        """Test RMSF mean per-residue rms aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mean metric, median reference, rms aggregator
    def test_rmsf_mean_per_residue_rms_to_median_reference_default(self):
        """Test RMSF mean per-residue rms aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mean', and aggregator='rms'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_rms_to_median_reference_custom(self):
        """Test RMSF mean per-residue rms aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mean metric, mean reference, rms_median aggregator
    def test_rmsf_mean_per_residue_rms_median_to_mean_reference_default(self):
        """Test RMSF mean per-residue rms_median aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mean', and aggregator='rms_median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_rms_median_to_mean_reference_custom(self):
        """Test RMSF mean per-residue rms_median aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mean metric, median reference, rms_median aggregator
    def test_rmsf_mean_per_residue_rms_median_to_median_reference_default(self):
        """Test RMSF mean per-residue rms_median aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mean', and aggregator='rms_median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mean_per_residue_rms_median_to_median_reference_custom(self):
        """Test RMSF mean per-residue rms_median aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mean", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    # ========== MEDIAN METRIC TESTS ==========

    # Median metric, mean reference, mean aggregator
    def test_rmsf_median_per_residue_mean_to_mean_reference_default(self):
        """Test RMSF median per-residue mean aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='median', and aggregator='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_mean_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_mean_to_mean_reference_custom(self):
        """Test RMSF median per-residue mean aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_mean_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    # Median metric, median reference, mean aggregator
    def test_rmsf_median_per_residue_mean_to_median_reference_default(self):
        """Test RMSF median per-residue mean aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='median', and aggregator='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_mean_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_mean_to_median_reference_custom(self):
        """Test RMSF median per-residue mean aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_mean_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    # Median metric, mean reference, median aggregator
    def test_rmsf_median_per_residue_median_to_mean_reference_default(self):
        """Test RMSF median per-residue median aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='median', and aggregator='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_median_to_mean_reference_custom(self):
        """Test RMSF median per-residue median aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Median metric, median reference, median aggregator
    def test_rmsf_median_per_residue_median_to_median_reference_default(self):
        """Test RMSF median per-residue median aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='median', and aggregator='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_median_to_median_reference_custom(self):
        """Test RMSF median per-residue median aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Median metric, mean reference, rms aggregator
    def test_rmsf_median_per_residue_rms_to_mean_reference_default(self):
        """Test RMSF median per-residue rms aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='median', and aggregator='rms'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_rms_to_mean_reference_custom(self):
        """Test RMSF median per-residue rms aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    # Median metric, median reference, rms aggregator
    def test_rmsf_median_per_residue_rms_to_median_reference_default(self):
        """Test RMSF median per-residue rms aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='median', and aggregator='rms'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_rms_to_median_reference_custom(self):
        """Test RMSF median per-residue rms aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    # Median metric, mean reference, rms_median aggregator
    def test_rmsf_median_per_residue_rms_median_to_mean_reference_default(self):
        """Test RMSF median per-residue rms_median aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='median', and aggregator='rms_median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_rms_median_to_mean_reference_custom(self):
        """Test RMSF median per-residue rms_median aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="median", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Median metric, median reference, rms_median aggregator
    def test_rmsf_median_per_residue_rms_median_to_median_reference_default(self):
        """Test RMSF median per-residue rms_median aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='median', and aggregator='rms_median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_median_per_residue_rms_median_to_median_reference_custom(self):
        """Test RMSF median per-residue rms_median aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.median.per_residue.with_rms_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="median", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    # ========== MAD METRIC TESTS ==========

    # Mad metric, mean reference, mean aggregator
    def test_rmsf_mad_per_residue_mean_to_mean_reference_default(self):
        """Test RMSF mad per-residue mean aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mad', and aggregator='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_mean_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_mean_to_mean_reference_custom(self):
        """Test RMSF mad per-residue mean aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_mean_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mad metric, median reference, mean aggregator
    def test_rmsf_mad_per_residue_mean_to_median_reference_default(self):
        """Test RMSF mad per-residue mean aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mad', and aggregator='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_mean_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_mean_to_median_reference_custom(self):
        """Test RMSF mad per-residue mean aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_mean_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mad metric, mean reference, median aggregator
    def test_rmsf_mad_per_residue_median_to_mean_reference_default(self):
        """Test RMSF mad per-residue median aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mad', and aggregator='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_median_to_mean_reference_custom(self):
        """Test RMSF mad per-residue median aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mad metric, median reference, median aggregator
    def test_rmsf_mad_per_residue_median_to_median_reference_default(self):
        """Test RMSF mad per-residue median aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mad', and aggregator='median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_median_to_median_reference_custom(self):
        """Test RMSF mad per-residue median aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mad metric, mean reference, rms aggregator
    def test_rmsf_mad_per_residue_rms_to_mean_reference_default(self):
        """Test RMSF mad per-residue rms aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mad', and aggregator='rms'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_rms_to_mean_reference_custom(self):
        """Test RMSF mad per-residue rms aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mad metric, median reference, rms aggregator
    def test_rmsf_mad_per_residue_rms_to_median_reference_default(self):
        """Test RMSF mad per-residue rms aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mad', and aggregator='rms'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_rms_to_median_reference_custom(self):
        """Test RMSF mad per-residue rms aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="rms")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mad metric, mean reference, rms_median aggregator
    def test_rmsf_mad_per_residue_rms_median_to_mean_reference_default(self):
        """Test RMSF mad per-residue rms_median aggregation to mean reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='mean', metric='mad', and aggregator='rms_median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_rms_median_to_mean_reference_custom(self):
        """Test RMSF mad per-residue rms_median aggregation to mean reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_median_aggregation.to_mean_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mad", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    # Mad metric, median reference, rms_median aggregator
    def test_rmsf_mad_per_residue_rms_median_to_median_reference_default(self):
        """Test RMSF mad per-residue rms_median aggregation to median reference with default parameters.

        Expected: Calculator initialized with all trajectories, calculate_per_residue
        called with reference_mode='median', metric='mad', and aggregator='rms_median'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_mad_per_residue_rms_median_to_median_reference_custom(self):
        """Test RMSF mad per-residue rms_median aggregation to median reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, calculate_per_residue
        called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.mad.per_residue.with_rms_median_aggregation.to_median_reference(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="median", metric="mad", aggregator="rms_median")
        assert result == mock_instance.calculate_per_residue.return_value

    # ========================================
    # RMSD TESTS
    # ========================================

    # RMSD mean metric tests
    def test_rmsd_mean_to_reference_default(self):
        """Test RMSD mean to_reference with default parameters.

        Expected: Calculator initialized with all trajectories, rmsd_to_reference
        called with reference frame from trajectory 0.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.mean.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mean")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_mean_to_reference_custom(self):
        """Test RMSD mean to_reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, rmsd_to_reference
        called with custom reference and atom selection.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.mean.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mean")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_mean_frame_to_frame_default(self):
        """Test RMSD mean frame_to_frame with default parameters.

        Expected: Calculator initialized with all trajectories, frame_to_frame
        called with lag=1.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mean.frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_frame_to_frame_call(mock_instance, args, "mean")
        assert result == mock_instance.frame_to_frame.return_value

    def test_rmsd_mean_frame_to_frame_custom(self):
        """Test RMSD mean frame_to_frame with custom parameters.

        Expected: Calculator initialized with selected trajectories, frame_to_frame
        called with custom lag.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mean.frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_frame_to_frame_call(mock_instance, args, "mean")
        assert result == mock_instance.frame_to_frame.return_value

    def test_rmsd_mean_window_frame_to_start_default(self):
        """Test RMSD mean window_frame_to_start with default parameters.

        Expected: Calculator initialized with all trajectories, window
        called with mode='frame_to_start'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_window_frame_to_start()
        result = self.pipeline.analysis.structure.rmsd.mean.window_frame_to_start(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mean", "frame_to_start")
        assert result == mock_instance.window.return_value

    def test_rmsd_mean_window_frame_to_start_custom(self):
        """Test RMSD mean window_frame_to_start with custom parameters.

        Expected: Calculator initialized with selected trajectories, window
        called with custom parameters and mode='frame_to_start'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_window_frame_to_start()
        result = self.pipeline.analysis.structure.rmsd.mean.window_frame_to_start(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mean", "frame_to_start")
        assert result == mock_instance.window.return_value

    def test_rmsd_mean_window_frame_to_frame_default(self):
        """Test RMSD mean window_frame_to_frame with default parameters.

        Expected: Calculator initialized with all trajectories, window
        called with mode='frame_to_frame'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_window_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mean.window_frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mean", "frame_to_frame")
        assert result == mock_instance.window.return_value

    def test_rmsd_mean_window_frame_to_frame_custom(self):
        """Test RMSD mean window_frame_to_frame with custom parameters.

        Expected: Calculator initialized with selected trajectories, window
        called with custom parameters and mode='frame_to_frame'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_window_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mean.window_frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mean", "frame_to_frame")
        assert result == mock_instance.window.return_value

    # RMSD median metric tests
    def test_rmsd_median_to_reference_default(self):
        """Test RMSD median to_reference with default parameters.

        Expected: Calculator initialized with all trajectories, rmsd_to_reference
        called with reference frame from trajectory 0.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.median.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "median")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_median_to_reference_custom(self):
        """Test RMSD median to_reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, rmsd_to_reference
        called with custom reference and atom selection.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.median.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "median")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_median_frame_to_frame_default(self):
        """Test RMSD median frame_to_frame with default parameters.

        Expected: Calculator initialized with all trajectories, frame_to_frame
        called with lag=1.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.median.frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_frame_to_frame_call(mock_instance, args, "median")
        assert result == mock_instance.frame_to_frame.return_value

    def test_rmsd_median_frame_to_frame_custom(self):
        """Test RMSD median frame_to_frame with custom parameters.

        Expected: Calculator initialized with selected trajectories, frame_to_frame
        called with custom lag.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.median.frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_frame_to_frame_call(mock_instance, args, "median")
        assert result == mock_instance.frame_to_frame.return_value

    def test_rmsd_median_window_frame_to_start_default(self):
        """Test RMSD median window_frame_to_start with default parameters.

        Expected: Calculator initialized with all trajectories, window
        called with mode='frame_to_start'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_window_frame_to_start()
        result = self.pipeline.analysis.structure.rmsd.median.window_frame_to_start(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "median", "frame_to_start")
        assert result == mock_instance.window.return_value

    def test_rmsd_median_window_frame_to_start_custom(self):
        """Test RMSD median window_frame_to_start with custom parameters.

        Expected: Calculator initialized with selected trajectories, window
        called with custom parameters and mode='frame_to_start'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_window_frame_to_start()
        result = self.pipeline.analysis.structure.rmsd.median.window_frame_to_start(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "median", "frame_to_start")
        assert result == mock_instance.window.return_value

    def test_rmsd_median_window_frame_to_frame_default(self):
        """Test RMSD median window_frame_to_frame with default parameters.

        Expected: Calculator initialized with all trajectories, window
        called with mode='frame_to_frame'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_window_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.median.window_frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "median", "frame_to_frame")
        assert result == mock_instance.window.return_value

    def test_rmsd_median_window_frame_to_frame_custom(self):
        """Test RMSD median window_frame_to_frame with custom parameters.

        Expected: Calculator initialized with selected trajectories, window
        called with custom parameters and mode='frame_to_frame'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_window_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.median.window_frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "median", "frame_to_frame")
        assert result == mock_instance.window.return_value

    # RMSD mad metric tests
    def test_rmsd_mad_to_reference_default(self):
        """Test RMSD mad to_reference with default parameters.

        Expected: Calculator initialized with all trajectories, rmsd_to_reference
        called with reference frame from trajectory 0.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.mad.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mad")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_mad_to_reference_custom(self):
        """Test RMSD mad to_reference with custom parameters.

        Expected: Calculator initialized with selected trajectories, rmsd_to_reference
        called with custom reference and atom selection.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.mad.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mad")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_mad_frame_to_frame_default(self):
        """Test RMSD mad frame_to_frame with default parameters.

        Expected: Calculator initialized with all trajectories, frame_to_frame
        called with lag=1.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mad.frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_frame_to_frame_call(mock_instance, args, "mad")
        assert result == mock_instance.frame_to_frame.return_value

    def test_rmsd_mad_frame_to_frame_custom(self):
        """Test RMSD mad frame_to_frame with custom parameters.

        Expected: Calculator initialized with selected trajectories, frame_to_frame
        called with custom lag.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mad.frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_frame_to_frame_call(mock_instance, args, "mad")
        assert result == mock_instance.frame_to_frame.return_value

    def test_rmsd_mad_window_frame_to_start_default(self):
        """Test RMSD mad window_frame_to_start with default parameters.

        Expected: Calculator initialized with all trajectories, window
        called with mode='frame_to_start'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_window_frame_to_start()
        result = self.pipeline.analysis.structure.rmsd.mad.window_frame_to_start(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mad", "frame_to_start")
        assert result == mock_instance.window.return_value

    def test_rmsd_mad_window_frame_to_start_custom(self):
        """Test RMSD mad window_frame_to_start with custom parameters.

        Expected: Calculator initialized with selected trajectories, window
        called with custom parameters and mode='frame_to_start'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_window_frame_to_start()
        result = self.pipeline.analysis.structure.rmsd.mad.window_frame_to_start(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mad", "frame_to_start")
        assert result == mock_instance.window.return_value

    def test_rmsd_mad_window_frame_to_frame_default(self):
        """Test RMSD mad window_frame_to_frame with default parameters.

        Expected: Calculator initialized with all trajectories, window
        called with mode='frame_to_frame'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_window_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mad.window_frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mad", "frame_to_frame")
        assert result == mock_instance.window.return_value

    def test_rmsd_mad_window_frame_to_frame_custom(self):
        """Test RMSD mad window_frame_to_frame with custom parameters.

        Expected: Calculator initialized with selected trajectories, window
        called with custom parameters and mode='frame_to_frame'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_window_frame_to_frame()
        result = self.pipeline.analysis.structure.rmsd.mad.window_frame_to_frame(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_window_call(mock_instance, args, "mad", "frame_to_frame")
        assert result == mock_instance.window.return_value

    # ========================================
    # SHORTCUT TESTS
    # ========================================

    # RMSD shortcut tests (no metric specified - should default to mean)
    def test_rmsd_shortcut_to_reference_default(self):
        """Test RMSD shortcut to_reference with default parameters.

        Expected: Delegates to mean.to_reference, Calculator initialized with all trajectories,
        rmsd_to_reference called with metric='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_default_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mean")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_shortcut_to_reference_custom(self):
        """Test RMSD shortcut to_reference with custom parameters.

        Expected: Delegates to mean.to_reference, Calculator initialized with selected trajectories,
        rmsd_to_reference called with custom parameters and metric='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args
        args = self.get_custom_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd.to_reference(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mean")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_shortcut_call_default(self):
        """Test RMSD __call__ shortcut with default parameters.

        Expected: Delegates to mean.to_reference(), Calculator initialized with all trajectories,
        rmsd_to_reference called with metric='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args - using __call__ shortcut
        args = self.get_default_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mean")
        assert result == mock_instance.rmsd_to_reference.return_value

    def test_rmsd_shortcut_call_custom(self):
        """Test RMSD __call__ shortcut with custom parameters.

        Expected: Delegates to mean.to_reference(), Calculator initialized with selected trajectories,
        rmsd_to_reference called with custom parameters and metric='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with args - using __call__ shortcut
        args = self.get_custom_args_rmsd_to_reference()
        result = self.pipeline.analysis.structure.rmsd(**args)

        # Verify
        self.verify_rmsd_calculator_init(mock_calc, args)
        self.verify_rmsd_to_reference_call(mock_instance, args, "mean")
        assert result == mock_instance.rmsd_to_reference.return_value

    # RMSF shortcut tests
    def test_rmsf_shortcut_call_default(self):
        """Test RMSF __call__ shortcut with default parameters.

        Expected: Delegates to mean.per_atom.to_mean_reference(), Calculator initialized with all trajectories,
        calculate_per_atom called with reference_mode='mean' and metric='mean'.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args - using __call__ shortcut
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_shortcut_call_custom(self):
        """Test RMSF __call__ shortcut with custom parameters.

        Expected: Delegates to mean.per_atom.to_mean_reference(), Calculator initialized with selected trajectories,
        calculate_per_atom called with custom parameters.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args - using __call__ shortcut
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_shortcut_per_atom_default(self):
        """Test RMSF per_atom property shortcut with default parameters.

        Expected: Returns mean.per_atom service, delegates to to_mean_reference().
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args - using per_atom property shortcut
        args = self.get_default_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.per_atom(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_shortcut_per_atom_custom(self):
        """Test RMSF per_atom property shortcut with custom parameters.

        Expected: Returns mean.per_atom service, delegates to to_mean_reference().
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args - using per_atom property shortcut
        args = self.get_custom_args_rmsf_per_atom()
        result = self.pipeline.analysis.structure.rmsf.per_atom(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_atom_call(mock_instance, args, reference_mode="mean", metric="mean")
        assert result == mock_instance.calculate_per_atom.return_value

    def test_rmsf_shortcut_per_residue_default(self):
        """Test RMSF per_residue property shortcut with default parameters.

        Expected: Returns mean.per_residue service, delegates to with_mean_aggregation.to_mean_reference().
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args - using per_residue property shortcut
        args = self.get_default_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.per_residue(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    def test_rmsf_shortcut_per_residue_custom(self):
        """Test RMSF per_residue property shortcut with custom parameters.

        Expected: Returns mean.per_residue service, delegates to with_mean_aggregation.to_mean_reference().
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with args - using per_residue property shortcut
        args = self.get_custom_args_rmsf_per_residue()
        result = self.pipeline.analysis.structure.rmsf.per_residue(**args)

        # Verify
        self.verify_rmsf_calculator_init(mock_calc, args)
        self.verify_rmsf_per_residue_call(mock_instance, args, reference_mode="mean", metric="mean", aggregator="mean")
        assert result == mock_instance.calculate_per_residue.return_value

    # ========================================
    # CORNER CASE TESTS
    # ========================================

    # Empty trajectory selection tests
    def test_rmsf_mean_per_atom_empty_trajectory_selection(self):
        """Test RMSF mean per-atom with empty trajectory selection.

        Expected: Raises ValueError for empty trajectory selection.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with empty trajectory selection
        args = self.get_default_args_rmsf_per_atom()
        args['traj_selection'] = []

        with pytest.raises(ValueError, match="No trajectories found"):
            self.pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(**args)

    def test_rmsf_mean_per_residue_empty_trajectory_selection(self):
        """Test RMSF mean per-residue with empty trajectory selection.

        Expected: Raises ValueError for empty trajectory selection.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with empty trajectory selection
        args = self.get_default_args_rmsf_per_residue()
        args['traj_selection'] = []

        with pytest.raises(ValueError, match="No trajectories found"):
            self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_mean_reference(**args)

    def test_rmsd_mean_empty_trajectory_selection(self):
        """Test RMSD mean with empty trajectory selection.

        Expected: Raises ValueError for empty trajectory selection.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with empty trajectory selection
        args = self.get_default_args_rmsd_to_reference()
        args['traj_selection'] = []

        with pytest.raises(ValueError, match="No trajectories found for the requested selection"):
            self.pipeline.analysis.structure.rmsd.mean.to_reference(**args)

    # Invalid atom selection tests
    def test_rmsf_mean_per_atom_invalid_atom_selection(self):
        """Test RMSF mean per-atom with invalid atom selection.

        Expected: Raises ValueError for empty atom selection result.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with invalid atom selection
        args = self.get_default_args_rmsf_per_atom()
        args['atom_selection'] = "name INVALID"

        with pytest.raises(ValueError, match="Atom selection.*produced no atoms"):
            self.pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(**args)

    def test_rmsf_mean_per_residue_invalid_atom_selection(self):
        """Test RMSF mean per-residue with invalid atom selection.

        Expected: Raises ValueError for empty atom selection result.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsf_calculator_mock()

        # Execute with invalid atom selection
        args = self.get_default_args_rmsf_per_residue()
        args['atom_selection'] = "name INVALID"

        with pytest.raises(ValueError, match="Atom selection.*produced no atoms"):
            self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_mean_reference(**args)

    def test_rmsd_mean_invalid_atom_selection(self):
        """Test RMSD mean with invalid atom selection.

        Expected: Raises ValueError for empty atom selection result.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with invalid atom selection
        args = self.get_default_args_rmsd_to_reference()
        args['atom_selection'] = "name INVALID"

        with pytest.raises(ValueError, match="Atom selection.*produced no atoms"):
            self.pipeline.analysis.structure.rmsd.mean.to_reference(**args)

    # Invalid reference trajectory/frame tests
    def test_rmsd_mean_invalid_reference_trajectory_index(self):
        """Test RMSD mean with invalid reference trajectory index.

        Expected: Raises ValueError for out-of-range reference trajectory.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with invalid reference trajectory index
        args = self.get_default_args_rmsd_to_reference()
        args['reference_traj'] = 999  # Out of range

        with pytest.raises(ValueError, match="Reference trajectory index is out of range"):
            self.pipeline.analysis.structure.rmsd.mean.to_reference(**args)

    def test_rmsd_mean_invalid_reference_frame_index(self):
        """Test RMSD mean with invalid reference frame index.

        Expected: Raises ValueError for out-of-range reference frame.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with invalid reference frame index
        args = self.get_default_args_rmsd_to_reference()
        args['reference_frame'] = 9999  # Out of range

        with pytest.raises(ValueError, match="Reference frame index is out of range"):
            self.pipeline.analysis.structure.rmsd.mean.to_reference(**args)

    # Invalid lag tests
    def test_rmsd_mean_invalid_lag_negative(self):
        """Test RMSD mean frame_to_frame with negative lag.

        Expected: Raises ValueError for negative lag.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with negative lag
        args = self.get_default_args_rmsd_frame_to_frame()
        args['lag'] = -1

        # Verify raises ValueError
        with pytest.raises(ValueError, match="lag must be positive"):
            self.pipeline.analysis.structure.rmsd.mean.frame_to_frame(**args)

    def test_rmsd_mean_invalid_lag_zero(self):
        """Test RMSD mean frame_to_frame with zero lag.

        Expected: Raises ValueError for zero lag.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with zero lag
        args = self.get_default_args_rmsd_frame_to_frame()
        args['lag'] = 0

        # Verify raises ValueError
        with pytest.raises(ValueError, match="lag must be positive"):
            self.pipeline.analysis.structure.rmsd.mean.frame_to_frame(**args)

    # Invalid window parameters tests
    def test_rmsd_mean_invalid_window_size_negative(self):
        """Test RMSD mean window with negative window size.

        Expected: Raises ValueError for negative window size.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with negative window size
        args = self.get_default_args_rmsd_window_frame_to_start()
        args['window_size'] = -1

        # Verify raises ValueError
        with pytest.raises(ValueError, match="window_size must be positive"):
            self.pipeline.analysis.structure.rmsd.mean.window_frame_to_start(**args)

    def test_rmsd_mean_invalid_window_size_zero(self):
        """Test RMSD mean window with zero window size.

        Expected: Raises ValueError for zero window size.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with zero window size
        args = self.get_default_args_rmsd_window_frame_to_start()
        args['window_size'] = 0

        # Verify raises ValueError
        with pytest.raises(ValueError, match="window_size must be positive"):
            self.pipeline.analysis.structure.rmsd.mean.window_frame_to_start(**args)

    def test_rmsd_mean_invalid_stride_negative(self):
        """Test RMSD mean window with negative stride.

        Expected: Raises ValueError for negative stride.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with negative stride
        args = self.get_default_args_rmsd_window_frame_to_start()
        args['stride'] = -1

        # Verify raises ValueError
        with pytest.raises(ValueError, match="stride must be positive"):
            self.pipeline.analysis.structure.rmsd.mean.window_frame_to_start(**args)

    def test_rmsd_mean_invalid_stride_zero(self):
        """Test RMSD mean window with zero stride.

        Expected: Raises ValueError for zero stride.
        """
        # Setup
        mock_calc, mock_instance = self.setup_rmsd_calculator_mock()

        # Execute with zero stride
        args = self.get_default_args_rmsd_window_frame_to_start()
        args['stride'] = 0

        # Verify raises ValueError
        with pytest.raises(ValueError, match="stride must be positive"):
            self.pipeline.analysis.structure.rmsd.mean.window_frame_to_start(**args)

    # Pipeline data validation tests
    def test_rmsf_mean_per_atom_none_pipeline_data(self):
        """Test RMSF mean per-atom with None pipeline data.

        Expected: Raises ValueError for None pipeline data.
        """
        # Verify raises ValueError during service construction
        with pytest.raises(ValueError, match="RMSFPerAtomService requires pipeline_data"):
            RMSFPerAtomService(pipeline_data=None, metric="mean")

    def test_rmsf_mean_per_residue_none_pipeline_data(self):
        """Test RMSF mean per-residue with None pipeline data.

        Expected: Raises ValueError for None pipeline data.
        """
        # Verify raises ValueError during service construction
        with pytest.raises(ValueError, match="RMSFPerResidueService requires pipeline_data"):
            RMSFPerResidueService(pipeline_data=None, metric="mean", aggregator="mean")

    def test_rmsd_mean_none_pipeline_data(self):
        """Test RMSD mean with None pipeline data.

        Expected: Raises ValueError for None pipeline data.
        """
        # Verify raises ValueError during service construction
        with pytest.raises(ValueError, match="RMSDMeanService requires pipeline_data"):
            RMSDMeanService(pipeline_data=None)

    # Inconsistent atom selection tests
    def test_rmsf_mean_per_atom_inconsistent_atom_selection(self):
        """Test RMSF mean per-atom with inconsistent atom selections between trajectories.

        Expected: Raises ValueError for inconsistent atom selection results.
        """
        # Execute with atom selection that produces different counts
        args = self.get_default_args_rmsf_per_atom()
        args['atom_selection'] = "INCONSISTENT"  # Returns 5 atoms for traj 0, 7 atoms for others
        args['cross_trajectory'] = True  # Only validate atom consistency for cross_trajectory=True

        with pytest.raises(ValueError, match="Atom selection results differ between trajectories"):
            self.pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(**args)

    def test_rmsf_mean_per_residue_inconsistent_atom_selection(self):
        """Test RMSF mean per-residue with inconsistent atom selections between trajectories.

        Expected: Raises ValueError for inconsistent atom selection results.
        """
        # Execute with atom selection that produces different counts
        args = self.get_default_args_rmsf_per_residue()
        args['atom_selection'] = "INCONSISTENT"  # Returns 5 atoms for traj 0, 7 atoms for others
        args['cross_trajectory'] = True  # Only validate atom consistency for cross_trajectory=True

        with pytest.raises(ValueError, match="Atom selection results differ between trajectories"):
            self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_mean_reference(**args)

    def test_rmsd_mean_inconsistent_atom_selection(self):
        """Test RMSD mean with inconsistent atom selections between trajectories.

        Expected: Raises ValueError for inconsistent atom selection results.
        """
        # Execute with atom selection that produces different counts
        args = self.get_default_args_rmsd_to_reference()
        args['atom_selection'] = "INCONSISTENT"  # Returns 5 atoms for traj 0, 7 atoms for others

        with pytest.raises(ValueError, match="Atom selection results differ between trajectories"):
            self.pipeline.analysis.structure.rmsd.mean.to_reference(**args)

    # Window size larger than trajectory tests
    def test_rmsd_mean_window_size_larger_than_trajectory(self):
        """Test RMSD mean window with window size larger than trajectory length.

        Expected: Raises ValueError for window size exceeding trajectory frames.
        """
        # Execute with window size larger than trajectory
        args = self.get_default_args_rmsd_window_frame_to_start()
        args['window_size'] = 10000  # Larger than trajectory length

        with pytest.raises(ValueError, match="window_size.*exceeds trajectory length"):
            self.pipeline.analysis.structure.rmsd.mean.window_frame_to_start(**args)

    # Lag larger than trajectory tests
    def test_rmsd_mean_lag_larger_than_trajectory(self):
        """Test RMSD mean frame_to_frame with lag larger than trajectory length.

        Expected: Raises ValueError for lag exceeding trajectory frames.
        """
        # Execute with lag larger than trajectory
        args = self.get_default_args_rmsd_frame_to_frame()
        args['lag'] = 10000  # Larger than trajectory length

        with pytest.raises(ValueError, match="lag.*exceeds trajectory length"):
            self.pipeline.analysis.structure.rmsd.mean.frame_to_frame(**args)
    # Invalid metric tests
    def test_rmsf_invalid_metric_type(self):
        """Test RMSF with invalid metric type.

        Expected: Raises ValueError for unsupported metric.
        """
        # Create a minimal mock pipeline_data
        mock_pipeline_data = Mock()
        mock_pipeline_data.chunk_size = 1000
        mock_pipeline_data.use_memmap = False

        # Verify raises ValueError for invalid metric during service construction
        with pytest.raises(ValueError, match="metric must be"):
            RMSFPerAtomService(pipeline_data=mock_pipeline_data, metric="invalid_metric")

    # NOTE: test_rmsd_invalid_metric_type removed - metric is now compile-time checked via type hints
    # Invalid metric is caught at type-checking time with RMSDMeanService/RMSDMedianService/RMSDMadService

    # Empty trajectories tests
    def test_rmsf_mean_per_atom_empty_trajectories_list(self):
        """Test RMSF mean per-atom with empty trajectories list in pipeline data.

        Expected: Raises ValueError for empty trajectories list.
        """
        # Test with empty trajectory selection (empty list)
        with pytest.raises(ValueError, match="No trajectories found for the requested selection"):
            self.pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(traj_selection=[])

    def test_rmsd_mean_empty_trajectories_list(self):
        """Test RMSD mean with empty trajectories list in pipeline data.

        Expected: Raises ValueError for empty trajectories list.
        """
        # Test with empty trajectory selection (empty list)
        with pytest.raises(ValueError, match="No trajectories found for the requested selection"):
            self.pipeline.analysis.structure.rmsd.mean.to_reference(traj_selection=[])

    # Invalid aggregator tests for RMSF per-residue
    def test_rmsf_per_residue_invalid_aggregator_type(self):
        """Test RMSF per-residue with invalid aggregator type.

        Expected: Raises ValueError for unsupported aggregator.
        """
        # Create a minimal mock pipeline_data
        mock_pipeline_data = Mock()
        mock_pipeline_data.chunk_size = 1000
        mock_pipeline_data.use_memmap = False

        # Verify raises ValueError for invalid aggregator during service construction
        with pytest.raises(ValueError, match="aggregator must be"):
            RMSFPerResidueService(pipeline_data=mock_pipeline_data, metric="mean", aggregator="invalid_aggregator")

    def test_rmsf_mean_per_residue_invalid_reference_traj_selection(self):
        """Test RMSF mean per-residue with reference_traj_selection not in traj_selection.

        Expected: Raises ValueError for reference trajectory not in selected trajectories.
        """
        args = self.get_default_args_rmsf_per_residue()
        args['traj_selection'] = [0, 1, 2]  # Select trajectories 0, 1, 2
        args['reference_traj_selection'] = 5  # Reference trajectory not in selection

        with pytest.raises(ValueError, match="Reference trajectory 5 not in selected trajectories"):
            self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_mean_reference(**args)

    def test_rmsf_mean_per_residue_invalid_reference_traj_selection_median(self):
        """Test RMSF mean per-residue with reference_traj_selection not in traj_selection (median reference).

        Expected: Raises ValueError for reference trajectory not in selected trajectories.
        """
        args = self.get_default_args_rmsf_per_residue()
        args['traj_selection'] = [0, 1, 2]  # Select trajectories 0, 1, 2
        args['reference_traj_selection'] = 5  # Reference trajectory not in selection

        with pytest.raises(ValueError, match="Reference trajectory 5 not in selected trajectories"):
            self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_median_reference(**args)

    def test_rmsf_mean_per_atom_zero_frames_trajectory(self):
        """Test RMSF mean per-atom with trajectory containing zero frames.

        Expected: Raises ValueError for trajectory with no frames.
        """
        # Create trajectory with zero frames
        zero_frames_traj = self._create_mock_trajectory(0)  # traj_index=0
        zero_frames_traj.n_frames = 0  # Set to zero frames

        # Replace first trajectory temporarily
        self.pipeline._data.trajectory_data.trajectories[0] = zero_frames_traj

        args = self.get_default_args_rmsf_per_atom()
        args['traj_selection'] = [0]  # Select only the zero-frames trajectory

        with pytest.raises(ValueError, match="Trajectory .* contains no frames"):
            self.pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(**args)

    def test_rmsf_mean_per_residue_zero_frames_trajectory(self):
        """Test RMSF mean per-residue with trajectory containing zero frames.

        Expected: Raises ValueError for trajectory with no frames.
        """
        # Create trajectory with zero frames
        zero_frames_traj = self._create_mock_trajectory(0)  # traj_index=0
        zero_frames_traj.n_frames = 0  # Set to zero frames

        # Replace first trajectory temporarily
        self.pipeline._data.trajectory_data.trajectories[0] = zero_frames_traj

        args = self.get_default_args_rmsf_per_residue()
        args['traj_selection'] = [0]  # Select only the zero-frames trajectory

        with pytest.raises(ValueError, match="Trajectory .* contains no frames"):
            self.pipeline.analysis.structure.rmsf.mean.per_residue.with_mean_aggregation.to_mean_reference(**args)

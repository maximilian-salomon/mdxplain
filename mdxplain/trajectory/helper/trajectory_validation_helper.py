# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
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

"""
Trajectory validation helper for pipeline operations.

Provides validation logic for trajectory-changing operations in the pipeline,
ensuring that existing features are properly handled before modifications.
"""


class TrajectoryValidationHelper:
    """Helper class for validating trajectory operations in pipeline context."""

    @staticmethod
    def check_features_before_trajectory_changes(
        pipeline_data, force: bool, operation_name: str
    ):
        """
        Check if features exist before changing trajectories and handle accordingly.

        This method validates whether trajectory-changing operations should be allowed
        based on existing computed features. Features become invalid when trajectories
        are modified, so users must explicitly force the operation or clear features first.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        force : bool
            Whether to force the operation despite existing features
        operation_name : str
            Name of the operation for error messages ("load", "add", "remove", "cut", "select_atoms")

        Returns:
        --------
        None
            Validates operation or prints warning messages

        Raises:
        -------
        ValueError
            If features exist and force=False

        Examples:
        ---------
        >>> # Check before loading new trajectories
        >>> TrajectoryValidationHelper.check_features_before_trajectory_changes(
        ...     pipeline_data, False, "load"
        ... )
        ValueError: Cannot load trajectories: 2 feature(s) already computed...

        >>> # Force the operation
        >>> TrajectoryValidationHelper.check_features_before_trajectory_changes(
        ...     pipeline_data, True, "load"
        ... )
        WARNING: Loading new trajectories will invalidate 2 existing features...
        """
        # No features computed - operation is safe
        if not pipeline_data.feature_data:
            return

        feature_list = list(pipeline_data.feature_data.keys())
        feature_count = len(feature_list)

        if not force:
            # Features exist and force=False - raise error
            raise ValueError(
                f"Cannot {operation_name} trajectories: {feature_count} feature(s) "
                f"already computed: {', '.join(feature_list)}. "
                f"{operation_name.capitalize()}ing trajectories would invalidate these "
                f"features. "
                f"Use force=True to proceed, or clear features first with FeatureManager. "
                f"The whole analysis is based on the trajectory data, "
                f"so if you want to change this base, you need to make the analysis again. "
                f"Maybe create a new PipelineData object and start from scratch, "
                f"to not lose your results."
            )

        # Features exist and force=True - print warning
        print(
            f"WARNING: {operation_name.capitalize()}ing trajectories will invalidate "
            f"{feature_count} existing features. Features must be recalculated. "
            f"Existing features: {', '.join(feature_list)}\n"
            f"The whole analysis is based on the trajectory data, so if you want to change "
            f"this base, you need to make the analysis again. Maybe create a new PipelineData "
            f"object and start from scratch, to not lose your results."
        )

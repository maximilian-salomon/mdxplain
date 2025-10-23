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

"""
Helper for calculating probability densities with feature-type awareness.

Provides methods for computing smooth density curves from feature values,
with special handling for discrete features (contacts, DSSP) using
height-dependent Gaussian bells.
"""

from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
from scipy.stats import gaussian_kde

from mdxplain.plots.helper.discrete_feature_helper import DiscreteFeatureHelper


class DensityCalculationHelper:
    """
    Helper class for feature-type-aware density calculations.

    Provides methods to calculate probability densities using:
    - Discrete features (contacts, DSSP): Height-dependent Gaussian bells
    - Continuous features: Standard Kernel Density Estimation

    Examples
    --------
    >>> # Continuous feature (distances)
    >>> x, density = DensityCalculationHelper.calculate_density(
    ...     distance_data, kde_bandwidth="scott"
    ... )

    >>> # Discrete feature (contacts)
    >>> contact_metadata = {"visualization": {"is_discrete": True}}
    >>> x, density = DensityCalculationHelper.calculate_density(
    ...     contact_data, contact_metadata,
    ...     base_sigma=0.05, max_sigma=0.12
    ... )
    """

    @staticmethod
    def calculate_density(
        data: np.ndarray,
        feature_metadata: Optional[Dict[str, Any]] = None,
        kde_bandwidth: Union[str, float] = "scott",
        base_sigma: float = 0.05,
        max_sigma: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate probability density with feature-type-aware method.

        For discrete features (contacts, DSSP), creates Gaussian bells at
        discrete positions with widths proportional to probabilities. For
        continuous features, uses standard Kernel Density Estimation.

        Parameters
        ----------
        data : np.ndarray
            Feature values
        feature_metadata : dict, optional
            Feature metadata containing visualization information.
            If None, falls back to continuous density estimation.
        kde_bandwidth : str or float, default="scott"
            KDE bandwidth for continuous features only:
            - "scott": Scott's rule (automatic bandwidth selection)
            - "silverman": Silverman's rule
            - float: Manual bandwidth value
        base_sigma : float, default=0.05
            Minimum Gaussian width for discrete features (narrowest peak)
        max_sigma : float, default=0.12
            Maximum Gaussian width for discrete features (widest peak)

        Returns
        -------
        x_range : np.ndarray
            X-axis values for plotting (300 points)
        density : np.ndarray
            Probability density values corresponding to x_range

        Examples
        --------
        >>> # Continuous feature with default KDE
        >>> distances = np.random.normal(5.0, 1.5, 1000)
        >>> x, density = DensityCalculationHelper.calculate_density(distances)

        >>> # Discrete feature (contacts) with custom Gaussian widths
        >>> contacts = np.array([0]*900 + [1]*100)  # 90% no contact
        >>> contact_metadata = {"visualization": {"is_discrete": True}}
        >>> x, density = DensityCalculationHelper.calculate_density(
        ...     contacts, contact_metadata, base_sigma=0.04, max_sigma=0.15
        ... )

        >>> # Discrete feature (DSSP) with full visualization metadata
        >>> dssp_data = np.array([0, 1, 2, 0, 1])  # H, E, C, H, E
        >>> dssp_metadata = {
        ...     "visualization": {
        ...         "is_discrete": True,
        ...         "tick_labels": {"short": ["H", "E", "C"]}
        ...     }
        ... }
        >>> x, density = DensityCalculationHelper.calculate_density(
        ...     dssp_data, dssp_metadata
        ... )

        Notes
        -----
        Discrete features create natural-looking distributions where:
        - Dominant states (high probability) => Tall AND wide peaks
        - Rare states (low probability) => Short AND narrow peaks

        This allows multiple DataSelectors to be overlaid without
        visual overlap, unlike bar plots where overlapping bars
        obscure differences.

        Backwards compatibility: When metadata parameters are None,
        falls back to binary data detection for discrete features.
        """
        # Get visualization metadata if available
        viz = feature_metadata.get("visualization", {}) if feature_metadata else {}

        # Check if feature is discrete
        if viz.get("is_discrete", False):
            # Build axis_config from visualization metadata
            tick_labels_dict = viz.get("tick_labels", {})
            tick_labels = tick_labels_dict.get("short", [])

            if tick_labels:
                # Use tick labels to build positions
                n_positions = len(tick_labels)
                positions = list(range(n_positions))
                value_to_position = {i: i for i in range(n_positions)}
                xlim = (-0.3, n_positions - 1 + 0.3)
            else:
                # Fallback for binary without tick labels
                positions = [0, 1]
                value_to_position = {0: 0, 1: 1}
                xlim = (-0.3, 1.3)

            axis_config = {
                "positions": positions,
                "value_to_position": value_to_position,
                "xlim": xlim
            }

            return DensityCalculationHelper._calculate_discrete_gaussians(
                data, axis_config, base_sigma, max_sigma
            )
        else:
            return DensityCalculationHelper._calculate_continuous_density(
                data, kde_bandwidth
            )

    @staticmethod
    def _calculate_discrete_gaussians(
        data: np.ndarray,
        axis_config: Dict[str, Any],
        base_sigma: float,
        max_sigma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create smooth density from discrete data using height-dependent Gaussians.

        Places Gaussian bells at discrete positions with widths proportional
        to their respective probabilities. Works for binary (contacts) and
        multi-class (DSSP) discrete features.

        Important: Not normalized to area=1. Wider and higher peaks have larger area.

        Parameters
        ----------
        data : np.ndarray
            Discrete values (integers, chars, or binary)
        axis_config : dict
            Axis configuration from DiscreteFeatureHelper with keys:
            positions, value_to_position, xlim
        base_sigma : float
            Minimum width for narrowest peak (low probability)
        max_sigma : float
            Maximum width for widest peak (high probability)

        Returns
        -------
        x_range : np.ndarray
            X-axis from xlim[0] to xlim[1] (allows for Gaussian tails)
        density : np.ndarray
            Sum of all Gaussian bells

        Examples
        --------
        >>> # Binary contacts
        >>> data = np.array([0]*90 + [1]*10)
        >>> config = {"positions": [0, 1], "value_to_position": {0:0, 1:1},
        ...          "xlim": (-0.3, 1.3)}
        >>> x, density = DensityCalculationHelper._calculate_discrete_gaussians(
        ...     data, config, base_sigma=0.05, max_sigma=0.12
        ... )

        >>> # DSSP with 3 classes
        >>> data = np.array([0, 1, 2, 0, 0, 1])  # H, E, C distribution
        >>> config = {"positions": [0, 1, 2], "value_to_position": {0:0, 1:1, 2:2},
        ...          "xlim": (-0.3, 2.3)}
        >>> x, density = DensityCalculationHelper._calculate_discrete_gaussians(
        ...     data, config, base_sigma=0.05, max_sigma=0.12
        ... )

        Notes
        -----
        Width scaling formula:
            σ = base_sigma + (max_sigma - base_sigma) x probability

        This ensures:
        - P=1.0 (100%) => σ = max_sigma (widest possible)
        - P=0.0 (0%)   => σ = base_sigma (narrowest possible)
        - P=0.5 (50%)  => σ = (base_sigma + max_sigma) / 2 (medium)

        The Gaussian bell formula:
            gaussian = probability x exp(-0.5 x ((x - center) / σ)²)
        """
        # 1. Extract config components
        positions = axis_config["positions"]
        value_to_position = axis_config["value_to_position"]
        xlim = axis_config["xlim"]

        # 2. Convert data to positions if needed
        position_data = DiscreteFeatureHelper.prepare_discrete_data(
            data, value_to_position
        )

        # 3. Create x-axis grid based on xlim
        x_range = np.linspace(xlim[0], xlim[1], 300)

        # 4. Initialize density as zeros
        density = np.zeros_like(x_range)

        # 5. Create Gaussian bell for each position
        for pos in positions:
            # Calculate probability for this position
            prob = np.sum(position_data == pos) / len(position_data)

            # Calculate width based on probability
            sigma = base_sigma + (max_sigma - base_sigma) * prob

            # Create Gaussian bell at this position
            gaussian = prob * np.exp(-0.5 * ((x_range - pos) / sigma) ** 2)

            # Add to total density
            density += gaussian

        return x_range, density

    @staticmethod
    def _calculate_continuous_density(
        data: np.ndarray,
        kde_bandwidth: Union[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate standard Kernel Density Estimation for continuous features.

        Uses scipy's gaussian_kde with automatic or manual bandwidth selection
        to create smooth probability density curves.

        Parameters
        ----------
        data : np.ndarray
            Continuous feature values (e.g., distances, angles, SASA)
        kde_bandwidth : str or float
            Bandwidth selection method:
            - "scott": Scott's rule (default, works well for most cases)
            - "silverman": Silverman's rule
            - float: Manual bandwidth factor

        Returns
        -------
        x_range : np.ndarray
            X-axis with 20% padding beyond data range (300 points)
        density : np.ndarray
            KDE probability density values

        Examples
        --------
        >>> # Distance feature with automatic bandwidth
        >>> distances = np.array([4.5, 5.2, 4.8, 6.1, 5.5])
        >>> x, density = DensityCalculationHelper._calculate_continuous_density(
        ...     distances, "scott"
        ... )

        >>> # Custom bandwidth
        >>> x, density = DensityCalculationHelper._calculate_continuous_density(
        ...     distances, 0.5
        ... )

        Notes
        -----
        The x-range is automatically calculated with 20% padding beyond the
        data range to show the full distribution tails without truncation.

        For the bandwidth parameter:
        - "scott": Bandwidth = n^(-1/5) x σ (good default)
        - "silverman": Bandwidth = n^(-1/5) x min(σ, IQR/1.34)
        - float: Multiply default bandwidth by this factor
        """
        # Create Gaussian KDE with specified bandwidth
        kde = gaussian_kde(data, bw_method=kde_bandwidth)

        # Calculate auto-range with 20% padding for nice visualization
        data_range = data.max() - data.min()
        x_min = data.min() - 0.2 * data_range
        x_max = data.max() + 0.2 * data_range
        x_range = np.linspace(x_min, x_max, 300)

        # Evaluate KDE at x_range points
        density = kde(x_range)

        return x_range, density

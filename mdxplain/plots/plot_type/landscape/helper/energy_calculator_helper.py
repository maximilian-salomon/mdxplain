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
Helper class for energy landscape calculations.

Provides methods for free energy surface calculation from decomposition
data using Boltzmann inversion of probability distributions.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.stats import gaussian_kde

# Boltzmann constant in kcal/(mol*K)
KB_KCAL = 0.001987204


class EnergyCalculatorHelper:
    """
    Helper class for energy landscape calculations.

    Provides static methods for calculating KDE-based free energy landscapes
    using Boltzmann inversion, determining energy ranges, and masking
    high-energy regions.

    Examples
    --------
    >>> # Calculate KDE-based energy landscape
    >>> X, Y, energy = EnergyCalculatorHelper.calculate_kde_energy_landscape(
    ...     data_x, data_y, bins=50, temperature=310.15
    ... )

    >>> # Get energy range for colormap
    >>> vmin, vmax = EnergyCalculatorHelper.get_energy_range(energy, percentile=99.0)
    """

    @staticmethod
    def calculate_kde_grid(
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate KDE on 2D grid.

        Computes Gaussian kernel density estimation on a regular grid
        for 2D data.

        Parameters
        ----------
        data_x : numpy.ndarray, shape (n_frames,)
            Data for x-axis
        data_y : numpy.ndarray, shape (n_frames,)
            Data for y-axis
        bins : int
            Number of grid points per dimension
        xlim : Tuple[float, float], optional
            X-axis limits for grid. If None, uses data min/max
        ylim : Tuple[float, float], optional
            Y-axis limits for grid. If None, uses data min/max

        Returns
        -------
        X : numpy.ndarray, shape (bins, bins)
            X-axis grid coordinates
        Y : numpy.ndarray, shape (bins, bins)
            Y-axis grid coordinates
        density : numpy.ndarray, shape (bins, bins)
            KDE-evaluated density values

        Notes
        -----
        Uses scipy.stats.gaussian_kde for density estimation.
        Grid is created using linspace between min and max data values
        or provided limits.
        """
        kde = gaussian_kde([data_x, data_y])

        # Use provided limits or data extent
        x_min, x_max = xlim if xlim is not None else (data_x.min(), data_x.max())
        y_min, y_max = ylim if ylim is not None else (data_y.min(), data_y.max())

        x_grid = np.linspace(x_min, x_max, bins)
        y_grid = np.linspace(y_min, y_max, bins)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])

        density = kde(positions).reshape(X.shape)

        return X, Y, density

    @staticmethod
    def calculate_kde_energy_landscape(
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        temperature: float,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate KDE-based free energy landscape.

        Uses Gaussian kernel density estimation to compute smooth density,
        then applies Boltzmann inversion: F = -kT * ln(P/P_max)

        Parameters
        ----------
        data_x : numpy.ndarray, shape (n_frames,)
            Data for x-axis (first dimension)
        data_y : numpy.ndarray, shape (n_frames,)
            Data for y-axis (second dimension)
        bins : int
            Number of grid points per dimension
        temperature : float
            Temperature in Kelvin for Boltzmann factor
        xlim : Tuple[float, float], optional
            X-axis limits for grid. If None, uses data min/max
        ylim : Tuple[float, float], optional
            Y-axis limits for grid. If None, uses data min/max

        Returns
        -------
        X : numpy.ndarray, shape (bins, bins)
            X-axis grid coordinates
        Y : numpy.ndarray, shape (bins, bins)
            Y-axis grid coordinates
        energy : numpy.ndarray, shape (bins, bins)
            Free energy values in kcal/mol

        Notes
        -----
        Uses scipy.stats.gaussian_kde for smooth density estimation.
        The Boltzmann relation is: F = -kT * ln(P/P_max)
        where:
        - F: Free energy
        - k: Boltzmann constant (0.001987204 kcal/(mol*K))
        - T: Temperature in Kelvin
        - P: KDE-estimated probability density
        - P_max: Maximum density across grid
        """
        # Use shared KDE grid calculation with extended limits
        X, Y, density = EnergyCalculatorHelper.calculate_kde_grid(
            data_x, data_y, bins, xlim, ylim
        )

        # Boltzmann inversion
        kT = KB_KCAL * temperature
        with np.errstate(divide='ignore', invalid='ignore'):
            energy = -kT * np.log(density / density.max())

        return X, Y, energy

    @staticmethod
    def get_energy_range(
        energy: np.ndarray,
        percentile: float = 99.0
    ) -> Tuple[float, float]:
        """
        Get reasonable energy range for colormap limits.

        Excludes extreme outliers by using percentile cutoff.

        Parameters
        ----------
        energy : numpy.ndarray
            Free energy landscape
        percentile : float, default=99.0
            Percentile for maximum energy cutoff

        Returns
        -------
        vmin : float
            Minimum energy (always 0.0 for reference state)
        vmax : float
            Maximum energy at given percentile

        Examples
        --------
        >>> energy = np.random.exponential(2.0, (50, 50))
        >>> vmin, vmax = EnergyCalculatorHelper.get_energy_range(energy, percentile=95.0)
        >>> print(f"Energy range: {vmin:.1f} to {vmax:.1f} kcal/mol")

        Notes
        -----
        Using percentile cutoff prevents a few high-energy outliers
        from dominating the colormap, making the landscape more
        visually informative.
        """
        # Minimum is always 0 (reference state at maximum probability)
        vmin = 0.0

        # Maximum uses percentile to exclude extreme outliers
        vmax = np.nanpercentile(energy, percentile)

        return vmin, vmax

    @staticmethod
    def mask_high_energy_regions(
        energy: np.ndarray,
        threshold: float = 10.0
    ) -> np.ma.MaskedArray:
        """
        Mask high-energy regions for cleaner visualization.

        Regions with energy above threshold are masked (not shown).

        Parameters
        ----------
        energy : numpy.ndarray
            Free energy landscape
        threshold : float, default=10.0
            Energy threshold in kcal/mol

        Returns
        -------
        numpy.ma.MaskedArray
            Energy array with high values masked

        Examples
        --------
        >>> energy = np.random.exponential(2.0, (50, 50))
        >>> masked = EnergyCalculatorHelper.mask_high_energy_regions(energy, threshold=5.0)
        >>> print(f"Masked {masked.mask.sum()} of {energy.size} bins")

        Notes
        -----
        Masking helps focus visualization on relevant conformational
        space by hiding rarely-sampled high-energy regions.
        """
        # Mask both NaN and high-energy regions
        mask = np.isnan(energy) | (energy > threshold)
        return np.ma.masked_array(energy, mask=mask)

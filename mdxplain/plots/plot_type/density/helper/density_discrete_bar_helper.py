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
Density helper for rendering grouped discrete probability bars.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ....helper.discrete_feature_helper import DiscreteFeatureHelper


class DensityDiscreteBarHelper:
    """
    Renderer for grouped probability bars of discrete features.

    Encapsulates the discrete-bar drawing strategy used by density plots
    so the main plotter remains focused on orchestration.
    """

    @staticmethod
    def plot_grouped_probability_bars(
        ax,
        selector_data: Dict[str, np.ndarray],
        data_selector_colors: Dict[str, str],
        viz: Dict[str, Any],
        long_labels: bool,
        alpha: float = 0.85
    ) -> None:
        """
        Plot grouped probability bars for one discrete feature.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Subplot axes.
        selector_data : Dict[str, np.ndarray]
            Selector to discrete values.
        data_selector_colors : Dict[str, str]
            Selector color mapping.
        viz : Dict[str, Any]
            Visualization metadata for the feature.
        long_labels : bool
            Whether long labels should be used on x-axis.
        alpha : float, default=0.85
            Bar transparency.

        Returns
        -------
        None
            Modifies the provided `ax` in place.

        Notes
        -----
        Bars are grouped per discrete state. For multiple selectors, bars are
        placed side-by-side with adaptive width to preserve readability.
        """
        axis_config = DiscreteFeatureHelper.build_axis_config(
            selector_data=selector_data,
            viz=viz,
            long_labels=long_labels,
            x_padding=0.5,
            fallback_from_data=True
        )
        positions = axis_config["positions"]
        value_to_position = axis_config["value_to_position"]
        tick_labels = axis_config["tick_labels"]
        xlim = axis_config["xlim"]

        selector_names = list(selector_data.keys())
        if not selector_names or len(positions) == 0:
            return

        n_selectors = len(selector_names)
        group_width = 0.8
        if n_selectors == 1:
            bar_width = 0.5
        else:
            bar_width = min(group_width / n_selectors, 0.18)

        total_width = bar_width * n_selectors
        start_offset = -0.5 * total_width + 0.5 * bar_width

        for selector_idx, selector_name in enumerate(selector_names):
            probabilities = DiscreteFeatureHelper.calculate_discrete_probabilities(
                selector_data[selector_name], value_to_position, len(positions)
            )
            x_positions = positions + start_offset + selector_idx * bar_width
            ax.bar(
                x_positions, probabilities,
                width=bar_width * 0.95,
                color=data_selector_colors[selector_name],
                alpha=alpha,
                label=selector_name
            )

        ax.set_xticks(positions)
        if tick_labels:
            if long_labels:
                ax.set_xticklabels(tick_labels, rotation=90, ha="center")
            else:
                ax.set_xticklabels(tick_labels)
        ax.set_xlim(*xlim)

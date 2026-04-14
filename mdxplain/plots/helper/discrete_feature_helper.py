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
Helper for discrete feature data preparation.

Provides data conversion for discrete features to prepare them for
visualization by mapping categorical values to integer positions.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class DiscreteFeatureHelper:
    """
    Helper for discrete feature data preparation.

    Provides data conversion for discrete features to prepare them for
    visualization by mapping categorical values to integer positions.

    Examples
    --------
    >>> # Convert character data to positions
    >>> data = np.array(['H', 'E', 'C', 'H'])
    >>> mapping = {'H': 0, 'E': 1, 'C': 2}
    >>> positions = DiscreteFeatureHelper.prepare_discrete_data(data, mapping)
    >>> print(positions)
    [0 1 2 0]
    """

    @staticmethod
    def prepare_discrete_data(
        data: np.ndarray,
        value_to_position: Dict
    ) -> np.ndarray:
        """
        Convert discrete data to integer positions for plotting.

        Maps categorical data values (integers or characters) to sequential
        integer positions suitable for visualization.

        Parameters
        ----------
        data : np.ndarray
            Original data (integers or character strings)
        value_to_position : dict
            Mapping from data values to plot positions

        Returns
        -------
        np.ndarray
            Data converted to integer positions

        Examples
        --------
        >>> # Integer data (no conversion needed)
        >>> data = np.array([0, 1, 2, 0, 1])
        >>> mapping = {0: 0, 1: 1, 2: 2}
        >>> positions = DiscreteFeatureHelper.prepare_discrete_data(data, mapping)
        >>> print(positions)
        [0 1 2 0 1]

        >>> # Character data (needs conversion)
        >>> data = np.array(['H', 'E', 'C', 'H', 'E'])
        >>> mapping = {'H': 0, 'E': 1, 'C': 2}
        >>> positions = DiscreteFeatureHelper.prepare_discrete_data(data, mapping)
        >>> print(positions)
        [0 1 2 0 1]

        Notes
        -----
        Values not present in value_to_position are returned as -1.
        """
        values = np.asarray(data).ravel()
        if values.size == 0:
            return np.array([], dtype=int)

        mapped_positions = [
            DiscreteFeatureHelper._map_value_to_position(value, value_to_position)
            for value in values
        ]
        return np.asarray(mapped_positions, dtype=int)

    @staticmethod
    def build_axis_config(
        selector_data: Optional[Dict[str, np.ndarray]] = None,
        viz: Optional[Dict[str, Any]] = None,
        long_labels: bool = False,
        x_padding: float = 0.3,
        fallback_from_data: bool = False
    ) -> Dict[str, Any]:
        """
        Build axis configuration for discrete plotting.

        Parameters
        ----------
        selector_data : Dict[str, np.ndarray], optional
            Selector values used for optional data-driven fallback.
        viz : Dict[str, Any], optional
            Visualization metadata (expects optional tick_labels).
        long_labels : bool, default=False
            If True, uses long tick labels from metadata when available.
        x_padding : float, default=0.3
            Horizontal padding applied to x-limits.
        fallback_from_data : bool, default=False
            If True and metadata has no tick labels, build positions from
            unique observed values in selector_data. If False, binary fallback.

        Returns
        -------
        Dict[str, Any]
            Axis configuration with keys:
            
            - positions: np.ndarray
            - value_to_position: Dict[Any, int]
            - tick_labels: List[str]
            - xlim: tuple(float, float)
        """
        metadata_axis_config = DiscreteFeatureHelper._build_axis_config_from_metadata(
            viz=viz,
            long_labels=long_labels,
            x_padding=x_padding
        )
        if metadata_axis_config is not None:
            return metadata_axis_config

        if fallback_from_data and selector_data:
            return DiscreteFeatureHelper._build_axis_config_from_data(
                selector_data=selector_data,
                x_padding=x_padding
            )

        # Conservative fallback used when no metadata labels are available.
        return DiscreteFeatureHelper._build_binary_axis_config(x_padding=x_padding)

    @staticmethod
    def _build_axis_config_from_metadata(
        viz: Optional[Dict[str, Any]],
        long_labels: bool,
        x_padding: float
    ) -> Optional[Dict[str, Any]]:
        """
        Build discrete axis configuration from visualization metadata.

        Parameters
        ----------
        viz : Dict[str, Any], optional
            Visualization metadata that may contain `tick_labels`.
        long_labels : bool
            Whether to prefer long labels (`tick_labels["long"]`) over
            short labels (`tick_labels["short"]`).
        x_padding : float
            Horizontal padding applied to x-limits.

        Returns
        -------
        Dict[str, Any] or None
            Axis configuration if metadata labels are available,
            otherwise `None`.
        """
        viz = viz or {}
        tick_labels_dict = viz.get("tick_labels", {})
        label_key = "long" if long_labels else "short"
        tick_labels = list(tick_labels_dict.get(label_key, []))

        if not tick_labels:
            return None

        n_positions = len(tick_labels)
        positions = np.arange(n_positions, dtype=float)
        value_to_position = {i: i for i in range(n_positions)}
        return DiscreteFeatureHelper._compose_axis_config(
            positions=positions,
            value_to_position=value_to_position,
            tick_labels=tick_labels,
            x_padding=x_padding
        )

    @staticmethod
    def _build_axis_config_from_data(
        selector_data: Dict[str, np.ndarray],
        x_padding: float
    ) -> Dict[str, Any]:
        """
        Build discrete axis configuration from observed selector values.

        Parameters
        ----------
        selector_data : Dict[str, np.ndarray]
            Mapping of selector names to discrete value arrays.
        x_padding : float
            Horizontal padding applied to x-limits.

        Returns
        -------
        Dict[str, Any]
            Axis configuration derived from unique observed values.
            Falls back to a single zero category if all arrays are empty.
        """
        values = DiscreteFeatureHelper._extract_values_from_selector_data(
            selector_data
        )
        if not values:
            return DiscreteFeatureHelper._build_singleton_zero_axis_config(
                x_padding=x_padding
            )

        unique_values = DiscreteFeatureHelper._extract_sorted_unique_values(values)
        positions = np.arange(len(unique_values), dtype=float)
        value_to_position = {value: i for i, value in enumerate(unique_values)}
        tick_labels = [str(v) for v in unique_values]
        return DiscreteFeatureHelper._compose_axis_config(
            positions=positions,
            value_to_position=value_to_position,
            tick_labels=tick_labels,
            x_padding=x_padding
        )

    @staticmethod
    def _extract_sorted_unique_values(values: List[Any]) -> List[Any]:
        """
        Extract sorted unique discrete values with stable type handling.

        Parameters
        ----------
        values : List[Any]
            Raw normalized discrete values.

        Returns
        -------
        List[Any]
            Sorted unique values.

        Notes
        -----
        If any value is string-like, all values are converted to strings
        before deduplication/sorting to avoid mixed-type ordering issues.
        Otherwise, values are coerced to integers.
        """
        if any(isinstance(v, str) for v in values):
            return sorted({str(v) for v in values})
        return sorted({int(v) for v in values})

    @staticmethod
    def _build_singleton_zero_axis_config(x_padding: float) -> Dict[str, Any]:
        """
        Build axis configuration for an empty-data single category fallback.

        Parameters
        ----------
        x_padding : float
            Horizontal padding applied to x-limits.

        Returns
        -------
        Dict[str, Any]
            Axis configuration with one category labeled `"0"`.
        """
        positions = np.array([0.0], dtype=float)
        return DiscreteFeatureHelper._compose_axis_config(
            positions=positions,
            value_to_position={0: 0},
            tick_labels=["0"],
            x_padding=x_padding
        )

    @staticmethod
    def _build_binary_axis_config(x_padding: float) -> Dict[str, Any]:
        """
        Build conservative binary axis fallback configuration.

        Parameters
        ----------
        x_padding : float
            Horizontal padding applied to x-limits.

        Returns
        -------
        Dict[str, Any]
            Axis configuration with categories `0` and `1`.
        """
        positions = np.array([0.0, 1.0], dtype=float)
        return DiscreteFeatureHelper._compose_axis_config(
            positions=positions,
            value_to_position={0: 0, 1: 1},
            tick_labels=["0", "1"],
            x_padding=x_padding
        )

    @staticmethod
    def _compose_axis_config(
        positions: np.ndarray,
        value_to_position: Dict[Any, int],
        tick_labels: List[str],
        x_padding: float
    ) -> Dict[str, Any]:
        """
        Compose the standard axis configuration dictionary.

        Parameters
        ----------
        positions : np.ndarray
            Position coordinates on discrete axis.
        value_to_position : Dict[Any, int]
            Mapping from raw values to positions.
        tick_labels : List[str]
            Tick labels aligned with `positions`.
        x_padding : float
            Horizontal padding applied to x-limits.

        Returns
        -------
        Dict[str, Any]
            Axis configuration containing positions, mapping, labels, and xlim.
        """
        return {
            "positions": positions,
            "value_to_position": value_to_position,
            "tick_labels": tick_labels,
            "xlim": DiscreteFeatureHelper._build_xlim(positions, x_padding)
        }

    @staticmethod
    def calculate_discrete_probabilities(
        data: np.ndarray,
        value_to_position: Dict[Any, int],
        n_positions: int
    ) -> np.ndarray:
        """
        Convert discrete samples into per-position probabilities.

        Parameters
        ----------
        data : np.ndarray
            Raw discrete samples for one selector.
        value_to_position : Dict[Any, int]
            Mapping from raw discrete values to axis position indices.
        n_positions : int
            Total number of discrete axis positions.

        Returns
        -------
        np.ndarray
            Probability vector of length `n_positions`.

        Notes
        -----
        Samples that cannot be mapped to a valid position are ignored.
        """
        counts = np.zeros(n_positions, dtype=float)
        if n_positions <= 0:
            return counts

        mapped_positions = DiscreteFeatureHelper.prepare_discrete_data(
            data, value_to_position
        )
        if mapped_positions.size == 0:
            return counts

        for mapped in mapped_positions:
            if 0 <= mapped < n_positions:
                counts[mapped] += 1.0

        total = counts.sum()
        if total <= 0:
            return counts
        return counts / total

    @staticmethod
    def _extract_values_from_selector_data(
        selector_data: Dict[str, np.ndarray]
    ) -> List[Any]:
        """
        Extract normalized values from selector data dictionary.

        Parameters
        ----------
        selector_data : Dict[str, np.ndarray]
            Mapping of selector name to raw discrete values.

        Returns
        -------
        List[Any]
            Flattened list of normalized values across all selectors.
        """
        values: List[Any] = []
        for data in selector_data.values():
            arr = np.asarray(data).ravel()
            if arr.size == 0:
                continue
            for raw_value in arr:
                values.append(
                    DiscreteFeatureHelper._normalize_discrete_value(raw_value)
                )
        return values

    @staticmethod
    def _map_value_to_position(value: Any, value_to_position: Dict[Any, int]) -> int:
        """
        Map one raw value to a discrete position index, or -1 if unknown.

        Parameters
        ----------
        value : Any
            Raw discrete value from input data.
        value_to_position : Dict[Any, int]
            Mapping from raw values to position indices.

        Returns
        -------
        int
            Position index if mapping succeeds, otherwise `-1`.

        Notes
        -----
        Includes tolerant fallback matching between numeric and string
        representations (e.g., `1` vs `"1"`).
        """
        normalized_value = DiscreteFeatureHelper._normalize_discrete_value(value)
        mapped_idx = value_to_position.get(normalized_value)
        if mapped_idx is not None:
            return int(mapped_idx)

        if isinstance(normalized_value, str):
            try:
                mapped_idx = value_to_position.get(int(normalized_value))
            except (TypeError, ValueError):
                mapped_idx = None
        else:
            mapped_idx = value_to_position.get(str(normalized_value))
            if mapped_idx is not None:
                return int(mapped_idx)
            try:
                mapped_idx = value_to_position.get(int(normalized_value))
            except (TypeError, ValueError):
                mapped_idx = None

        if mapped_idx is None:
            return -1
        return int(mapped_idx)

    @staticmethod
    def _normalize_discrete_value(value: Any) -> Any:
        """
        Normalize numpy scalar values to native Python values.

        Parameters
        ----------
        value : Any
            Raw input value, potentially a numpy scalar.

        Returns
        -------
        Any
            Native Python scalar if conversion is possible, otherwise
            unchanged input value.
        """
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _build_xlim(positions: np.ndarray, x_padding: float) -> tuple:
        """
        Build x-limits around discrete positions with padding.

        Parameters
        ----------
        positions : np.ndarray
            Ordered position coordinates on the discrete axis.
        x_padding : float
            Symmetric padding added left and right.

        Returns
        -------
        tuple
            `(x_min, x_max)` axis limits.
        """
        if positions.size == 0:
            return (-0.5, 0.5)
        return (float(positions[0] - x_padding), float(positions[-1] + x_padding))

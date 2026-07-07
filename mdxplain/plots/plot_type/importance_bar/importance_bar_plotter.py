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
Importance bar plotter for feature importance visualization.

Renders ranked feature importance scores as horizontal bar charts, one
subplot per sub-comparison. Works for any feature importance analyzer: when
per-tree standard deviations are stored (Random Forest with GINI importance),
they are drawn as error bars; otherwise plain bars are drawn.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ....feature_importance.helper.top_features_helper import TopFeaturesHelper
from ....utils.path_utils import PathUtils
from ...helper.svg_export_helper import SvgExportHelper


class ImportanceBarPlotter:
    """
    Bar plotter for feature importance scores.

    Draws a horizontal bar chart of the top feature importance scores for each
    sub-comparison of a feature importance analysis. Error bars are drawn when
    the analysis stores a per-tree importance standard deviation (Random Forest
    with GINI importance); for analyzers without it (Decision Tree, SHAP) the
    bars are drawn without error bars.

    Examples
    --------
    >>> plotter = ImportanceBarPlotter(pipeline_data, cache_dir)
    >>> fig = plotter.plot("forest_gini", n_top=10)
    """

    def __init__(self, pipeline_data, cache_dir: str) -> None:
        """
        Initialize the importance bar plotter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        cache_dir : str
            Cache directory path

        Returns
        -------
        None
            Initializes ImportanceBarPlotter instance
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = cache_dir

    def plot(
        self,
        feature_importance_name: str,
        n_top: int = 10,
        max_cols: int = 2,
        color: str = "#4C72B0",
        title: Optional[str] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300,
        title_fontsize: Optional[int] = None,
        label_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None,
    ) -> Figure:
        """
        Create a bar plot of feature importance scores.

        Draws one subplot per sub-comparison, each showing the top ``n_top``
        features ranked by importance. Error bars are drawn when the analysis
        stores an importance standard deviation.

        Parameters
        ----------
        feature_importance_name : str
            Name of the feature importance analysis
        n_top : int, default=10
            Number of top features to show per sub-comparison
        max_cols : int, default=2
            Maximum number of subplot columns
        color : str, default="#4C72B0"
            Bar color
        title : str, optional
            Custom figure title. Auto-generated if None.
        save_fig : bool, default=False
            Whether to save the figure to a file
        filename : str, optional
            Custom filename. Auto-generated if None.
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.)
        dpi : int, default=300
            Resolution for the saved figure in dots per inch
        title_fontsize : int, optional
            Font size for the figure title
        label_fontsize : int, optional
            Font size for axis and subplot-title labels
        tick_fontsize : int, optional
            Font size for the feature tick labels

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the bar plots

        Raises
        ------
        ValueError
            If the feature importance analysis is not found

        Examples
        --------
        >>> # Random Forest GINI importance (with error bars)
        >>> fig = plotter.plot("forest_gini", n_top=10)

        >>> # Decision Tree importance (plain bars)
        >>> fig = plotter.plot("tree_analysis", n_top=8)
        """
        fi_data = self._get_fi_data(feature_importance_name)
        entries = self._collect_bar_data(fi_data, n_top)
        method_label = self._method_label(fi_data)

        fig, axes = self._make_figure_and_axes(len(entries), max_cols)
        for ax, entry in zip(axes, entries):
            self._render_bars(ax, entry, color, label_fontsize, tick_fontsize)
        self._hide_unused_axes(axes, len(entries))

        self._finalize_figure(
            fig, title, feature_importance_name, method_label, title_fontsize
        )
        if save_fig:
            self._save_figure(
                fig, filename, feature_importance_name, n_top, file_format, dpi
            )
        return fig

    def _get_fi_data(self, feature_importance_name: str):
        """
        Retrieve the feature importance data for an analysis.

        Parameters
        ----------
        feature_importance_name : str
            Name of the feature importance analysis

        Returns
        -------
        FeatureImportanceData
            The stored feature importance data

        Raises
        ------
        ValueError
            If the analysis is not found
        """
        store = self.pipeline_data.feature_importance_data
        if feature_importance_name not in store:
            raise ValueError(
                f"Feature importance analysis '{feature_importance_name}' "
                f"not found. Available: {list(store.keys())}"
            )
        return store[feature_importance_name]

    def _collect_bar_data(self, fi_data, n_top: int) -> List[Dict[str, Any]]:
        """
        Build the per-sub-comparison bar data.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data object
        n_top : int
            Number of top features per sub-comparison

        Returns
        -------
        List[Dict[str, Any]]
            One entry per sub-comparison with labels, values and errors
        """
        entries = []
        for comp_name in fi_data.list_comparisons():
            entries.append(
                self._build_comparison_entry(fi_data, comp_name, n_top)
            )
        return entries

    def _build_comparison_entry(
        self, fi_data, comp_name: str, n_top: int
    ) -> Dict[str, Any]:
        """
        Build the bar data for a single sub-comparison.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data object
        comp_name : str
            Sub-comparison name
        n_top : int
            Number of top features to include

        Returns
        -------
        Dict[str, Any]
            Entry with keys comparison, labels, values and errors
        """
        top_features = TopFeaturesHelper.get_top_features_with_names(
            self.pipeline_data, fi_data, comp_name, n_top
        )
        _, metadata = fi_data.get_comparison(comp_name)
        feature_indices = [f["feature_index"] for f in top_features]
        merged_counts = metadata.get("merged_counts", {})
        return {
            "comparison": comp_name,
            "labels": [
                self._feature_label(f, merged_counts) for f in top_features
            ],
            "values": [f["importance_score"] for f in top_features],
            "errors": self._extract_std(metadata, feature_indices),
        }

    @staticmethod
    def _feature_label(
        feature: Dict[str, Any],
        merged_counts: Optional[Dict[int, int]] = None,
    ) -> str:
        """
        Build a bar label from a feature info dict.

        Appends the merged-neighbour count in parentheses when the feature is a
        filter representative (see FeatureImportanceManager.filter_importance).

        Parameters
        ----------
        feature : Dict[str, Any]
            Feature info dict with feature_type and feature_name
        merged_counts : Dict[int, int], optional
            Mapping from representative feature index to merged-neighbour count

        Returns
        -------
        str
            Label in the form "feature_type: feature_name" (with " (+N)" when a
            merged-neighbour count is available)
        """
        label = f"{feature['feature_type']}: {feature['feature_name']}"
        count = (merged_counts or {}).get(feature["feature_index"])
        if count:
            return f"{label} (+{count})"
        return label

    @staticmethod
    def _extract_std(
        metadata: Dict[str, Any], feature_indices: List[int]
    ) -> Optional[List[float]]:
        """
        Extract per-feature importance standard deviations.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Sub-comparison metadata
        feature_indices : List[int]
            Feature indices to extract standard deviations for

        Returns
        -------
        List[float] or None
            Standard deviations aligned with feature_indices, or None if the
            analysis does not store an importance standard deviation

        Notes
        -----
        The calculator metadata (with importance_std) is stored nested under
        the ``analysis_metadata`` key by the analysis runner.
        """
        analysis_metadata = metadata.get("analysis_metadata", {})
        importance_std = analysis_metadata.get("importance_std")
        if not importance_std:
            return None
        std_array = np.asarray(importance_std)
        return [float(std_array[idx]) for idx in feature_indices]

    def _method_label(self, fi_data) -> str:
        """
        Build a label describing the importance algorithm and method.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data object

        Returns
        -------
        str
            Label such as "random_forest (gini)" or "decision_tree"
        """
        comparisons = fi_data.list_comparisons()
        if not comparisons:
            return fi_data.analyzer_type
        _, metadata = fi_data.get_comparison(comparisons[0])
        analysis_metadata = metadata.get("analysis_metadata", {})
        algorithm = analysis_metadata.get("algorithm", fi_data.analyzer_type)
        method = analysis_metadata.get("importance_method")
        if method:
            return f"{algorithm} ({method})"
        return algorithm

    @staticmethod
    def _make_figure_and_axes(n_items: int, max_cols: int):
        """
        Create the figure and a flat list of axes.

        Parameters
        ----------
        n_items : int
            Number of sub-comparisons to plot
        max_cols : int
            Maximum number of subplot columns

        Returns
        -------
        Tuple[Figure, list]
            The figure and a flat list of axes
        """
        n_cols = max(1, min(max_cols, n_items))
        n_rows = max(1, int(np.ceil(n_items / n_cols)))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(7, n_cols * 6), max(4, n_rows * 4)),
        )
        axes_flat = np.atleast_1d(axes).ravel().tolist()
        return fig, axes_flat

    @staticmethod
    def _render_bars(
        ax,
        entry: Dict[str, Any],
        color: str,
        label_fontsize: Optional[int],
        tick_fontsize: Optional[int],
    ) -> None:
        """
        Draw the horizontal bars for one sub-comparison.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on
        entry : Dict[str, Any]
            Bar data with labels, values and errors
        color : str
            Bar color
        label_fontsize : int or None
            Font size for axis and title labels
        tick_fontsize : int or None
            Font size for the feature tick labels

        Returns
        -------
        None
            Draws onto the axes in place
        """
        labels = entry["labels"]
        positions = np.arange(len(labels))
        ax.barh(
            positions,
            entry["values"],
            xerr=entry["errors"],
            color=color,
            edgecolor="black",
            capsize=3,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=tick_fontsize or 9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=label_fontsize or 11)
        ax.set_title(entry["comparison"], fontsize=label_fontsize or 11)

    @staticmethod
    def _hide_unused_axes(axes: List, n_used: int) -> None:
        """
        Hide axes that have no sub-comparison to display.

        Parameters
        ----------
        axes : list
            Flat list of axes
        n_used : int
            Number of axes actually used

        Returns
        -------
        None
            Turns off the unused axes in place
        """
        for ax in axes[n_used:]:
            ax.axis("off")

    @staticmethod
    def _finalize_figure(
        fig: Figure,
        title: Optional[str],
        feature_importance_name: str,
        method_label: str,
        title_fontsize: Optional[int],
    ) -> None:
        """
        Add the figure title and tighten the layout.

        Parameters
        ----------
        fig : Figure
            Figure to finalize
        title : str or None
            Custom title, or None to auto-generate
        feature_importance_name : str
            Analysis name used in the auto title
        method_label : str
            Algorithm/method label used in the auto title
        title_fontsize : int or None
            Font size for the title

        Returns
        -------
        None
            Modifies the figure in place
        """
        title_text = title or (
            f"Feature Importance: {feature_importance_name} [{method_label}]"
        )
        fig.suptitle(title_text, fontsize=title_fontsize or 14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    def _save_figure(
        self,
        fig: Figure,
        filename: Optional[str],
        feature_importance_name: str,
        n_top: int,
        file_format: str,
        dpi: int,
    ) -> None:
        """
        Save the figure to a file in the cache directory.

        Parameters
        ----------
        fig : Figure
            Figure to save
        filename : str or None
            Custom filename, or None to auto-generate
        feature_importance_name : str
            Analysis name used in the auto filename
        n_top : int
            Number of top features used in the auto filename
        file_format : str
            File format extension (png, pdf, svg, etc.)
        dpi : int
            Resolution in dots per inch

        Returns
        -------
        None
            Saves the figure and prints the file path
        """
        if filename is None:
            filename = (
                f"importance_bars_{feature_importance_name}"
                f"_top{n_top}.{file_format}"
            )
        if not filename.endswith(f".{file_format}"):
            filename = f"{filename}.{file_format}"
        filepath = PathUtils.get_cache_file_path(filename, self.cache_dir)
        SvgExportHelper.save_figure_with_export_optimizations(
            fig=fig,
            filepath=filepath,
            file_format=file_format,
            dpi=dpi,
            bbox_inches="tight",
        )
        print(f"Figure saved: {filepath}")

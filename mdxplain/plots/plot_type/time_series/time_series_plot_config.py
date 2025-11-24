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

"""Central configuration dataclass for time series plotting."""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ....pipeline.entities.pipeline_data import PipelineData
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from matplotlib.axes import Axes


@dataclass
class TimeSeriesPlotConfig:
    """
    Central configuration for all time series plotting operations.

    Groups all parameters for time series plotting to avoid parameter explosion
    across multiple methods.

    Attributes
    ----------
    pipeline_data : PipelineData
        Pipeline data container
    feature_importance_name : str
        Feature importance analysis name
    n_top : int
        Top N features to plot
    traj_selection : Union[int, str, List, "all"]
        Trajectory selection for lines
    use_time : bool
        Use time axis (True) or frames (False)
    tags_for_coloring : Optional[List[str]]
        Tags for coloring trajectories
    allow_multi_tag_plotting : bool
        Plot multi-tag trajectories multiple times
    clustering_name : Optional[str]
        Clustering name for membership bars
    membership_per_feature : bool
        Show membership bar per feature (True) or once at bottom (False)
    membership_traj_selection : Union[str, int, List]
        Trajectory selection for membership bars
    contact_transformation : bool
        Convert contacts to distances
    max_cols : int
        Maximum columns in grid layout
    long_labels : bool
        Use long labels for discrete features
    subplot_height : float
        Height per feature subplot
    membership_bar_height : float
        Height per trajectory in membership bar
    show_legend : bool
        Show legend
    contact_threshold : Optional[float]
        Contact threshold line
    title : Optional[str]
        Custom title
    save_fig : bool
        Save figure
    filename : Optional[str]
        Custom filename
    file_format : str
        File format (png, pdf, svg, etc.)
    dpi : int
        Resolution
    smoothing : bool
        Enable or disable data smoothing
    smoothing_method : str
        Smoothing method ("moving_average" or "savitzky")
    smoothing_window : int
        Window size for smoothing (frames)
    smoothing_polyorder : int
        Polynomial order for Savitzky-Golay filter
    show_unsmoothed_background : bool
        Show unsmoothed data as transparent background when smoothing is enabled
    feature_data : Dict
        Prepared feature data
    feature_indices : Dict[int, str]
        Feature index to name mapping
    metadata_map : Dict
        Feature metadata mapping
    contact_cutoff : Optional[float]
        Contact cutoff value
    feature_selector_name : str
        Feature selector name
    is_temporary : bool
        Whether selector is temporary
    tag_map : Dict[int, List[str]]
        Trajectory to tags mapping
    all_features : List[Tuple[str, str]]
        List of (feature_type, feature_name) tuples
    layout : List
        Grid layout information
    n_rows : int
        Number of grid rows
    n_cols : int
        Number of grid columns
    n_frames : int
        Number of frames in trajectories
    n_membership_rows : int
        Number of membership rows
    membership_row_height_inches : float
        Membership row height in inches
    use_tag_coloring : bool
        Use tag-based coloring (True) or trajectory-based (False)
    tag_colors : Dict[str, str]
        Tag to color mapping
    traj_colors : Dict[str, str]
        Trajectory name to color mapping
    fig : Optional[Figure]
        Matplotlib figure (set after creation)
    gs : Optional[GridSpec]
        GridSpec layout (set after creation)
    wrapped_title : Optional[str]
        Wrapped title text (set after creation)
    top : Optional[float]
        Top position for title (set after creation)
    rightmost_ax_first_row : Optional[Axes]
        Rightmost axes in first row (set after creation)

    Examples
    --------
    >>> config = TimeSeriesPlotConfig(
    ...     pipeline_data=data,
    ...     feature_importance_name="analysis",
    ...     n_top=5,
    ...     traj_selection="all",
    ...     use_time=True,
    ...     tags_for_coloring=None,
    ...     allow_multi_tag_plotting=False,
    ...     clustering_name=None,
    ...     membership_per_feature=False,
    ...     membership_traj_selection="all",
    ...     contact_transformation=True,
    ...     max_cols=2,
    ...     long_labels=False,
    ...     subplot_height=2.5,
    ...     membership_bar_height=0.5,
    ...     show_legend=True,
    ...     contact_threshold=4.5,
    ...     title=None,
    ...     save_fig=False,
    ...     filename=None,
    ...     file_format="png",
    ...     dpi=300
    ... )
    """

    # User input parameters
    pipeline_data: PipelineData
    mode_type: str
    mode_name: str
    feature_importance_name: Optional[str]
    n_top: int
    feature_selector: Optional[str]
    traj_selection: Union[int, str, List, "all"]
    use_time: bool
    tags_for_coloring: Optional[List[str]]
    allow_multi_tag_plotting: bool
    clustering_name: Optional[str]
    membership_per_feature: bool
    membership_traj_selection: Union[str, int, List]
    contact_transformation: bool
    max_cols: int
    long_labels: bool
    subplot_height: float
    membership_bar_height: float
    show_legend: bool
    contact_threshold: Optional[float]
    title: Optional[str]
    save_fig: bool
    filename: Optional[str]
    file_format: str
    dpi: int
    smoothing: bool = True
    smoothing_method: str = "savitzky"
    smoothing_window: int = 51
    smoothing_polyorder: int = 3
    show_unsmoothed_background: bool = True

    # Font size parameters
    title_fontsize: int = None
    subplot_title_fontsize: int = None
    xlabel_fontsize: int = None
    ylabel_fontsize: int = None
    tick_fontsize: int = None
    legend_fontsize: int = None
    legend_title_fontsize: int = None

    # Prepared data (set during plotting)
    feature_data: Dict = field(default_factory=dict)
    feature_indices: Dict[int, str] = field(default_factory=dict)
    metadata_map: Dict = field(default_factory=dict)
    contact_cutoff: Optional[float] = None
    feature_selector_name: str = ""
    is_temporary: bool = False
    tag_map: Dict[int, List[str]] = field(default_factory=dict)
    all_features: List[Tuple[str, str]] = field(default_factory=list)

    # Layout data (set during plotting)
    layout: List = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    n_frames: int = 0
    n_membership_rows: int = 0
    membership_row_height_inches: float = 0.0

    # Color data (set during plotting)
    use_tag_coloring: bool = False
    tag_colors: Dict[str, str] = field(default_factory=dict)
    traj_colors: Dict[str, str] = field(default_factory=dict)

    # Figure data (set during plotting)
    fig: Optional[Figure] = None
    gs: Optional[GridSpec] = None
    wrapped_title: Optional[str] = None
    top: Optional[float] = None
    rightmost_ax_first_row: Optional[Axes] = None

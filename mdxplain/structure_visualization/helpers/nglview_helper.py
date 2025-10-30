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
NGLView widget helper for Jupyter notebook visualization.

This module provides helper functions to create interactive 3D molecular
visualizations using nglview in Jupyter notebooks with beta-factor coloring
and feature highlighting.
"""

import nglview as nv
import ipywidgets as widgets
from typing import Dict, List, Any, Tuple
from datetime import datetime

from .color_conversion_helper import ColorConversionHelper
from .feature_overlap_helper import FeatureOverlapHelper


class NGLViewHelper:
    """
    Helper for creating NGLView widgets with focus-based visualization.

    Provides static methods to create interactive 3D molecular structure
    viewers with beta-factor gradient coloring, feature highlights,
    structure selection dropdowns, and 2x4 checkbox grid for independent
    control of own vs other structures and features.

    Examples
    --------
    >>> pdb_info = {
    ...     "cluster_0": {"path": "/path/to/c0.pdb", "color": "#bf4242"},
    ...     "cluster_1": {"path": "/path/to/c1.pdb", "color": "#4242bf"}
    ... }
    >>> features = [{"feature_name": "ALA_5_CA-GLU_10_CA", ...}]
    >>> colors = {"ALA_5_CA-GLU_10_CA": "#ff0000"}
    >>> ui, view = NGLViewHelper.create_widget(pdb_info, features, colors)
    """

    @staticmethod
    def create_widget(
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        feature_own_color: bool = True,
        show_feature_legend: bool = True,
        show_structure_legend: bool = True,
        viz_name: str = "structure"
    ) -> Tuple:
        """
        Create NGLView widget with focus-based structure visualization.

        Creates interactive 3D viewer with beta-factor gradient coloring
        and feature highlights. Supports both global features (averaged across
        all clusters) and local features (cluster-specific). Includes dropdown
        for structure selection and 3x4 checkbox grid for independent control.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure information with 'path' and 'color' keys per structure
        top_features_global : List[Dict[str, Any]]
            Global features to highlight (averaged across all clusters)
        feature_colors_global : Dict[str, str]
            Global feature name to HEX color mapping
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure {struct_name: features}
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors {struct_name: {feat_name: color}}
        feature_own_color : bool, default=True
            If True, features use their own color from feature_colors.
            If False, features use the color of their structure.
        show_feature_legend : bool, default=True
            If True, display feature color legend below controls
        show_structure_legend : bool, default=True
            If True, display structure color legend below controls
        viz_name : str, default="structure"
            Visualization name for filename generation in save buttons

        Returns
        -------
        Tuple[widgets.VBox, nv.NGLWidget]
            UI container (dropdown + checkbox grid + legend) and NGLWidget

        Examples
        --------
        >>> pdb_info = {"cluster_0": {"path": "c0.pdb", "color": "#bf4242"}}
        >>> global_feats = [{"feature_name": "ALA_5_CA-GLU_10_CA", ...}]
        >>> global_colors = {"ALA_5_CA-GLU_10_CA": "#ff0000"}
        >>> local_feats = {"cluster_0": [{"feature_name": "GLY_3_phi", ...}]}
        >>> local_colors = {"cluster_0": {"GLY_3_phi": "#00ff00"}}
        >>> ui, view = NGLViewHelper.create_widget(
        ...     pdb_info, global_feats, global_colors,
        ...     local_feats, local_colors
        ... )
        >>> from IPython.display import display
        >>> display(ui, view)

        Notes
        -----
        - Beta-factors: 0.0 (base color) → 1.0 (white)
        - PDBs must be pre-aligned for multi-structure visualization
        - Dropdown selects focus structure
        - 3x4 grid controls: own/other x struct/global/local
        - Global features: shown for all structures
        - Local features: only for own structure
        - Each checkbox controls visibility and transparency independently
        """
        # Setup view and load structures
        view, component_map, overlaps_global, overlaps_local = NGLViewHelper._setup_view_components(
            pdb_info, top_features_global, feature_colors_global,
            top_features_local, feature_colors_local, feature_own_color
        )

        # Create control widgets (dropdown + checkbox grid)
        dropdown, checkbox_grid = NGLViewHelper._create_control_widgets(
            component_map, pdb_info, top_features_global, feature_colors_global,
            top_features_local, feature_colors_local,
            overlaps_global, overlaps_local, feature_own_color
        )

        # Setup legend widgets
        legend_widgets = NGLViewHelper._setup_legend_widgets(
            pdb_info, top_features_global, feature_colors_global,
            top_features_local, feature_colors_local,
            overlaps_global, overlaps_local,
            feature_own_color, show_structure_legend, show_feature_legend,
            list(pdb_info.keys())[0] if pdb_info else None  # Initial selection
        )

        # Register callbacks (after legend creation so it's in scope)
        NGLViewHelper._register_callbacks(
            dropdown, checkbox_grid, component_map, pdb_info,
            top_features_global, feature_colors_global, overlaps_global,
            top_features_local, feature_colors_local, overlaps_local,
            feature_own_color, legend_widgets
        )

        # Create download button
        download_button, download_output = NGLViewHelper._create_download_button(
            view, viz_name, dropdown
        )

        # Assemble final UI
        ui = NGLViewHelper._assemble_ui(
            dropdown, checkbox_grid, legend_widgets,
            download_button, download_output
        )

        return ui, view

    @staticmethod
    def _load_structure_component(
        view: nv.NGLWidget,
        pdb_path: str,
        base_color: str,
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: List[Dict[str, Any]],
        feature_colors_local: Dict[str, str],
        overlaps_local: Dict[int, List[int]],
        feature_own_color: bool
    ):
        """
        Load single structure component with representations.

        Parameters
        ----------
        view : nv.NGLWidget
            NGLView widget
        pdb_path : str
            Path to PDB file
        base_color : str
            Base color for beta-factor gradient (HEX) and features
        top_features_global : List[Dict[str, Any]]
            Global features to highlight
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Pre-computed overlapping residues for global features
        top_features_local : List[Dict[str, Any]]
            Local features for this structure
        feature_colors_local : Dict[str, str]
            Local feature color mapping
        overlaps_local : Dict[int, List[int]]
            Pre-computed overlapping residues for local features
        feature_own_color : bool
            If True, use feature colors; if False, use base_color

        Returns
        -------
        nv.ComponentViewer
            Component handle with representations added
        """
        component = view.add_component(pdb_path, default=False)
        component.clear_representations()

        # Cartoon with beta-factor coloring
        component.add_representation(
            "cartoon",
            selection="not hydrogen",
            colorScheme="bfactor",
            colorScale=[base_color, "white"],
            colorDomain=[0.0, 1.0],
            opacity=1.0
        )

        # Add global feature highlights
        for feature in top_features_global:
            NGLViewHelper._add_feature_highlight(
                component, feature, feature_colors_global, overlaps_global,
                1.0, base_color, feature_own_color
            )
        NGLViewHelper._add_overlap_highlights(
            component, overlaps_global, top_features_global, feature_colors_global,
            1.0, base_color, feature_own_color
        )

        # Add local feature highlights
        for feature in top_features_local:
            NGLViewHelper._add_feature_highlight(
                component, feature, feature_colors_local, overlaps_local,
                1.0, base_color, feature_own_color
            )
        NGLViewHelper._add_overlap_highlights(
            component, overlaps_local, top_features_local, feature_colors_local,
            1.0, base_color, feature_own_color
        )

        return component

    @staticmethod
    def _add_feature_highlight(
        component,
        feature: Dict[str, Any],
        feature_colors: Dict[str, str],
        all_overlaps: Dict[int, List[int]],
        opacity: float,
        struct_color: str,
        feature_own_color: bool
    ) -> None:
        """
        Add licorice highlight for non-overlapping feature residues.

        Only highlights residues that don't appear in multiple features.
        Overlapping residues are handled separately with mixed colors.

        Parameters
        ----------
        component : nv.ComponentViewer
            Component to add highlight to
        feature : Dict[str, Any]
            Feature dictionary with residue_seqids from metadata
        feature_colors : Dict[str, str]
            Feature color mapping
        all_overlaps : Dict[int, List[int]]
            Mapping of overlapping residues (to exclude)
        opacity : float
            Opacity for licorice representation
        struct_color : str
            Structure color (used if feature_own_color=False)
        feature_own_color : bool
            If True, use feature colors; if False, use struct_color

        Returns
        -------
        None
            Adds representation to component
        """
        non_overlap_residues = NGLViewHelper._filter_non_overlaps(
            feature, all_overlaps
        )
        if not non_overlap_residues:
            return

        color_hex = NGLViewHelper._get_feature_color(
            feature, feature_colors, struct_color, feature_own_color
        )
        res_selection = ' or '.join(map(str, non_overlap_residues))

        component.add_representation(
            "licorice",
            selection=res_selection,
            color=color_hex,
            opacity=opacity
        )

    @staticmethod
    def _filter_non_overlaps(
        feature: Dict[str, Any],
        all_overlaps: Dict[int, List[int]]
    ) -> List[int]:
        """
        Filter out overlapping residues from feature.

        Parameters
        ----------
        feature : Dict[str, Any]
            Feature dictionary with residue_seqids
        all_overlaps : Dict[int, List[int]]
            Mapping of overlapping residues

        Returns
        -------
        List[int]
            Non-overlapping residue IDs
        """
        residues = feature.get("residue_seqids", [])
        return [s for s in residues if s not in all_overlaps]

    @staticmethod
    def _get_feature_color(
        feature: Dict[str, Any],
        feature_colors: Dict[str, str],
        struct_color: str,
        feature_own_color: bool
    ) -> str:
        """
        Get color for feature based on settings.

        Parameters
        ----------
        feature : Dict[str, Any]
            Feature dictionary
        feature_colors : Dict[str, str]
            Feature color mapping
        struct_color : str
            Structure color
        feature_own_color : bool
            If True, use feature color; if False, use struct_color

        Returns
        -------
        str
            HEX color for feature
        """
        if feature_own_color:
            feature_name = feature.get('feature_name', '')
            return feature_colors.get(feature_name, '#808080')
        return struct_color

    @staticmethod
    def _create_focus_checkboxes() -> widgets.GridBox:
        """
        Create 3x4 checkbox grid for focus visualization control.

        Grid layout:
        Row 1: [Own Struct] [Own Struct Trans] [Own Global Feat] [Own Global Trans]
        Row 2: [Other Struct] [Other Struct Trans] [Other Global Feat] [Other Global Trans]
        Row 3: [Own Local Feat] [Own Local Trans] [Other Local Feat] [Other Local Trans]

        Returns
        -------
        widgets.GridBox
            3x4 checkbox grid
        """
        checkboxes = [
            # Row 1: Own Structure + Global Features
            widgets.Checkbox(value=True, description="Own Structure"),
            widgets.Checkbox(value=False, description="Own Struct Transparent"),
            widgets.Checkbox(value=True, description="Own Global Features"),
            widgets.Checkbox(value=False, description="Own Global Trans"),
            # Row 2: Other Structures + Global Features
            widgets.Checkbox(value=False, description="Other Structures"),
            widgets.Checkbox(value=True, description="Other Struct Transparent"),
            widgets.Checkbox(value=False, description="Other Global Features"),
            widgets.Checkbox(value=True, description="Other Global Trans"),
            # Row 3: Local Features (Own + Other)
            widgets.Checkbox(value=True, description="Own Local Features"),
            widgets.Checkbox(value=False, description="Own Local Trans"),
            widgets.Checkbox(value=False, description="Other Local Features"),
            widgets.Checkbox(value=True, description="Other Local Trans"),
        ]

        grid = widgets.GridBox(
            checkboxes,
            layout=widgets.Layout(
                width='100%',
                grid_template_columns='repeat(4, 25%)',
                grid_template_rows='auto auto auto',
                grid_gap='5px 5px'
            )
        )

        return grid

    @staticmethod
    def _create_dropdown(
        component_map: Dict,
        checkbox_grid: widgets.GridBox,
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_global: Dict[int, List[int]],
        overlaps_local: Dict,
        feature_own_color: bool
    ) -> widgets.Dropdown:
        """
        Create dropdown for structure selection.

        Parameters
        ----------
        component_map : Dict
            Structure name to component mapping
        checkbox_grid : widgets.GridBox
            Checkbox grid container
        pdb_info : Dict[str, Dict[str, str]]
            Structure information with paths and colors
        top_features_global : List[Dict[str, Any]]
            Global features to highlight
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_global : Dict[int, List[int]]
            Pre-computed overlapping residues for global features
        overlaps_local : Dict
            Pre-computed overlapping residues for local features per structure
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors

        Returns
        -------
        widgets.Dropdown
            Dropdown widget with change handler attached
        """
        structure_names = list(component_map.keys())

        dropdown = widgets.Dropdown(
            options=structure_names,
            value=structure_names[0] if structure_names else None,
            description='Structure:',
            style={'description_width': 'initial'}
        )

        # Note: Callbacks registered later in create_widget() after legend creation
        return dropdown

    @staticmethod
    def _register_callbacks(
        dropdown: widgets.Dropdown,
        checkbox_grid: widgets.GridBox,
        component_map: Dict,
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        feature_own_color: bool,
        legend_widgets: List[widgets.HTML]
    ) -> None:
        """
        Register dropdown and checkbox callbacks with legend update support.

        Must be called after legend_widgets is created to avoid scope issues.

        Parameters
        ----------
        dropdown : widgets.Dropdown
            Dropdown widget
        checkbox_grid : widgets.GridBox
            Checkbox grid widget
        component_map : Dict
            Structure name to component mapping
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features_global : List[Dict[str, Any]]
            Global features
        feature_colors_global : Dict[str, str]
            Global feature colors
        overlaps_global : Dict[int, List[int]]
            Global overlaps
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        feature_own_color : bool
            Feature color mode
        legend_widgets : List[widgets.HTML]
            Legend widgets to update

        Returns
        -------
        None
            Registers callbacks on dropdown and checkboxes
        """
        def on_structure_change(change):
            # Update legend for new selection (if feature legend exists)
            if legend_widgets:
                # Feature legend is last widget
                feature_legend_idx = -1
                new_html = NGLViewHelper._create_feature_legend_widget(
                    top_features_global, feature_colors_global, overlaps_global,
                    top_features_local, feature_colors_local, overlaps_local,
                    feature_own_color, change['new']
                )
                legend_widgets[feature_legend_idx].value = new_html.value

            NGLViewHelper._handle_dropdown_change(
                change, component_map, checkbox_grid, pdb_info,
                top_features_global, feature_colors_global, overlaps_global,
                top_features_local, feature_colors_local, overlaps_local,
                feature_own_color
            )

        dropdown.observe(on_structure_change, names='value')

        # Attach change handlers to checkboxes
        checkboxes = checkbox_grid.children
        for checkbox in checkboxes:
            checkbox.observe(
                lambda change, cbs=checkboxes: NGLViewHelper._handle_checkbox_change(
                    change, dropdown, component_map, cbs, pdb_info,
                    top_features_global, feature_colors_global, overlaps_global,
                    top_features_local, feature_colors_local, overlaps_local,
                    feature_own_color
                ),
                names='value'
            )

    @staticmethod
    def _handle_dropdown_change(
        change: Dict,
        component_map: Dict,
        checkbox_grid: widgets.GridBox,
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        feature_own_color: bool
    ) -> None:
        """
        Handle dropdown selection change (focus structure).

        Parameters
        ----------
        change : Dict
            Change event dictionary
        component_map : Dict
            Structure name to component mapping
        checkbox_grid : widgets.GridBox
            Checkbox grid container
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Pre-computed global overlapping residues
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors

        Returns
        -------
        None
            Updates all components based on checkboxes
        """
        NGLViewHelper._update_view_from_checkboxes(
            change['owner'], component_map, checkbox_grid.children,
            pdb_info, top_features_global, feature_colors_global, overlaps_global,
            top_features_local, feature_colors_local, overlaps_local,
            feature_own_color
        )

    @staticmethod
    def _handle_checkbox_change(
        _change: Dict,
        dropdown: widgets.Dropdown,
        component_map: Dict,
        checkboxes: Tuple,
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        feature_own_color: bool
    ) -> None:
        """
        Handle checkbox state change.

        Parameters
        ----------
        _change : Dict
            Change event dictionary (unused, required by ipywidgets)
        dropdown : widgets.Dropdown
            Dropdown widget for structure selection
        component_map : Dict
            Structure name to component mapping
        checkboxes : Tuple
            12 checkboxes from 3x4 grid
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Pre-computed global overlapping residues
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors

        Returns
        -------
        None
            Updates all components based on checkboxes
        """
        NGLViewHelper._update_view_from_checkboxes(
            dropdown, component_map, checkboxes,
            pdb_info, top_features_global, feature_colors_global, overlaps_global,
            top_features_local, feature_colors_local, overlaps_local,
            feature_own_color
        )

    @staticmethod
    def _update_view_from_checkboxes(
        dropdown: widgets.Dropdown,
        component_map: Dict,
        checkboxes: Tuple,
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        feature_own_color: bool
    ) -> None:
        """
        Update all components based on checkbox states.

        Parameters
        ----------
        dropdown : widgets.Dropdown
            Dropdown widget
        component_map : Dict
            Structure name to component mapping
        checkboxes : Tuple
            12 checkboxes from 3x4 grid
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Pre-computed global overlapping residues
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors

        Returns
        -------
        None
            Updates all components
        """
        selected_struct = dropdown.value
        opacities = NGLViewHelper._extract_opacity_settings(checkboxes)

        for struct_name, component in component_map.items():
            NGLViewHelper._update_single_component(
                component, struct_name, selected_struct, pdb_info,
                top_features_global, feature_colors_global, overlaps_global,
                top_features_local, feature_colors_local, overlaps_local,
                feature_own_color, opacities
            )

    @staticmethod
    def _calc_opacity(enabled: bool, transparent: bool) -> float:
        """
        Calculate opacity from enabled and transparent checkboxes.

        Parameters
        ----------
        enabled : bool
            Enabled checkbox value
        transparent : bool
            Transparent checkbox value

        Returns
        -------
        float
            Opacity value (0.0, 0.2, or 1.0)
        """
        if not enabled:
            return 0.0
        return 0.2 if transparent else 1.0

    @staticmethod
    def _extract_opacity_settings(checkboxes: Tuple) -> Dict[str, float]:
        """
        Extract opacity settings from checkbox grid.

        Parameters
        ----------
        checkboxes : Tuple
            12 checkboxes from 3x4 grid

        Returns
        -------
        Dict[str, float]
            Dictionary with opacity values for own/other structures and features
        """
        return {
            'own_struct': NGLViewHelper._calc_opacity(
                checkboxes[0].value, checkboxes[1].value
            ),
            'own_global_feat': NGLViewHelper._calc_opacity(
                checkboxes[2].value, checkboxes[3].value
            ),
            'other_struct': NGLViewHelper._calc_opacity(
                checkboxes[4].value, checkboxes[5].value
            ),
            'other_global_feat': NGLViewHelper._calc_opacity(
                checkboxes[6].value, checkboxes[7].value
            ),
            'own_local_feat': NGLViewHelper._calc_opacity(
                checkboxes[8].value, checkboxes[9].value
            ),
            'other_local_feat': NGLViewHelper._calc_opacity(
                checkboxes[10].value, checkboxes[11].value
            )
        }

    @staticmethod
    def _update_single_component(
        component,
        struct_name: str,
        selected_struct: str,
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        feature_own_color: bool,
        opacities: Dict[str, float]
    ) -> None:
        """
        Update single component based on opacity settings.

        Parameters
        ----------
        component : nv.Component
            NGLView component to update
        struct_name : str
            Name of this structure
        selected_struct : str
            Currently selected structure
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features_global : List[Dict[str, Any]]
            Global features
        feature_colors_global : Dict[str, str]
            Global feature colors
        overlaps_global : Dict[int, List[int]]
            Global overlaps
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        feature_own_color : bool
            Color mode for features
        opacities : Dict[str, float]
            Opacity settings from checkboxes

        Returns
        -------
        None
            Updates component visibility and representations
        """
        is_own = (struct_name == selected_struct)
        struct_opacity = opacities['own_struct' if is_own else 'other_struct']

        if struct_opacity == 0.0:
            component.hide()
            return

        global_feat_opacity = opacities['own_global_feat' if is_own else 'other_global_feat']
        local_feat_opacity = opacities['own_local_feat' if is_own else 'other_local_feat']

        NGLViewHelper._recreate_structure_with_focus(
            component, pdb_info[struct_name]['color'],
            top_features_global, feature_colors_global, overlaps_global,
            top_features_local, feature_colors_local, overlaps_local,
            struct_name, struct_opacity, global_feat_opacity,
            local_feat_opacity, feature_own_color
        )
        component.show()

    @staticmethod
    def _recreate_structure_with_focus(
        component,
        base_color: str,
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        struct_name: str,
        struct_opacity: float,
        global_feat_opacity: float,
        local_feat_opacity: float,
        feature_own_color: bool
    ) -> None:
        """
        Recreate structure with separate opacities for structure and features.

        Parameters
        ----------
        component : nv.Component
            NGLView component
        base_color : str
            Base color for beta-factor gradient and features
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Pre-computed global overlapping residues
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        struct_name : str
            Structure name (for local feature lookup)
        struct_opacity : float
            Opacity for structure (cartoon)
        global_feat_opacity : float
            Opacity for global features (licorice)
        local_feat_opacity : float
            Opacity for local features (licorice)
        feature_own_color : bool
            If True, use feature colors; if False, use base_color

        Returns
        -------
        None
            Recreates component representations
        """
        component.clear_representations()

        NGLViewHelper._add_cartoon_to_component(
            component, base_color, struct_opacity
        )
        NGLViewHelper._add_global_features_to_component(
            component, top_features_global, feature_colors_global,
            overlaps_global, global_feat_opacity, base_color, feature_own_color
        )
        NGLViewHelper._add_local_features_to_component(
            component, struct_name, top_features_local, feature_colors_local,
            overlaps_local, local_feat_opacity, base_color, feature_own_color
        )

    @staticmethod
    def _add_cartoon_to_component(
        component,
        base_color: str,
        struct_opacity: float
    ) -> None:
        """
        Add cartoon representation to component.

        Parameters
        ----------
        component : nv.Component
            NGLView component
        base_color : str
            Base color for beta-factor gradient
        struct_opacity : float
            Opacity for cartoon

        Returns
        -------
        None
            Adds cartoon representation to component
        """
        component.add_representation(
            "cartoon",
            selection="not hydrogen",
            colorScheme="bfactor",
            colorScale=[base_color, "white"],
            colorDomain=[0.0, 1.0],
            opacity=struct_opacity
        )

    @staticmethod
    def _add_global_features_to_component(
        component,
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        global_feat_opacity: float,
        base_color: str,
        feature_own_color: bool
    ) -> None:
        """
        Add global feature representations to component.

        Parameters
        ----------
        component : nv.Component
            NGLView component
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Global overlapping residues
        global_feat_opacity : float
            Opacity for global features
        base_color : str
            Base color for features
        feature_own_color : bool
            If True, use feature colors; if False, use base_color

        Returns
        -------
        None
            Adds global feature representations to component
        """
        if global_feat_opacity <= 0.0:
            return

        for feature in top_features_global:
            NGLViewHelper._add_feature_highlight(
                component, feature, feature_colors_global, overlaps_global,
                global_feat_opacity, base_color, feature_own_color
            )
        NGLViewHelper._add_overlap_highlights(
            component, overlaps_global, top_features_global,
            feature_colors_global, global_feat_opacity, base_color,
            feature_own_color
        )

    @staticmethod
    def _add_local_features_to_component(
        component,
        struct_name: str,
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        local_feat_opacity: float,
        base_color: str,
        feature_own_color: bool
    ) -> None:
        """
        Add local feature representations to component.

        Parameters
        ----------
        component : nv.Component
            NGLView component
        struct_name : str
            Structure name for local feature lookup
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        local_feat_opacity : float
            Opacity for local features
        base_color : str
            Base color for features
        feature_own_color : bool
            If True, use feature colors; if False, use base_color

        Returns
        -------
        None
            Adds local feature representations to component
        """
        if local_feat_opacity <= 0.0 or struct_name not in top_features_local:
            return

        local_features = top_features_local[struct_name]
        local_colors = feature_colors_local[struct_name]
        local_overlaps = overlaps_local[struct_name]

        for feature in local_features:
            NGLViewHelper._add_feature_highlight(
                component, feature, local_colors, local_overlaps,
                local_feat_opacity, base_color, feature_own_color
            )
        NGLViewHelper._add_overlap_highlights(
            component, local_overlaps, local_features, local_colors,
            local_feat_opacity, base_color, feature_own_color
        )

    @staticmethod
    def _set_initial_visibility(component_map: Dict) -> None:
        """
        Set initial component visibility (first visible only).

        Parameters
        ----------
        component_map : Dict
            Structure name to component mapping

        Returns
        -------
        None
            Updates component visibility
        """
        if not component_map:
            return

        for i, component in enumerate(component_map.values()):
            if i == 0:
                component.show()
            else:
                component.hide()

    @staticmethod
    def _add_overlap_highlights(
        component,
        overlaps: Dict[int, List[int]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        opacity: float,
        struct_color: str,
        feature_own_color: bool
    ) -> None:
        """
        Add licorice representations for overlapping residues with mixed colors.

        Parameters
        ----------
        component : nv.ComponentViewer
            Component to add highlights to
        overlaps : Dict[int, List[int]]
            Mapping from seqid to list of feature indices
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries
        feature_colors : Dict[str, str]
            Feature name to color mapping
        opacity : float
            Opacity value for representations
        struct_color : str
            Structure color (used if feature_own_color=False)
        feature_own_color : bool
            If True, use mixed feature colors; if False, use struct_color

        Returns
        -------
        None
            Adds representations to component
        """
        for seqid, feature_indices in overlaps.items():
            if feature_own_color:
                # Collect colors from all features containing this residue
                colors = [
                    feature_colors[top_features[i]["feature_name"]]
                    for i in feature_indices
                ]
                mixed_color = ColorConversionHelper.mix_hex_colors(colors)
            else:
                mixed_color = struct_color

            # Add representation with mixed color
            component.add_representation(
                "licorice",
                selection=str(seqid),
                color=mixed_color,
                opacity=opacity
            )

    @staticmethod
    def _create_legend_widget(
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
        feature_own_color: bool
    ) -> widgets.HTML:
        """
        Create legend widget showing feature colors, types, and overlaps.

        Creates HTML widget displaying a legend that maps licorice
        representation colors to their corresponding feature names with
        feature types. Also shows overlapping residues with mixed colors.
        If feature_own_color is False, no legend is shown since features
        use structure colors.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries with feature names and types
        feature_colors : Dict[str, str]
            Mapping from feature name to HEX color
        overlaps : Dict[int, List[int]]
            Mapping from residue seqid to list of feature indices
        feature_own_color : bool
            If False, no legend is shown

        Returns
        -------
        widgets.HTML
            HTML widget with styled legend or empty widget

        Examples
        --------
        >>> features = [{"feature_name": "ALA_5_CA-GLU_10_CA",
        ...              "feature_type": "distances"}]
        >>> colors = {"ALA_5_CA-GLU_10_CA": "#ff0000"}
        >>> overlaps = {33: [0, 1]}
        >>> legend = NGLViewHelper._create_legend_widget(
        ...     features, colors, overlaps, True
        ... )
        """
        # No legend needed if features use structure colors
        if not feature_own_color:
            return widgets.HTML(value='')

        html_content = ['<div style="padding:10px; border:1px solid #ddd; '
                       'background:#f9f9f9; margin-top:10px; border-radius:4px;">']
        html_content.append('<b style="font-size:14px;">Feature Legend:</b>')
        html_content.append('<div style="margin-top:8px;">')

        NGLViewHelper._add_feature_legend_items(
            html_content, top_features, feature_colors
        )

        if overlaps:
            NGLViewHelper._add_overlap_legend_items(
                html_content, overlaps, top_features, feature_colors
            )

        html_content.append('</div>')

        return widgets.HTML(value=''.join(html_content))

    @staticmethod
    def _create_structure_legend_widget(
        pdb_info: Dict[str, Dict[str, str]]
    ) -> widgets.HTML:
        """
        Create legend widget showing structure colors.

        Creates HTML widget displaying a legend that maps structure
        names to their assigned colors.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with 'color' key per structure

        Returns
        -------
        widgets.HTML
            HTML widget with styled structure legend

        Examples
        --------
        >>> pdb_info = {
        ...     "cluster_0": {"path": "c0.pdb", "color": "#bf4040"},
        ...     "cluster_1": {"path": "c1.pdb", "color": "#4040bf"}
        ... }
        >>> legend = NGLViewHelper._create_structure_legend_widget(pdb_info)
        """
        html_content = ['<div style="padding:10px; border:1px solid #ddd; '
                       'background:#f9f9f9; margin-top:10px; border-radius:4px;">']
        html_content.append('<b style="font-size:14px;">Structure Colors:</b>')
        html_content.append('<div style="margin-top:8px;">')

        for struct_name, info in pdb_info.items():
            color = info['color']
            html_content.append(
                f'<div style="margin:4px 0; display:flex; align-items:center;">'
                f'<span style="display:inline-block; width:20px; height:20px; '
                f'background:{color}; margin-right:8px; border:1px solid #333; '
                f'border-radius:2px;"></span>'
                f'<span style="font-size:12px; font-family:monospace;">'
                f'{struct_name}</span>'
                f'</div>'
            )

        html_content.append('</div>')
        html_content.append('</div>')

        return widgets.HTML(value=''.join(html_content))

    @staticmethod
    def _create_feature_legend_widget(
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]],
        feature_own_color: bool,
        selected_struct: str = None
    ) -> widgets.HTML:
        """
        Create feature legend widget with global and selected local features.

        Creates HTML widget displaying global features and local features
        only for the currently selected structure. Dynamic legend that
        updates with dropdown selection.

        Parameters
        ----------
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Global overlapping residues
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure
        feature_own_color : bool
            If False, no legend is shown
        selected_struct : str, optional
            Currently selected structure (shows only its local features)

        Returns
        -------
        widgets.HTML
            HTML widget with styled legend or empty widget

        Examples
        --------
        >>> legend = NGLViewHelper._create_feature_legend_widget(
        ...     global_features, global_colors, global_overlaps,
        ...     local_features, local_colors, local_overlaps, True, "cluster_0"
        ... )
        """
        if not feature_own_color:
            return widgets.HTML(value='')

        html_content = NGLViewHelper._init_legend_html()

        NGLViewHelper._add_global_features_section(
            html_content, top_features_global, feature_colors_global,
            overlaps_global
        )
        NGLViewHelper._add_local_features_section(
            html_content, selected_struct, top_features_local,
            feature_colors_local, overlaps_local
        )

        html_content.append('</div>')
        return widgets.HTML(value=''.join(html_content))

    @staticmethod
    def _init_legend_html() -> List[str]:
        """
        Initialize HTML content list for feature legend.

        Returns
        -------
        List[str]
            HTML content with opening div and header
        """
        html_content = [
            '<div style="padding:10px; border:1px solid #ddd; '
            'background:#f9f9f9; margin-top:10px; border-radius:4px;">'
        ]
        html_content.append('<b style="font-size:14px;">Feature Legend:</b>')
        return html_content

    @staticmethod
    def _add_global_features_section(
        html_content: List[str],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        overlaps_global: Dict[int, List[int]]
    ) -> None:
        """
        Add global features section to legend HTML.

        Parameters
        ----------
        html_content : List[str]
            HTML content list to append to
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature color mapping
        overlaps_global : Dict[int, List[int]]
            Global overlapping residues

        Returns
        -------
        None
            Modifies html_content in-place
        """
        if not top_features_global:
            return

        html_content.append('<div style="margin-top:8px;">')
        html_content.append('<b style="font-size:12px; color:#666;">Global Features:</b>')
        html_content.append('<div style="margin-left:10px;">')
        NGLViewHelper._add_feature_legend_items(
            html_content, top_features_global, feature_colors_global
        )
        if overlaps_global:
            NGLViewHelper._add_overlap_legend_items(
                html_content, overlaps_global, top_features_global,
                feature_colors_global
            )
        html_content.append('</div>')
        html_content.append('</div>')

    @staticmethod
    def _add_local_features_section(
        html_content: List[str],
        selected_struct: str,
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_local: Dict[str, Dict[int, List[int]]]
    ) -> None:
        """
        Add local features section to legend HTML for selected structure.

        Parameters
        ----------
        html_content : List[str]
            HTML content list to append to
        selected_struct : str
            Currently selected structure
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_local : Dict[str, Dict[int, List[int]]]
            Local overlaps per structure

        Returns
        -------
        None
            Modifies html_content in-place
        """
        if not selected_struct or selected_struct not in top_features_local:
            return

        local_features = top_features_local[selected_struct]
        local_colors = feature_colors_local[selected_struct]
        local_overlaps = overlaps_local.get(selected_struct, {})

        html_content.append('<div style="margin-top:8px;">')
        html_content.append('<b style="font-size:12px; color:#666;">Local Features:</b>')
        html_content.append(f'<div style="margin-left:10px; margin-top:4px;">')
        html_content.append(f'<i style="font-size:11px; color:#888;">{selected_struct}:</i>')
        html_content.append('<div style="margin-left:10px;">')
        NGLViewHelper._add_feature_legend_items(
            html_content, local_features, local_colors
        )
        if local_overlaps:
            NGLViewHelper._add_overlap_legend_items(
                html_content, local_overlaps, local_features, local_colors
            )
        html_content.append('</div>')
        html_content.append('</div>')
        html_content.append('</div>')

    @staticmethod
    def _add_feature_legend_items(
        html_content: List[str],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str]
    ) -> None:
        """
        Add feature legend items to HTML content.

        Parameters
        ----------
        html_content : List[str]
            HTML content list to append to
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries
        feature_colors : Dict[str, str]
            Feature color mapping

        Returns
        -------
        None
            Appends HTML to html_content
        """
        for feature in top_features:
            feature_name = feature.get('feature_name', 'Unknown')
            feature_type = feature.get('feature_type', 'unknown').capitalize()
            color = feature_colors.get(feature_name, '#808080')

            html_content.append(
                f'<div style="margin:4px 0; display:flex; align-items:center;">'
                f'<span style="display:inline-block; width:20px; height:20px; '
                f'background:{color}; margin-right:8px; border:1px solid #333; '
                f'border-radius:2px;"></span>'
                f'<span style="font-size:12px; font-family:monospace;">'
                f'{feature_type}: {feature_name}</span>'
                f'</div>'
            )

    @staticmethod
    def _add_overlap_legend_items(
        html_content: List[str],
        overlaps: Dict[int, List[int]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str]
    ) -> None:
        """
        Add overlap legend items to HTML content.

        Parameters
        ----------
        html_content : List[str]
            HTML content list to append to
        overlaps : Dict[int, List[int]]
            Mapping from seqid to feature indices
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries
        feature_colors : Dict[str, str]
            Feature color mapping

        Returns
        -------
        None
            Appends HTML to html_content
        """
        html_content.append('<hr style="margin:12px 0; border:0; '
                          'border-top:1px solid #ddd;">')
        html_content.append('<b style="font-size:14px;">Overlapping Residues:</b>')
        html_content.append('<div style="margin-top:8px;">')

        for seqid, feature_indices in overlaps.items():
            # Calculate mixed color
            colors_to_mix = [
                feature_colors[top_features[i]["feature_name"]]
                for i in feature_indices
            ]
            mixed_color = ColorConversionHelper.mix_hex_colors(colors_to_mix)

            html_content.append(
                f'<div style="margin:4px 0; display:flex; align-items:center;">'
                f'<span style="display:inline-block; width:20px; height:20px; '
                f'background:{mixed_color}; margin-right:8px; border:1px solid #333; '
                f'border-radius:2px;"></span>'
                f'<span style="font-size:12px; font-family:monospace;">'
                f'Residue {seqid} ({len(feature_indices)} features)</span>'
                f'</div>'
            )

        html_content.append('</div>')

    @staticmethod
    def _create_download_button(
        view: nv.NGLWidget,
        viz_name: str,
        dropdown: widgets.Dropdown
    ) -> Tuple[widgets.Button, widgets.Output]:
        """
        Create download button for exporting NGLView as PNG.

        Creates a button that triggers browser download of the current
        NGLView widget using NGLView's download_image() method.

        Parameters
        ----------
        view : nv.NGLWidget
            NGLView widget to download
        viz_name : str
            Visualization name for filename generation
        dropdown : widgets.Dropdown
            Dropdown widget to get selected structure name

        Returns
        -------
        Tuple[widgets.Button, widgets.Output]
            Download button and Output widget for status messages

        Examples
        --------
        >>> button, output = NGLViewHelper._create_download_button(
        ...     view, "my_viz", dropdown
        ... )

        Notes
        -----
        Uses NGLView.download_image() which opens browser download dialog.
        Filename format: {viz_name}_{structure}_{timestamp}.png
        """
        # Create output widget for status messages
        output = widgets.Output()

        # Download button
        download_btn = widgets.Button(
            description='Download Image',
            button_style='info',
            icon='download',
            tooltip='Download current view as PNG'
        )

        # Attach click handler
        download_btn.on_click(
            lambda _b: NGLViewHelper._handle_download_click(
                view, output, viz_name, dropdown
            )
        )

        return download_btn, output

    @staticmethod
    def _handle_download_click(
        view: nv.NGLWidget,
        output: widgets.Output,
        viz_name: str,
        dropdown: widgets.Dropdown
    ):
        """
        Handle download button click event.

        Generates filename and triggers browser download of NGLView widget.

        Parameters
        ----------
        view : nv.NGLWidget
            NGLView widget to download
        output : widgets.Output
            Output widget for status messages
        viz_name : str
            Visualization name for filename
        dropdown : widgets.Dropdown
            Dropdown to get current structure name

        Notes
        -----
        Filename format: {viz_name}_{structure}_{timestamp}.png
        """
        with output:
            # Get current structure name for filename
            struct_name = dropdown.value if dropdown.value else "unknown"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{viz_name}_{struct_name}_{timestamp}.png"

            # Trigger browser download
            view.download_image(filename=filename)

    @staticmethod
    def _setup_view_components(
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        feature_own_color: bool
    ) -> Tuple[nv.NGLWidget, Dict, Dict, Dict]:
        """
        Setup NGLView widget with loaded structures.

        Creates NGLView widget, detects overlapping features for both
        global and local features, loads all structures with representations,
        and centers the view.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure information with 'path' and 'color' keys
        top_features_global : List[Dict[str, Any]]
            Global features to highlight
        feature_colors_global : Dict[str, str]
            Global feature name to color mapping
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        feature_own_color : bool
            If True, features use own colors; if False, use structure color

        Returns
        -------
        Tuple[nv.NGLWidget, Dict, Dict, Dict]
            NGLView widget, component map, global overlaps, local overlaps
        """
        view = nv.NGLWidget()
        component_map = {}

        # Detect overlapping residues for global features
        overlaps_global = FeatureOverlapHelper.detect_residue_overlaps(top_features_global)

        # Detect overlapping residues for local features (per structure)
        overlaps_local = {}
        for struct_name, local_features in top_features_local.items():
            overlaps_local[struct_name] = FeatureOverlapHelper.detect_residue_overlaps(local_features)

        # Load all structures and setup representations
        for struct_name, info in pdb_info.items():
            local_features = top_features_local.get(struct_name, [])
            local_colors = feature_colors_local.get(struct_name, {})
            local_overlaps = overlaps_local.get(struct_name, {})

            component = NGLViewHelper._load_structure_component(
                view, info['path'], info['color'],
                top_features_global, feature_colors_global, overlaps_global,
                local_features, local_colors, local_overlaps,
                feature_own_color
            )
            component_map[struct_name] = component

        view.center()

        return view, component_map, overlaps_global, overlaps_local

    @staticmethod
    def _create_control_widgets(
        component_map: Dict,
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_global: Dict,
        overlaps_local: Dict,
        feature_own_color: bool
    ) -> Tuple[widgets.Dropdown, widgets.GridBox]:
        """
        Create control widgets (dropdown and checkbox grid).

        Creates focus checkboxes (3x4 grid), dropdown for structure selection,
        and sets initial visibility (first structure only).

        Parameters
        ----------
        component_map : Dict
            Mapping of structure names to NGLView components
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features_global : List[Dict[str, Any]]
            Global features to highlight
        feature_colors_global : Dict[str, str]
            Global feature name to color mapping
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_global : Dict
            Overlap dictionary for global features
        overlaps_local : Dict
            Overlap dictionaries for local features
        feature_own_color : bool
            Color mode for features

        Returns
        -------
        Tuple[widgets.Dropdown, widgets.GridBox]
            Dropdown and checkbox grid widgets
        """
        # Create focus checkboxes (3x4 grid)
        checkbox_grid = NGLViewHelper._create_focus_checkboxes()

        # Create dropdown
        dropdown = NGLViewHelper._create_dropdown(
            component_map, checkbox_grid, pdb_info,
            top_features_global, feature_colors_global,
            top_features_local, feature_colors_local,
            overlaps_global, overlaps_local, feature_own_color
        )

        # Set initial visibility (first structure only)
        NGLViewHelper._set_initial_visibility(component_map)

        return dropdown, checkbox_grid

    @staticmethod
    def _setup_legend_widgets(
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        overlaps_global: Dict,
        overlaps_local: Dict,
        feature_own_color: bool,
        show_structure_legend: bool,
        show_feature_legend: bool,
        selected_struct: str = None
    ) -> List[widgets.HTML]:
        """
        Setup legend widgets based on configuration.

        Creates structure and/or feature legend widgets based on flags.
        Shows global features and local features only for selected structure.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features_global : List[Dict[str, Any]]
            Global features to highlight
        feature_colors_global : Dict[str, str]
            Global feature name to color mapping
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        overlaps_global : Dict
            Overlap dictionary for global features
        overlaps_local : Dict
            Overlap dictionaries for local features
        feature_own_color : bool
            Color mode for features
        show_structure_legend : bool
            If True, add structure legend
        show_feature_legend : bool
            If True, add feature legend
        selected_struct : str, optional
            Currently selected structure (for filtering local features)

        Returns
        -------
        List[widgets.HTML]
            List of legend HTML widgets
        """
        legend_widgets = []

        if show_structure_legend:
            structure_legend = NGLViewHelper._create_structure_legend_widget(
                pdb_info
            )
            legend_widgets.append(structure_legend)

        if show_feature_legend:
            feature_legend = NGLViewHelper._create_feature_legend_widget(
                top_features_global, feature_colors_global, overlaps_global,
                top_features_local, feature_colors_local, overlaps_local,
                feature_own_color, selected_struct
            )
            legend_widgets.append(feature_legend)

        return legend_widgets

    @staticmethod
    def _assemble_ui(
        dropdown: widgets.Dropdown,
        checkbox_grid: widgets.GridBox,
        legend_widgets: List[widgets.HTML],
        download_button: widgets.Button,
        download_output: widgets.Output
    ) -> widgets.VBox:
        """
        Assemble UI elements into final VBox layout.

        Combines all UI elements (dropdown, checkbox grid, legends,
        download button, and output) into a single VBox container.

        Parameters
        ----------
        dropdown : widgets.Dropdown
            Structure selection dropdown
        checkbox_grid : widgets.GridBox
            Focus control checkbox grid
        legend_widgets : List[widgets.HTML]
            Legend widgets
        download_button : widgets.Button
            Download button
        download_output : widgets.Output
            Download status output widget

        Returns
        -------
        widgets.VBox
            Combined UI container
        """
        ui_elements = [dropdown, checkbox_grid]
        ui_elements.extend(legend_widgets)
        ui_elements.append(download_button)
        ui_elements.append(download_output)

        return widgets.VBox(ui_elements)

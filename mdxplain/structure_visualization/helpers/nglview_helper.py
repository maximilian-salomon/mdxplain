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
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        feature_own_color: bool = True
    ) -> Tuple:
        """
        Create NGLView widget with focus-based structure visualization.

        Creates interactive 3D viewer with beta-factor gradient coloring
        and feature highlights. Includes dropdown for structure selection
        and 2x4 checkbox grid for controlling visibility and transparency
        of own vs other structures and features.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure information with 'path' and 'color' keys per structure
        top_features : List[Dict[str, Any]]
            Features to highlight with licorice representation
        feature_colors : Dict[str, str]
            Feature name to HEX color mapping
        feature_own_color : bool, default=True
            If True, features use their own color from feature_colors.
            If False, features use the color of their structure.

        Returns
        -------
        Tuple[widgets.VBox, nv.NGLWidget]
            UI container (dropdown + checkbox grid + legend) and NGLWidget

        Examples
        --------
        >>> pdb_info = {
        ...     "cluster_0": {"path": "c0.pdb", "color": "#bf4242"}
        ... }
        >>> features = [
        ...     {"feature_name": "ALA_5_CA-GLU_10_CA", ...}
        ... ]
        >>> colors = {"ALA_5_CA-GLU_10_CA": "#ff0000"}
        >>> ui, view = NGLViewHelper.create_widget(pdb_info, features, colors)
        >>> from IPython.display import display
        >>> display(ui, view)

        Notes
        -----
        - Beta-factors: 0.0 (base color) → 1.0 (white)
        - PDBs must be pre-aligned for multi-structure visualization
        - Dropdown selects focus structure
        - 2x4 grid controls own/other structures and features separately
        - Each checkbox controls visibility and transparency independently
        """
        view = nv.NGLWidget()
        component_map = {}

        # Detect overlapping residues once (used throughout)
        overlaps = FeatureOverlapHelper.detect_residue_overlaps(top_features)

        # Load all structures and setup representations
        for struct_name, info in pdb_info.items():
            component = NGLViewHelper._load_structure_component(
                view, info['path'], info['color'], top_features, feature_colors,
                overlaps, feature_own_color
            )
            component_map[struct_name] = component

        view.center()

        # Create focus checkboxes (2x4 grid)
        checkbox_grid = NGLViewHelper._create_focus_checkboxes()

        # Create dropdown
        dropdown = NGLViewHelper._create_dropdown(
            component_map, checkbox_grid, pdb_info, top_features, feature_colors,
            overlaps, feature_own_color
        )

        # Set initial visibility (first structure only)
        NGLViewHelper._set_initial_visibility(component_map)

        # Create legend widget
        legend_widget = NGLViewHelper._create_legend_widget(
            top_features, feature_colors, overlaps, feature_own_color
        )

        # Combine UI elements
        ui = widgets.VBox([dropdown, checkbox_grid, legend_widget])
        return ui, view

    @staticmethod
    def _load_structure_component(
        view: nv.NGLWidget,
        pdb_path: str,
        base_color: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
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
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues
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

        # Add feature highlights (non-overlapping residues only)
        for feature in top_features:
            NGLViewHelper._add_feature_highlight(
                component, feature, feature_colors, overlaps,
                1.0, base_color, feature_own_color
            )

        # Add overlap highlights with mixed colors
        NGLViewHelper._add_overlap_highlights(
            component, overlaps, top_features, feature_colors,
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
        Create 2x4 checkbox grid for focus visualization control.

        Grid layout:
        Row 1: [Own Struct Enabled] [Own Struct Trans] [Own Feat Enabled] [Own Feat Trans]
        Row 2: [Other Struct Enabled] [Other Struct Trans] [Other Feat Enabled] [Other Feat Trans]

        Returns
        -------
        widgets.GridBox
            2x4 checkbox grid
        """
        checkboxes = [
            widgets.Checkbox(value=True, description="Own Structure"),
            widgets.Checkbox(value=False, description="Own Struct Transparent"),
            widgets.Checkbox(value=True, description="Own Features"),
            widgets.Checkbox(value=False, description="Own Feat Transparent"),
            widgets.Checkbox(value=False, description="Other Structures"),
            widgets.Checkbox(value=True, description="Other Struct Transparent"),
            widgets.Checkbox(value=False, description="Other Features"),
            widgets.Checkbox(value=True, description="Other Feat Transparent"),
        ]

        grid = widgets.GridBox(
            checkboxes,
            layout=widgets.Layout(
                width='100%',
                grid_template_columns='repeat(4, 25%)',
                grid_template_rows='auto auto',
                grid_gap='5px 5px'
            )
        )

        return grid

    @staticmethod
    def _create_dropdown(
        component_map: Dict,
        checkbox_grid: widgets.GridBox,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
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
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues
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

        def on_structure_change(change):
            NGLViewHelper._handle_dropdown_change(
                change, component_map, checkbox_grid,
                pdb_info, top_features, feature_colors, overlaps, feature_own_color
            )

        dropdown.observe(on_structure_change, names='value')

        # Attach change handlers to checkboxes
        checkboxes = checkbox_grid.children
        for checkbox in checkboxes:
            checkbox.observe(
                lambda change, cbs=checkboxes: NGLViewHelper._handle_checkbox_change(
                    change, dropdown, component_map, cbs, pdb_info,
                    top_features, feature_colors, overlaps, feature_own_color
                ),
                names='value'
            )

        return dropdown

    @staticmethod
    def _handle_dropdown_change(
        change: Dict,
        component_map: Dict,
        checkbox_grid: widgets.GridBox,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
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
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors

        Returns
        -------
        None
            Updates all components based on checkboxes
        """
        NGLViewHelper._update_view_from_checkboxes(
            change['owner'], component_map, checkbox_grid.children,
            pdb_info, top_features, feature_colors, overlaps, feature_own_color
        )

    @staticmethod
    def _handle_checkbox_change(
        _change: Dict,
        dropdown: widgets.Dropdown,
        component_map: Dict,
        checkboxes: Tuple,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
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
            8 checkboxes from grid
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors

        Returns
        -------
        None
            Updates all components based on checkboxes
        """
        NGLViewHelper._update_view_from_checkboxes(
            dropdown, component_map, checkboxes,
            pdb_info, top_features, feature_colors, overlaps, feature_own_color
        )

    @staticmethod
    def _update_view_from_checkboxes(
        dropdown: widgets.Dropdown,
        component_map: Dict,
        checkboxes: Tuple,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
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
            8 checkboxes from grid
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors

        Returns
        -------
        None
            Updates all components
        """
        selected_struct = dropdown.value

        # Extract checkbox values
        own_struct_enabled = checkboxes[0].value
        own_struct_trans = checkboxes[1].value
        own_feat_enabled = checkboxes[2].value
        own_feat_trans = checkboxes[3].value
        other_struct_enabled = checkboxes[4].value
        other_struct_trans = checkboxes[5].value
        other_feat_enabled = checkboxes[6].value
        other_feat_trans = checkboxes[7].value

        # Calculate opacities
        own_struct_opacity = NGLViewHelper._calc_opacity(
            own_struct_enabled, own_struct_trans
        )
        own_feat_opacity = NGLViewHelper._calc_opacity(
            own_feat_enabled, own_feat_trans
        )
        other_struct_opacity = NGLViewHelper._calc_opacity(
            other_struct_enabled, other_struct_trans
        )
        other_feat_opacity = NGLViewHelper._calc_opacity(
            other_feat_enabled, other_feat_trans
        )

        # Update all components
        for struct_name, component in component_map.items():
            is_own = (struct_name == selected_struct)
            struct_opacity = own_struct_opacity if is_own else other_struct_opacity
            feat_opacity = own_feat_opacity if is_own else other_feat_opacity

            if struct_opacity == 0.0:
                component.hide()
            else:
                NGLViewHelper._recreate_structure_with_focus(
                    component, pdb_info[struct_name]['color'],
                    top_features, feature_colors, overlaps,
                    struct_opacity, feat_opacity, feature_own_color
                )
                component.show()

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
    def _recreate_structure_with_focus(
        component,
        base_color: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
        struct_opacity: float,
        feat_opacity: float,
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
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues
        struct_opacity : float
            Opacity for structure (cartoon)
        feat_opacity : float
            Opacity for features (licorice)
        feature_own_color : bool
            If True, use feature colors; if False, use base_color

        Returns
        -------
        None
            Recreates component representations
        """
        component.clear_representations()

        # Add cartoon with structure opacity
        component.add_representation(
            "cartoon",
            selection="not hydrogen",
            colorScheme="bfactor",
            colorScale=[base_color, "white"],
            colorDomain=[0.0, 1.0],
            opacity=struct_opacity
        )

        # Add features with feature opacity
        if feat_opacity > 0.0:
            for feature in top_features:
                NGLViewHelper._add_feature_highlight(
                    component, feature, feature_colors, overlaps,
                    feat_opacity, base_color, feature_own_color
                )
            NGLViewHelper._add_overlap_highlights(
                component, overlaps, top_features, feature_colors,
                feat_opacity, base_color, feature_own_color
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

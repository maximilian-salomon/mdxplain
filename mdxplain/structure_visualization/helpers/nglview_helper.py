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


class NGLViewHelper:
    """
    Helper for creating NGLView widgets with advanced features.

    Provides static methods to create interactive 3D molecular structure
    viewers with beta-factor gradient coloring, feature highlights,
    structure selection dropdowns, and multi-view mode.

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
        feature_colors: Dict[str, str]
    ) -> Tuple:
        """
        Create NGLView widget with structure selection and features.

        Creates interactive 3D viewer with beta-factor gradient coloring
        and feature highlights. Includes dropdown for structure selection
        and multi-view mode with checkboxes for opacity control.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure information with 'path' and 'color' keys per structure
        top_features : List[Dict[str, Any]]
            Features to highlight with licorice representation
        feature_colors : Dict[str, str]
            Feature name to HEX color mapping

        Returns
        -------
        Tuple[widgets.VBox, nv.NGLWidget]
            UI container (dropdown + checkboxes) and NGLWidget

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
        - PDBs must be pre-aligned if using multi-view mode
        - Dropdown allows switching between individual structures or "multiple"
        - Multi-view mode shows all structures with opacity checkboxes
        """
        view = nv.NGLWidget()
        component_map = {}

        # Detect overlapping residues once (used throughout)
        overlaps = NGLViewHelper._detect_residue_overlaps(top_features)

        # Load all structures and setup representations
        for struct_name, info in pdb_info.items():
            component = NGLViewHelper._load_structure_component(
                view, info['path'], info['color'], top_features, feature_colors, overlaps
            )
            component_map[struct_name] = component

        view.center()

        # Create UI widgets
        checkbox_ui = NGLViewHelper._create_checkboxes(
            component_map, pdb_info, top_features, feature_colors, overlaps
        )

        dropdown = NGLViewHelper._create_dropdown(
            component_map, checkbox_ui, pdb_info, top_features, feature_colors, overlaps
        )

        # Set initial visibility (first structure only)
        NGLViewHelper._set_initial_visibility(component_map)

        # Create legend widget
        legend_widget = NGLViewHelper._create_legend_widget(
            top_features, feature_colors, overlaps
        )

        # Combine UI elements
        ui = widgets.VBox([dropdown, checkbox_ui, legend_widget])
        return ui, view

    @staticmethod
    def _load_structure_component(
        view: nv.NGLWidget,
        pdb_path: str,
        base_color: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]]
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
            Base color for beta-factor gradient (HEX)
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues

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
                component, feature, feature_colors, overlaps
            )

        # Add overlap highlights with mixed colors
        NGLViewHelper._add_overlap_highlights(
            component, overlaps, top_features, feature_colors
        )

        return component

    @staticmethod
    def _add_feature_highlight(
        component,
        feature: Dict[str, Any],
        feature_colors: Dict[str, str],
        all_overlaps: Dict[int, List[int]],
        opacity: float = 1.0
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
        opacity : float, default=1.0
            Opacity for licorice representation

        Returns
        -------
        None
            Adds representation to component
        """
        residues = feature.get("residue_seqids", [])
        if not residues:
            return

        # Filter out overlapping residues
        non_overlap_residues = [s for s in residues if s not in all_overlaps]
        if not non_overlap_residues:
            return

        feature_name = feature.get('feature_name', '')
        color_hex = feature_colors.get(feature_name, '#808080')
        res_selection = ' or '.join(map(str, non_overlap_residues))

        component.add_representation(
            "licorice",
            selection=res_selection,
            color=color_hex,
            opacity=opacity
        )

    @staticmethod
    def _create_checkboxes(
        component_map: Dict,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]]
    ) -> widgets.VBox:
        """
        Create checkboxes for multi-view opacity control.

        Parameters
        ----------
        component_map : Dict
            Structure name to component mapping
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues

        Returns
        -------
        widgets.VBox
            Checkbox UI container
        """
        checkbox_widgets = []

        for name in component_map.keys():
            checkbox = widgets.Checkbox(
                value=True,
                description=name,
                indent=False
            )
            checkbox_widgets.append(checkbox)

        checkbox_ui = widgets.VBox(checkbox_widgets)
        checkbox_ui.layout.display = 'none'

        # Attach change handler
        def on_checkbox_change(change):
            NGLViewHelper._handle_checkbox_change(
                change, component_map,
                pdb_info, top_features, feature_colors, overlaps
            )

        for checkbox in checkbox_widgets:
            checkbox.observe(on_checkbox_change, names='value')

        return checkbox_ui

    @staticmethod
    def _handle_checkbox_change(
        change: Dict,
        component_map: Dict,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]]
    ) -> None:
        """
        Handle checkbox state change by updating opacity.

        Parameters
        ----------
        change : Dict
            Change event dictionary
        component_map : Dict
            Structure name to component mapping
        pdb_info : Dict[str, Dict[str, str]]
            Structure information
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues

        Returns
        -------
        None
            Updates component representations
        """
        checkbox = change['owner']
        is_active = change['new']
        struct_name = checkbox.description
        component = component_map[struct_name]
        info = pdb_info[struct_name]

        opacity_value = 1.0 if is_active else 0.2

        # Recreate all representations with new opacity
        NGLViewHelper._recreate_structure_representations(
            component, info['color'], top_features, feature_colors, overlaps, opacity_value
        )

    @staticmethod
    def _create_dropdown(
        component_map: Dict,
        checkbox_ui: widgets.VBox,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]]
    ) -> widgets.Dropdown:
        """
        Create dropdown for structure selection.

        Parameters
        ----------
        component_map : Dict
            Structure name to component mapping
        checkbox_ui : widgets.VBox
            Checkbox UI container
        pdb_info : Dict[str, Dict[str, str]]
            Structure information with paths and colors
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues

        Returns
        -------
        widgets.Dropdown
            Dropdown widget with change handler attached
        """
        structure_names = list(component_map.keys())
        dropdown_options = structure_names + ["All structures"]

        dropdown = widgets.Dropdown(
            options=dropdown_options,
            value=structure_names[0] if structure_names else None,
            description='Structure:',
            style={'description_width': 'initial'}
        )

        def on_structure_change(change):
            NGLViewHelper._handle_dropdown_change(
                change, component_map, checkbox_ui,
                pdb_info, top_features, feature_colors, overlaps
            )

        dropdown.observe(on_structure_change, names='value')

        return dropdown

    @staticmethod
    def _handle_dropdown_change(
        change: Dict,
        component_map: Dict,
        checkbox_ui: widgets.VBox,
        pdb_info: Dict[str, Dict[str, str]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]]
    ) -> None:
        """
        Handle dropdown selection change.

        Parameters
        ----------
        change : Dict
            Change event dictionary
        component_map : Dict
            Structure name to component mapping
        checkbox_ui : widgets.VBox
            Checkbox UI container
        pdb_info : Dict[str, Dict[str, str]]
            Structure information with paths and colors
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues

        Returns
        -------
        None
            Updates component visibility and checkbox visibility
        """
        selected = change['new']

        if selected == "All structures":
            # Multi-view mode: show all, display checkboxes
            for component in component_map.values():
                component.show()
            checkbox_ui.layout.display = 'flex'
        else:
            # Single-view mode: show only selected, reset opacity to 1.0
            for name, component in component_map.items():
                if name == selected:
                    # Reset representations with full opacity
                    NGLViewHelper._recreate_structure_representations(
                        component, pdb_info[name]['color'],
                        top_features, feature_colors, overlaps, opacity=1.0
                    )
                    component.show()
                else:
                    component.hide()
            checkbox_ui.layout.display = 'none'

    @staticmethod
    def _recreate_structure_representations(
        component,
        base_color: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        overlaps: Dict[int, List[int]],
        opacity: float = 1.0
    ) -> None:
        """
        Recreate all representations for a component.

        Clears existing representations and creates beta-factor cartoon,
        feature highlights, and overlap highlights with specified opacity.

        Parameters
        ----------
        component : nv.Component
            NGLView component
        base_color : str
            Base color for beta-factor gradient (HEX)
        top_features : List[Dict[str, Any]]
            Features to highlight
        feature_colors : Dict[str, str]
            Feature color mapping
        overlaps : Dict[int, List[int]]
            Pre-computed overlapping residues
        opacity : float, default=1.0
            Opacity value for all representations

        Returns
        -------
        None
            Recreates component representations
        """
        component.clear_representations()

        # Add beta-factor cartoon
        component.add_representation(
            "cartoon",
            selection="not hydrogen",
            colorScheme="bfactor",
            colorScale=[base_color, "white"],
            colorDomain=[0.0, 1.0],
            opacity=opacity
        )

        # Add feature highlights (non-overlapping)
        for feature in top_features:
            NGLViewHelper._add_feature_highlight(
                component, feature, feature_colors, overlaps, opacity
            )

        # Add overlap highlights
        NGLViewHelper._add_overlap_highlights(
            component, overlaps, top_features, feature_colors, opacity
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
    def _hex_to_rgb(hex_color: str) -> tuple:
        """
        Convert HEX color to RGB tuple.

        Parameters
        ----------
        hex_color : str
            HEX color string (e.g., "#ff0000")

        Returns
        -------
        tuple
            RGB values (r, g, b) as integers
        """
        hex_clean = hex_color.lstrip('#')
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
        return (r, g, b)

    @staticmethod
    def _calculate_average_rgb(rgb_values: List[tuple]) -> tuple:
        """
        Calculate average RGB values.

        Parameters
        ----------
        rgb_values : List[tuple]
            List of RGB tuples

        Returns
        -------
        tuple
            Average (r, g, b) values
        """
        n = len(rgb_values)
        avg_r = sum(rgb[0] for rgb in rgb_values) // n
        avg_g = sum(rgb[1] for rgb in rgb_values) // n
        avg_b = sum(rgb[2] for rgb in rgb_values) // n
        return (avg_r, avg_g, avg_b)

    @staticmethod
    def _mix_hex_colors(hex_colors: List[str]) -> str:
        """
        Calculate RGB average of multiple HEX colors.

        Mixes multiple colors by averaging their RGB components.
        Used for visualizing overlapping features.

        Parameters
        ----------
        hex_colors : List[str]
            List of HEX color strings (e.g., ["#ff0000", "#0000ff"])

        Returns
        -------
        str
            Mixed color as HEX (#RRGGBB)

        Examples
        --------
        >>> # Mix red and blue
        >>> mixed = NGLViewHelper._mix_hex_colors(["#ff0000", "#0000ff"])
        >>> print(mixed)
        '#7f007f'
        """
        if not hex_colors:
            return "#808080"

        rgb_values = [NGLViewHelper._hex_to_rgb(color) for color in hex_colors]
        avg_r, avg_g, avg_b = NGLViewHelper._calculate_average_rgb(rgb_values)

        return f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"

    @staticmethod
    def _build_residue_feature_map(
        top_features: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """
        Build mapping from residue seqid to feature indices.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries

        Returns
        -------
        Dict[int, List[int]]
            Mapping from seqid to list of feature indices
        """
        residue_to_features = {}

        for feat_idx, feature in enumerate(top_features):
            seqids = feature.get("residue_seqids", [])
            for seqid in seqids:
                if seqid not in residue_to_features:
                    residue_to_features[seqid] = []
                residue_to_features[seqid].append(feat_idx)

        return residue_to_features

    @staticmethod
    def _detect_residue_overlaps(
        top_features: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """
        Detect residues occurring in multiple features.

        Identifies residues that appear in more than one feature,
        which need mixed color representation.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries with residue_seqids

        Returns
        -------
        Dict[int, List[int]]
            Mapping from seqid to list of feature indices.
            Only includes residues appearing in 2+ features.

        Examples
        --------
        >>> features = [
        ...     {"residue_seqids": [16, 33]},
        ...     {"residue_seqids": [27, 33]}
        ... ]
        >>> overlaps = NGLViewHelper._detect_residue_overlaps(features)
        >>> print(overlaps)
        {33: [0, 1]}
        """
        residue_to_features = NGLViewHelper._build_residue_feature_map(
            top_features
        )

        return {
            seqid: feat_indices
            for seqid, feat_indices in residue_to_features.items()
            if len(feat_indices) > 1
        }

    @staticmethod
    def _add_overlap_highlights(
        component,
        overlaps: Dict[int, List[int]],
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        opacity: float = 1.0
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
        opacity : float, default=1.0
            Opacity value for representations

        Returns
        -------
        None
            Adds representations to component
        """
        for seqid, feature_indices in overlaps.items():
            # Collect colors from all features containing this residue
            colors = [
                feature_colors[top_features[i]["feature_name"]]
                for i in feature_indices
            ]
            mixed_color = NGLViewHelper._mix_hex_colors(colors)

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
        overlaps: Dict[int, List[int]]
    ) -> widgets.HTML:
        """
        Create legend widget showing feature colors, types, and overlaps.

        Creates HTML widget displaying a legend that maps licorice
        representation colors to their corresponding feature names with
        feature types. Also shows overlapping residues with mixed colors.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            List of top feature dictionaries with feature names and types
        feature_colors : Dict[str, str]
            Mapping from feature name to HEX color
        overlaps : Dict[int, List[int]]
            Mapping from residue seqid to list of feature indices

        Returns
        -------
        widgets.HTML
            HTML widget with styled legend

        Examples
        --------
        >>> features = [{"feature_name": "ALA_5_CA-GLU_10_CA",
        ...              "feature_type": "distances"}]
        >>> colors = {"ALA_5_CA-GLU_10_CA": "#ff0000"}
        >>> overlaps = {33: [0, 1]}
        >>> legend = NGLViewHelper._create_legend_widget(
        ...     features, colors, overlaps
        ... )
        """
        html_content = ['<div style="padding:10px; border:1px solid #ddd; '
                       'background:#f9f9f9; margin-top:10px; border-radius:4px;">']
        html_content.append('<b style="font-size:14px;">Feature Legend:</b>')
        html_content.append('<div style="margin-top:8px;">')

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

        # Add overlapping residues section if overlaps exist
        if overlaps:
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
                mixed_color = NGLViewHelper._mix_hex_colors(colors_to_mix)

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

        html_content.append('</div>')

        return widgets.HTML(value=''.join(html_content))

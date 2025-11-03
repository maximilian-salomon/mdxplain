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
PyMOL script generator for structure visualization.

This module generates PyMOL scripts (.pml) for visualizing molecular
structures with beta-factor-based thickness and coloring, plus feature
highlights with licorice representation.
"""

from typing import Dict, List, Any

from .color_conversion_helper import ColorConversionHelper


class PyMolScriptGenerator:
    """
    Generator for PyMOL visualization scripts.

    Creates .pml scripts with putty cartoons (thickness from beta-factor),
    color gradients, and focus groups for context visualization.
    Each focus group contains own structure/features (opak) plus other
    structures/features (transparent) for context.

    Examples
    --------
    >>> pdb_info = {
    ...     "cluster_0": {"path": "/path/to/c0.pdb", "color": "#ff0000"}
    ... }
    >>> features = [{"feature_name": "ALA5-GLU10", "residue_seqids": [5, 10]}]
    >>> colors = {"ALA5-GLU10": "#00ff00"}
    >>> script = PyMolScriptGenerator.generate_script(
    ...     pdb_info, features, colors
    ... )
    """

    @staticmethod
    def generate_script(
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        feature_own_color: bool = True,
        use_putty: bool = True
    ) -> str:
        """
        Generate complete PyMOL script with focus groups.

        Creates script with background and focus groups only.
        Each group contains up to 6 objects: own structure, own global features,
        own local features, other structures (combined), other global features,
        other local features (all transparent for context).

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure info: {name: {"path": pdb_path, "color": hex_color}}
        top_features_global : List[Dict[str, Any]]
            Global top features (averaged across all clusters)
        feature_colors_global : Dict[str, str]
            Mapping from global feature_name to HEX color
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local top features per structure {struct_name: features}
        feature_colors_local : Dict[str, Dict[str, str]]
            Mapping from struct_name to feature colors {struct_name: {feat_name: color}}
        feature_own_color : bool, default=True
            If True, features use their own color from feature_colors.
            If False, features use the color of their structure.
        use_putty : bool, default=True
            If True, use putty cartoon with beta-factor thickness.
            If False, use normal cartoon with uniform thickness.

        Returns
        -------
        str
            Complete PyMOL script as string

        Examples
        --------
        >>> # With global and local features
        >>> script = PyMolScriptGenerator.generate_script(
        ...     pdb_info, global_features, global_colors,
        ...     local_features, local_colors, use_putty=True
        ... )
        >>> # Only global features (empty dict for local)
        >>> script = PyMolScriptGenerator.generate_script(
        ...     pdb_info, global_features, global_colors,
        ...     {}, {}, use_putty=True
        ... )

        Notes
        -----
        - Only focus groups in script
        - First group enabled, rest disabled
        - Each group: up to 6 objects (struct, global/local features x own/other)
        - Global and local features independently toggleable
        - Putty cartoons: thickness from beta-factor (if use_putty=True)
        - Normal cartoons: uniform thickness (if use_putty=False)
        - Color gradient: base_color → white
        """
        lines = []

        # Background
        lines.append(PyMolScriptGenerator._add_background())
        lines.append("")

        # Create single comparison group
        lines.append(PyMolScriptGenerator._add_single_comparison_group(
            pdb_info, top_features_global, feature_colors_global,
            top_features_local, feature_colors_local, feature_own_color, use_putty
        ))

        # Create focus groups (loads structures directly)
        lines.append(PyMolScriptGenerator._add_focus_groups(
            pdb_info, top_features_global, feature_colors_global,
            top_features_local, feature_colors_local, feature_own_color, use_putty
        ))

        return "\n".join(lines)

    @staticmethod
    def _add_background() -> str:
        """
        Add background color command.

        Returns
        -------
        str
            PyMOL command for white background
        """
        return "bg_color white"

    @staticmethod
    def _add_focus_groups(
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        feature_own_color: bool,
        use_putty: bool
    ) -> str:
        """
        Create focus groups for each structure.

        Each group contains own (opaque) + others (transparent) for both
        global and local features.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with paths and colors
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature colors
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        str
            PyMOL commands for all focus groups
        """
        lines = ["# === Focus Groups ===", ""]
        struct_names = list(pdb_info.keys())

        for i, (struct_name, info) in enumerate(pdb_info.items()):
            PyMolScriptGenerator._create_single_focus_group(
                lines, struct_name, info, struct_names,
                top_features_global, feature_colors_global,
                top_features_local, feature_colors_local,
                pdb_info, feature_own_color, use_putty
            )
            PyMolScriptGenerator._finalize_focus_group(
                lines, struct_name, i
            )

            # Cleanup temporary context structures (after group finalization)
            other_structs = [name for name in struct_names if name != struct_name]
            if other_structs:
                PyMolScriptGenerator._cleanup_context_temps(lines, other_structs)

        return "\n".join(lines)

    @staticmethod
    def _create_single_focus_group(
        lines: List[str],
        struct_name: str,
        info: Dict[str, str],
        struct_names: List[str],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        pdb_info: Dict[str, Dict[str, str]],
        feature_own_color: bool,
        use_putty: bool
    ) -> None:
        """
        Create single focus group with own and context objects.

        Creates up to 6 objects: own structure, own global features,
        own local features, other structures, other global features,
        other local features (all transparent for context).

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        struct_name : str
            Current structure name
        info : Dict[str, str]
            Structure info with path and color
        struct_names : List[str]
            All structure names
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature colors
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features per structure
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors per structure
        pdb_info : Dict[str, Dict[str, str]]
            All structure info
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        None
            Appends commands to lines
        """
        group_name = f"all_focus_struct_{struct_name}"
        lines.append(f"# Focus group for {struct_name}")
        lines.append(f"group {group_name}, open")
        lines.append("")

        main_obj = f"{group_name}_main"
        PyMolScriptGenerator._add_own_structure(
            lines, group_name, info['path'], info['color'], main_obj, use_putty
        )

        # Add own global features
        PyMolScriptGenerator._add_own_features(
            lines, group_name, main_obj, top_features_global, feature_colors_global,
            info['color'], feature_own_color, suffix="_features_global"
        )

        # Add own local features (if available for this structure)
        if struct_name in top_features_local:
            local_features = top_features_local[struct_name]
            local_colors = feature_colors_local.get(struct_name, {})
            PyMolScriptGenerator._add_own_features(
                lines, group_name, main_obj, local_features, local_colors,
                info['color'], feature_own_color, suffix="_features_local"
            )

        other_structs = [name for name in struct_names if name != struct_name]
        if other_structs:
            PyMolScriptGenerator._add_context_structures(
                lines, group_name, main_obj, other_structs, pdb_info, use_putty
            )

            # Add context global features
            PyMolScriptGenerator._add_context_features(
                lines, group_name, other_structs, top_features_global,
                feature_colors_global, pdb_info, feature_own_color,
                suffix="_context_feat_global"
            )

            # Add context local features
            PyMolScriptGenerator._add_context_features(
                lines, group_name, other_structs, top_features_local,
                feature_colors_local, pdb_info, feature_own_color,
                suffix="_context_feat_local", is_local=True
            )

    @staticmethod
    def _finalize_focus_group(
        lines: List[str],
        struct_name: str,
        group_index: int
    ) -> None:
        """
        Close and optionally disable focus group.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        struct_name : str
            Structure name
        group_index : int
            Index of group (0 stays enabled, >0 disabled)

        Returns
        -------
        None
            Appends commands to lines
        """
        group_name = f"all_focus_struct_{struct_name}"
        lines.append(f"group {group_name}, close")
        if group_index > 0:
            lines.append(f"disable {group_name}")
        lines.append("")

    @staticmethod
    def _add_single_comparison_group(
        pdb_info: Dict[str, Dict[str, str]],
        top_features_global: List[Dict[str, Any]],
        feature_colors_global: Dict[str, str],
        top_features_local: Dict[str, List[Dict[str, Any]]],
        feature_colors_local: Dict[str, Dict[str, str]],
        feature_own_color: bool,
        use_putty: bool
    ) -> str:
        """
        Create single comparison group with individual structure objects.

        Each structure gets its own combined object (structure + features)
        for direct comparison. All structures are aligned to the first.
        Only global features are shown in this view for simplicity.

        Parameters
        ----------
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with paths and colors
        top_features_global : List[Dict[str, Any]]
            Global feature list
        feature_colors_global : Dict[str, str]
            Global feature colors
        top_features_local : Dict[str, List[Dict[str, Any]]]
            Local features (unused in this view)
        feature_colors_local : Dict[str, Dict[str, str]]
            Local feature colors (unused in this view)
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        str
            PyMOL commands for single comparison group

        Notes
        -----
        Only global features are displayed in single comparison view.
        Local features are omitted for clarity in direct comparison mode.
        """
        lines = ["# === Single Comparison Group ===", ""]
        lines.append("group single_comparison, open")
        lines.append("")

        struct_names = list(pdb_info.keys())
        reference_struct = struct_names[0] if struct_names else None

        for struct_name in struct_names:
            PyMolScriptGenerator._create_single_comparison_object(
                lines, struct_name, pdb_info[struct_name],
                reference_struct, top_features_global, feature_colors_global,
                feature_own_color, use_putty
            )

        lines.append("group single_comparison, close")
        lines.append("disable single_comparison")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _create_single_comparison_object(
        lines: List[str],
        struct_name: str,
        struct_info: Dict[str, str],
        reference_struct: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        feature_own_color: bool,
        use_putty: bool
    ) -> None:
        """
        Create combined object (structure + features) for single comparison.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        struct_name : str
            Structure name
        struct_info : Dict[str, str]
            Structure info with path and color
        reference_struct : str
            Reference structure name for alignment
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature colors
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        None
            Appends commands to lines
        """
        temp_obj = f"temp_single_{struct_name}"
        final_obj = f"single_comparison_{struct_name}"

        # Load and align structure
        lines.append(f'load "{struct_info["path"]}", {temp_obj}')
        if struct_name != reference_struct:
            ref_obj = f'single_comparison_{reference_struct}'
            lines.append(f'align {temp_obj}, {ref_obj}')

        # Rename and add to group
        lines.append(f'set_name {temp_obj}, {final_obj}')
        lines.append(f'group single_comparison, {final_obj}, add')

        # Apply styling and coloring
        PyMolScriptGenerator._apply_combined_styling(
            lines, final_obj, struct_name, struct_info['color'],
            top_features, feature_colors, feature_own_color, use_putty
        )

        lines.append("")

    @staticmethod
    def _apply_combined_styling(
        lines: List[str],
        obj_name: str,
        struct_name: str,
        color_hex: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        feature_own_color: bool,
        use_putty: bool
    ) -> None:
        """
        Apply styling and coloring to combined object.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        obj_name : str
            Object name
        struct_name : str
            Structure name for color gradient
        color_hex : str
            HEX color for gradient and features
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature colors
        feature_own_color : bool
            If True, use feature colors; if False, use structure color
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        None
            Appends commands to lines
        """
        all_residues, resi_selection = PyMolScriptGenerator._collect_feature_residues(
            top_features
        )

        # Hide everything first
        lines.append(f'hide everything, {obj_name}')

        # Show cartoon for entire structure (continuous)
        lines.append(f'show cartoon, {obj_name}')

        # Show features as sticks overlay
        if all_residues:
            lines.append(f'show sticks, {obj_name} and resi {resi_selection}')

        # Apply cartoon and colors
        PyMolScriptGenerator._add_cartoon(lines, obj_name, use_putty)
        PyMolScriptGenerator._add_color_gradient(
            lines, struct_name, color_hex, obj_name
        )

        if all_residues:
            PyMolScriptGenerator._color_feature_sticks(
                lines, obj_name, top_features, feature_colors,
                not feature_own_color, color_hex
            )

    @staticmethod
    def _add_own_structure(
        lines: List[str],
        group_name: str,
        pdb_path: str,
        color_hex: str,
        main_obj: str,
        use_putty: bool
    ) -> None:
        """
        Add own structure (opaque) with cartoon and gradient.

        Loads structure directly from PDB file and assigns to group.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        group_name : str
            PyMOL group name for assignment
        pdb_path : str
            Path to PDB file
        color_hex : str
            HEX color for gradient
        main_obj : str
            PyMOL object name for structure
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        None
            Appends commands to lines
        """
        lines.append("# Own structure (opak)")
        lines.append(f'load "{pdb_path}", {main_obj}')
        lines.append(f'group {group_name}, {main_obj}, add')

        PyMolScriptGenerator._add_cartoon(lines, main_obj, use_putty)
        PyMolScriptGenerator._add_color_gradient(
            lines, main_obj, color_hex, main_obj
        )
        lines.append("")

    @staticmethod
    def _add_own_features(
        lines: List[str],
        group_name: str,
        main_obj: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        struct_color_hex: str,
        feature_own_color: bool,
        suffix: str = "_features"
    ) -> None:
        """
        Add own features (opaque) with sticks.

        Selects residues directly from main structure object and assigns to group.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        group_name : str
            PyMOL group name for assignment
        main_obj : str
            Main structure object name
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature color mapping
        struct_color_hex : str
            Structure color for features (if feature_own_color=False)
        feature_own_color : bool
            If True, use feature colors; if False, use structure color
        suffix : str, default="_features"
            Object name suffix (e.g., "_features_global", "_features_local")

        Returns
        -------
        None
            Appends commands to lines
        """
        lines.append(f"# Own {suffix.replace('_', ' ').strip()} (opak)")

        all_residues, resi_selection = PyMolScriptGenerator._collect_feature_residues(
            top_features
        )

        if not all_residues:
            lines.append("")
            return

        feat_obj = PyMolScriptGenerator._create_own_feature_object(
            lines, group_name, main_obj, resi_selection, suffix
        )

        PyMolScriptGenerator._color_feature_sticks(
            lines, feat_obj, top_features, feature_colors,
            not feature_own_color, struct_color_hex
        )

        lines.append("delete temp_own_feat")
        lines.append("")

    @staticmethod
    def _create_own_feature_object(
        lines: List[str],
        group_name: str,
        main_obj: str,
        resi_selection: str,
        suffix: str = "_features"
    ) -> str:
        """
        Create feature object from main structure.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        group_name : str
            PyMOL group name
        main_obj : str
            Main structure object name
        resi_selection : str
            Residue selection string
        suffix : str, default="_features"
            Object name suffix

        Returns
        -------
        str
            Feature object name
        """
        feat_obj = f"{group_name}{suffix}"
        lines.append(f"select temp_own_feat, {main_obj} and resi {resi_selection}")
        lines.append(f"create {feat_obj}, temp_own_feat")
        lines.append(f"group {group_name}, {feat_obj}, add")
        lines.append(f"hide everything, {feat_obj}")
        lines.append(f"show sticks, {feat_obj}")
        return feat_obj

    @staticmethod
    def _add_context_structures(
        lines: List[str],
        group_name: str,
        main_obj: str,
        other_structs: List[str],
        pdb_info: Dict[str, Dict[str, str]],
        use_putty: bool
    ) -> None:
        """
        Add other structures (80% transparent) combined into one object.

        Uses segment IDs to track structures after combining, then colors
        each structure separately. Aligns all to main structure.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        group_name : str
            PyMOL group name for assignment
        main_obj : str
            Main structure object name for alignment
        other_structs : List[str]
            Other structure names
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with paths and colors
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        None
            Appends commands to lines
        """
        lines.append("# Other structures (80% transparent)")

        temp_objs = PyMolScriptGenerator._load_and_align_structures(
            lines, other_structs, main_obj, pdb_info
        )

        ctx_obj = PyMolScriptGenerator._combine_and_style_structures(
            lines, group_name, temp_objs, use_putty
        )

        PyMolScriptGenerator._color_structures_by_segi(
            lines, ctx_obj, other_structs, pdb_info
        )

        lines.append(f"set cartoon_transparency, 0.8, {ctx_obj}")
        lines.append("")

    @staticmethod
    def _load_and_align_structures(
        lines: List[str],
        other_structs: List[str],
        main_obj: str,
        pdb_info: Dict[str, Dict[str, str]]
    ) -> List[str]:
        """
        Load structures, mark with segi, and align.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        other_structs : List[str]
            Other structure names
        main_obj : str
            Main structure for alignment
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with paths

        Returns
        -------
        List[str]
            Temp object names
        """
        temp_objs = []
        for other_name in other_structs:
            temp_obj = f"temp_ctx_{other_name}"
            lines.append(f'load "{pdb_info[other_name]["path"]}", {temp_obj}')
            lines.append(f'alter {temp_obj}, segi="{other_name}"')
            lines.append(f'align {temp_obj}, {main_obj}')
            temp_objs.append(temp_obj)
        return temp_objs

    @staticmethod
    def _combine_and_style_structures(
        lines: List[str],
        group_name: str,
        temp_objs: List[str],
        use_putty: bool
    ) -> str:
        """
        Combine structures and apply styling.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        group_name : str
            PyMOL group name
        temp_objs : List[str]
            Temp structure object names
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        str
            Combined structure object name
        """
        ctx_obj = f"{group_name}_context_struct"
        selection = " or ".join(temp_objs)
        lines.append(f"create {ctx_obj}, {selection}")
        lines.append(f"group {group_name}, {ctx_obj}, add")
        lines.append(f"hide everything, {ctx_obj}")
        lines.append(f"show cartoon, {ctx_obj}")
        PyMolScriptGenerator._add_cartoon(lines, ctx_obj, use_putty)
        return ctx_obj

    @staticmethod
    def _color_structures_by_segi(
        lines: List[str],
        ctx_obj: str,
        other_structs: List[str],
        pdb_info: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Color structures using segi selection.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        ctx_obj : str
            Combined structure object name
        other_structs : List[str]
            Other structure names
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with colors

        Returns
        -------
        None
            Appends commands to lines
        """
        for other_name in other_structs:
            lines.append(f'select temp_segi_sel, {ctx_obj} and segi {other_name}')
            PyMolScriptGenerator._add_color_gradient(
                lines, f"{other_name}_ctx", pdb_info[other_name]['color'],
                "temp_segi_sel"
            )
            lines.append("delete temp_segi_sel")

    @staticmethod
    def _add_context_features(
        lines: List[str],
        group_name: str,
        other_structs: List[str],
        top_features,
        feature_colors,
        pdb_info: Dict[str, Dict[str, str]],
        feature_own_color: bool,
        suffix: str = "_context_feat",
        is_local: bool = False
    ) -> None:
        """
        Add other features (80% transparent) combined into one object.

        Creates features from temp structures (which have segi), combines them,
        then colors using segi to identify structure parts. Supports both global
        features (List) and local features (Dict per structure).

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        group_name : str
            PyMOL group name for assignment
        other_structs : List[str]
            Other structure names
        top_features : List[Dict[str, Any]] or Dict[str, List[Dict[str, Any]]]
            Global features (List) or local features per structure (Dict)
        feature_colors : Dict[str, str] or Dict[str, Dict[str, str]]
            Global feature colors (Dict) or local colors per structure (nested Dict)
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with colors
        feature_own_color : bool
            If True, use feature colors; if False, use structure colors
        suffix : str, default="_context_feat"
            Object name suffix
        is_local : bool, default=False
            If True, top_features and feature_colors are per-structure dicts

        Returns
        -------
        None
            Appends commands to lines
        """
        label = suffix.replace("_", " ").replace("context", "Other").strip()
        lines.append(f"# {label} (80% transparent)")

        # Collect features
        all_features, all_residues, resi_selection = PyMolScriptGenerator._collect_context_features(
            top_features, other_structs, is_local
        )

        if not all_residues:
            PyMolScriptGenerator._cleanup_context_temps(lines, other_structs)
            lines.append("")
            return

        # Create and combine feature objects
        temp_feat_objs = PyMolScriptGenerator._create_temp_feature_objects(
            lines, other_structs, resi_selection
        )
        ctx_feat = PyMolScriptGenerator._combine_context_features(
            lines, group_name, temp_feat_objs, suffix
        )

        # Color features
        PyMolScriptGenerator._color_context_features_by_segi(
            lines, ctx_feat, other_structs, all_features, feature_colors,
            pdb_info, not feature_own_color, is_local
        )

        lines.append(f"set stick_transparency, 0.8, {ctx_feat}")
        lines.append("")

    @staticmethod
    def _collect_context_features(
        top_features,
        other_structs: List[str],
        is_local: bool
    ) -> tuple:
        """
        Collect features for context visualization.

        Parameters
        ----------
        top_features : List[Dict] or Dict[str, List[Dict]]
            Global features (List) or local features per structure (Dict)
        other_structs : List[str]
            Other structure names
        is_local : bool
            If True, collect from per-structure dicts

        Returns
        -------
        tuple
            (all_features, all_residues, resi_selection)

        Examples
        --------
        >>> feats, residues, selection = PyMolScriptGenerator._collect_context_features(
        ...     global_features, ["cluster_1"], is_local=False
        ... )
        """
        if is_local:
            all_features = []
            for struct_name in other_structs:
                if struct_name in top_features:
                    all_features.extend(top_features[struct_name])
        else:
            all_features = top_features

        all_residues, resi_selection = PyMolScriptGenerator._collect_feature_residues(
            all_features
        )
        return all_features, all_residues, resi_selection

    @staticmethod
    def _collect_feature_residues(
        top_features: List[Dict[str, Any]]
    ) -> tuple:
        """
        Collect all residues from features.

        Parameters
        ----------
        top_features : List[Dict[str, Any]]
            Feature list with residue_seqids

        Returns
        -------
        tuple
            (all_residues: set, resi_selection: str)
        """
        all_residues = set()
        for feature in top_features:
            residues = feature.get("residue_seqids", [])
            all_residues.update(residues)

        if not all_residues:
            return set(), ""

        resi_selection = "+".join(map(str, sorted(all_residues)))
        return all_residues, resi_selection

    @staticmethod
    def _create_temp_feature_objects(
        lines: List[str],
        other_structs: List[str],
        resi_selection: str
    ) -> List[str]:
        """
        Create temp feature objects from structures.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        other_structs : List[str]
            Other structure names
        resi_selection : str
            Residue selection string

        Returns
        -------
        List[str]
            Temp feature object names
        """
        temp_feat_objs = []
        for other_name in other_structs:
            temp_struct = f"temp_ctx_{other_name}"
            temp_feat = f"temp_ctx_feat_{other_name}"

            lines.append(f"select temp_sel, {temp_struct} and resi {resi_selection}")
            lines.append(f"create {temp_feat}, temp_sel")
            lines.append("delete temp_sel")

            temp_feat_objs.append(temp_feat)

        return temp_feat_objs

    @staticmethod
    def _combine_context_features(
        lines: List[str],
        group_name: str,
        temp_feat_objs: List[str],
        suffix: str = "_context_feat"
    ) -> str:
        """
        Combine temp features into single object.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        group_name : str
            PyMOL group name
        temp_feat_objs : List[str]
            Temp feature object names
        suffix : str, default="_context_feat"
            Object name suffix

        Returns
        -------
        str
            Combined feature object name
        """
        ctx_feat = f"{group_name}{suffix}"
        selection = " or ".join(temp_feat_objs)
        lines.append(f"create {ctx_feat}, {selection}")
        lines.append(f"group {group_name}, {ctx_feat}, add")
        lines.append(f"hide everything, {ctx_feat}")
        lines.append(f"show sticks, {ctx_feat}")
        return ctx_feat

    @staticmethod
    def _color_context_features_by_segi(
        lines: List[str],
        ctx_feat: str,
        other_structs: List[str],
        top_features: List[Dict[str, Any]],
        feature_colors,
        pdb_info: Dict[str, Dict[str, str]],
        use_struct_color: bool,
        is_local: bool = False
    ) -> None:
        """
        Color features using segi to identify structures.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        ctx_feat : str
            Combined feature object name
        other_structs : List[str]
            Other structure names
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str] or Dict[str, Dict[str, str]]
            Global color mapping (Dict) or local per structure (nested Dict)
        pdb_info : Dict[str, Dict[str, str]]
            Structure info with colors
        use_struct_color : bool
            If True, use structure colors; if False, use feature colors
        is_local : bool, default=False
            If True, feature_colors is nested dict per structure

        Returns
        -------
        None
            Appends commands to lines
        """
        for idx, feature in enumerate(top_features):
            feature_name = feature.get("feature_name", f"feature_{idx}")
            residues = feature.get("residue_seqids", [])
            if not residues:
                continue

            resi_sel = "+".join(map(str, residues))
            PyMolScriptGenerator._color_feature_for_structures(
                lines, ctx_feat, other_structs, feature_name, resi_sel,
                feature_colors, pdb_info, use_struct_color, is_local
            )

    @staticmethod
    def _color_feature_for_structures(
        lines: List[str],
        ctx_feat: str,
        other_structs: List[str],
        feature_name: str,
        resi_sel: str,
        feature_colors,
        pdb_info: Dict[str, Dict[str, str]],
        use_struct_color: bool,
        is_local: bool
    ) -> None:
        """
        Color single feature for all structures.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        ctx_feat : str
            Combined feature object name
        other_structs : List[str]
            Other structure names
        feature_name : str
            Feature name for color lookup
        resi_sel : str
            Residue selection string
        feature_colors : Dict or nested Dict
            Color mapping
        pdb_info : Dict[str, Dict[str, str]]
            Structure info
        use_struct_color : bool
            Color mode flag
        is_local : bool
            Local features flag

        Returns
        -------
        None
            Appends commands to lines
        """
        for other_name in other_structs:
            color_hex = PyMolScriptGenerator._get_feature_color_for_struct(
                feature_name, other_name, feature_colors, pdb_info,
                use_struct_color, is_local
            )
            PyMolScriptGenerator._apply_feature_color(
                lines, ctx_feat, other_name, resi_sel, color_hex
            )

    @staticmethod
    def _get_feature_color_for_struct(
        feature_name: str,
        struct_name: str,
        feature_colors,
        pdb_info: Dict[str, Dict[str, str]],
        use_struct_color: bool,
        is_local: bool
    ) -> str:
        """
        Get color for feature in specific structure.

        Parameters
        ----------
        feature_name : str
            Feature name
        struct_name : str
            Structure name
        feature_colors : Dict or nested Dict
            Color mapping
        pdb_info : Dict[str, Dict[str, str]]
            Structure info
        use_struct_color : bool
            If True, use structure color
        is_local : bool
            If True, feature_colors is nested

        Returns
        -------
        str
            HEX color string
        """
        if use_struct_color:
            return pdb_info[struct_name]['color']

        if is_local:
            struct_colors = feature_colors.get(struct_name, {})
            return struct_colors.get(feature_name, "#808080")

        return feature_colors.get(feature_name, "#808080")

    @staticmethod
    def _apply_feature_color(
        lines: List[str],
        ctx_feat: str,
        struct_name: str,
        resi_sel: str,
        color_hex: str
    ) -> None:
        """
        Apply color to feature selection.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        ctx_feat : str
            Feature object name
        struct_name : str
            Structure name
        resi_sel : str
            Residue selection
        color_hex : str
            HEX color

        Returns
        -------
        None
            Appends commands to lines
        """
        color_pymol = ColorConversionHelper.hex_to_pymol_rgb(color_hex)
        lines.append(
            f"select temp_feat_sel, {ctx_feat} and segi {struct_name} and resi {resi_sel}"
        )
        lines.append(f"set stick_color, {color_pymol}, temp_feat_sel")
        lines.append("delete temp_feat_sel")

    @staticmethod
    def _cleanup_context_temps(
        lines: List[str],
        other_structs: List[str]
    ) -> None:
        """
        Delete all temp context objects.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        other_structs : List[str]
            Other structure names

        Returns
        -------
        None
            Appends commands to lines
        """
        for other_name in other_structs:
            lines.append(f"delete temp_ctx_{other_name}")
            if f"temp_ctx_feat_{other_name}" in " ".join(lines):
                lines.append(f"delete temp_ctx_feat_{other_name}")

    @staticmethod
    def _add_cartoon(
        lines: List[str],
        object_name: str,
        use_putty: bool
    ) -> None:
        """
        Add cartoon representation to object.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        object_name : str
            PyMOL object name
        use_putty : bool
            If True, use putty cartoon; if False, use normal cartoon

        Returns
        -------
        None
            Appends commands to lines
        """
        if use_putty:
            PyMolScriptGenerator._add_putty_cartoon(lines, object_name)
        else:
            PyMolScriptGenerator._add_normal_cartoon(lines, object_name)

    @staticmethod
    def _add_putty_cartoon(lines: List[str], object_name: str) -> None:
        """
        Add putty cartoon settings to object.

        Putty cartoon uses beta-factor values to control thickness.
        Requires non-uniform beta-factors to avoid division by zero.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        object_name : str
            PyMOL object name

        Returns
        -------
        None
            Appends commands to lines

        Notes
        -----
        Only use with structures that have varying beta-factors.
        For uniform beta-factors (all 0.0), use _add_normal_cartoon().
        """
        lines.append(f"cartoon putty, {object_name}")
        lines.append(f"set cartoon_putty_scale_min, 1.0, {object_name}")
        lines.append(f"set cartoon_putty_scale_max, 3.0, {object_name}")

    @staticmethod
    def _add_normal_cartoon(lines: List[str], object_name: str) -> None:
        """
        Add normal cartoon settings to object.

        Normal cartoon has uniform thickness regardless of beta-factors.
        Use this for structures with uniform beta-factors (all 0.0).

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        object_name : str
            PyMOL object name

        Returns
        -------
        None
            Appends commands to lines

        Notes
        -----
        Normal cartoon avoids PyMOL putty division-by-zero warnings
        when all beta-factors are identical.
        """
        # Normal cartoon is already the default, no special settings needed
        # Just ensure it's not in putty mode
        lines.append(f"cartoon automatic, {object_name}")

    @staticmethod
    def _add_color_gradient(
        lines: List[str],
        color_name: str,
        color_hex: str,
        object_name: str
    ) -> None:
        """
        Add color definition and gradient to object.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        color_name : str
            PyMOL color name to define
        color_hex : str
            HEX color value
        object_name : str
            PyMOL object name

        Returns
        -------
        None
            Appends commands to lines
        """
        rgb = ColorConversionHelper.hex_to_rgb(color_hex)
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        lines.append(f"set_color color_{color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]")
        lines.append(f"spectrum b, color_{color_name} white, {object_name}, minimum=0.0, maximum=1.0")

    @staticmethod
    def _color_feature_sticks(
        lines: List[str],
        feature_obj: str,
        top_features: List[Dict[str, Any]],
        feature_colors: Dict[str, str],
        use_struct_color: bool,
        struct_color_hex: str
    ) -> None:
        """
        Color feature sticks by residue selection.

        Colors each feature's residues directly without requiring
        separate feature objects.

        Parameters
        ----------
        lines : List[str]
            Command list to append to
        feature_obj : str
            Feature object name
        top_features : List[Dict[str, Any]]
            Feature list
        feature_colors : Dict[str, str]
            Feature color mapping
        use_struct_color : bool
            If True, use struct_color_hex for all features
        struct_color_hex : str
            Structure color to use if use_struct_color=True

        Returns
        -------
        None
            Appends commands to lines
        """
        for idx, feature in enumerate(top_features):
            feature_name = feature.get("feature_name", f"feature_{idx}")
            residues = feature.get("residue_seqids", [])
            if not residues:
                continue

            if use_struct_color:
                color_hex = struct_color_hex
            else:
                color_hex = feature_colors.get(feature_name, "#808080")

            color_pymol = ColorConversionHelper.hex_to_pymol_rgb(color_hex)

            resi_selection = "+".join(map(str, residues))
            lines.append(f"select temp_color_sel, {feature_obj} and resi {resi_selection}")
            lines.append(f"set stick_color, {color_pymol}, temp_color_sel")
            lines.append(f"delete temp_color_sel")

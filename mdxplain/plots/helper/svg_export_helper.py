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
SVG export helper for editable text and large-plot export stability.

Provides utilities that keep SVG text editable and apply save-time
rendering heuristics for very large artists to reduce vector export cost.
"""

from pathlib import Path
import re
from typing import Dict, Iterator, List, Optional, Tuple, Union
from xml.sax.saxutils import escape
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class SvgExportHelper:
    """
    Helper for plot export configuration and optimizations.

    Centralizes save-time behavior for plot outputs:

    - keep SVG text editable (`svg.fonttype='none'`)
    - selectively rasterize very heavy artists in vector outputs
    - temporarily tune path rcParams for very large polylines

    Examples
    --------
    >>> SvgExportHelper.apply_svg_config_if_needed("svg")
    >>> fig.savefig("plot.svg", format="svg")
    >>> SvgExportHelper.save_figure_with_export_optimizations(
    ...     fig=fig, filepath="plot.svg", file_format="svg", dpi=300
    ... )
    """

    _VECTOR_FORMATS = {"svg", "pdf", "eps", "ps"}
    _SCATTER_RASTERIZE_THRESHOLD = 50_000
    _LINE_RASTERIZE_THRESHOLD = 80_000
    _POLY_RASTERIZE_THRESHOLD = 80_000
    _LINE_CHUNKSIZE_THRESHOLD = 100_000
    _LINE_CHUNKSIZE_VALUE = 20_000
    _CONSERVATIVE_SIMPLIFY_THRESHOLD = 0.05

    @staticmethod
    def configure_svg_text_editability() -> None:
        """
        Configure matplotlib for editable SVG text export.

        Sets matplotlib rcParams to prevent text-to-path conversion in
        SVG exports. After calling this method, all text elements in
        exported SVG files will be selectable and editable in SVG editors.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Modifies global matplotlib rcParams

        Notes
        -----
        - Sets 'svg.fonttype' to 'none' to preserve text as <text> elements
        - This is a global setting that affects all subsequent SVG exports
        - To restore default behavior, call restore_default_svg_settings()

        Examples
        --------
        >>> # Configure once at beginning
        >>> SvgExportHelper.configure_svg_text_editability()
        >>> fig.savefig("plot1.svg", format="svg")
        >>> fig2.savefig("plot2.svg", format="svg")
        """
        plt.rcParams['svg.fonttype'] = 'none'

    @staticmethod
    def restore_default_svg_settings() -> None:
        """
        Restore default matplotlib SVG export settings.

        Resets SVG font configuration to matplotlib defaults, which
        converts text to paths for better portability but reduced
        editability.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Modifies global matplotlib rcParams

        Examples
        --------
        >>> # Temporarily use editable text
        >>> SvgExportHelper.configure_svg_text_editability()
        >>> fig.savefig("editable.svg", format="svg")
        >>>
        >>> # Restore defaults for other exports
        >>> SvgExportHelper.restore_default_svg_settings()
        >>> fig.savefig("paths.svg", format="svg")
        """
        plt.rcParams['svg.fonttype'] = 'path'

    @staticmethod
    def get_current_svg_settings() -> Dict[str, str]:
        """
        Get current SVG export settings.

        Returns a dictionary of current SVG-related rcParams settings.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, str]
            Dictionary with current SVG settings

        Examples
        --------
        >>> settings = SvgExportHelper.get_current_svg_settings()
        >>> print(settings['svg.fonttype'])
        'none'
        """
        return {
            'svg.fonttype': plt.rcParams['svg.fonttype']
        }

    @staticmethod
    def apply_svg_config_if_needed(file_format: str) -> None:
        """
        Apply SVG configuration if format is SVG.

        Convenience method that checks the file format and applies
        editable text configuration only if the format is 'svg'.

        Parameters
        ----------
        file_format : str
            The export file format (e.g., 'svg', 'png', 'pdf')

        Returns
        -------
        None
            Conditionally modifies matplotlib rcParams

        Examples
        --------
        >>> # Automatically configure based on format
        >>> SvgExportHelper.apply_svg_config_if_needed("svg")  # Applies config
        >>> SvgExportHelper.apply_svg_config_if_needed("png")  # Does nothing
        """
        if file_format.lower() == 'svg':
            SvgExportHelper.configure_svg_text_editability()

    @staticmethod
    def save_figure_with_export_optimizations(
        fig: Figure,
        filepath: Union[str, Path],
        file_format: str,
        dpi: int,
        bbox_inches: str = "tight",
    ) -> None:
        """
        Save figure with export-time performance optimizations.

        Applies optional, human-equivalent optimizations for large plots:

        - editable text config for SVG output
        - selective rasterization of very heavy artists on vector backends
        - temporary path rcParam tuning for very large polylines

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filepath : str or Path
            Output file path.
        file_format : str
            Export format.
        dpi : int
            Output resolution in dots per inch.
        bbox_inches : str, default="tight"
            Bounding box mode forwarded to ``Figure.savefig``.

        Returns
        -------
        None
            Figure is written to disk.

        Notes
        -----
        Temporary artist and rcParam changes are restored in ``finally``.
        """
        SvgExportHelper.apply_svg_config_if_needed(file_format)

        old_path_rcparams = SvgExportHelper._apply_path_rendering_rcparams_if_needed(fig)
        rasterized_states = SvgExportHelper._apply_rasterization_if_needed(fig, file_format)
        try:
            fig.savefig(filepath, dpi=dpi, format=file_format, bbox_inches=bbox_inches)
            if file_format.lower() == "svg":
                SvgExportHelper._postprocess_svg_superscripts(Path(filepath))
        finally:
            SvgExportHelper._restore_rasterized_states(rasterized_states)
            SvgExportHelper._restore_path_rendering_rcparams(old_path_rcparams)

    @staticmethod
    def _postprocess_svg_superscripts(filepath: Path) -> None:
        """
        Rewrite matplotlib mathtext superscripts into editable SVG tspans.

        Matplotlib's editable SVG export serializes mathtext as many absolute-
        positioned tspans. That keeps text editable, but some SVG editors render
        those fragments differently than the on-screen/PNG output. This step
        collapses those blocks into a single SVG <text> element with a normal
        baseline text run plus a superscript tspan.

        Parameters
        ----------
        filepath : Path
            Path to the SVG file to process.

        Returns
        -------
        None
            Modifies the SVG file in-place.
        """
        if not filepath.exists():
            return

        svg_text = filepath.read_text(encoding="utf-8")
        updated = SvgExportHelper._rewrite_mathtext_groups(svg_text)
        if updated != svg_text:
            filepath.write_text(updated, encoding="utf-8")

    @staticmethod
    def _rewrite_mathtext_groups(svg_text: str) -> str:
        """
        Replace mathtext-exported feature labels with cleaner editable SVG text.

        Parameters
        ----------
        svg_text : str
            Raw SVG document content.

        Returns
        -------
        str
            SVG content with supported mathtext feature labels rewritten to
            editable SVG ``<tspan>`` superscripts.
        """
        group_pattern = re.compile(
            r"(?P<indent>[ \t]*)<!-- (?P<label>.*?) -->\s*"
            r"(?P<group><g(?P<g_attrs>[^>]*)>\s*<text>\s*(?P<body>.*?)\s*</text>\s*</g>)",
            re.DOTALL,
        )
        return group_pattern.sub(SvgExportHelper._replace_mathtext_group_match, svg_text)

    @staticmethod
    def _replace_mathtext_group_match(match: re.Match) -> str:
        """
        Rewrite one SVG mathtext group match if it contains mathtext.

        Parameters
        ----------
        match : re.Match
            Regex match for a comment + ``<g><text>...</text></g>`` block.

        Returns
        -------
        str
            Rewritten SVG fragment, or the original fragment if no mathtext
            superscript could be derived from the label comment.
        """
        label = match.group("label")
        if "$" not in label:
            return match.group(0)

        base_style = SvgExportHelper._extract_base_text_style(match.group("body"))
        if not base_style:
            return match.group(0)

        rewritten_runs = SvgExportHelper._build_svg_runs_from_mathtext_label(label)
        if rewritten_runs is None:
            return match.group(0)

        indent = match.group("indent")
        g_attrs = match.group("g_attrs")
        return (
            f"{indent}<!-- {label} -->\n"
            f"{indent}<g{g_attrs}>\n"
            f"{indent} <text style=\"{escape(base_style)}\">{rewritten_runs}</text>\n"
            f"{indent}</g>"
        )

    @staticmethod
    def _extract_base_text_style(body: str) -> Optional[str]:
        """
        Extract the style string from the first text-bearing tspan.

        Parameters
        ----------
        body : str
            Inner SVG markup of a ``<text>`` element.

        Returns
        -------
        Optional[str]
            Normalized style string to apply to the rewritten parent text
            element, or ``None`` if no tspan style could be found.
        """
        text_tspan_pattern = re.compile(
            r'<tspan[^>]*style="(?P<style>[^"]*)"[^>]*>(?P<text>.*?)</tspan>',
            re.DOTALL,
        )

        for tspan_match in text_tspan_pattern.finditer(body):
            tspan_text = tspan_match.group("text").strip()
            if not tspan_text:
                continue
            return SvgExportHelper._normalize_text_style(tspan_match.group("style"))

        return None

    @staticmethod
    def _normalize_text_style(style: str) -> str:
        """
        Normalize rewritten SVG text style for non-math editable text.

        Parameters
        ----------
        style : str
            Style string extracted from a mathtext-generated ``<tspan>``.

        Returns
        -------
        str
            Style string suitable for a normal SVG ``<text>`` node.
        """
        font_stack = (
            "'DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', "
            "'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', "
            "'Avant Garde', sans-serif"
        )

        if "font-family:" in style:
            style = re.sub(
                r"font-family:\s*'DejaVu Sans'",
                f"font-family: {font_stack}",
                style,
            )
        else:
            style = f"{style.rstrip(';')}; font-family: {font_stack}"
        return style

    @staticmethod
    def _build_svg_runs_from_mathtext_label(label: str) -> Optional[str]:
        """
        Convert a mathtext label comment into SVG text/tspan runs.

        Parameters
        ----------
        label : str
            Comment text emitted by matplotlib for a mathtext label.

        Returns
        -------
        Optional[str]
            SVG text fragment containing plain text plus rewritten mathtext
            superscripts, or ``None`` when no supported mathtext fragment was
            found.
        """
        superscript_style = "font-size: 70%; baseline-shift: super;"
        token_pattern = re.compile(
            r"(?:"
            r"\$\\(?:mathregular|mathrm)\{([^}]*)\}\^\{\\(?:mathregular|mathrm)\{([^}]*)\}\}\$"
            r"|"
            r"\$([^$]*?)\^\{([^}]*)\}\$"
            r")"
        )

        parts: List[str] = []
        last_end = 0

        for token in token_pattern.finditer(label):
            plain_prefix = label[last_end:token.start()]
            if plain_prefix:
                parts.append(escape(plain_prefix))

            math_base, math_super, plain_base, plain_super = token.groups()
            base_text = math_base if math_base is not None else plain_base
            superscript_text = math_super if math_super is not None else plain_super
            parts.append(escape(base_text))
            parts.append(
                f'<tspan style="{superscript_style}">'
                f"{escape(superscript_text)}</tspan>"
            )
            last_end = token.end()

        if not parts and last_end == 0:
            return None

        trailing = label[last_end:]
        if trailing:
            parts.append(escape(trailing))

        return "".join(parts)

    @staticmethod
    def _is_vector_format(file_format: str) -> bool:
        """
        Check whether output format is treated as vector backend.

        Parameters
        ----------
        file_format : str
            Requested export format.

        Returns
        -------
        bool
            True when format is one of SVG/PDF/EPS/PS.
        """
        return file_format.lower() in SvgExportHelper._VECTOR_FORMATS

    @staticmethod
    def _iter_data_artists(fig: Figure) -> Iterator[Artist]:
        """
        Iterate over data-carrying artists in all figure axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure containing axes and artists.

        Yields
        ------
        matplotlib.artist.Artist
            Line artists and collection artists from each axis.
        """
        for ax in fig.get_axes():
            for line in ax.lines:
                yield line
            for collection in ax.collections:
                yield collection

    @staticmethod
    def _estimate_artist_points(artist: Artist) -> int:
        """
        Estimate plotted point/vertex count for supported artists.

        Parameters
        ----------
        artist : matplotlib.artist.Artist
            Artist instance to inspect.

        Returns
        -------
        int
            Estimated number of points/vertices for threshold checks.
        """
        if isinstance(artist, Line2D):
            x_data = artist.get_xdata(orig=False)
            return int(len(x_data))

        if isinstance(artist, PathCollection):
            offsets = artist.get_offsets()
            shape = getattr(offsets, "shape", None)
            if shape is not None and len(shape) > 0:
                return int(shape[0])
            return int(len(offsets))

        if isinstance(artist, LineCollection):
            segments = artist.get_segments()
            return int(sum(len(seg) for seg in segments))

        if isinstance(artist, PolyCollection):
            paths = artist.get_paths()
            return int(sum(len(path.vertices) for path in paths))

        return 0

    @staticmethod
    def _should_rasterize_artist(artist: Artist, n_points: int) -> bool:
        """
        Decide if an artist should be rasterized by size heuristic.

        Parameters
        ----------
        artist : matplotlib.artist.Artist
            Candidate artist.
        n_points : int
            Estimated number of points/vertices in artist.

        Returns
        -------
        bool
            True when artist is above the configured rasterization threshold.
        """
        if isinstance(artist, PathCollection):
            return n_points >= SvgExportHelper._SCATTER_RASTERIZE_THRESHOLD
        if isinstance(artist, Line2D):
            return n_points >= SvgExportHelper._LINE_RASTERIZE_THRESHOLD
        if isinstance(artist, (LineCollection, PolyCollection)):
            return n_points >= SvgExportHelper._POLY_RASTERIZE_THRESHOLD
        return False

    @staticmethod
    def _apply_rasterization_if_needed(
        fig: Figure,
        file_format: str
    ) -> List[Tuple[Artist, Optional[bool]]]:
        """
        Rasterize heavy data artists for vector outputs only.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to inspect and modify before save.
        file_format : str
            Output format to check for vector rendering.

        Returns
        -------
        List[Tuple[matplotlib.artist.Artist, Optional[bool]]]
            Pairs of ``(artist, previous_rasterized_state)`` for restoration.
        """
        if not SvgExportHelper._is_vector_format(file_format):
            return []

        original_states: List[Tuple[Artist, Optional[bool]]] = []
        for artist in SvgExportHelper._iter_data_artists(fig):
            n_points = SvgExportHelper._estimate_artist_points(artist)
            if not SvgExportHelper._should_rasterize_artist(artist, n_points):
                continue
            original_states.append((artist, artist.get_rasterized()))
            artist.set_rasterized(True)
        return original_states

    @staticmethod
    def _restore_rasterized_states(
        rasterized_states: List[Tuple[Artist, Optional[bool]]]
    ) -> None:
        """
        Restore artist rasterization flags after save operation.

        Parameters
        ----------
        rasterized_states : List[Tuple[matplotlib.artist.Artist, Optional[bool]]]
            Previously captured ``(artist, state)`` entries.

        Returns
        -------
        None
            Restores in-memory artist flags.
        """
        for artist, old_state in rasterized_states:
            artist.set_rasterized(old_state)

    @staticmethod
    def _get_max_polyline_points(fig: Figure) -> int:
        """
        Get maximum point count among line-like artists in figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to inspect.

        Returns
        -------
        int
            Maximum estimated point count of ``Line2D``/``LineCollection``.
        """
        max_points = 0
        for artist in SvgExportHelper._iter_data_artists(fig):
            if isinstance(artist, (Line2D, LineCollection)):
                max_points = max(max_points, SvgExportHelper._estimate_artist_points(artist))
        return max_points

    @staticmethod
    def _apply_path_rendering_rcparams_if_needed(
        fig: Figure
    ) -> Optional[Tuple[int, bool, float]]:
        """
        Apply path-rendering rcParams when very large polylines are present.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to inspect.

        Returns
        -------
        Optional[Tuple[int, bool, float]]
            Captured old values for ``agg.path.chunksize``,
            ``path.simplify``, and ``path.simplify_threshold``.
            Returns None when no changes were necessary.
        """
        max_points = SvgExportHelper._get_max_polyline_points(fig)
        if max_points < SvgExportHelper._LINE_CHUNKSIZE_THRESHOLD:
            return None

        old_chunksize = int(plt.rcParams.get('agg.path.chunksize', 0))
        old_simplify = bool(plt.rcParams.get('path.simplify', True))
        old_simplify_threshold = float(plt.rcParams.get('path.simplify_threshold', 0.0))

        plt.rcParams['agg.path.chunksize'] = max(
            old_chunksize,
            SvgExportHelper._LINE_CHUNKSIZE_VALUE
        )
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = min(
            old_simplify_threshold,
            SvgExportHelper._CONSERVATIVE_SIMPLIFY_THRESHOLD
        )

        return old_chunksize, old_simplify, old_simplify_threshold

    @staticmethod
    def _restore_path_rendering_rcparams(
        old_values: Optional[Tuple[int, bool, float]]
    ) -> None:
        """
        Restore previously captured path-rendering rcParams.

        Parameters
        ----------
        old_values : Optional[Tuple[int, bool, float]]
            Tuple returned by
            ``_apply_path_rendering_rcparams_if_needed``.

        Returns
        -------
        None
            Restores global matplotlib rcParams when values are provided.
        """
        if old_values is None:
            return
        old_chunksize, old_simplify, old_simplify_threshold = old_values
        plt.rcParams['agg.path.chunksize'] = old_chunksize
        plt.rcParams['path.simplify'] = old_simplify
        plt.rcParams['path.simplify_threshold'] = old_simplify_threshold

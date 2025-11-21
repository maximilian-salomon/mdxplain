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
SVG export helper for editable text elements.

Configures matplotlib to export SVG files with editable text elements
instead of converting text to paths, allowing full editability in
SVG editors like Inkscape or Adobe Illustrator.
"""

import matplotlib.pyplot as plt
from typing import Dict


class SvgExportHelper:
    """
    Helper for configuring SVG export with editable text.

    Provides methods to configure matplotlib's SVG backend to preserve
    text as editable <text> elements instead of converting them to
    <path> elements, ensuring compatibility with SVG editors.

    Examples
    --------
    >>> # Before saving as SVG
    >>> SvgExportHelper.configure_svg_text_editability()
    >>> fig.savefig("plot.svg", format="svg")

    >>> # Use context manager for temporary configuration
    >>> with SvgExportHelper.svg_text_editable():
    ...     fig.savefig("plot.svg", format="svg")
    """

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

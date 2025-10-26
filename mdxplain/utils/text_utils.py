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
Text utilities for string manipulation operations.

Provides utility methods for text wrapping, cleaning, and formatting.
"""

import textwrap


class TextUtils:
    r"""
    Utility class for text manipulation operations.

    Provides static methods for text wrapping, newline removal, and
    other string formatting operations commonly needed across the codebase.

    Examples
    --------
    >>> from mdxplain.utils.text_utils import TextUtils
    >>> text = "Very Long\nLabel Text"
    >>> wrapped = TextUtils.wrap_text(text, max_length=10)
    >>> print(wrapped)
    Very Long
    Label Text
    """

    @staticmethod
    def wrap_text(text: str, max_length: int = 40) -> str:
        r"""
        Remove newlines and wrap text at word boundaries.

        Removes all \n characters from the text and wraps the resulting
        text at word boundaries to fit within max_length characters per line.

        Parameters
        ----------
        text : str
            Text that may contain \n characters
        max_length : int, default=40
            Maximum characters per line

        Returns
        -------
        str
            Cleaned and wrapped text

        Examples
        --------
        >>> text = "Very Long\nLabel Text"
        >>> wrapped = TextUtils.wrap_text(text, max_length=10)
        >>> print(wrapped)
        Very Long
        Label Text

        >>> text = "Contact"
        >>> wrapped = TextUtils.wrap_text(text, max_length=20)
        >>> print(wrapped)
        Contact

        Notes
        -----
        Uses textwrap.fill with break_long_words=False to ensure words
        are not broken mid-word, even if they exceed max_length.
        """
        clean_text = text.replace('\n', ' ')
        return textwrap.fill(clean_text, width=max_length, break_long_words=False)

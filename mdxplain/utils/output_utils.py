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
Output utilities for suppressing print statements.

Provides utility methods for controlling stdout behavior.
"""

from contextlib import contextmanager
import sys
import io


class OutputUtils:
    """
    Utility class for output management operations.

    Provides static methods for controlling stdout behavior, such as
    suppressing print output in specific code blocks.

    Examples
    --------
    >>> from mdxplain.utils.output_utils import OutputUtils
    >>> with OutputUtils.suppress_output():
    ...     print("This will not be shown")
    >>> print("This will be shown")
    This will be shown
    """

    @staticmethod
    @contextmanager
    def suppress_output():
        """
        Context manager to suppress all print output.

        Redirects stdout to a null buffer, effectively silencing all print
        statements within the context. Automatically restores original stdout
        even if exceptions occur within the context block.

        Yields
        ------
        None
            Context manager that temporarily redirects stdout

        Examples
        --------
        >>> from mdxplain.utils.output_utils import OutputUtils
        >>> with OutputUtils.suppress_output():
        ...     print("This will not be shown")
        ...     some_function_with_prints()
        >>> print("This will be shown")
        This will be shown

        Notes
        -----
        This only suppresses stdout (print statements). stderr output
        (warnings, errors) will still be visible. The original stdout is
        guaranteed to be restored even if an exception occurs within the
        context block due to the try/finally pattern.
        """
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = original_stdout

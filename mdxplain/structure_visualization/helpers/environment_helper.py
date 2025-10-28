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
Environment detection helper for structure visualization.

This module provides utilities for detecting the execution environment,
particularly for identifying Jupyter notebook contexts where certain
visualization methods may not work properly.
"""


class EnvironmentHelper:
    """
    Helper class for detecting execution environment.

    Provides methods to identify whether code is running in a Jupyter
    notebook, terminal, or other environment. This is useful for
    adapting visualization strategies.

    Examples
    --------
    >>> if EnvironmentHelper.is_jupyter_environment():
    ...     # Use Jupyter-compatible visualization
    ...     print("Running in Jupyter")
    ... else:
    ...     # Use standard terminal visualization
    ...     print("Running in terminal")
    """

    @staticmethod
    def is_jupyter_environment() -> bool:
        """
        Detect if code is running in Jupyter notebook.

        Checks for IPython kernel presence which indicates Jupyter
        notebook environment. Returns False for standard Python
        terminals and scripts.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if running in Jupyter notebook, False otherwise

        Examples
        --------
        >>> is_jupyter = EnvironmentHelper.is_jupyter_environment()
        >>> if is_jupyter:
        ...     print("Jupyter environment detected")
        """
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            return ipython is not None and hasattr(ipython, 'kernel')
        except ImportError:
            return False

    @staticmethod
    def is_pymol_available() -> bool:
        """
        Detect if PyMOL Python module is available.

        Attempts to import pymol module to check if PyMOL is
        installed and accessible in the current environment.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if pymol can be imported, False otherwise

        Examples
        --------
        >>> if EnvironmentHelper.is_pymol_available():
        ...     print("PyMOL is available")
        ... else:
        ...     print("PyMOL not found")
        """
        try:
            import pymol
            return True
        except ImportError:
            return False

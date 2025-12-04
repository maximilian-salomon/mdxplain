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
Progress control utilities for progress bar handling.

Provides a central toggle to enable or disable tqdm-based progress bars
without touching environment variables or external programs. The controller
is accessed by managers to keep progress handling consistent within this
process.
"""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm


T = TypeVar("T")


class ProgressUtils:
    """
    Toggle and wrap progress bar rendering.

    Uses an internal flag to enable or disable tqdm progress bars. This avoids
    side effects on the environment and allows runtime updates through the
    pipeline configuration.
    """

    _enabled: bool = True

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        """
        Enable or disable progress bars globally within this process.

        Parameters
        ----------
        enabled : bool
            Flag to control progress bar visibility.

        Returns
        -------
        None
            Progress bar state is updated for subsequent calls.

        Examples
        --------
        >>> ProgressUtils.set_enabled(False)
        """
        cls._enabled = enabled

    @classmethod
    def iterate(cls, iterable: Iterable[T], **kwargs) -> Iterator[T]:
        """
        Wrap an iterable with tqdm respecting the current enable flag.

        Parameters
        ----------
        iterable : Iterable
            Sequence or generator to wrap with a progress bar.
        kwargs
            Additional tqdm keyword arguments (e.g., desc, total, unit).

        Returns
        -------
        Iterator
            Iterator that yields from the input iterable, optionally showing a bar.

        Examples
        --------
        >>> for i in ProgressUtils.iterate(range(3), desc="Work"):
        ...     pass
        """
        return tqdm(iterable, disable=not cls._enabled, **kwargs)

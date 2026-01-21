# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex GPT 5.2 Codex High.
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
Performance configuration for PipelineManager.

This module defines a small configuration container that triggers a callback
whenever its values change. PipelineManager uses it to re-apply process-level
resource limits when the user edits pipeline.config.performance at runtime.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class PerformanceConfig:
    """
    Mutable performance settings that apply process limits on change.

    The configuration stores process-level tuning knobs such as CPU priority,
    I/O priority, CPU affinity, and BLAS/OpenMP thread limits. It does not
    apply any settings by itself; instead it calls a provided callback whenever
    one of its fields changes. This keeps the configuration lightweight while
    allowing PipelineManager to centralize the actual resource management.

    Fields
    ------
    auto_resource_limits : bool
        If True, compute a recommended CPU affinity based on reserve_cores and
        apply process-level limits when configuration changes.
    reserve_cores : int
        Number of CPU cores to keep free when auto_resource_limits is enabled.
    resource_nice : int or None
        POSIX nice value (or Windows priority mapping). None means "do not set".
    resource_io_priority : str or None
        I/O priority hint ("idle", "low", "normal", "high"). None means "do not set".
    resource_cpu_affinity : sequence of int or None
        Explicit CPU affinity list. When set, it overrides auto selection.
    auto_blas_thread_limit : bool
        If True, cap BLAS/OpenMP threads to the active CPU set size.
    """

    _FIELDS = {
        "auto_resource_limits",
        "reserve_cores",
        "resource_nice",
        "resource_io_priority",
        "resource_cpu_affinity",
        "auto_blas_thread_limit",
    }

    def __init__(
        self,
        defaults: Dict[str, Any],
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Create a performance configuration with explicit defaults.

        Parameters
        ----------
        defaults : dict
            Mapping from field name to initial value. All fields must be
            provided so that pipeline.config.performance has a complete set
            of tunables visible to the user.
        on_change : callable, optional
            Callback invoked after a value changes. PipelineManager passes a
            function that re-applies resource limits based on this config.

        Returns
        -------
        None
            Creates a configuration object with defaults applied.
        """
        missing = self._FIELDS - set(defaults)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing performance defaults: {missing_list}")

        object.__setattr__(self, "_on_change", on_change)
        object.__setattr__(self, "_suspend_apply", True)
        for key, value in defaults.items():
            setattr(self, key, value)
        object.__setattr__(self, "_suspend_apply", False)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set a configuration field and trigger the change callback.

        Parameters
        ----------
        name : str
            Name of the field to update.
        value : Any
            New value for the field.

        Returns
        -------
        None
            Updates the field and triggers the on_change callback.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self._FIELDS:
            raise AttributeError(f"Unknown performance setting: {name}")

        old = getattr(self, name, None)
        object.__setattr__(self, name, value)
        if old == value:
            return
        if not getattr(self, "_suspend_apply", False):
            self._notify_change()

    def update(self, **kwargs: Any) -> None:
        """
        Update multiple fields and apply once.

        This helper avoids repeated re-application when several settings are
        changed together. It validates field names and triggers the callback
        only once after all updates are applied.

        Parameters
        ----------
        **kwargs
            Field names and values to update.

        Returns
        -------
        None
            Applies all updates and triggers a single on_change callback.
        """
        changed = False
        object.__setattr__(self, "_suspend_apply", True)
        for key, value in kwargs.items():
            if key not in self._FIELDS:
                raise AttributeError(f"Unknown performance setting: {key}")
            if getattr(self, key) != value:
                changed = True
            setattr(self, key, value)
        object.__setattr__(self, "_suspend_apply", False)
        if changed:
            self._notify_change()

    def _notify_change(self) -> None:
        """
        Invoke the on_change callback unless updates are suspended.

        Returns
        -------
        None
            Calls the callback if present.
        """
        on_change = getattr(self, "_on_change", None)
        if on_change is not None and not getattr(self, "_suspend_apply", False):
            on_change()

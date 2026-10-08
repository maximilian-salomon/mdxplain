# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maeve Branwen Butler
# Created with assistance from GitHub Copilot (Claude Sonnet 5.0).
#
# Copyright (C) 2026 Maximilian Salomon and Maeve Branwen Butler
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
Pure JSON (de)serialization for SpecData - no mdxplain coupling.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union
from pathlib import Path

from ..entities.spec_data import SpecData


class SpecIOHelper:
    """Static helper converting between `SpecData` and plain JSON."""

    @staticmethod
    def to_dict(spec_data: SpecData) -> Dict[str, Any]:
        """
        Convert a `SpecData` instance to a plain JSON-serializable dict.

        Parameters
        ----------
        spec_data : SpecData
            The spec data container to serialize.

        Returns
        -------
        Dict[str, Any]
            Dict with ``"pipeline"``, one key per module in `SpecData.MODULES`,
            and ``"studies"`` (only if non-empty).
        """
        result: Dict[str, Any] = {"pipeline": spec_data.pipeline}
        for module_name in spec_data.MODULES:
            result[module_name] = getattr(spec_data, module_name)
        if spec_data.studies:
            result["studies"] = spec_data.studies
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SpecData:
        """
        Build a `SpecData` instance from a plain dict (the inverse of
        ``to_dict``).

        Parameters
        ----------
        data : Dict[str, Any]
            Dict as produced by ``to_dict``/``json.load``.

        Returns
        -------
        SpecData
            A new `SpecData` instance populated from ``data``.
        """
        spec_data = SpecData()
        spec_data.pipeline = data.get("pipeline", {"modifiers": []})
        for module_name in spec_data.MODULES:
            setattr(spec_data, module_name, data.get(module_name, {}))
        spec_data.studies = data.get("studies", {})
        return spec_data

    @staticmethod
    def write(spec_data: SpecData, path: Union[str, Path]) -> None:
        """
        Write a `SpecData` instance to a spec.json file.

        Parameters
        ----------
        spec_data : SpecData
            The spec data container to serialize.
        path : str or Path
            Destination file path.

        Returns
        -------
        None
            Non-JSON-safe config values (e.g. a raw ``numpy`` dtype logged
            from a real pipeline) are stringified via ``str()`` instead of
            raising.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SpecIOHelper.to_dict(spec_data), f, indent=2, default=str)
            f.write("\n")

    @staticmethod
    def read(path: Union[str, Path]) -> SpecData:
        """
        Read a spec.json file into a `SpecData` instance.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        SpecData
            A new `SpecData` instance populated from the file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SpecIOHelper.from_dict(data)

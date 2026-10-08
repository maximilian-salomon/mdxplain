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

"""Study-level facade over `SpecManager`, exposed as `SpecManager.studies`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from ..helper.spec_validator_helper import SpecValidatorHelper

if TYPE_CHECKING:
    from ..manager.spec_manager import SpecManager
    from ..entities.spec_data import SpecData


class SpecStudyService:
    """
    Facade for named study reference bundles, exposed as `SpecManager.studies`.

    A study groups instance references per domain:
    ``{study_name: {domain: {instance_name: "all" | [mod_name, ...]}}}``
    (see repo memory spec_module_design.md). ``"all"`` is resolved
    dynamically at use time rather than snapshotted, so it tracks modifiers
    added to the instance after the study was defined. ``add_to``/
    ``remove_from`` mutate single references within a study; ``delete``
    removes a whole study. Each mutating method re-validates
    ``spec_data.studies`` afterwards via
    ``SpecValidatorHelper.check_study_references``.
    """

    def __init__(self, manager: SpecManager, data: SpecData) -> None:
        """
        Initialize the facade over an existing manager/data pair.

        Parameters
        ----------
        manager : SpecManager
            The manager owning this facade's spec data (unused for now,
            kept for consistency with the other facades, e.g.
            `SpecInstancesService`).
        data : SpecData
            The spec data container to read from/mutate.

        Returns
        -------
        None
        """
        self._manager = manager
        self._data = data

    def add_new(self, study_name: str, exist_ok: bool = False) -> None:
        """
        Create a new, empty named study.

        Parameters
        ----------
        study_name : str
            Name of the study to create.
        exist_ok : bool, default=False
            If False, raise when a study of this name already exists.

        Returns
        -------
        None
        """
        self._data.add_new_study(study_name, exist_ok=exist_ok)

    def delete(self, study_name: str) -> None:
        """
        Delete an entire named study.

        Parameters
        ----------
        study_name : str
            Name of the study to delete.

        Returns
        -------
        None
        """
        self._data.remove_study(study_name)

    def add_to(
        self,
        study_name: str,
        domain: str,
        instance_name: str,
        mod_names: Union[str, List[str]] = "all",
    ) -> None:
        """
        Add or overwrite one instance reference within a study.

        Parameters
        ----------
        study_name : str
            Name of the study to add to (created if it does not exist yet).
        domain : str
            Domain/module name the referenced instance belongs to.
        instance_name : str
            Name of the referenced instance.
        mod_names : str or List[str], default="all"
            Either the literal ``"all"`` or an explicit list of
            ``mod_name``s to reference.

        Returns
        -------
        None
        """
        self._data.add_to_study(study_name, domain, instance_name, mod_names)
        SpecValidatorHelper.check_study_references(self._data)

    def update(
        self,
        study_name: str,
        domain: str,
        instance_name: str,
        mod_names: Union[str, List[str]],
    ) -> None:
        """
        Update an existing instance reference within a study.

        Parameters
        ----------
        study_name : str
            Name of the study to update.
        domain : str
            Domain/module name the referenced instance belongs to.
        instance_name : str
            Name of the referenced instance.
        mod_names : str or List[str]
            New value, either the literal ``"all"`` or an explicit list of
            ``mod_name``s.

        Returns
        -------
        None
        """
        self._data.update_study(study_name, domain, instance_name, mod_names)
        SpecValidatorHelper.check_study_references(self._data)

    def remove_from(
        self,
        study_name: str,
        domain: str,
        instance_name: Optional[str] = None,
    ) -> None:
        """
        Remove one instance reference, or a whole domain group, from a study.

        Parameters
        ----------
        study_name : str
            Name of the study to remove from.
        domain : str
            Domain/module name to remove from, or to remove an instance
            reference within.
        instance_name : str, optional
            If given, only this instance reference is removed. If None,
            the whole domain group is removed.

        Returns
        -------
        None
        """
        self._data.remove_from_study(study_name, domain, instance_name)

    def get(
        self,
        study_name: str,
        domain: Optional[str] = None,
        instance_name: Optional[str] = None,
    ) -> Any:
        """
        Read a study, a domain group within it, or a single reference value.

        Parameters
        ----------
        study_name : str
            Name of the study to read from.
        domain : str, optional
            If given, scope the read to this domain group.
        instance_name : str, optional
            If given (requires ``domain``), scope the read to this single
            instance's ``mod_names`` value.

        Returns
        -------
        Any
            The whole study dict, the domain group dict, or the single
            ``mod_names`` value, depending on how far ``domain``/
            ``instance_name`` are given.
        """
        return self._data.get_study(study_name, domain, instance_name)

    def list(self) -> List[str]:
        """
        List the names of every defined study.

        Returns
        -------
        List[str]
            Names of all studies currently defined.
        """
        return list(self._data.studies.keys())

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

from typing import Any, TYPE_CHECKING


from ..helper.spec_builder_helper import SpecBuilderHelper
from ..helper.spec_validator_helper import SpecValidatorHelper
from ...utils.operation_registry_utils import OperationRegistryUtils
from typing import Optional

if TYPE_CHECKING:
    from ..manager.spec_manager import SpecManager
    from ..entities.spec_data import SpecData


class SpecStudyService:
    """
    Facade for named study reference bundles, exposed as `SpecManager.studies`.

    A study groups instance references per module, e.g. ``{"study_name":
    {"clustering": [...], "feature_importance": [...]}}`` (see repo memory
    spec_module_design.md). Each method is a thin pass-through to the
    corresponding ``SpecData`` method; not all backing ``SpecData`` methods
    are implemented yet.
    """

    def __init__(self, manager: SpecManager, data: SpecData) -> None:
        """
        Initialize the facade over an existing manager/data pair.

        Parameters
        ----------
        manager : SpecManager
            The manager to delegate mutating operations to.
        data : SpecData
            The spec data container to read from.

        Returns
        -------
        None
        """
        self._manager = manager
        self._data = data

    def add_new(self, study_name, **kwargs):
        """
        Create a new, empty named study.

        Parameters
        ----------
        study_name : str
            Name of the study to create.
        **kwargs : Any
            Forwarded to `SpecData.add_new_study`.

        Returns
        -------
        None
        """
        self._data.add_new_study(study_name, **kwargs)

    def get(self, study_name, ref, **kwargs):
        """
        Retrieve a reference entry from a study.

        Parameters
        ----------
        study_name : str
            Name of the study to read from.
        ref : Any
            Reference identifying the entry to retrieve.
        **kwargs : Any
            Forwarded to `SpecData.get_study`.

        Returns
        -------
        Any
            The requested study entry, as returned by `SpecData.get_study`.
        """
        return self._data.get_study(study_name, ref, **kwargs)

    def list(self, **kwargs):
        """
        List all studies.

        Parameters
        ----------
        **kwargs : Any
            Currently unused.

        Returns
        -------
        Any
            Result of `SpecManager.list` restricted to the ``studies``
            domain.
        """
        study_domain = ["studies"]
        return self._manager.list(
            domains=study_domain,
        )

    def update(self, study_name, ref, **kwargs):
        """
        Update a reference entry within a study.

        Parameters
        ----------
        study_name : str
            Name of the study to update.
        ref : Any
            Reference identifying the entry to update.
        **kwargs : Any
            Forwarded to `SpecData.update_study`.

        Returns
        -------
        None
        """
        self._data.update_study(study_name, ref, **kwargs)

    def remove(self, study_name, ref, **kwargs):
        """
        Remove a reference entry from a study.

        Parameters
        ----------
        study_name : str
            Name of the study to remove from.
        ref : Any
            Reference identifying the entry to remove.
        **kwargs : Any
            Forwarded to `SpecData.remove_from_study`.

        Returns
        -------
        None
        """
        self._data.remove_from_study(study_name, ref, **kwargs)

    def add_to_study(self, study_name, ref, **kwargs):
        """
        Add a reference entry to an existing study.

        Parameters
        ----------
        study_name : str
            Name of the study to add to.
        ref : Any
            Reference to add.
        **kwargs : Any
            Forwarded to `SpecData.add_to_study`.

        Returns
        -------
        None
        """
        self._data.add_to_study(study_name, ref, **kwargs)
        # update

    def remove_from_study(self, study_name, ref, **kwargs):
        """
        Remove a reference entry from a study.

        Parameters
        ----------
        study_name : str
            Name of the study to remove from.
        ref : Any
            Reference identifying the entry to remove.
        **kwargs : Any
            Forwarded to `SpecData.remove_from_study`.

        Returns
        -------
        None
        """
        self._data.remove_from_study(study_name, ref, **kwargs)

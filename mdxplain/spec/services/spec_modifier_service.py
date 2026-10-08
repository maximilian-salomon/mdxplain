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

"""Modifier-level facade service, exposed via
`SpecInstancesService.modifiers`."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, List, Optional


from ..helper.spec_builder_helper import SpecBuilderHelper
from ..helper.spec_validator_helper import SpecValidatorHelper
from ...utils.operation_registry_utils import OperationRegistryUtils
from typing import Optional

if TYPE_CHECKING:
    from ..manager.spec_manager import SpecManager
    from ..entities.spec_data import SpecData


class SpecModifierService:
    """
    Facade for adding, reading, updating, removing, and reordering modifiers.

    A modifier is one logged operation call (e.g. one ``slice_traj``) that
    was recorded against an instance; this service operates on individual
    modifiers rather than the instance as a whole.
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

    def add(
        self,
        instance_name: str,
        operation_type: str,
        depends_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Append a new modifier to an instance, creating the instance if needed.

        Parameters
        ----------
        instance_name : str
            Name of the instance the modifier is attached to.
        operation_type : str
            Registered operation type of the call (e.g.
            ``"DBSCANAddService.dbscan"``), used to resolve the domain and
            registry entry.
        depends_on : List[str], optional
            Explicit dependency override. If omitted, dependencies are
            auto-resolved from the registry's tag declarations.
        **kwargs : Any
            Config values for the call; resolved against the operation's
            registered ``technical_params`` (required params enforced,
            missing optional params filled with their defaults).

        Returns
        -------
        str or None
            The new modifier's name (``mod_name``).
        """
        domain = OperationRegistryUtils.get_domain(operation_type)
        entry = OperationRegistryUtils.get_registry_entry(operation_type)
        params = SpecBuilderHelper.write_params(
            kwargs, entry.get("technical_params", {})
        )

        return self._manager.add(
            domain=domain,
            instance_name=instance_name,
            operation_type=operation_type,
            params=params,
            depends_on=depends_on,
        )

    def get(
        self,
        domain,
        instance_name,
        modifier_name,
    ):
        """
        Retrieve a single modifier from an instance by its name.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        modifier_name : str
            Name of the modifier to retrieve.

        Returns
        -------
        dict
            The modifier dictionary.
        """
        return self._data.get_modifier(domain, instance_name, modifier_name)

    def list(
        self,
        domain,
        instance_name,
    ):
        """
        List all modifiers of a single instance, in storage order.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance whose modifiers should be listed.

        Returns
        -------
        list
            Copy of the instance's modifier list.
        """
        return self._data.list_modifiers(domain, instance_name)

    def update(
        self,
        domain,
        instance_name,
        modifier_name,
        depends_on: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Update a modifier's config and/or explicit ``depends_on``.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        modifier_name : str
            Name of the modifier to update.
        depends_on : List[str], optional
            Explicit dependency override; if given, auto-resolution is
            disabled for this modifier until cleared via ``set_depends_on``.
        **kwargs : Any
            Config values merged into the modifier's existing config.

        Returns
        -------
        None
        """
        self._manager.update_modifier(
            domain,
            instance_name,
            modifier_name,
            config=kwargs or None,
            depends_on=depends_on,
        )

    def set_depends_on(
        self,
        domain,
        instance_name,
        modifier_name,
        depends_on,
    ):
        """
        Set or clear a modifier's explicit ``depends_on`` override.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        modifier_name : str
            Name of the modifier to update.
        depends_on : list or None
            New explicit dependency list, or None to clear the override
            and fall back to automatic resolution.

        Returns
        -------
        None
        """
        self._manager.set_modifier_depends_on(
            domain, instance_name, modifier_name, depends_on
        )

    def remove(
        self,
        domain,
        instance_name,
        modifier_name,
    ):
        """
        Remove a single modifier from an instance.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        modifier_name : str
            Name of the modifier to remove.

        Returns
        -------
        None
        """
        self._manager.remove_modifier(domain, instance_name, modifier_name)

    def reorder(
        self,
        domain,
        instance_name,
        modifier_name,
        new_position,
    ):
        """
        Move a modifier to ``new_position`` among its instance's siblings.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance holding the modifier.
        modifier_name : str
            Name of the modifier to reposition.
        new_position : int
            Target 0-based index among the instance's other modifiers.

        Returns
        -------
        None
        """
        self._manager.reorder_modifier(
            domain, instance_name, modifier_name, new_position
        )

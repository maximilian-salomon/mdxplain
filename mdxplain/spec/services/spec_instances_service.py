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

"""Instance-level facade service, exposed via `SpecManager.instances`."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .spec_modifier_service import SpecModifierService
from ..helper.spec_builder_helper import SpecBuilderHelper
from ..helper.spec_validator_helper import SpecValidatorHelper
from ...utils.operation_registry_utils import OperationRegistryUtils

if TYPE_CHECKING:
    from ..manager.spec_manager import SpecManager
    from ..entities.spec_data import SpecData


class SpecInstancesService:
    """
    Facade for adding, reading, updating, and removing instances.

    An instance is a named, addressable resource within one domain (e.g.
    a named ``feature_selection``), holding an ordered ``modifiers`` list.
    Exposes `.modifiers` for modifier-level access on the same instance.
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

    @property
    def modifiers(self) -> SpecModifierService:
        """Modifier-level facade for the same manager/data pair."""
        return SpecModifierService(self._manager, self._data)

    def add(
        self,
        instance_name: Optional[str] = None,
        *,
        domain: Optional[str] = None,
        operation_type: Optional[str] = None,
        depends_on: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Add a new, empty instance or create one via its first operation call.

        Two modes: pass ``domain`` alone to create a bare empty instance
        (``{"modifiers": []}``), or pass ``operation_type`` (and optionally
        ``**kwargs`` config) to create the instance via its actual creating
        call, delegating to ``self.modifiers.add``.

        Parameters
        ----------
        instance_name : str, optional
            Explicit instance name. If omitted, a name is auto-resolved
            from ``operation_type``/``kwargs`` (see
            ``SpecBuilderHelper.resolve_instance_name``).
        domain : str, optional
            Domain/module name for the new instance. Required if
            ``operation_type`` is not given; if both are given, must match
            the domain implied by ``operation_type``.
        operation_type : str, optional
            Registered operation type of the creating call (e.g.
            ``"FeatureSelectorManager.create"``).
        depends_on : list, optional
            Explicit dependency override for the creating modifier.
        **kwargs : Any
            Config values for the creating call.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If neither ``domain`` nor ``operation_type`` is given, or if
            ``domain`` does not match the domain implied by
            ``operation_type``.
        """
        if operation_type is None:
            if domain is None:
                raise ValueError(
                    "Either operation_type or domain must be specified."
                )
            return self._manager.add(domain=domain, instance_name=instance_name)
        else:
            if domain is not None:
                calc_domain = OperationRegistryUtils.get_domain(operation_type)
                if calc_domain != domain:
                    raise ValueError(
                        f"Specified domain '{domain}' does not match the "
                        f"calculated domain '{calc_domain}' for "
                        f"operation_type '{operation_type}'."
                    )
            domain = OperationRegistryUtils.get_domain(operation_type)
            instance_name = SpecBuilderHelper.resolve_instance_name(
                target=getattr(self._data, domain),
                method_name=operation_type.rsplit(".", 1)[-1],
                name=instance_name,
                instance_params=OperationRegistryUtils.get_instance_params(
                    domain
                ),
                kwargs=kwargs,
            )
            return self.modifiers.add(
                instance_name=instance_name,
                operation_type=operation_type,
                depends_on=depends_on,
                **kwargs,
            )

    def get(
        self,
        domain,
        instance_name,
    ):
        """
        Retrieve a single instance by name from a domain.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance to retrieve.

        Returns
        -------
        dict
            The instance dictionary, as returned by `SpecManager.get`.
        """
        return self._manager.get(
            domain=domain,
            name=instance_name,
        )

    def list(
        self,
    ):
        """
        List instances across every registered module domain.

        Returns
        -------
        Any
            Result of `SpecManager.list` for all module domains.
        """
        module_domains = OperationRegistryUtils.get_all_domains()
        return self._manager.list(domains=module_domains)

    def update(
        self,
        domain,
        instance_name,
        depends_on: Optional[list] = None,
        **kwargs,
    ):
        """
        Update the creating (first) modifier of an existing instance.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance to update.
        depends_on : list, optional
            Explicit dependency override for the creating modifier.
        **kwargs : Any
            Config values merged into the creating modifier's config.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the instance does not exist or has no modifiers.
        """
        instance = self._data.get_instance(instance_name, domain)
        if instance is None or not instance["modifiers"]:
            raise ValueError(
                f"Instance '{instance_name}' not found in domain '{domain}'."
            )
        # the first modifier is the creating call, the only one an
        # instance "owns"
        self._manager.update_modifier(
            domain,
            instance_name,
            instance["modifiers"][0]["mod_name"],
            config=kwargs or None,
            depends_on=depends_on,
        )

    def remove(
        self,
        domain,
        instance_name,
    ):
        """
        Remove an instance together with all of its modifiers.

        Parameters
        ----------
        domain : str
            Domain/module name where the instance is located.
        instance_name : str
            Name of the instance to remove.

        Returns
        -------
        None
        """
        self._manager.remove_instance(domain, instance_name)

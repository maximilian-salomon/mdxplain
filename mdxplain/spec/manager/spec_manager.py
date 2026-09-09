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

"""Spec manager: single public access point for spec.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from ..entities.spec_data import SpecData
from ..helper.log_graph_helper import LogGraphHelper
from ..helper.spec_builder_helper import SpecBuilderHelper
from ..helper.spec_io_helper import SpecIOHelper
from ..helper.spec_registry_helper import SpecRegistryHelper
from ..helper.spec_validator_helper import SpecValidatorHelper


class SpecManager:
    """
    Manager owning one SpecData instance for the one pipeline/project spec.json.

    Examples
    --------
    Manual construction:

    >>> spec_manager = SpecManager()
    >>> spec_manager.add("FeatureSelectorManager.create", name="key_features")
    'key_features'
    >>> spec_manager.add(
    ...     "DBSCANAddService.dbscan", selection_name="key_features", eps=0.5
    ... )
    'dbscan'

    From a real pipeline:

    >>> spec_manager = SpecManager()
    >>> spec_manager.read_from_pipeline(pipeline)
    >>> spec_manager.write_json("spec.json")
    """

    def __init__(self) -> None:
        """
        Initialize the spec manager with its own, empty SpecData instance.

        Returns
        -------
        None
        """
        self.data = SpecData()

    def add(
        self, operation_type: str, instance_name: Optional[str] = None, **kwargs: Any
    ) -> str:
        """
        Add one logged operation's config to the spec, registry-driven.

        Parameters
        ----------
        operation_type : str
            Registered operation type name (``"ClassName.method_name"``, or
            an explicit override like ``"pipeline_init"``).
        instance_name : str, optional
            Explicit instance name override. Only meaningful for resource
            types with no name-carrying technical_param (e.g. trajectory,
            plots, analysis) - named resources (clustering, decomposition,
            ...) resolve their name from ``kwargs`` instead (see
            ``SpecBuilderHelper.resolve_name``). Named ``instance_name``
            (not ``name``) to avoid colliding with the many operations whose
            own technical_param is literally called ``"name"``.
        **kwargs : Any
            The call's config values.

        Returns
        -------
        str
            The resolved instance name (``"pipeline"`` for the `pipeline`
            singleton domain).

        Raises
        ------
        KeyError
            If ``operation_type`` is not registered.
        ValueError
            If a required param is missing, or a cross-reference points to a
            not-(yet)-existing instance.
        """
        entry = SpecRegistryHelper.get_entry(operation_type)
        SpecValidatorHelper.check_required_params(operation_type, entry, kwargs)

        domain = entry["domain"]
        if domain == "pipeline":
            SpecBuilderHelper.add_singleton_modifier(self.data.pipeline, operation_type, kwargs)
            return "pipeline"

        SpecValidatorHelper.check_cross_references(self.data, entry, domain, kwargs)

        target = getattr(self.data, domain)
        method_name = operation_type.rsplit(".", 1)[-1]
        instance_params = SpecRegistryHelper.get_instance_params(domain)
        resolved_name = SpecBuilderHelper.resolve_name(
            target, method_name, instance_name, instance_params, kwargs
        )
        SpecBuilderHelper.add_modifier(target, resolved_name, operation_type, kwargs)
        return resolved_name

    def read_from_pipeline(self, pipeline: Any) -> None:
        """
        Populate ``self.data`` from a real, already-executed pipeline.

        Replays every logged operation (in dependency order) as an ``add()``
        call.

        Parameters
        ----------
        pipeline : PipelineManager
            A real pipeline instance whose ``pipeline.data.log`` to read.

        Returns
        -------
        None
            Mutates ``self.data`` in place.
        """
        log = pipeline.data.log
        graph = LogGraphHelper.build_graph(log)
        for layer in LogGraphHelper.layers(graph):
            for entry_id in layer:
                operation = log["operations"][entry_id]
                self.add(operation["type"], **operation["config"])

    def write_json(self, path: Union[str, Path]) -> None:
        """
        Write ``self.data`` to a spec.json file.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Returns
        -------
        None
        """
        SpecIOHelper.write(self.data, path)

    def read_json(self, path: Union[str, Path]) -> None:
        """
        Replace ``self.data`` with the contents of a spec.json file.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        None
            Mutates ``self.data`` in place.
        """
        self.data = SpecIOHelper.read(path)

    def validate(self) -> None:
        """
        Validate every modifier in ``self.data`` (required params + cross-refs).

        For specs loaded via ``read_json()`` (which does not go through
        ``add()``, so never gets incrementally validated).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any modifier is missing a required param, or references a
            not-existing instance.
        """
        for module_name in self.data.MODULES:
            for instance in getattr(self.data, module_name).values():
                for modifier in instance["modifiers"]:
                    entry = SpecRegistryHelper.get_entry(modifier["type"])
                    SpecValidatorHelper.check_required_params(
                        modifier["type"], entry, modifier["config"]
                    )
                    SpecValidatorHelper.check_cross_references(
                        self.data, entry, module_name, modifier["config"]
                    )
        for modifier in self.data.pipeline["modifiers"]:
            entry = SpecRegistryHelper.get_entry(modifier["type"])
            SpecValidatorHelper.check_required_params(modifier["type"], entry, modifier["config"])

    def add_study(self, study_name: str, module: str, ref_name: str) -> None:
        """
        Add a reference to an existing instance under a named study, grouped by module.

        Parameters
        ----------
        study_name : str
            Name of the study (created if not yet present).
        module : str
            Module the referenced instance belongs to (one of `SpecData.MODULES`).
        ref_name : str
            Name of an existing instance in that module.

        Returns
        -------
        None
            Mutates ``self.data.studies`` in place.

        Raises
        ------
        ValueError
            If ``module`` is not a known module, or ``ref_name`` does not
            exist in it.
        """
        if module not in self.data.MODULES:
            raise ValueError(f"Unknown module '{module}'")
        if ref_name not in getattr(self.data, module):
            raise ValueError(f"'{ref_name}' does not exist in module '{module}'")

        study = self.data.studies.setdefault(study_name, {})
        refs = study.setdefault(module, [])
        if ref_name not in refs:
            refs.append(ref_name)

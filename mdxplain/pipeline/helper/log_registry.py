# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
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
Manual registry describing which manager/service methods are logged as
pipeline operations, and which of their parameters are captured.

This is a first implementation slice: only the operations exercised by
``spec/tests/test.ipynb`` (cell 2) are registered. Registry entries are
looked up by ``AutoInjectProxy``/``LogHelper`` via the (owner class, method
name) dispatch key - see ``mdxplain/pipeline/helper/log_helper.py``.

The registry *data* lives in ``log_registry.json`` (same directory) - this
module only loads that file and resolves it into the runtime shape used by
the rest of the pipeline logging code. Keeping the data in JSON means it can
be validated/extended by the ``dev_scripts/check_log_registry.py`` script
without touching Python code.

JSON shape (arbitrary nesting depth, grouped by pipeline domain and, within
it, however closely the folder/class structure needs to be mirrored for
readability):

.. code-block:: json

    {
      "<domain>": {
        "<any nesting of grouping keys>": {
          "<json_key>": {
            "operation_type": "...",
            "method_name": "...",
            "class": "ClassName",
            "module": "module.path",
            "emits_tags": [],
            "affected_by_tags": [],
            "technical_params": [...],
            "gui_param_info": {}
          }
        }
      }
    }

``<domain>`` is a readable grouping key (e.g. "trajectory", "feature") and
does NOT necessarily match a single owner class - some domains (e.g.
"feature_selector") mix a Manager class (``create``/``select``) with a
Service class reached through a nested ``.add`` property
(``contacts`` -> ``ContactsSelectionService``). ``class``/``module`` are
therefore always given explicitly per method entry.

Below a domain, any number of intermediate grouping keys is allowed (e.g.
to mirror ``analysis.structure.rmsf.per_atom_service``) - they carry no
dispatch meaning and exist purely for human readability. A dict node is a
leaf entry once it has both ``module`` and ``class`` keys; every other dict
node is recursed into (see ``_iter_leaf_entries``).

``<json_key>`` only needs to be unique *within* its parent object (plain
dict key requirement) - it carries no dispatch meaning. By convention it
equals ``method_name``, and is only qualified as ``"ClassName.method_name"``
when two different classes in the same domain happen to share a method
name (common with generic stat method names like "mean"/"std"/"max" across
several ``*AnalysisService``/``*ReduceService`` classes).

``method_name`` is the exact method name captured at runtime by
``AutoInjectProxy``/``LoggingServiceProxy`` - for callable services reached
through a ``.add.<name>(...)`` property (e.g. ``ContactsSelectionService``),
this is the *property name* (e.g. ``"contacts"``), NOT the literal
``__call__`` dunder - see ``LoggingServiceProxy.__call__``, which now logs
under the access name it was obtained through instead of the hardcoded
``"__call__"``. Since such a name does not exist as a literal attribute on
the owner class, ``_resolve_dispatch`` falls back to the class's
``__call__`` method whenever it needs a real callable (e.g. for signature
introspection) - see ``_resolve_dispatch``. Together with ``class``, this
is what forms the dispatch identity (``(owner_class, method_name)``) used
by ``get_operation_type`` - it must be unique per (module, class) pair, but
MAY repeat across different classes (even within the same domain), which is
exactly why it is decoupled from the JSON key above.

After loading, each registry entry (flattened, keyed by ``operation_type``)
is a dict with the following fields (see plan.md, Abschnitt 4, for the full
design):

- ``dispatch``: (owner class, method name) - reverse of operation_type lookup.
- ``emits_tags``: List[str] - structural tags this operation sets.
- ``affected_by_tags``: List[str] - structural tags this operation depends on.
- ``technical_params``: List[str] - parameter names captured in the log entry.
- ``gui_param_info``: Dict[str, dict] - GUI metadata for parameters (empty
  for this first slice - not yet populated).

NOTE: emits_tags/affected_by_tags are left mostly empty in this first slice.
Only the trajectory-slicing-style dependency chain is not yet exercised by
cell 2, so structural tag dependencies are deferred to a follow-up pass.
"""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Type

_REGISTRY_JSON_PATH = Path(__file__).parent / "log_registry.json"


class LogRegistry:
    """
    Namespace for the manual operation registry (build/lookup/register).

    All state lives inside the ``_build_registry`` cache (an
    ``lru_cache``-wrapped staticmethod), never as bare module-level
    variables. Use the public staticmethods below to interact with it.
    """

    @staticmethod
    def _iter_leaf_entries(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Recursively yield every leaf entry (dict with "module" and "class") in ``node``."""
        if "module" in node and "class" in node:
            yield node
            return
        for value in node.values():
            if isinstance(value, dict):
                yield from LogRegistry._iter_leaf_entries(value)

    @staticmethod
    def _resolve_dispatch(
        operation_type: str,
        module_path: str,
        class_name: str,
        method_name: str,
    ) -> Tuple[Type, str]:
        """
        Resolve a ``module``/``class``/``method_name`` registry entry to a
        real ``(owner_class, method_name)`` dispatch tuple.

        Parameters
        ----------
        operation_type : str
            Name of the operation being resolved, only used for error context.
        module_path : str
            Dotted module path, e.g.
            ``"mdxplain.trajectory.manager.trajectory_manager"``.
        class_name : str
            Name of the class within ``module_path``, e.g. ``"TrajectoryManager"``.
        method_name : str
            Method name as captured at runtime (may be an access-name alias
            for ``__call__`` on callable services - not necessarily a literal
            attribute of the resolved class, see module docstring).

        Returns
        -------
        Tuple[Type, str]
            The resolved ``(owner_class, method_name)`` tuple. ``method_name``
            is returned unchanged (not resolved to ``__call__``) since it must
            match exactly what is captured at runtime for dispatch lookup.

        Raises
        ------
        ValueError
            If the module cannot be imported or has no such class.
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"log_registry.json: cannot import module '{module_path}' "
                f"for dispatch of '{operation_type}'"
            ) from exc

        try:
            owner = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"log_registry.json: module '{module_path}' has no class "
                f"'{class_name}' (dispatch of '{operation_type}')"
            ) from exc

        return owner, method_name

    @staticmethod
    @lru_cache(maxsize=1)
    def _build_registry() -> Dict[str, Dict[str, Any]]:
        """
        Build and cache the base operation registry (built once, on first use).

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The single shared registry dict, keyed by operation_type. The same
            dict instance is returned on every call (cached), so
            ``LogRegistry.register_operation`` can mutate it in-place to add
            further entries (e.g. "pipeline_init", registered by
            ``LogHelper.log_pipeline_init``).
        """
        with open(_REGISTRY_JSON_PATH, "r", encoding="utf-8") as f:
            raw_domains = json.load(f)

        registry: Dict[str, Dict[str, Any]] = {}
        for domain_node in raw_domains.values():
            for raw_entry in LogRegistry._iter_leaf_entries(domain_node):
                method_name = raw_entry["method_name"]
                operation_type = f"{raw_entry['class']}.{method_name}"
                registry[operation_type] = {
                    "emits_tags": raw_entry["emits_tags"],
                    "affected_by_tags": raw_entry["affected_by_tags"],
                    "technical_params": raw_entry["technical_params"],
                    "dispatch": LogRegistry._resolve_dispatch(
                        operation_type,
                        raw_entry["module"],
                        raw_entry["class"],
                        method_name,
                    ),
                }
        return registry

    @staticmethod
    def get_operation_type(owner: Type, method_name: str) -> str | None:
        """
        Look up the operation_type registered for a given owner class + method.

        Parameters
        ----------
        owner : Type
            The class that owns the method (Manager or Service class).
        method_name : str
            The name of the called method (use "__call__" for callable services).

        Returns
        -------
        str or None
            The registered operation_type, or None if this method is not logged.
        """
        dispatch = (owner, method_name)
        for operation_type, entry in LogRegistry._build_registry().items():
            if entry["dispatch"] == dispatch:
                return operation_type
        return None

    @staticmethod
    def get_registry_entry(operation_type: str) -> Dict[str, Any]:
        """
        Look up the full registry entry for a given operation_type.

        Parameters
        ----------
        operation_type : str
            Registered operation type name.

        Returns
        -------
        Dict[str, Any]
            The registry entry for ``operation_type``.
        """
        return LogRegistry._build_registry()[operation_type]

    @staticmethod
    def register_operation(operation_type: str, entry: Dict[str, Any]) -> None:
        """
        Register an operation type from a module that cannot be imported here.

        Some owner classes (e.g. ``PipelineManager``) cannot be imported by this
        module without creating a circular import, since they themselves import
        (transitively) from this module. Such modules call this function after
        their class definition instead of adding an entry directly.

        Parameters
        ----------
        operation_type : str
            Name of the operation type to register.
        entry : Dict[str, Any]
            Registry entry, see ``_build_registry`` for the expected shape.

        Returns
        -------
        None
            Updates the shared registry in-place.
        """
        LogRegistry._build_registry()[operation_type] = entry

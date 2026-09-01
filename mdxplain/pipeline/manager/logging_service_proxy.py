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
Logging proxy for service objects returned by manager properties.

This module provides the ``LoggingServiceProxy`` class used by
``AutoInjectProxy`` (see
``mdxplain/pipeline/manager/auto_inject_proxy.py``) to wrap service
objects (e.g. ``FeatureSelectorAddService``) so that their public calls are
logged the same way as directly injected manager methods.
"""
from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, TYPE_CHECKING

from ..helper.log_helper import LogHelper

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData


class LoggingServiceProxy:
    """
    Recursively wraps a service object (e.g. FeatureSelectorAddService) so
    that its methods - and any nested sub-service properties (e.g.
    ClusterAddService.dpa returning a callable DPAAddService) - log their
    calls via LogHelper, exactly like directly injected manager methods.

    Only wraps calls; does not perform any pipeline_data injection itself
    (the wrapped service already received pipeline_data via its constructor,
    see AutoInjectProxy.__getattr__).
    """

    _PRIMITIVE_TYPES = (str, int, float, bool, bytes, list, dict, tuple, set, type(None))

    def __init__(
        self, service: Any, pipeline_data: "PipelineData", access_name: str | None = None
    ):
        """
        Initialize the logging service proxy.

        Parameters
        ----------
        service : Any
            The service instance to wrap with call logging.
        pipeline_data : PipelineData
            The PipelineData instance to log operations into.
        access_name : str, optional
            Name of the property/attribute this ``service`` was obtained
            through (e.g. ``"contacts"`` for ``add.contacts``). Used instead
            of the literal ``"__call__"`` when the wrapped service itself is
            invoked directly, so the registry can dispatch on a readable name
            that matches how the operation is actually reached in the
            pipeline API, rather than the dunder method - see
            ``mdxplain/pipeline/helper/log_registry.py``.

        Returns
        -------
        None
            Initializes the proxy with service and pipeline_data references.
        """
        object.__setattr__(self, "_service", service)
        object.__setattr__(self, "_pipeline_data", pipeline_data)
        object.__setattr__(self, "_access_name", access_name)

    def __getattr__(self, name: str) -> Any:
        """
        Return a logging-wrapped method, a plain value, or a nested proxy.

        Parameters
        ----------
        name : str
            Name of the attribute being accessed on the wrapped service.

        Returns
        -------
        Any
            Logging-wrapped callable, primitive value, or nested
            ``LoggingServiceProxy`` depending on the attribute type.
        """
        attr = getattr(self._service, name)

        if inspect.ismethod(attr) or inspect.isfunction(attr):
            owner = type(self._service)
            sig = inspect.signature(attr)

            @wraps(attr)
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                LogHelper.log_call(self._pipeline_data, owner, name, sig, args, kwargs)
                return result

            return wrapper

        if isinstance(attr, self._PRIMITIVE_TYPES):
            return attr

        # Non-method, non-primitive attribute (e.g. a nested sub-service
        # returned by a property): wrap recursively so its own methods /
        # __call__ also get logged. Passing `name` as access_name lets the
        # nested proxy log a readable name (e.g. "contacts") instead of
        # "__call__" if that nested service is itself called directly.
        return LoggingServiceProxy(attr, self._pipeline_data, access_name=name)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the wrapped service directly and log the call.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the wrapped service.
        **kwargs : Any
            Keyword arguments forwarded to the wrapped service.

        Returns
        -------
        Any
            The wrapped service's return value.
        """
        owner = type(self._service)
        method_name = self._access_name or "__call__"
        sig = inspect.signature(self._service.__call__)
        result = self._service(*args, **kwargs)
        LogHelper.log_call(self._pipeline_data, owner, method_name, sig, args, kwargs)
        return result

    def __repr__(self) -> str:
        """
        Return string representation of the proxy.

        Returns
        -------
        str
            String representation showing the wrapped service type.
        """
        return f"LoggingServiceProxy({type(self._service).__name__})"

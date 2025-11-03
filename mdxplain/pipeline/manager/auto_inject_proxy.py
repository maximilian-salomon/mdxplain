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
Auto-injection proxy for pipeline managers.

This module provides the AutoInjectProxy class that automatically detects
whether a method needs PipelineData and injects it as the first parameter
when needed. This eliminates the need for manual wrapper methods while
maintaining clean manager APIs.
"""
from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Union, Tuple, Dict, TYPE_CHECKING
from inspect import Parameter

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData


class AutoInjectProxy:
    """
    Proxy that automatically injects PipelineData into manager methods.

    This proxy analyzes method signatures to determine if a method expects
    PipelineData as any parameter. If so, it automatically injects
    the PipelineData instance at the correct position when called through the proxy.

    Methods that don't expect PipelineData are called directly without
    modification, allowing for both stateful pipeline operations and
    stateless utility functions within the same manager.

    Examples
    --------
    >>> from mdxplain.pipeline import PipelineManager
    >>> pipeline = PipelineManager()
    >>>
    >>> # Methods with pipeline_data parameter get auto-injection
    >>> pipeline.trajectory.load_trajectories('../data')  # pipeline_data injected
    >>>
    >>> # Utility methods without pipeline_data work normally
    >>> valid = pipeline.trajectory.validate_selection('res CA')  # no injection
    """

    def __init__(self, manager: Any, pipeline_data: PipelineData):
        """
        Initialize the auto-injection proxy.

        Parameters
        ----------
        manager : object
            The manager instance to wrap with auto-injection
        pipeline_data : PipelineData
            The PipelineData instance to inject into methods

        Returns
        -------
        None
            Initializes the proxy with manager and data references
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
        self._service_properties = {}  # Store service properties per instance
        self._initialize_manager_method_proxies()

    def __dir__(self):
        """
        Return list of available attributes for IDE autocompletion.

        Returns
        -------
        list
            List of public method names from the wrapped manager
        """
        return [attr for attr in dir(self._manager) if not attr.startswith("_")]

    def _initialize_manager_method_proxies(self):
        """
        Initialize proxy methods for all manager methods with auto-injection support.

        This method creates static proxy methods for each callable attribute of the
        wrapped manager. Each proxy method automatically handles pipeline_data
        injection when needed, enabling both IDE autocompletion and seamless
        auto-injection functionality.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Initializes the proxy with manager and data references
        """
        for attr_name in dir(self._manager):
            # Skip private attributes
            if attr_name.startswith("_"):
                continue

            # Get attribute from class to check if it's a property
            attr_from_class = getattr(type(self._manager), attr_name, None)
            
            # Handle properties (service properties)
            if isinstance(attr_from_class, property):
                # Store service property in instance instead of setting on class
                self._service_properties[attr_name] = attr_from_class
            else:
                # Get attribute from instance for callable check
                attr = getattr(self._manager, attr_name)
                # Only wrap callable attributes (methods)
                if callable(attr):
                    # Check if the current manager has this as a property
                    manager_class_attr = getattr(type(self._manager), attr_name, None)
                    if isinstance(manager_class_attr, property):
                        # This is a property from the manager - don't wrap as method
                        continue
                    
                    # This is a method from the manager - wrap it
                    wrapper_method = self._create_auto_inject_wrapper(attr)
                    # Only set if not already a property on the AutoInjectProxy class
                    if not isinstance(getattr(self.__class__, attr_name, None), property):
                        # Set method directly on instance to override any class-level attributes
                        setattr(self, attr_name, wrapper_method)

    def _create_auto_inject_wrapper(self, method: callable) -> callable:
        """
        Create auto-injection wrapper for a manager method.

        Analyzes the method signature to determine if pipeline_data injection
        is needed. If the method expects a pipeline_data parameter, creates
        a wrapper that automatically injects it at the correct position.
        Also validates that users don't manually pass pipeline_data in Pipeline mode.

        Parameters
        ----------
        method : callable
            The original method to wrap

        Returns
        -------
        callable
            Wrapped method with intelligent auto-injection or original method unchanged
        """
        sig = inspect.signature(method)
        if "pipeline_data" not in sig.parameters:
            # Method doesn't need pipeline_data - return as-is
            return method

        # Analyze pipeline_data parameter position and type
        params = list(sig.parameters.items())
        pipeline_data_param = None
        pipeline_data_index = None

        for i, (name, param) in enumerate(params):
            if name == "pipeline_data":
                pipeline_data_param = param
                pipeline_data_index = i
                break

        def injected_method(*args, **kwargs):
            # Validate: User should never pass pipeline_data manually in Pipeline mode
            self._validate_no_manual_pipeline_data(args, kwargs)

            # Intelligent injection based on parameter type and position
            args, kwargs = self._inject_pipeline_data_intelligently(
                pipeline_data_param, pipeline_data_index, args, kwargs
            )

            # Call method as bound method - self is already included
            return method(*args, **kwargs)

        # Copy all metadata from original method to preserve function introspection
        injected_method.__name__ = getattr(method, '__name__', 'injected_method')
        injected_method.__qualname__ = getattr(method, '__qualname__', 'injected_method')
        injected_method.__module__ = getattr(method, '__module__', None)
        injected_method.__doc__ = getattr(method, '__doc__', None)
        injected_method.__annotations__ = getattr(method, '__annotations__', {})
        injected_method.__signature__ = sig

        # Copy any additional attributes from the original method
        if hasattr(method, '__dict__'):
            injected_method.__dict__.update(method.__dict__)

        return injected_method

    def _validate_no_manual_pipeline_data(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """
        Validate that user doesn't manually pass pipeline_data in Pipeline mode.

        Pipeline mode rule: User should NEVER pass pipeline_data manually,
        whether as positional or keyword argument. AutoInjectProxy handles
        injection automatically.

        Parameters
        ----------
        args : tuple
            Positional arguments from user
        kwargs : dict
            Keyword arguments from user

        Raises
        ------
        ValueError
            If user tries to pass pipeline_data manually
        """
        # Check 1: No PipelineData objects in positional arguments
        for i, arg in enumerate(args):
            if self._looks_like_pipeline_data(arg):
                raise ValueError(
                    f"Pipeline mode: Don't pass PipelineData as positional argument #{i}! "
                    "Use standalone manager for manual data control: "
                    "TrajectoryManager().method(pipeline_data, ...)"
                )

        # Check 2: No 'pipeline_data' in keyword arguments
        if "pipeline_data" in kwargs:
            raise ValueError(
                "Pipeline mode: Don't specify 'pipeline_data' keyword - it's auto-injected! "
                "Use standalone manager for manual control: "
                "TrajectoryManager().method(..., pipeline_data=your_data)"
            )

    def _looks_like_pipeline_data(self, obj: PipelineData) -> bool:
        """
        Check if object looks like a pipeline-related data instance.

        Uses conservative heuristics to identify pipeline-related objects:
        1. Class name indicates pipeline data type
        2. Object has all pipeline data attributes (complete PipelineData)

        This conservative approach minimizes false positives while catching
        the most common cases where users mistakenly pass pipeline data manually.

        Parameters
        ----------
        obj : PipelineData
            PipelineData to check

        Returns
        -------
        bool
            True if object looks like pipeline-related data, False otherwise
        """
        # Duck typing: check if object is a PipelineData-like instance => If it walks like a duck...
        # Level 1: Class name indicates pipeline data type
        class_name = type(obj).__name__
        pipeline_class_names = [
            "PipelineData",  # Main pipeline container
            "TrajectoryData",  # Trajectory-specific data
            "FeatureData",  # Feature-specific data
            "ClusterData",  # Clustering-specific data
            "DecompositionData",  # Decomposition-specific data
            "FeatureSelectorData",  # Selected features for analysis
        ]
        if any(name in class_name for name in pipeline_class_names):
            return True

        # Level 2: Complete pipeline data
        pipeline_attrs = [
            "trajectory_data",
            "feature_data",
            "cluster_data",
            "decomposition_data",
            "selected_feature_data",
        ]
        if all(hasattr(obj, attr) for attr in pipeline_attrs):
            return True

        return False

    def _create_service_property(self, original_property: property) -> property:
        """
        Create property wrapper that handles different service initialization patterns.
        
        Supports two service patterns:
        1. Manager-based services: Need (manager, pipeline_data) - e.g. FeatureAddService
        2. Direct services: Already initialized with data - e.g. FeatureAnalysisService
        
        Intelligently detects the pattern by inspecting the service constructor signature
        and handles each appropriately.
        
        Parameters
        ----------
        original_property : property
            The original property from the manager
            
        Returns
        -------
        property
            New property that handles service creation correctly
        """
        def service_getter(proxy_self):
            # Get the service from the original property
            service = original_property.fget(proxy_self._manager)
            
            # Check what parameters the service constructor expects
            try:
                sig = inspect.signature(service.__class__.__init__)
                params = set(sig.parameters.keys()) - {'self'}  # Parameter names ohne 'self'
                
                # Determine what to pass based on what the service expects
                if 'manager' in params and 'pipeline_data' in params:
                    # Service wants both
                    return service.__class__(proxy_self._manager, proxy_self._pipeline_data)
                elif 'pipeline_data' in params:
                    # Service only wants pipeline_data
                    return service.__class__(pipeline_data=proxy_self._pipeline_data)
                elif 'manager' in params:
                    # Service only wants manager (unusual but possible)
                    return service.__class__(manager=proxy_self._manager)
                else:
                    # Service doesn't need injection - return as is
                    return service
                    
            except Exception:
                # If we can't inspect, return as is (safe fallback)
                return service
        
        # Create new property with proper docstring
        return property(service_getter, doc=original_property.__doc__)

    def _inject_pipeline_data_intelligently(
        self, pipeline_data_param: Parameter, pipeline_data_index: int, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Inject pipeline_data at the correct position based on parameter definition.

        Handles all parameter types: positional, keyword-only, with defaults, etc.
        Always injects pipeline_data regardless of how it's defined in the original method.

        Parameters
        ----------
        pipeline_data_param : Parameter
            The pipeline_data parameter object from signature inspection
        pipeline_data_index : int
            Index position of pipeline_data parameter
        args : tuple
            User-provided positional arguments
        kwargs : dict
            User-provided keyword arguments

        Returns
        -------
        Tuple[tuple, dict]
            Modified (args, kwargs) with pipeline_data injected
        """
        if pipeline_data_param.kind == Parameter.KEYWORD_ONLY:
            # Keyword-only parameter: ALWAYS inject as keyword
            kwargs["pipeline_data"] = self._pipeline_data

        elif pipeline_data_param.kind == Parameter.VAR_KEYWORD:
            # **kwargs parameter: shouldn't happen, but handle gracefully
            kwargs["pipeline_data"] = self._pipeline_data

        elif len(args) > 0:
            # User provided positional arguments: inject pipeline_data at correct position
            # pipeline_data_index is the parameter position (bound method has no 'self' in signature)
            injection_index = pipeline_data_index
            args_list = list(args)
            args_list.insert(injection_index, self._pipeline_data)
            args = tuple(args_list)

        else:
            # Not enough positional args: inject as keyword
            kwargs["pipeline_data"] = self._pipeline_data

        return args, kwargs

    def __getattr__(self, name: str) -> Any:
        """
        Fallback method for dynamic attribute access.

        Parameters
        ----------
        name : str
            Name of the method or attribute being accessed

        Returns
        -------
        Any
            Method wrapper with auto-injection if needed, or original attribute
        """
        # Check if this is a stored service property first
        if name in self._service_properties:
            original_property = self._service_properties[name]
            service = original_property.fget(self._manager)
            
            # Handle service creation correctly
            try:
                sig = inspect.signature(service.__class__.__init__)
                params = set(sig.parameters.keys()) - {'self'}
                
                if 'manager' in params and 'pipeline_data' in params:
                    return service.__class__(self._manager, self._pipeline_data)
                elif 'pipeline_data' in params:
                    return service.__class__(self._pipeline_data)
                elif 'manager' in params:
                    return service.__class__(self._manager)
                else:
                    return service
            except Exception:
                return service
        
        # Default behavior for regular attributes and methods
        attr = getattr(self._manager, name)
        if not callable(attr):
            return attr
        return self._create_auto_inject_wrapper(attr)

    def __repr__(self) -> str:
        """
        Return string representation of the proxy.

        Returns
        -------
        str
            String representation showing wrapped manager type
        """
        return f"AutoInjectProxy({type(self._manager).__name__})"

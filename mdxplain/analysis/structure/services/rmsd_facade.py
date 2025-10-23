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

"""Facade exposing RMSD variant services."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsd_variant_service import RMSDVariantService


class RMSDFacade:
    """Provide access to RMSD variants with a shared pipeline context.

    The facade lazily creates :class:`RMSDVariantService` instances for the
    ``mean``, ``median``, and ``mad`` metrics. Each variant reuses the pipeline
    configuration supplied via :class:`PipelineData`, ensuring consistent
    behaviour across helper methods.

    Examples
    --------
    >>> facade = RMSDFacade(pipeline_data)
    >>> isinstance(facade.mean, RMSDVariantService)
    True
    """

    def __init__(self, pipeline_data: PipelineData | None) -> None:
        """
        Initialise the facade with shared pipeline data.

        Ensures that pipeline data is available before any RMSD computations are
        requested. The facade caches created variants for reuse.

        Parameters
        ----------
        pipeline_data : PipelineData | None
            Pipeline context injected by the analysis manager.

        Returns
        -------
        None
            The initializer returns ``None``.

        Examples
        --------
        >>> facade = RMSDFacade(pipeline_data)
        >>> isinstance(facade, RMSDFacade)
        True
        """

        if pipeline_data is None:
            raise ValueError("RMSDFacade requires pipeline_data")

        self._pipeline_data = pipeline_data
        self._variants: Dict[str, RMSDVariantService] = {}

    @property
    def mean(self) -> RMSDVariantService:
        """
        Access the mean RMSD variant.

        
        Provides the RMSD service that aggregates atom-wise deviations via the
        arithmetic mean. This is the default variant for most workflows.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSDVariantService
            Mean RMSD service exposing helpers such as :meth:`to_reference`,
            :meth:`frame_to_frame`, :meth:`window_frame_to_start`, and
            :meth:`window_frame_to_frame`.

        Examples
        --------
        >>> facade = RMSDFacade(pipeline_data)
        >>> facade.mean.to_reference(reference_traj=0, reference_frame=0)  # doctest: +ELLIPSIS
        {...}
        """

        return self._get_variant("mean")

    @property
    def median(self) -> RMSDVariantService:
        """
        Access the median RMSD variant.

        Provides the RMSD service that aggregates atom-wise deviations using the
        statistical median, offering robustness against outliers.

        Parameters
        ----------
        None
            This property does not accept parameters.
            
        Returns
        -------
        RMSDVariantService
            Median RMSD service exposing the same helper methods as the mean
            variant.

        Examples
        --------
        >>> facade = RMSDFacade(pipeline_data)
        >>> facade.median.frame_to_frame(lag=2)  # doctest: +ELLIPSIS
        {...}
        """

        return self._get_variant("median")

    @property
    def mad(self) -> RMSDVariantService:
        """
        Access the MAD RMSD variant.
    
        Provides the RMSD service based on the median absolute deviation (MAD),
        delivering outlier-resistant RMSD values.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSDVariantService
            MAD RMSD service exposing helpers such as :meth:`to_reference` and
            the window-based calculations.

        Examples
        --------
        >>> facade = RMSDFacade(pipeline_data)
        >>> facade.mad.window_frame_to_start(window_size=5, stride=2)  # doctest: +ELLIPSIS
        {...}
        """

        return self._get_variant("mad")

    def _get_variant(self, metric: str) -> RMSDVariantService:
        """
        Return a cached variant for the requested metric.
        Retrieves an existing variant service from the cache or instantiates a
        new one when the metric is requested for the first time.

        Parameters
        ----------
        metric : str
            Name of the robust RMSD metric (``"mean"``, ``"median"``, or
            ``"mad"``).

        Returns
        -------
        RMSDVariantService
            Variant service configured for the requested metric.

        Examples
        --------
        >>> facade = RMSDFacade(pipeline_data)
        >>> isinstance(facade._get_variant("mean"), RMSDVariantService)
        True
        """

        if metric not in self._variants:
            self._variants[metric] = RMSDVariantService(self._pipeline_data, metric)
        return self._variants[metric]

    def to_reference(
        self,
        reference_traj: int = 0,
        reference_frame: int = 0,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
    ) -> Dict[str, np.ndarray]:
        """
        Calculate RMSD values against a reference frame.
        
        Delegates to :meth:`RMSDVariantService.to_reference` on the mean variant
        to compute RMSD values for the selected trajectories.

        Parameters
        ----------
        reference_traj : int, optional
            Index of the trajectory containing the reference frame. Defaults to
            ``0``.
        reference_frame : int, optional
            Frame index within the reference trajectory. Defaults to ``0``.
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to process. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string forwarded to the calculation. Defaults
            to ``"all"``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to RMSD arrays.

        Examples
        --------
        >>> facade = RMSDFacade(pipeline_data)
        >>> facade.to_reference(reference_traj=0, reference_frame=0)  # doctest: +ELLIPSIS
        {...}
        """

        return self.mean.to_reference(
            reference_traj=reference_traj,
            reference_frame=reference_frame,
            traj_selection=traj_selection,
            atom_selection=atom_selection,
        )

    def __call__(
        self,
        reference_traj: int = 0,
        reference_frame: int = 0,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
    ) -> Dict[str, np.ndarray]:
        """
        Shortcut for :meth:`to_reference` on the mean RMSD variant.

        Calls :meth:`to_reference` with the provided parameters. This method is
        a convenience wrapper so that users can execute the mean RMSD workflow
        via call syntax.
        
        Parameters
        ----------
        reference_traj : int, optional
            Index of the trajectory containing the reference frame. Defaults to
            ``0``.
        reference_frame : int, optional
            Frame index within the reference trajectory. Defaults to ``0``.
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to process. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string forwarded to the calculation. Defaults
            to ``"all"``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names to RMSD arrays.

        Examples
        --------
        >>> facade = RMSDFacade(pipeline_data)
        >>> facade(reference_traj=0, reference_frame=0)  # doctest: +ELLIPSIS
        {...}
        """

        return self.to_reference(
            reference_traj=reference_traj,
            reference_frame=reference_frame,
            traj_selection=traj_selection,
            atom_selection=atom_selection,
        )

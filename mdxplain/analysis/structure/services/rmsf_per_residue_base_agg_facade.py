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

"""Base facade for per-residue RMSF aggregation services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

from .rmsf_per_residue_service import RMSFPerResidueService


class BaseRMSFPerResidueAggFacade(ABC):
    """Abstract base facade for per-residue RMSF aggregation services.

    Provides common delegation methods for per-residue RMSF calculations that
    forward to the concrete service. Subclasses must implement the service
    property to provide their specific aggregation configuration.

    Returns
    -------
    BaseRMSFPerResidueAggFacade
        Abstract base facade exposing delegation methods.

    Examples
    --------
    >>> # Subclasses provide concrete service
    >>> facade = RMSFPerResidueMeanAggFacade(pipeline_data, "mean")
    >>> result = facade.to_mean_reference()  # doctest: +ELLIPSIS
    {...}
    """

    @property
    @abstractmethod
    def service(self) -> RMSFPerResidueService:
        """Return the configured per-residue RMSF service.

        Subclasses must implement this property to provide a service instance
        with the appropriate metric and aggregator configuration.

        Parameters
        ----------
        None
            The property accepts no parameters.

        Returns
        -------
        RMSFPerResidueService
            Service configured with specific metric and aggregator.

        Examples
        --------
        >>> facade = RMSFPerResidueMeanAggFacade(pipeline_data, "mean")
        >>> isinstance(facade.service, RMSFPerResidueService)
        True
        """

    def to_mean_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: Union[int, str, List[Union[int, str]], "all"] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate per-residue RMSF values for the mean reference structure.

        Delegates to the configured service's to_mean_reference method while
        forwarding all trajectory/atom selections and reference configuration.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string forwarded to the calculation. Defaults
            to ``"all"``.
        cross_trajectory : bool, optional
            Combine all selected trajectories into a single RMSF profile when
            ``True``. Defaults to ``False``.
        reference_traj_selection : Union[int, str, list[Union[int, str]], 'all'] | None, optional
            Trajectories used to compute the mean reference structure. Uses
            ``traj_selection`` when ``None``. Defaults to ``None``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> facade = RMSFPerResidueMeanAggFacade(pipeline_data, "mean")
        >>> facade.to_mean_reference(traj_selection="all")  # doctest: +ELLIPSIS
        {...}
        """
        return self.service.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
            reference_traj_selection=reference_traj_selection,
        )

    def to_median_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: Union[int, str, List[Union[int, str]], "all"] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate per-residue RMSF values for the median reference structure.

        Delegates to the configured service's to_median_reference method while
        forwarding all trajectory/atom selections and reference configuration.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string forwarded to the calculation. Defaults
            to ``"all"``.
        cross_trajectory : bool, optional
            Combine all selected trajectories into a single RMSF profile when
            ``True``. Defaults to ``False``.
        reference_traj_selection : Union[int, str, list[Union[int, str]], 'all'] | None, optional
            Trajectories used to compute the median reference structure. Uses
            ``traj_selection`` when ``None``. Defaults to ``None``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> facade = RMSFPerResidueMeanAggFacade(pipeline_data, "mean")
        >>> facade.to_median_reference(traj_selection="all")  # doctest: +ELLIPSIS
        {...}
        """
        return self.service.to_median_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
            reference_traj_selection=reference_traj_selection,
        )

    def __call__(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
        reference_traj_selection: Union[int, str, List[Union[int, str]], "all"] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Shortcut for to_mean_reference on the configured service.

        Executes to_mean_reference with the provided parameters for a concise
        per-residue RMSF workflow using the mean reference structure.

        Parameters
        ----------
        traj_selection : Union[int, str, list[Union[int, str]], 'all'], optional
            Selection describing which trajectories to analyse. Defaults to
            ``"all"``.
        atom_selection : str, optional
            MDTraj atom selection string forwarded to the calculation. Defaults
            to ``"all"``.
        cross_trajectory : bool, optional
            Combine all selected trajectories into a single RMSF profile when
            ``True``. Defaults to ``False``.
        reference_traj_selection : Union[int, str, list[Union[int, str]], 'all'] | None, optional
            Trajectories used to compute the mean reference structure. Uses
            ``traj_selection`` when ``None``. Defaults to ``None``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-residue RMSF arrays.

        Examples
        --------
        >>> facade = RMSFPerResidueMeanAggFacade(pipeline_data, "mean")
        >>> facade(traj_selection="all", atom_selection="backbone")  # doctest: +ELLIPSIS
        {...}
        """
        return self.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
            reference_traj_selection=reference_traj_selection,
        )

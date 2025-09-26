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

"""Facade exposing RMSF variant services."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from mdxplain.pipeline.entities.pipeline_data import PipelineData

from .rmsf_variant_facade import RMSFVariantFacade
from .rmsf_per_atom_service import RMSFPerAtomService
from .rmsf_per_residue_aggregation_facade import RMSFPerResidueAggregationFacade


class RMSFFacade:
    """Provide access to RMSF variants with a shared pipeline context.

    The facade lazily constructs :class:`RMSFVariantFacade` instances for the
    ``mean``, ``median``, and ``mad`` deviation metrics. All variants reuse the
    supplied pipeline configuration, ensuring consistent behaviour across the
    per-atom and per-residue helper services.

    Examples
    --------
    >>> facade = RMSFFacade(pipeline_data)
    >>> isinstance(facade.mean, RMSFVariantFacade)
    True
    """

    def __init__(self, pipeline_data: PipelineData | None) -> None:
        """
        Initialise the facade with shared pipeline data.

        Ensures that the facade operates only after pipeline data injection. The
        created variants are cached for reuse.

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
        >>> facade = RMSFFacade(pipeline_data)
        >>> isinstance(facade, RMSFFacade)
        True
        """

        if pipeline_data is None:
            raise ValueError("RMSFFacade requires pipeline_data")

        self._pipeline_data = pipeline_data

    @property
    def mean(self) -> RMSFVariantFacade:
        """
        Access the mean RMSF variant.

        Provides the RMSF variant that uses the mean for RMSF calculations.
        So Root Mean Square Fluctuation.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFVariantFacade
            Mean RMSF variant exposing per-atom and per-residue helpers.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> isinstance(facade.mean.per_atom, RMSFPerAtomService)
        True
        """

        return self._get_variant("mean")

    @property
    def median(self) -> RMSFVariantFacade:
        """
        Access the median RMSF variant.

        Provides the RMSF variant that uses the median for robust RMSF
        calculations. So Root median Square Fluctuation.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFVariantFacade
            Median RMSF variant exposing per-atom and per-residue helpers.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> facade.median.per_residue.with_median_aggregation.to_median_reference()
        Traceback (most recent call last):
        ...
        """

        return self._get_variant("median")

    @property
    def mad(self) -> RMSFVariantFacade:
        """Access the MAD RMSF variant.

        Provides the RMSF variant based on the median absolute deviation (MAD)
        for outlier-resistant RMSF calculations.
        So MAD Flucatuation.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFVariantFacade
            MAD RMSF variant exposing per-atom and per-residue helpers.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> isinstance(facade.mad.per_atom, RMSFPerAtomService)
        True
        """

        return self._get_variant("mad")

    def _get_variant(self, metric: str) -> RMSFVariantFacade:
        """Return a cached variant for the requested metric.

        Retrieves an existing variant from the cache or creates a new one.

        Parameters
        ----------
        metric : str
            Name of the RMSF metric (``"mean"``, ``"median"``, or ``"mad"``).

        Returns
        -------
        RMSFVariantFacade
            Variant facade configured for the requested metric.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> isinstance(facade._get_variant("mean"), RMSFVariantFacade)
        True
        """
        return RMSFVariantFacade(self._pipeline_data, metric)

    @property
    def per_atom(self) -> RMSFPerAtomService:
        """
        Access the per-atom RMSF helper for the mean metric.

        Returns the per-atom service supplied by the classical mean RMSF variant.

        Returns
        -------
        RMSFPerAtomService
            Per-atom RMSF service providing ``to_*_reference`` helpers.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> facade.per_atom.to_mean_reference()  # doctest: +ELLIPSIS
        {...}
        """

        return self.mean.per_atom

    @property
    def per_residue(self) -> RMSFPerResidueAggregationFacade:
        """Access the per-residue RMSF helper for the mean metric.

        Returns the residue aggregation facade supplied by the classical mean RMSF
        variant.

        Parameters
        ----------
        None
            This property does not accept parameters.

        Returns
        -------
        RMSFPerResidueAggregationFacade
            Residue-level RMSF helper exposing aggregation variants.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> facade.per_residue.with_mean_aggregation.to_mean_reference()  # doctest: +ELLIPSIS
        {...}
        """

        return self.mean.per_residue

    def to_mean_reference(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Calculate per-atom RMSF values for the mean reference structure.

        Delegates to :meth:`RMSFPerAtomService.to_mean_reference` on the mean
        variant while forwarding the trajectory/atom selections and optional
        cross-trajectory settings.

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

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-atom RMSF arrays.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> facade.to_mean_reference(traj_selection="all", atom_selection="all")  # doctest: +ELLIPSIS
        {...}
        """

        return self.per_atom.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
        )

    def __call__(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        atom_selection: str = "all",
        cross_trajectory: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Shortcut for :meth:`to_mean_reference` on the mean RMSF variant.

        Executes :meth:`to_mean_reference` with the provided parameters for a
        concise per-atom mean RMSF workflow.
        So basically calculates the per-atom RMSF relative to the mean
        structure.

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

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of trajectory names – or ``"combined"`` when
            ``cross_trajectory`` is ``True`` – to per-atom RMSF arrays.

        Examples
        --------
        >>> facade = RMSFFacade(pipeline_data)
        >>> facade(traj_selection="all", atom_selection="backbone")  # doctest: +ELLIPSIS
        {...}
        """

        return self.to_mean_reference(
            traj_selection=traj_selection,
            atom_selection=atom_selection,
            cross_trajectory=cross_trajectory,
        )

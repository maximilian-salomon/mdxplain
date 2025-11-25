Feature Computation
===================

.. todo: Provide guidance on how to access arrays for external analysis.
    Optionally link to possible downstream applications.

mdxplain provides six feature types for molecular dynamics analysis:

- Distances
- Contacts
- Torsions
- DSSP secondary structure
- Solvent accessible surface area (SASA)
- Atomic coordinates

It is recommended to save computed features to disk for large datasets
to avoid recomputation. (:doc:`Saving and Loading <saving_and_loading>`)

.. code:: python
    
    # Residue-residue distances (closest heavy-atom pairs)
    pipeline.feature.add.distances(excluded_neighbors=1)

    # Binary contacts (interaction indicator)
    pipeline.feature.add.contacts(cutoff=4.5)

    # Backbone and side-chain torsion angles
    pipeline.feature.add.torsions()  # phi, psi, omega, chi1-4

    # Secondary structure assignment
    pipeline.feature.add.dssp()

    # Solvent accessible surface area
    pipeline.feature.add.sasa()

    # Atomic coordinates (xyz positions)
    pipeline.feature.add.coordinates()
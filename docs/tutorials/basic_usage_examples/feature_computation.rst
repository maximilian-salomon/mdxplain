Feature Computation
===================

mdxplain provides six feature types for molecular dynamics analysis:

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
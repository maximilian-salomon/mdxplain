Feature Reduction
=================

Statistical Feature Filtering
-----------------------------

Feature reduction applies statistical criteria to filter features AFTER they
have been computed but BEFORE analysis. This reduces dimensionality by keeping
only features that meet specific biological or statistical criteria.

So it does not change the feature-data but is specific for this selection.

What is Feature Reduction?
--------------------------

- Computed features are analyzed for statistical properties (frequency, variability, transitions)
- Features failing threshold criteria are removed from the feature set
- Reduces noise, computational cost, and focuses analysis on relevant features
- Two approaches: inline reduction (``.with_xxx_reduction()``) or pre-reduction (``feature.reduce_data()``)
- Pre reduction add this permanent to feature-data. It creates a new permanant data matrix.
- Post reduction is specified to this specific selection and does not create a specific matrix, but keep the indices.
- Available metrics depend on feature type. Concrete methods listed here: :doc:`Feature Sattistics <feature_statistics>`

When to Use Feature Reduction
-----------------------------

- **Too many features**: Thousands of distances/contacts overwhelm analysis
- **Focus on variability**: Only analyze features that actually change (cv, std, variance, range)
- **Focus on stability**: Only analyze persistent interactions (frequency, stability)
- **Focus on dynamics**: Only analyze features showing transitions
- **Multi-trajectory**: Ensure features exist across all systems (common_denominator)

Two Reduction Approaches
------------------------

Approach 1: Inline Reduction (during selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Apply reduction while selecting features
    # Advantage: Specific criteria per selection
    pipeline.feature_selector.add.contacts.with_frequency_reduction(
        "stable_contacts", "resid 100-200",
        threshold_min=0.7  # Only contacts formed in >70% of frames
    )

Approach 2: Pre-Reduction + ``use_reduced=True``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Step 1: Globally reduce features across all trajectories
    pipeline.feature.reduce_data(
        feature_type="distances",
        metric="cv",  # Coefficient of variation
        threshold_min=0.1,  # Only distances with CV > 0.1 (variable distances)
        cross_trajectory=True  # Feature must pass in ALL trajectories
    )

    # Step 2: Use pre-reduced features in selection
    pipeline.feature_selector.add.distances(
        "variable_distances", "all",
        use_reduced=True  # Uses reduced data from Step 1
    )

Reduction Methods by Feature Type
---------------------------------

Contacts
^^^^^^^^

Binary interaction indicators

- ``with_frequency_reduction()``: Contact formation frequency (0.0-1.0)

    - **Use**: Find stable interactions (high freq) or transient contacts (low freq)
    - **Example**: ``threshold_min=0.8`` → contacts formed in >80% of frames

- ``with_stability_reduction()``: Contact persistence over time

    - **Use**: Identify consistently maintained interactions vs. flickering contacts

- ``with_transitions_reduction()``: Contact formation/breaking events

    - **Use**: Find dynamic regions with frequent state changes

Distances
^^^^^^^^^

Continuous separation values

- ``with_cv_reduction()``: Coefficient of variation (std/mean)

    - **Use**: Normalized variability, independent of absolute distance scale
    - **Example**: ``threshold_min=0.15`` → distances varying by >15% of mean

- ``with_std_reduction()``, ``with_variance_reduction()``: Absolute variability

    - **Use**: Find distances with large absolute fluctuations

- ``with_range_reduction()``: max - min distance

    - **Use**: Identify distances exploring wide conformational space

- ``with_transitions_reduction()``: Distance change events

    - **Use**: Detect conformational switching between states

- ``with_mean/min/max/mad_reduction()``: Value-based filtering

    - **Use**: Filter by typical distance values

Coordinates
^^^^^^^^^^^

XYZ positions

- ``with_rmsf_reduction()``: Root mean square fluctuation

    - **Use**: Focus on flexible regions, identify mobile loops
    - **Example**: ``threshold_min=2.0`` → atoms fluctuating >2 Å

- ``with_std_reduction()``, ``with_cv_reduction()``: Position variability

    - **Use**: Similar to RMSF, identify dynamic vs. rigid regions

- ``with_range/variance/mad_reduction()``: Position spread metrics

Torsions
^^^^^^^^

Dihedral angles

- ``with_transitions_reduction()``: Angular transitions (rotamer changes)

    - **Use**: Identify side chains or backbone angles switching conformations
    - **Example**: ``threshold_min=10`` → angles with >10 transition events

- ``with_std/cv/variance_reduction()``: Angular variability

    - **Use**: Find flexible torsions vs. constrained angles

SASA
^^^^

Solvent accessible surface area

- ``with_cv_reduction()``: Exposure variability

    - **Use**: Find residues alternating between buried/exposed
    - **Example**: ``threshold_max=0.3`` → relatively constant exposure

- ``with_burial_fraction_reduction()``: Fraction of time buried

    - **Use**: Classify residues as core (high burial) vs. surface (low burial)

- ``with_range/std_reduction()``: Exposure dynamics

DSSP
^^^^

Secondary structure

- ``with_transitions_reduction()``: Structure type changes

    - **Use**: Find regions switching between helix/sheet/coil

- ``with_stability_reduction()``: Structure persistence

    - **Use**: Identify stable vs. unstable secondary structure elements
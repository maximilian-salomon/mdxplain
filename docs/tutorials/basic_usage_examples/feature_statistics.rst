Feature Statistics
==================

.. todo: Feature Reduction & Feature Statistics – overlapping functionalities; needs consolidation.

Feature statistics provide quantitative insights into molecular interaction
patterns, dynamic behavior, and conformational preferences. Use statistics
to understand contact formation frequencies, distance distributions, or
identify key dynamic residues.

Available Statistics
--------------------

- mean
- std
- min
- max
- median
- variance
- cv (coefficient of variation)
- frequency (contacts)
- stability (contacts)
- transitions (contacts/distances)

Use Cases
---------

- Contact frequency: "How often does residue X interact with residue Y?"
- Distance distributions: "What is the typical separation between domains?"
- Dynamic patterns: "Which contacts are stable vs. transient?"
- Trajectory-specific vs. global behavior

.. code:: python

    # Contact frequency analysis (global)
    contact_freq = pipeline.analysis.features.contacts.frequency()
    # Returns: contact formation frequency across all frames

    # Statistical analysis of distances
    mean_distances = pipeline.analysis.features.distances.mean()
    std_distances = pipeline.analysis.features.distances.std()
    variance_distances = pipeline.analysis.features.distances.variance()

    # Coefficient of variation (normalized variability)
    cv_distances = pipeline.analysis.features.distances.cv()

    # Trajectory-specific statistics
    traj0_mean = pipeline.analysis.features.contacts.mean(traj_selection=0)
    traj1_mean = pipeline.analysis.features.contacts.mean(traj_selection=1)

    # Interpretation example: Contact frequency
    # freq = 0.8 => contact formed in 80% of frames (stable interaction)
    # freq = 0.2 => contact formed in 20% of frames (transient interaction)
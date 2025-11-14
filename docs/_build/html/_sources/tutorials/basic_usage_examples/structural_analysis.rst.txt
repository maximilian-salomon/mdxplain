Structural Analysis
===================

Structural metrics provide quantitative measures of protein dynamics and conformational
changes. mdxplain offers multiple RMSD/RMSF variants optimized for different analysis
scenarios.

RMSD Metrics - Which Variant to Use
-----------------------------------

``.rmsd.mean`` - Standard RMSD (Root Mean Square Deviation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Classic RMSD using arithmetic mean
- **When to use**:

    - Standard conformational analysis and literature comparison
    - Systems without highly flexible regions
    - Fastest computation

- **Avoid when**: System has highly flexible regions that would dominate the metric

``.rmsd.median`` - Robust RMSD (Root Median Square Deviation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Square root of median of squared deviations (instead of mean)
- **When to use**:

    - Systems with occasional outlier atoms (flexible loops + rigid core)
    - Multi-domain proteins with independent movement
    - Combined folded/unstructured regions
    
- **Why**: Robust against outlier atoms affecting the metric

``.rmsd.mad`` - MAD RMSD (Median Absolute Deviation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Most robust RMSD variant based on MAD
- **When to use**:

    - Extremely flexible systems
    - Proteins with intrinsically disordered regions
    - Maximum outlier resistance needed

- **Why**: Statistically most robust, least affected by rare large-amplitude moves

RMSD Modes - What to Measure
----------------------------

``.to_reference(reference_traj, reference_frame, atom_selection="all")``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: RMSD to a fixed reference frame
- **Why use it**:

    - Track structural drift from starting structure
    - Measure equilibration (distance from crystal structure)
    - Monitor return to specific conformation
    - Stability relative to known structure (X-ray, cryo-EM)

- **Example**: ``rmsd.mean.to_reference(0, 0)`` → drift from initial frame

``.frame_to_frame(lag=1, atom_selection="all")``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: RMSD between consecutive or lag-separated frames
- **Why use it**:

    - Quantify local structural fluctuations
    - Identify smooth vs. jerky dynamics
    - Detect conformational transition events
    - lag=1: frame-to-frame noise, lag=10+: larger conformational shifts

- **Example**: ``rmsd.mean.frame_to_frame(lag=10)`` → local dynamics

``.window_frame_to_start()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Sliding window RMSD to window start
- **Why use it**:

    - Assess equilibration within time windows
    - Local stability analysis
    - Detect gradual structural drift    - Standard flexibility analysis
    - Comparison with experimental B-factors
    - Identify rigid vs. flexible regions

``.window_frame_to_frame()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Sliding window frame-to-frame RMSD
- **Why use it**:

    - Track local fluctuations over time
    - Identify periods of high/low dynamics


RMSF Metrics - Which Variant
----------------------------

``.rmsf.mean`` - Standard RMSF (Root Mean Square Fluctuation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Classic per-residue flexibility metric
- **When to use**:

    - Standard flexibility analysis
    - Comparison with experimental B-factors
    - Identify rigid vs. flexible regions

``.rmsf.median`` - Robust RMSF (Root Median Square Fluctuation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Square root of median of squared fluctuations (instead of mean)
- **When to use**: Systems with occasional large-amplitude rare events
- **Why**: Robust against rare outlier movements

``.rmsf.mad`` - MAD RMSF (Median Absolute Deviation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Most robust fluctuation measure
- **When to use**: Very flexible systems requiring maximum robustness

RMSF Modes - Resolution Level
-----------------------------

``.per_atom``
^^^^^^^^^^^^^

- **What**: Atom-level fluctuations
- **Why use it**:

    - Detailed side-chain dynamics
    - Specific functional atoms (active site, binding residues)
    - High-resolution flexibility mapping

- **Example**: Side-chain rotamer dynamics

``.per_residue``
^^^^^^^^^^^^^^^^

- **What**: Residue-aggregated fluctuations
- **Why use it**:

    - Protein-wide flexibility overview
    - Domain mobility comparison
    - Experimental B-factor comparison
    - Flexibility profiles

- **Example**: Identify flexible loops vs. rigid helices

Per-Residue Aggregation Methods
-------------------------------

When using ``.per_residue``, atom-level RMSF values within each residue must be
aggregated to a single per-residue value. Four aggregation strategies are available:

``.with_mean_aggregation`` (Default, Simple Average)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
- **What**: Arithmetic mean of atom-level RMSF values: ``mean(RMSF_atoms)``
- **When to use**:

    - Standard flexibility profiles
    - All atoms in residue have similar flexibility
    - No extreme outlier atoms

- **Sensitivity**: Affected by outlier atoms (very flexible side-chain tips)
- **Example**: Backbone CA atoms or residues with uniform flexibility

``.with_median_aggregation`` (Robust to Outliers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Median of atom-level RMSF values: ``median(RMSF_atoms)``
- **When to use**:

    - Long flexible side chains (e.g., LYS, ARG) with rigid backbone
    - Mixed flexibility within residue (some atoms rigid, others flexible)
    - Want typical flexibility, not influenced by extreme atoms

- **Benefit**: Terminal side-chain atoms don't dominate the residue score
- **Example**: Surface residues with flexible tips but stable core

``.with_rms_aggregation`` (Emphasize Larger Values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Root-mean-square of RMSF values: ``sqrt(mean(RMSF_atoms²))``
- **When to use**:

    - Want to emphasize larger fluctuations more strongly
    - Quadratic weighting is desired (larger RMSF = disproportionately higher weight)

- **Effect**: Residue score dominated by most flexible atoms
- **Example**: Identifying highly dynamic regions where any flexible atom matters

``.with_rms_median_aggregation`` (Emphasize + Robust)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What**: Root-median-square of RMSF values: ``sqrt(median(RMSF_atoms²))``
- **When to use**:

    - Want quadratic weighting but robust against extreme outliers
    - Very flexible side chains with occasional large-amplitude jumps

- **Effect**: Emphasizes typical flexibility, ignores rare extremes
- **Example**: Disordered regions with occasional extreme conformations

Aggregation Selection Guide
---------------------------

+-------------------------------------+----------------------+-------------------------------------------------+
| Your Goal                           | Recommended Method   | Why                                             |
+=====================================+======================+=================================================+
| Standard flexibility profile        | ``mean``             | Simple average, standard representation         |
+-------------------------------------+----------------------+-------------------------------------------------+
| Residues with very flexible tips    | ``median``           | Ignores outlier atoms at side-chain ends        |
+-------------------------------------+----------------------+-------------------------------------------------+
| Emphasize most flexible atoms       | ``rms``              | Larger fluctuations get more weight (quadratic) |
+-------------------------------------+----------------------+-------------------------------------------------+
| Flexibility with outlier protection | ``rms_median``       | Emphasizes flexibility but ignores extremes     |
+-------------------------------------+----------------------+-------------------------------------------------+

Practical Examples
------------------

Equilibration Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^

- **Question**: Has the simulation equilibrated? How far does structure drift from starting point?
- **Why**: mean.to_reference = standard metric for comparing to crystal structure
- **Why**: backbone = ignores side-chain noise, focuses on secondary structure stability
- **Use** case: Quality control, detect slow conformational drift, assess convergence

.. code:: python

    rmsd_crystal = pipeline.analysis.structure.rmsd.mean.to_reference(
        reference_traj=0, reference_frame=0, atom_selection="backbone"
    )

Flexible Multi-Domain Protein
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Question**: What is overall conformational change in protein with mobile loops?
- **Why**: median.to_reference = robust against flexible loop outliers affecting global RMSD
- **Why**: "protein" selection = all protein atoms, including flexible regions
- **Use case**: Multi-domain proteins, antibodies, disordered regions that would dominate mean

.. code:: python

    rmsd_robust = pipeline.analysis.structure.rmsd.median.to_reference(
        reference_traj=0, reference_frame=0, atom_selection="protein"
    )

Conformational Transition Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Question**: Are there sudden conformational changes or smooth dynamics?
- **Why**: frame_to_frame(lag=5) = local structural changes every 5 frames
- **Why**: mean metric = sufficient for well-behaved dynamics
- **Why**: CA atoms = coarse-grained, computationally efficient
- **Use case**: Identify transition events, measure local stability, detect jerky vs smooth motion

.. code:: python

    rmsd_dynamics = pipeline.analysis.structure.rmsd.mean.frame_to_frame(
        lag=5, atom_selection="name CA"
    )

Standard Flexibility Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Question**: Which regions are rigid vs flexible?
- **Why**: rmsf.mean.per_residue = standard
- **Why**: with_mean_aggregation (default) = balanced residue flexibility
- **Why**: CA atoms = one value per residue, standard representation
- **Use case**: identify flexible loops/hinges

.. code:: python

    rmsf_profile = pipeline.analysis.structure.rmsf.mean.per_residue.to_mean_reference(
        atom_selection="name CA"
    )

Binding Site Side-Chain Flexibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Question**: How do side chains move in the binding pocket?
- **Why**: per_atom = high-resolution, individual atom fluctuations
- **Why**: mean metric = standard for well-defined binding site
- **Why**: resid 120-140 = specific binding site region
- **Use case**: Ligand binding analysis, induced fit, side-chain conformational sampling

.. code:: python

    rmsf_detailed = pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(
        atom_selection="resid 120-140"  # binding site
    )

Multi-Domain Protein Flexibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Question**: Which domains are rigid vs mobile, with robustness against outliers?
- **Why**: rmsf.median = robust against occasional large-amplitude motions
- **Why**: per_residue.to_median_reference = consistent robust statistics at all levels
- **Why**: with_median_aggregation (implicit) = prevents outlier atoms from dominating
- **Use case**: Domain motion analysis, linker flexibility, proteins with mobile tails

.. code:: python

    rmsf_robust = pipeline.analysis.structure.rmsf.median.per_residue.to_median_reference(
        atom_selection="protein"
    )

Side-Chain Rotamer Switching (Advanced Aggregation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Question**: Which residues show side-chain flexibility with robust aggregation?
- **Why**: with_rms_aggregation = combines backbone and side-chain motion correctly
- **Why**: Emphasizes larger fluctuations, suitable for rotamer analysis
- **Use case**: Functional side-chain dynamics, allosteric communication pathways

.. code:: python

    rmsf_rotamers = pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_aggregation(
        atom_selection="protein"
    )
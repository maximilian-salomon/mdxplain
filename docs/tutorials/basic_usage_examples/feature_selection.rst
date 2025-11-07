Feature Selection
=================

Feature selection defines which molecular features (matrix columns) enter your analysis.
mdxplain provides a custom parsing syntax that supports residue names, IDs, consensus
nomenclature, and logical operations for flexible feature filtering.

How the Parser Works
--------------------

The selection language is case-insensitive and automatically recognizes keywords. Ranges
use ``-`` (e.g., ``10-50``), lists are space-separated (e.g., ``ALA HIS GLY``), ``and``
combines conditions (intersection), and ``not`` excludes matches (difference).

Selection Keywords
^^^^^^^^^^^^^^^^^^

``res`` - Residue Names
"""""""""""""""""""""""

- **What**: Select by amino acid type
- **Why use it**:

    - Analyze hydrophobic core: ``res ALA VAL LEU ILE PHE``
    - Study charged interactions: ``res ARG LYS ASP GLU``
    - Focus on specific residue types for functional analysis
    - Test chemical property hypotheses (aromatic, polar, etc.)

- **Example**: ``res ARG LYS``

``resid`` - Residue IDs
"""""""""""""""""""""""

- **What**: Select by residue numbers in PDB
- **Why use it**:

    - Target known functional regions from literature: ``resid 120-140`` → binding site
    - Test structure-based hypotheses from crystal structures
    - Focus on specific structural elements (loops, termini)
    - Region-specific analysis based on domain knowledge

- **Example**: ``resid 10-50`` → N-terminal region

``seqid`` - Sequence IDs
""""""""""""""""""""""""

- **What**: Select by sequence position
- **Why use it**: Alignment-based analyses, homologous position comparisons across proteins
- **Example**: Useful in multi-chain complexes

``consensus`` - Consensus Nomenclature
""""""""""""""""""""""""""""""""""""""

- **What**: Structure-based biological numbering (GPCRdb, CGN, kinase numbering)
- **Why use it**:

  - **Functional motifs**: ``consensus 3x50`` → DRY motif in GPCRs (conserved activation
    switch)
  - **Structural elements**: ``consensus 7x*`` → entire TM7 helix
  - **Conserved networks**: ``consensus *40-*50`` → positions 40-50 e.g. across all TM
    helices (allosteric pathways)
  - **Cross-protein comparison**: Same consensus position = functionally equivalent

- **Patterns**:

    - Single: ``7x53`` → NPxxY tyrosine (Y7.53, activation-associated)
    - Range: ``7x50-8x50`` → TM7-TM8 interface region
    - Wildcard: ``7x*`` → all TM7 positions
    - Multi-pattern: ``*40-*50`` → positions 40-50 in all TM helices
    - G-proteins: ``G.H5.*`` → helix 5 of G-alpha
    - ``all`` **prefix**: Include residues without consensus labels

        - ``7x-8x`` → only residues WITH consensus labels from 7x to 8x
        - ``all 7x-8x`` → ALL residues from 7x to 8x, including those without consensus
          (loops, unstructured regions)
        - **Why**: Capture complete structural regions including flexible elements
        - **Example**: ``all 7x-8x`` includes TM7, intracellular loop, and TM8

- **Note**: This is not fixed to a consensus. ``*40-*50`` for example is looking for a
  pattern that matches this. It does not have to be a TM helix, if you have other
  consensus labels. It is a pattern matching. It takes the first consensus label entry
  with ``*40`` and then takes the next with ``*50`` and takes the range in between. If it
  does NOT find a start or end point it takes the next residue in the list starting or
  ending with seqid.

``all``
"""""""

- **What**: Select all features
- **Why use it**: Initial exploration, global pattern discovery, unbiased analysis

Logical Operators
^^^^^^^^^^^^^^^^^

- ``and`` - Intersection (both conditions must be true)

    - **Why**: Specific combinations: ``res ARG and resid 120-140`` → positive charges in
      binding site
- ``not`` - Exclusion (remove matches)

    - **Why**: "All except" patterns: ``res ALA and not resid 25`` → all alanines except
      position 25

.. code:: python

    # Step 1: Create named feature selection (empty container)
    pipeline.feature_selector.create("my_selection")

    # Step 2: Add features using different selection syntaxes
    # Add contacts matching residue names
    pipeline.feature_selector.add.contacts("my_selection", "res ALA HIS")

    # Add contacts in specific residue ID range (binding site region)
    pipeline.feature_selector.add.contacts("my_selection", "resid 10-50")

    # Add contacts using consensus nomenclature (GPCR activation pathway)
    pipeline.feature_selector.add.contacts("my_selection", "consensus 7x50-8x50")

    # Add distances with logical operators (all ALA except position 25)
    pipeline.feature_selector.add.distances("my_selection", "res ALA and not resid 25")

    # Step 3: Add features with reduction (statistical filtering during selection)
    # Only keep contacts formed in >30% of frames (stable interactions)
    pipeline.feature_selector.add.contacts.with_frequency_reduction(
        "my_selection", "resid 120-140", threshold_min=0.3
    )

    # Step 4: Multi-trajectory mode
    # common_denominator=True: Only Alanins present in ALL trajectories
    # (Useful for comparing different systems with slightly different structures)
    pipeline.feature_selector.add.contacts(
        "my_selection", "ALA", common_denominator=True
    )

    # Step 5: Use pre-reduced features
    # use_reduced=True: Uses features from feature.reduce_data() instead of raw data
    # (When you want to apply global reduction first, then select subset)
    pipeline.feature_selector.add.distances(
        "my_selection", "all", use_reduced=True
    )

    # Step 6: Apply selection to create final feature matrix
    # Combines all added selections into single feature matrix for analysis
    # Note each of the selections adds the features to the set. Its a union of all selectors before select call.
    pipeline.feature_selector.select("my_selection")

Practical Examples with Biological Context
------------------------------------------

GPCR Activation Contact (Ionic Lock)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Selects a known constraining interaction in GPCRs:
- 3x50: DRY motif arginine (R3.50)
- 6x30: conserved glutamate (E6.30) on TM6

.. code:: python

    pipeline.feature_selector.add.contacts("gpcr", "consensus 3x50 and consensus 6x30")

G-Protein Binding Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Combines two structural elements for GPCR-G-protein coupling:
- G.H5.*: Complete helix 5 of G-alpha protein (major contact surface)
- 8x*: Helix 8 of receptor (intracellular C-terminus)
- Interface contacts important for G-protein activation

.. code:: python

    pipeline.feature_selector.add.contacts("interface", "consensus G.H5.* and consensus 8x*")

Binding Pocket Aromatic Cage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Targets aromatic residues in known binding region:
- Aromatic amino acids (PHE, TRP, TYR) form π-stacking interactions
- Residues 100-150: Binding pocket from crystal structure
- Creates feature set for ligand-binding site characterization

.. code:: python

    pipeline.feature_selector.add.contacts("pocket", "res PHE TRP TYR and resid 100-150")

Hydrophobic Core Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Selects non-polar residues forming protein core:
- ALA, VAL, LEU, ILE, PHE: Hydrophobic amino acids
- Monitors core packing stability and folding integrity

.. code:: python

    pipeline.feature_selector.add.contacts("core", "res ALA VAL LEU ILE PHE")

TM Helix Interface Including Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- With 'all' prefix
- Captures complete structural region:
- Without 'all': Only labeled TM7/TM8 residues
- With 'all': Includes intracellular loop between helices
- Complete interface for conformational change analysis

.. code:: python

    pipeline.feature_selector.add.contacts("tm7_loop_tm8", "all 7x-8x")
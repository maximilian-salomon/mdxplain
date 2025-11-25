Trajectory Management
=====================

.. todo: Depth/nesting of folder structure for loading – how deep can it go?
    Trajectories not loaded alphabetically → plotting (e.g., membership) can be disordered → reduces comparability in plots
    Is there an option to sort trajectories?
    Depth/nesting of folder structure for loading – how deep can it go?
    --> Missing example for folder structure: e.g., max one PDB per folder, multiple XTCs possible.
    Cache folder for trajectories – relative to script execution path or pipeline cache directory?
    select_atoms: which strings are valid? Include brief note that these are mdtraj selection strings; optionally link or provide example.
    Consensus nomenclature for labels: What is the exact reference for nomenclature consensus?

mdxplain supports all MDTraj-compatible trajectory formats (xtc, dcd, trr, pdb,
h5, netcdf, etc.) and provides powerful tools for organizing and manipulating
multiple trajectories. Tags enable systematic organization for comparative
analyses, while labels add residue metadata like consensus nomenclature or
fragment definitions.

.. code:: python

    # Load trajectories (supports all MDTraj formats)
    # See: http://mdtraj.org/latest/load_functions.html
    pipeline.trajectory.load_trajectories("data/system_A/")
    pipeline.trajectory.load_trajectories("data/system_B/")

    # Add residue labels and metadata
    pipeline.trajectory.add_labels(traj_selection="all")

    # Add tags for organizing trajectories (enables tag-based selection)
    pipeline.trajectory.add_tags(0, ["wild_type", "system_A"])
    pipeline.trajectory.add_tags(1, ["mutant", "system_B"])

    # Trajectory manipulation
    pipeline.trajectory.cut_traj(0, start=100, end=5000)  # Trim frames
    pipeline.trajectory.select_atoms(selection="protein")  # Atom selection


Consensus Nomenclature for Structured Proteins
----------------------------------------------

For proteins with established numbering schemes (GPCRs, kinases, G-proteins),
consensus nomenclature enables biologically meaningful feature selection. This
allows selecting structurally/functionally equivalent positions across different
proteins.

.. code:: python
    
    # Example: GPCR-G-Protein complex (3SN6)
    # Define two fragments with different consensus schemes
    pipeline.trajectory.add_labels(
        fragment_definition={
            "cgn_a": (0, 349),                    # G-protein alpha subunit
            "beta2": (349+340+58, 349+340+58+284) # Beta-2 adrenergic receptor
        },
        fragment_type={
            "cgn_a": "cgn",    # Common Galpha Numbering
            "beta2": "gpcr"    # GPCRdb numbering
        },
        fragment_molecule_name={
            "cgn_a": "gnas2_bovin",  # UniProt entry for G-alpha-s
            "beta2": "adrb2_human"   # UniProt entry for beta-2 receptor
        },
        consensus=True,        # Enable consensus numbering
        aa_short=False         # Use full amino acid names (THR vs T)
    )

    # Now you can select features using consensus patterns:
    # "consensus 7x49-7x53" => NPxxY region on TM7
    # "consensus G.H5.*" => G-protein helix 5
    # "consensus 3x* and consensus 6x*" => TM3-TM6 interface
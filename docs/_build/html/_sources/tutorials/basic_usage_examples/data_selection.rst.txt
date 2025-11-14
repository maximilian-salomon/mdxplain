Data Selection (Frame/Row Selection)
====================================

While FeatureSelector defines which features (matrix columns) to analyze,
DataSelector chooses which trajectory frames (matrix rows) to include.
This enables subset-based analyses focusing on specific conformational
states, conditions, or time windows.

Core Concept
------------

- **FeatureSelector**: Defines matrix columns (which features: contacts, distances, etc.)
- **DataSelector**: Defines matrix rows (which frames: states, trajectories, conditions)
- **Combined**: Creates targeted analysis matrices for specific scientific questions

Why Use Data Selection
----------------------

- **State-specific analysis**: Focus on folded, unfolded, or intermediate conformations
- **Condition comparison**: Wild-type vs. mutant, ligand-bound vs. apo
- **Outlier removal**: Exclude noise clusters or equilibration frames
- **Data reduction**: Sample large datasets for manageable analysis
- **Combined criteria**: Intersection of multiple selection criteria

Methods
-------

``create(name)`` - Create Named Frame Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    pipeline.data_selector.create("folded_frames")
    pipeline.data_selector.create("active_state")


``select_by_tags(name, tags, match_all=True, mode="add", stride=1)`` - Tag-Based Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters
""""""""""

- ``tags``: List of trajectory tags to match
- ``match_all``:

    - ``True`` (default): Frame needs ALL tags (AND logic) - ``["wild_type", "production"]`` → both required
    - ``False``: Frame needs ANY tag (OR logic) - ``["system_A", "system_B"]`` → either suffices

- ``mode``:

    - ``"add"`` (default): Union - add frames to selection
    - ``"subtract"``: Difference - remove frames from selection
    - ``"intersect"``: Intersection - keep only overlap

- ``stride``: Sample every Nth frame (1=all frames, 10=every 10th)

Why use it
""""""""""

- Condition-based filtering: ``tags=["wild_type"]`` → only WT trajectory frames
- System organization: ``tags=["system_A", "biased"]`` → specific simulation type
- Data sampling: ``stride=10`` → reduce dataset size

``select_by_cluster(name, clustering_name, cluster_ids, mode="add", stride=1)`` - Cluster-Based Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters
""""""""""

- ``clustering_name``: Name of clustering result (e.g., "DPA", "DBSCAN")
- ``cluster_ids``: List of cluster numbers ``[0, 1, 2]`` or names
- ``mode``: Same as tags (add/subtract/intersect)
- ``stride``: Sample selected clusters

Why use it
""""""""""

- Conformational states: ``cluster_ids=[0]`` → only folded state
- Multi-state analysis: ``cluster_ids=[0, 2]`` Result → active + intermediate
- Comparative studies: Different selectors for each state
- Outlier removal: ``cluster_ids=[-1], mode="subtract"`` → exclude noise

``select_by_indices(name, trajectory_indices, mode="add")`` - Direct Frame Index Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most flexible selection method: specify exact frame numbers for each trajectory.
Supports various input formats for maximum convenience.

Parameters
""""""""""

``trajectory_indices``: Dictionary mapping trajectory selectors to frame specifications

Trajectory Selectors (Keys)
'''''''''''''''''''''''''''

- ``int``: Trajectory index → ``0``, ``1``, ``2``
- ``str``: Trajectory name → ``"system_A"``
- ``str``: Tag pattern → ``"tag:biased"`` (applies to all trajectories with tag)
- ``str``: Name pattern → ``"system_*"`` (glob-style matching)

Frame Specifications (Values)
'''''''''''''''''''''''''''''

- ``int``: Single frame → ``42``
- ``List[int]``: Explicit frames → ``[10, 20, 30, 50]``
- ``str``: Various string formats:

    - Single: ``"42"`` → frame 42
    - Range: ``"10-20"`` → frames 10, 11, ..., 20
    - Comma list: ``"10,20,30"`` → frames 10, 20, 30
    - Combined: ``"10-20,30-40,50"`` → frames 10-20, 30-40, and 50
    - All: ``"all"`` → all frames in trajectory

- ``dict``: With stride support:

    - ``{"frames": frame_spec, "stride": N}`` → apply stride to frame selection
    - Example: ``{"frames": "0-100", "stride": 10}`` → frames 0, 10, 20, ..., 100

``mode``: Selection mode (add/subtract/intersect)

Why use it
""""""""""

- **Manual frame selection**: Choose specific frames from analysis (e.g., representative structures)
- **Time window analysis**: Select equilibrated region, exclude equilibration
- **Sparse sampling**: Reduce dataset size with stride for computational efficiency
- **Complex patterns**: Combine multiple ranges and specific frames
- **Cross-trajectory patterns**: Use tags/names to apply same frame selection to multiple trajectories

Examples
--------

Simple Frame Selection
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Select specific frames from trajectory 0Result
    pipeline.data_selector.create("custom_frames")
    pipeline.data_selector.select_by_indices(
        "custom_frames",
        {0: [100, 200, 300, 500]}  # Four specific frames
    )

Time Window (Exclude Equilibration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Use frames 1000-5000 from all trajectories (first 1000 = equilibration)
    pipeline.data_selector.create("equilibrated")
    pipeline.data_selector.select_by_indices(
        "equilibrated",
        {"all": "1000-5000"}  # "all" applies to all loaded trajectories
    )

Range With Stride (Sparse Sampling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Every 50th frame from equilibrated region for computational efficiency
    pipeline.data_selector.create("sparse_equilibrated")
    pipeline.data_selector.select_by_indices(
        "sparse_equilibrated",
        {0: {"frames": "1000-5000", "stride": 50}}  # 1000, 1050, 1100, ..., 5000
    )

Complex Combined Ranges
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Multiple time windows and specific frames
    pipeline.data_selector.create("complex_selection")
    pipeline.data_selector.select_by_indices(
        "complex_selection",
        {
            0: "100-200,500-600,1000",  # Two ranges + single frame
            1: "200-400,800-1000"       # Different ranges for trajectory 1
        }
    )

Tag-Based Frame Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Apply same frame selection to all trajectories with "biased" tag
    pipeline.data_selector.create("biased_frames")
    pipeline.data_selector.select_by_indices(
        "biased_frames",
        {"tag:biased": "500-2000"}  # Frames 500-2000 from all biased trajectories
    )

Name Pattern Matching
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Use all frames from trajectories matching pattern
    pipeline.data_selector.create("system_A_all")
    pipeline.data_selector.select_by_indices(
        "system_A_all",
        {"system_A*": "all"}  # All frames from trajectories starting with "system_A"
    )

Multi-Trajectory With Different Frame Selections and Stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Complex real-world scenario: different selections per trajectory type
    pipeline.data_selector.create("production_analysis")
    pipeline.data_selector.select_by_indices(
        "production_analysis",
        {
            "tag:wild_type": {"frames": "2000-10000", "stride": 20},  # WT: sparse sampling
            "tag:mutant": "2000-10000",        # Mutant: all frames (smaller dataset)
            "system_C": [100, 500, 1000, 2000] # System C: specific snapshots only
        }
    )

Practical Examples
------------------

State-Specific Analysis
^^^^^^^^^^^^^^^^^^^^^^^

- **Scientific question**: What features characterize the folded state?
- **Why**: Focus feature importance analysis only on folded conformation
- **Use case**: Identify stabilizing interactions in folded state, compare folded vs other states
- **Result**: Only frames from cluster 0 (folded state) for downstream analysis

.. code:: python

    pipeline.data_selector.create("folded")
    pipeline.data_selector.select_by_cluster("folded", "DPA", [0])

Multi-State Conformational Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Scientific question**: What features distinguish active and intermediate from inactive?
- **Why**: Combine multiple conformational states for comparison against baseline
- **Use case**: Activation pathway analysis, identify shared features of non-inactive states
- **Result**: Frames from clusters 0 (active) + 2 (intermediate), excludes cluster 1 (inactive)

.. code:: python

    pipeline.data_selector.create("active_states")
    pipeline.data_selector.select_by_cluster("active_states", "DPA", [0, 2])

Production Run With Data Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Scientific question**: What is baseline behavior in wild-type production simulations?
- **Why**: match_all=True ensures only production WT (not equilibration WT)
- **Why**: stride=5 reduces computational cost while maintaining statistical validity
- **Use case**: Reference dataset for mutant comparison, representative WT behavior
- **Result**: Every 5th frame from wild-type production runs only

.. code:: python

    pipeline.data_selector.create("wt_prod")
    pipeline.data_selector.select_by_tags(
        "wt_prod",
        tags=["wild_type", "production"],  # Must have BOTH tags
        match_all=True,                    # AND logic: production AND wild_type
        stride=5                           # Sample every 5th frame for efficiency
    )
   
Multi-Step Complex Selection With Set Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Scientific question**: What features are specific to cluster 1 in biased simulations?
- **Why**: 3-step refinement isolates specific subset using set operations
- **Use case**: Enhanced sampling analysis, isolate converged non-noise states

.. code:: python

    pipeline.data_selector.create("biased_cluster1")

    # Step 1: START with all biased simulation frames (union operation)
    pipeline.data_selector.select_by_tags("biased_cluster1", ["biased"], mode="add")
    # Current selection: all frames from biased trajectories

    # Step 2: NARROW to cluster 1 only (intersection operation)
    pipeline.data_selector.select_by_cluster("biased_cluster1", "DBSCAN", [1], mode="intersect")
    # Current selection: biased frames that are ALSO in cluster 1

    # Step 3: CLEAN by removing noise cluster (subtraction operation)
    pipeline.data_selector.select_by_cluster("biased_cluster1", "DBSCAN", [-1], mode="subtract")
    # Final selection: biased cluster 1, excluding any noise assignments
    # Note: Safety step - DBSCAN can assign noise (-1), this removes artifacts

Systematic Condition Comparison (WT vs Mutant in Same State)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Scientific question**: What molecular differences exist between WT and mutant in folded state?
- **Why**: Compare same conformational state across different systems
- **Use case**: Mutation effect analysis, identify compensatory changes
- **Strategy**: Create two matched selectors for direct comparison

    - Both select folded state (cluster 0)
    - Different genetic backgrounds (WT vs mutant tags)
    - These two selectors enable apples-to-apples comparison:

        - Same conformational state, different genetic backgrounds
        - Use in ``comparison.create_comparison()`` to find mutation-specific features

.. rubric:: Wild-type in folded state

.. code:: python

    pipeline.data_selector.create("wt_folded")
    pipeline.data_selector.select_by_tags("wt_folded", ["wild_type"], mode="add")
    # Get all WT frames
    pipeline.data_selector.select_by_cluster("wt_folded", "conformations", [0], mode="intersect")
    # Narrow to folded conformation only
    # Result: WT folded state frames

.. rubric:: Selector B: Mutant in folded state

.. code:: python

    pipeline.data_selector.create("mutant_folded")
    pipeline.data_selector.select_by_tags("mutant_folded", ["mutant"], mode="add")
    # Get all mutant frames
    pipeline.data_selector.select_by_cluster("mutant_folded", "conformations", [0], mode="intersect")
    # Narrow to folded conformation only
    # Result: Mutant folded state frames

Outlier Removal (Quality Control)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Scientific question**: What is protein behavior excluding simulation artifacts?
- **Why**: DBSCAN noise cluster (-1) often contains artifacts, transitions, rare events
- **Use case**: Clean analysis dataset, focus on well-sampled conformations
- **Result**: Only well-defined clusters, excludes noise/artifacts

.. code:: python

    pipeline.data_selector.create("clean_conformations")
    # Start with all frames from main clustering
    pipeline.data_selector.select_by_cluster("clean_conformations", "DBSCAN", [0, 1, 2], mode="add")
    # Explicitly exclude noise (could also start with "all" and subtract)

Integration with Comparative Analysis
-------------------------------------

Data selectors define the frame sets (rows) for feature importance and comparison analyses:

.. code:: python

    # Create feature matrix (columns)
    pipeline.feature_selector.create("binding_site")
    pipeline.feature_selector.add.contacts("binding_site", "resid 120-140")

    # Create frame selections (rows)
    pipeline.data_selector.create("state_A")
    pipeline.data_selector.select_by_cluster("state_A", "DPA", [0])

    pipeline.data_selector.create("state_B")
    pipeline.data_selector.select_by_cluster("state_B", "DPA", [1])

    # Compare states using selected features and frames
    pipeline.comparison.create_comparison(
        name="state_comparison",
        mode="one_vs_rest",
        feature_selector="binding_site",      # Column selection
        data_selectors=["state_A", "state_B"] # Row selections
    )


Common Use Cases
^^^^^^^^^^^^^^^^

- **Conformational analysis**: Select frames by cluster assignment
- **Condition comparison**: Select frames by trajectory tags (WT/mutant, apo/holo)
- **Data reduction**: Stride sampling for large datasets
- **Quality control**: Exclude equilibration, outliers, or unstable frames
- **Multi-criteria filtering**: Combine tags and clusters with mode operations

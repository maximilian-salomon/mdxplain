# TODOs before Version 1.0
## Must Have
- Plot-Module:
    1. Distancen time-line und Verteilungs-plots
    2. Trees
    3. PDB mit Betafactor
    4. Violin-Plots
    5. Heatmap der Contact-Matrix
    6. PCA und Cluster Plots
    
    - Cluster Representatives
    - Simple-Plot of matrix
    - SnakePlot siehe GPCRdb
    - FlarePlot
    - Compare FlarePlot grün für 1. Kat und rot für 2. Kat und dann dicke
    - Ramachadran Plot
    - Balken-Diagramm für DSSP
    - Cluster-Zentren als 3D Struktur nglview
    - Plot 2 Features gegeneinander

- Analysis-Module:
    - RMSD, RMSF

- We need unit tests, example notebooks and documentation

- Cleanup

## Maybes
- Features should get more features: CCI (a normalized combination of contacts and distances)
cutoff using vdw radii. Soft Contact with linear and sigmoid cut

- Specific interaction Types would be nice

- In the nomenclature for consensus, fragments and nomenclature could be guessed with mdciao => This would simplify the use of consensus

- Save of PipelineManager auto

- config

- Errors and prints should work with line breaks

- Logging!

- FeatureImportance should get 1 or 2 more Methods like RandomForest for sure and maybe SVMs




## Documentation TODOs
*Last updated: 2025-11-25 10:49:56*

### `docs/tutorials/basic_usage_examples/comparative_analysis_and_feature_importance.rst`
- **Line 4**:
  currently too brief; expand content.

### `docs/tutorials/basic_usage_examples/data_selection.rst`
- **Line 4**:
  needs streamlining

### `docs/tutorials/basic_usage_examples/dimensionality_reduction.rst`
- **Line 4**:
  requires more explanation:
  Include prerequisites (e.g., output from feature selector for selection_name).

### `docs/tutorials/basic_usage_examples/feature_computation.rst`
- **Line 4**:
  Provide guidance on how to access arrays for external analysis.
  Optionally link to possible downstream applications.

### `docs/tutorials/basic_usage_examples/feature_reduction.rst`
- **Line 4**:
  Feature Reduction & Feature Statistics – overlapping functionalities; needs consolidation.

### `docs/tutorials/basic_usage_examples/feature_selection.rst`
- **Line 4**:
  generally fine; needs streamlining and clarification.
  Clarify on difference to Data Selector. When should each be used?

### `docs/tutorials/basic_usage_examples/feature_statistics.rst`
- **Line 4**:
  Feature Reduction & Feature Statistics – overlapping functionalities; needs consolidation.

### `docs/tutorials/basic_usage_examples/memory-efficient_processing.rst`
- **Line 4**:
  Is it possible to adjust chunksize after pipeline creation?

### `docs/tutorials/basic_usage_examples/plotting.rst`
- **Line 6**:
  Add plotting examples and explanations.

### `docs/tutorials/basic_usage_examples/structural_analysis.rst`
- **Line 4**:
  should be shortened;
  add instructions on how to access data as a dictionary.

### `docs/tutorials/basic_usage_examples/trajectory_management.rst`
- **Line 4**:
  Depth/nesting of folder structure for loading – how deep can it go?
  Trajectories not loaded alphabetically → plotting (e.g., membership) can be disordered → reduces comparability in plots
  Is there an option to sort trajectories?
  Depth/nesting of folder structure for loading – how deep can it go?
  --> Missing example for folder structure: e.g., max one PDB per folder, multiple XTCs possible.
  Cache folder for trajectories – relative to script execution path or pipeline cache directory?
  select_atoms: which strings are valid? Include brief note that these are mdtraj selection strings; optionally link or provide example.
  Consensus nomenclature for labels: What is the exact reference for nomenclature consensus?

### `docs/tutorials/learning.rst`
- **Line 21**:
  streamline existing base examples for clarity and efficiency
- **Line 51**:
  Provide Multiple Notebooks – offer various notebooks illustrating different workflows.
  Missing: A page showing where all data is stored (labels, tags, computed arrays, etc.).


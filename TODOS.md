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

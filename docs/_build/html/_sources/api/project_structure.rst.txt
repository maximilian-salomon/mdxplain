Project Structure
=================

- **pipeline/**: Central coordination layer providing the `PipelineManager` entry point
- **trajectory/**: Load and manage MD trajectories with tagging and labeling support
- **feature/**: Compute molecular features (contacts, distances, torsions, DSSP, SASA, coordinates)
- **feature_selection/**: Select and combine features into analysis-ready matrices  (Define columns of final analysis data-matrix)
- **decomposition/**: Dimensionality reduction (PCA, Kernel PCA, Contact Kernel PCA, Diffusion Maps)
- **clustering/**: Identify conformational states (DPA, DBSCAN, HDBSCAN)
- **data_selector/**: Select and combine trajectory frames into analysis-ready matrices by cluster assignment or custom criteria (Define rows of final analysis data-matrix)
- **comparison/**: Compare groups (one-vs-rest, pairwise) to identify discriminative features
- **feature_importance/**: Determine which features characterize two different systems, states, etc.
- **analysis/**: Structural metrics (RMSD, RMSF) and feature statistics
- **plots/**: Visualization tools for results
- **utils/**: Shared utilities and data handling functions

.. code:: text

    mdxplain/
    ├── mdxplain/                      # Main Python package
    │   ├── pipeline/                  # Central pipeline coordination
    │   │   ├── managers/              # PipelineManager (main entry point)
    │   │   ├── entities/              # PipelineData (central data container)
    │   │   └── helper/                # Pipeline utilities
    │   │
    │   ├── trajectory/                # Trajectory loading and management
    │   │   ├── managers/              # TrajectoryManager
    │   │   ├── entities/              # TrajectoryData
    │   │   └── helper/                # Selection, metadata, validation helpers
    │   │
    │   ├── feature/                   # Feature computation
    │   │   ├── managers/              # FeatureManager
    │   │   ├── entities/              # FeatureData
    │   │   ├── services/              # FeatureAddService (add.distances(), etc.)
    │   │   ├── feature_type/          # Feature implementations
    │   │   │   ├── contacts/          # Binary contact features
    │   │   │   ├── distances/         # Residue-residue distances
    │   │   │   ├── torsions/          # Backbone/side-chain angles
    │   │   │   ├── dssp/              # Secondary structure
    │   │   │   ├── sasa/              # Solvent accessible surface area
    │   │   │   ├── coordinates/       # XYZ coordinates
    │   │   │   └── interfaces/        # FeatureTypeBase
    │   │   └── helper/                # Feature computation utilities
    │   │
    │   ├── feature_selection/         # Feature matrix selection
    │   │   ├── managers/              # FeatureSelectorManager
    │   │   ├── entities/              # FeatureSelectorData
    │   │   ├── services/              # Selection services
    │   │   └── helpers/               # Selection utilities
    │   │
    │   ├── decomposition/             # Dimensionality reduction
    │   │   ├── managers/              # DecompositionManager
    │   │   ├── entities/              # DecompositionData
    │   │   ├── services/              # DecompositionAddService
    │   │   ├── decomposition_type/    # Decomposition implementations
    │   │   │   ├── pca/               # Principal Component Analysis
    │   │   │   ├── kernel_pca/        # Kernel PCA
    │   │   │   ├── contact_kernel_pca/# Contact-optimized Kernel PCA
    │   │   │   ├── diffusion_maps/    # Diffusion Maps
    │   │   │   └── interfaces/        # DecompositionTypeBase
    │   │
    │   ├── clustering/                # Clustering algorithms
    │   │   ├── managers/              # ClusterManager
    │   │   ├── entities/              # ClusterData
    │   │   ├── services/              # ClusterAddService
    │   │   ├── cluster_type/          # Clustering implementations
    │   │   │   ├── dpa/               # Density Peak Advanced
    │   │   │   ├── dbscan/            # DBSCAN
    │   │   │   ├── hdbscan/           # Hierarchical DBSCAN
    │   │   │   └── interfaces/        # ClusterTypeBase
    │   │
    │   ├── data_selector/             # Frame selection by cluster/criteria
    │   │   ├── managers/              # DataSelectorManager
    │   │   ├── entities/              # DataSelectorData
    │   │   └── helpers/               # Selection helpers
    │   │
    │   ├── comparison/                # Multi-group comparisons
    │   │   ├── managers/              # ComparisonManager
    │   │   ├── entities/              # ComparisonData
    │   │   └── helpers/               # Comparison utilities
    │   │
    │   ├── feature_importance/        # Feature importance analysis
    │   │   ├── managers/              # FeatureImportanceManager
    │   │   ├── entities/              # FeatureImportanceData
    │   │   ├── services/              # Importance calculation services
    │   │   ├── analyzer_types/        # Analyzer implementations
    │   │   │   └── decision_tree/     # Decision tree analyzer
    │   │   └── helpers/               # Analysis helpers
    │   │
    │   ├── analysis/                  # Structural and statistical analysis
    │   │   ├── managers/              # AnalysisManager
    │   │   ├── services/              # FeatureAnalysisService
    │   │   └── structure/             # Structural analysis
    │   │       ├── calculators/       # RMSD, RMSF calculators
    │   │       └── services/          # StructureAnalysisService
    │   │
    │   ├── plots/                     # Visualization utilities
    │   ├── utils/                     # General utilities and helpers
    │   └── __init__.py                # Package initialization
    │
    ├── tutorials/                     # Tutorial notebooks
    │   └── 00_introduction.ipynb      # Complete workflow example
    │
    ├── tests/                         # Comprehensive test suite
    ├── docs/                          # Documentation
    ├── data/                          # Example datasets
    ├── dev_scripts/                   # Development scripts
    │
    ├── Makefile                       # Build and development automation
    ├── pyproject.toml                 # Package configuration
    ├── README.md                      # This file
    ├── AI_USAGE.md                    # AI assistance declaration
    └── LICENSE                        # LGPL v3.0
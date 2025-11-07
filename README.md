# mdxplain

A Python toolkit for scalable molecular dynamics trajectory analysis, combining modular workflows, memory-efficient processing and interpretable machine learning via decision trees to identify key conformational features and streamline complex pipelines.

**Developer:** Maximilian Salomon (Software), Maeve Branwen Butler (ReadTheDocs Documentation)

**Version:** 1.0.0

## Common Use Cases

mdxplain is designed for **interpretable molecular dynamics trajectory analysis**, bridging the gap between raw simulation data and biological insights. It excels when you need to move from thousands of MD frames to actionable, explainable results while maintaining complete workflow reproducibility.

**Core Strengths:**
- **Automated conformational analysis** with interpretable feature importance
- **Memory-efficient processing** for trajectories exceeding available RAM
- **Multi-trajectory comparison workflows** with tag-based organization
- **Complete pipeline persistence** for reproducible science
- **Decision tree explainability** for understanding state-defining features

**Use Cases by Category:**

**Conformational Analysis**
- Identify distinct conformational states in protein dynamics
- Characterize states by specific molecular interactions (contacts, distances)
- Understand conformational transitions and mechanisms
- Map allosteric pathways and communication networks

**Comparative Studies**
- Wild-type vs. mutant: Find molecular features affected by mutations
- Ligand-bound vs. apo: Identify binding-induced conformational changes
- Condition comparisons: pH, temperature, force effects on structure
- Multi-system analysis: Compare different proteins, homologs, or variants

**Feature Engineering**
- Prepare structured feature matrices for machine learning
- Generate input data for Markov state models (MSM)
- Export custom feature sets for downstream analysis
- Create interpretable descriptors for QSAR/QSPR studies

**Trajectory Exploration**
- RMSD/RMSF analysis for structural characterization
- Contact frequency patterns to identify key interactions
- Statistical feature analysis across conditions
- Quality assessment and validation of simulations

**When to Use mdxplain:**
- Large-scale trajectory analysis requiring memory efficiency
- Need for interpretable, explainable results (not black-box ML)
- Multi-system comparisons with systematic organization
- Reproducible workflows with complete pipeline saving
- Automated conformational state discovery and characterization

## Installation

### Prerequisites

- Python >= 3.8 (Python 3.12 recommended)
- Virtual environment manager (venv or conda)

### Quick Setup

mdxplain uses a Makefile for streamlined installation and development workflows:

```bash
# Clone repository
git clone https://github.com/BioFreak95/mdxplain.git
cd mdxplain

# Full setup with development tools and Jupyter (recommended)
make setup-full-env
source mdxplain-venv/bin/activate  # Linux/Mac

# Or using conda
make setup-conda-full-env
conda activate mdxplain
```

### Installation Options

| Command | Environment | Description |
|---------|-------------|-------------|
| `make setup-env` | venv | Production dependencies only |
| `make setup-jupyter-env` | venv | Production + Jupyter |
| `make setup-dev-env` | venv | Development tools (linting, testing) |
| `make setup-full-env` | venv | Development + Jupyter (recommended) |
| `make setup-conda-env` | conda | Production dependencies only |
| `make setup-conda-jupyter-env` | conda | Production + Jupyter |
| `make setup-conda-dev-env` | conda | Development tools |
| `make setup-conda-full-env` | conda | Development + Jupyter |

### Development Commands

```bash
make help          # Show all available commands
make test          # Run tests
make lint          # Run code quality checks
make format        # Format code with black and isort
make jupyter       # Start JupyterLab
make html         # Make documentation html build
make clean         # Remove environments and cache files
```

## Project Structure

```
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
```

### Module Descriptions

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

## Declaration of AI Tool Usage

This project was developed with AI assistance. For complete details about AI tools used, development process, and author contributions, see [AI_USAGE.md](AI_USAGE.md).

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL v3.0). See [LICENSE](LICENSE) for details.

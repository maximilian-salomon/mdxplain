# mdxplain

A Python toolkit for scalable molecular dynamics trajectory analysis, combining modular workflows, memory-efficient processing and interpretable machine learning via decision trees to identify key conformational features and streamline complex pipelines.

Clone or download the repository, then open the [documentation](docs/_build/html/landing.html) in your browser for comprehensive examples and workflow tutorial.

**Developer:** Maximilian Salomon (Software), Maeve Branwen Butler (ReadTheDocs Documentation)

**Version:** 0.1.0

![mdxplain Overview](docs/images/mdxplain_overview.png)

**mdxplain** creates a complete analytical loop that transforms large-scale molecular dynamics trajectory data into explainable observations through feature calculations, dimensionality reduction, clustering, and comparisons.

The **pipeline** begins by importing massive datasets of 1-100+ simulations (each containing 100 to over a million frames) along with metadata tags that classify different simulation conditions, mutations, or experimental variants.

Central to the framework is a feature extraction engine that computes **descriptive features** including distances, contacts, torsions, DSSP secondary structure, SASA, and atomic coordinates, transforming high-dimensional data for efficient analysis.

These features undergo **dimensionality reduction** using methods like PCA, Kernel PCA, or Diffusion Maps, followed by **clustering** with DPA, DBSCAN, or HDBSCAN to identify conformational states.

The system then applies **feature selection** using metrics such as variance, range, or transitions to prioritize the most relevant features.

Users can select specific data subsets based on frames, clusters, or tags, which are organized into a **final datamatrix** with frames as rows and features as columns.

mdxplain supports **systematic comparison** of different datasets, such as mutated versus wildtype proteins.

At its core, the framework uses **feature importance** analysis to identify system-specific molecular "fingerprints". **Decision tree**-like visualizations highlight which feature combinations best separate systems, reframing "what happened?" to "why did it happen?".

The comprehensive output includes analysis metrics, data exports, and visualizations such as energy landscapes, 3D molecular structures, cluster dynamics, and decision trees.

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

## Module Descriptions

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

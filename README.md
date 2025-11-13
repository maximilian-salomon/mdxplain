# mdxplain

A Python toolkit for scalable molecular dynamics trajectory analysis, combining modular workflows, memory-efficient processing and interpretable machine learning via decision trees to identify key conformational features and streamline complex pipelines.

![mdxplain Overview](docs/images/mdxplain_overview.png)

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

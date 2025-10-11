# mdxplain

A Python toolkit for scalable molecular dynamics trajectory analysis, combining modular workflows, memory-efficient processing and interpretable machine learning via decision trees to identify key conformational features and streamline complex pipelines.

**Developer:** Maximilian Salomon (Software), Maeve Buttler (ReadTheDocs Documentation)

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
make clean         # Remove environments and cache files
```

## Getting Started

1. **Start Jupyter**: `make jupyter`
2. **Explore tutorial**: Open `tutorials/00_introduction.ipynb` for a comprehensive workflow example
3. **Read documentation**: Visit project documentation for API details

### Memory-Efficient Processing

For large trajectories, mdxplain supports memory-mapped processing:

```python
# Enable memory mapping for datasets larger than RAM
pipeline = PipelineManager(use_memmap=True, chunk_size=1000)
```

**Memory mapping guidelines:**
- **Enable** for trajectories approaching/exceeding available RAM
- **Enable** when analyzing multiple large trajectories simultaneously
- **Disable** for small/medium datasets that fit in RAM (faster processing)
- **Chunk size**: Start with 2000 frames; increase if RAM allows, decrease if memory pressure occurs

## Usage

### Core Concepts

mdxplain provides a **PipelineManager** as the central entry point for all molecular dynamics trajectory analysis. The architecture follows a **builder pattern**, where complex analyses are constructed step-by-step through a fluent, manager-based interface.

**Key Design Principles:**
- **PipelineManager**: Single entry point that coordinates all analysis operations
- **Manager-based Architecture**: Specialized managers for trajectories, features, clustering, decomposition, etc.
- **Pipeline Data**: Central data structure (`pipeline.data`) that accumulates all analysis results
- **Fluent API**: Intuitive, chainable methods like `pipeline.feature.add.contacts()`

### Quick Start Example

Here's a complete conformational analysis workflow:

```python
from mdxplain import PipelineManager

# Initialize pipeline
pipeline = PipelineManager(use_memmap=True, chunk_size=1000)

# Load trajectory and add residue labels
pipeline.trajectory.load_trajectories("data/2RJY/")
pipeline.trajectory.add_labels(traj_selection="all")

# Compute features
pipeline.feature.add.distances()
pipeline.feature.add.contacts(cutoff=4.5)

# Create feature selection
pipeline.feature_selector.create("contacts_only")
pipeline.feature_selector.add.contacts("contacts_only", "all")
pipeline.feature_selector.select("contacts_only")

# Dimensionality reduction
pipeline.decomposition.add.contact_kernel_pca(
    n_components=10, gamma=0.001, selection_name="contacts_only"
)

# Clustering
pipeline.clustering.add.dpa(selection_name="ContactKernelPCA", Z=2.0)

# Create data selectors for each cluster
n_clusters = pipeline.data.cluster_data["DPA"].get_n_clusters()
for i in range(n_clusters):
    pipeline.data_selector.create(f"cluster_{i}")
    pipeline.data_selector.select_by_cluster(f"cluster_{i}", "DPA", [i])

# Feature importance analysis
cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
pipeline.comparison.create_comparison(
    name="cluster_comparison", mode="one_vs_rest",
    feature_selector="contacts_only",
    data_selectors=cluster_names
)
pipeline.feature_importance.add.decision_tree(
    comparison_name="cluster_comparison", max_depth=3
)

# Get top discriminative features
top_features = pipeline.feature_importance.get_top_features(
    analysis_name="feature_importance",
    comparison_identifier="cluster_0_vs_rest", n=5
)

# Save complete analysis
pipeline.save("my_analysis.pkl")
```

See `tutorials/00_introduction.ipynb` for a detailed walkthrough of this workflow.

### Core Functionality

#### 1. Trajectory Management

mdxplain supports all MDTraj-compatible trajectory formats (xtc, dcd, trr, pdb, h5, netcdf, etc.) and provides powerful tools for organizing and manipulating multiple trajectories. Tags enable systematic organization for comparative analyses, while labels add residue metadata like consensus nomenclature or fragment definitions.

```python
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
```

**Consensus Nomenclature for Structured Proteins**

For proteins with established numbering schemes (GPCRs, kinases, G-proteins), consensus nomenclature enables biologically meaningful feature selection. This allows selecting structurally/functionally equivalent positions across different proteins.

```python
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
```

#### 2. Feature Computation

mdxplain provides six feature types for molecular dynamics analysis:

```python
# Residue-residue distances (closest heavy-atom pairs)
pipeline.feature.add.distances(excluded_neighbors=1)

# Binary contacts (interaction indicator)
pipeline.feature.add.contacts(cutoff=4.5)

# Backbone and side-chain torsion angles
pipeline.feature.add.torsions()  # phi, psi, omega, chi1-4

# Secondary structure assignment
pipeline.feature.add.dssp()

# Solvent accessible surface area
pipeline.feature.add.sasa()

# Atomic coordinates (xyz positions)
pipeline.feature.add.coordinates()
```

#### 3. Feature Selection

Feature selection defines which molecular features (matrix columns) enter your analysis. mdxplain provides a custom parsing syntax that supports residue names, IDs, consensus nomenclature, and logical operations for flexible feature filtering.

**How the Parser Works:**

The selection language is case-insensitive and automatically recognizes keywords. Ranges use `-` (e.g., `10-50`), lists are space-separated (e.g., `ALA HIS GLY`), `and` combines conditions (intersection), and `not` excludes matches (difference).

**Selection Keywords:**

**`res` - Residue Names**
- **What**: Select by amino acid type
- **Why use it**:
  - Analyze hydrophobic core: `res ALA VAL LEU ILE PHE`
  - Study charged interactions: `res ARG LYS ASP GLU`
  - Focus on specific residue types for functional analysis
  - Test chemical property hypotheses (aromatic, polar, etc.)
- **Example**: `res ARG LYS`

**`resid` - Residue IDs**
- **What**: Select by residue numbers in PDB
- **Why use it**:
  - Target known functional regions from literature: `resid 120-140` => binding site
  - Test structure-based hypotheses from crystal structures
  - Focus on specific structural elements (loops, termini)
  - Region-specific analysis based on domain knowledge
- **Example**: `resid 10-50` => N-terminal region

**`seqid` - Sequence IDs**
- **What**: Select by sequence position
- **Why use it**: Alignment-based analyses, homologous position comparisons across proteins
- **Example**: Useful in multi-chain complexes

**`consensus` - Consensus Nomenclature**
- **What**: Structure-based biological numbering (GPCRdb, CGN, kinase numbering)
- **Why use it**:
  - **Functional motifs**: `consensus 3x50` => DRY motif in GPCRs (conserved activation switch)
  - **Structural elements**: `consensus 7x*` => entire TM7 helix
  - **Conserved networks**: `consensus *40-*50` => positions 40-50 e.g. across all TM helices (allosteric pathways)
  - **Cross-protein comparison**: Same consensus position = functionally equivalent
- **Patterns**:
  - Single: `7x53` => NPxxY tyrosine (Y7.53, activation-associated)
  - Range: `7x50-8x50` => TM7-TM8 interface region
  - Wildcard: `7x*` => all TM7 positions
  - Multi-pattern: `*40-*50` => positions 40-50 in all TM helices
  - G-proteins: `G.H5.*` => helix 5 of G-alpha
  - **`all` prefix**: Include residues without consensus labels
    - `7x-8x` => only residues WITH consensus labels from 7x to 8x
    - `all 7x-8x` => ALL residues from 7x to 8x, including those without consensus (loops, unstructured regions)
    - **Why**: Capture complete structural regions including flexible elements
    - **Example**: `all 7x-8x` includes TM7, intracellular loop, and TM8
- **Note**: This is not fixed to a consensus. `*40-*50` for example is looking for a pattern that matches this. It does not have to be a TM helix, if you have other consensus labels. It is a pattern matching. It takes the first consensus label entry with `*40` and then takes the next with `*50` and takes the range in between. If it does NOT find a start or end point it takes the next residue in the list starting or ending with seqid.

**`all`**
- **What**: Select all features
- **Why use it**: Initial exploration, global pattern discovery, unbiased analysis

**Logical Operators**
- **`and`** - Intersection (both conditions must be true)
  - **Why**: Specific combinations: `res ARG and resid 120-140` => positive charges in binding site
- **`not`** - Exclusion (remove matches)
  - **Why**: "All except" patterns: `res ALA and not resid 25` => all alanines except position 25

```python
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
```

**Practical Examples with Biological Context:**

```python
# GPCR activation contact (ionic lock)
# Selects a known constraining interaction in GPCRs:
# - 3x50: DRY motif arginine (R3.50)
# - 6x30: conserved glutamate (E6.30) on TM6
pipeline.feature_selector.add.contacts("gpcr", "consensus 3x50 and consensus 6x30")

# G-protein binding interface
# Combines two structural elements for GPCR-G-protein coupling:
# - G.H5.*: Complete helix 5 of G-alpha protein (major contact surface)
# - 8x*: Helix 8 of receptor (intracellular C-terminus)
# Interface contacts important for G-protein activation
pipeline.feature_selector.add.contacts("interface", "consensus G.H5.* and consensus 8x*")

# Binding pocket aromatic cage
# Targets aromatic residues in known binding region:
# - Aromatic amino acids (PHE, TRP, TYR) form π-stacking interactions
# - Residues 100-150: Binding pocket from crystal structure
# Creates feature set for ligand-binding site characterization
pipeline.feature_selector.add.contacts("pocket", "res PHE TRP TYR and resid 100-150")

# Hydrophobic core interactions
# Selects non-polar residues forming protein core:
# - ALA, VAL, LEU, ILE, PHE: Hydrophobic amino acids
# Monitors core packing stability and folding integrity
pipeline.feature_selector.add.contacts("core", "res ALA VAL LEU ILE PHE")

# TM helix interface including loop (with 'all' prefix)
# Captures complete structural region:
# - Without 'all': Only labeled TM7/TM8 residues
# - With 'all': Includes intracellular loop between helices
# Complete interface for conformational change analysis
pipeline.feature_selector.add.contacts("tm7_loop_tm8", "all 7x-8x")
```

**Feature Reduction - Statistical Feature Filtering**

Feature reduction applies statistical criteria to filter features AFTER they have been computed but BEFORE analysis. This reduces dimensionality by keeping only features that meet specific biological or statistical criteria.
So it does not change the feature-data but is specific for this selection.

**What is Feature Reduction?**
- Computed features are analyzed for statistical properties (frequency, variability, transitions)
- Features failing threshold criteria are removed from the feature set
- Reduces noise, computational cost, and focuses analysis on relevant features
- Two approaches: inline reduction (`.with_xxx_reduction()`) or pre-reduction (`feature.reduce_data()`)
- Pre reduction add this permanent to feature-data. It creates a new permanant data matrix.
- Post reduction is specified to this specific selection and does not create a specific matrix, but keep the indices.

**When to Use Feature Reduction:**
- **Too many features**: Thousands of distances/contacts overwhelm analysis
- **Focus on variability**: Only analyze features that actually change (cv, std, variance, range)
- **Focus on stability**: Only analyze persistent interactions (frequency, stability)
- **Focus on dynamics**: Only analyze features showing transitions
- **Multi-trajectory**: Ensure features exist across all systems (common_denominator)

**Two Reduction Approaches:**

**Approach 1: Inline Reduction (during selection)**
```python
# Apply reduction while selecting features
# Advantage: Specific criteria per selection
pipeline.feature_selector.add.contacts.with_frequency_reduction(
    "stable_contacts", "resid 100-200",
    threshold_min=0.7  # Only contacts formed in >70% of frames
)
```

**Approach 2: Pre-Reduction + use_reduced=True**
```python
# Step 1: Globally reduce features across all trajectories
pipeline.feature.reduce_data(
    feature_type="distances",
    metric="cv",  # Coefficient of variation
    threshold_min=0.1,  # Only distances with CV > 0.1 (variable distances)
    cross_trajectory=True  # Feature must pass in ALL trajectories
)

# Step 2: Use pre-reduced features in selection
pipeline.feature_selector.add.distances(
    "variable_distances", "all",
    use_reduced=True  # Uses reduced data from Step 1
)
```

**Reduction Methods by Feature Type:**

**Contacts** (Binary interaction indicators):
- `with_frequency_reduction()`: Contact formation frequency (0.0-1.0)
  - **Use**: Find stable interactions (high freq) or transient contacts (low freq)
  - **Example**: `threshold_min=0.8` => contacts formed in >80% of frames
- `with_stability_reduction()`: Contact persistence over time
  - **Use**: Identify consistently maintained interactions vs. flickering contacts
- `with_transitions_reduction()`: Contact formation/breaking events
  - **Use**: Find dynamic regions with frequent state changes

**Distances** (Continuous separation values):
- `with_cv_reduction()`: Coefficient of variation (std/mean)
  - **Use**: Normalized variability, independent of absolute distance scale
  - **Example**: `threshold_min=0.15` => distances varying by >15% of mean
- `with_std_reduction()`, `with_variance_reduction()`: Absolute variability
  - **Use**: Find distances with large absolute fluctuations
- `with_range_reduction()`: max - min distance
  - **Use**: Identify distances exploring wide conformational space
- `with_transitions_reduction()`: Distance change events
  - **Use**: Detect conformational switching between states
- `with_mean/min/max/mad_reduction()`: Value-based filtering
  - **Use**: Filter by typical distance values

**Coordinates** (XYZ positions):
- `with_rmsf_reduction()`: Root mean square fluctuation
  - **Use**: Focus on flexible regions, identify mobile loops
  - **Example**: `threshold_min=2.0` => atoms fluctuating >2 Å
- `with_std_reduction()`, `with_cv_reduction()`: Position variability
  - **Use**: Similar to RMSF, identify dynamic vs. rigid regions
- `with_range/variance/mad_reduction()`: Position spread metrics

**Torsions** (Dihedral angles):
- `with_transitions_reduction()`: Angular transitions (rotamer changes)
  - **Use**: Identify side chains or backbone angles switching conformations
  - **Example**: `threshold_min=10` => angles with >10 transition events
- `with_std/cv/variance_reduction()`: Angular variability
  - **Use**: Find flexible torsions vs. constrained angles

**SASA** (Solvent accessible surface area):
- `with_cv_reduction()`: Exposure variability
  - **Use**: Find residues alternating between buried/exposed
  - **Example**: `threshold_max=0.3` => relatively constant exposure
- `with_burial_fraction_reduction()`: Fraction of time buried
  - **Use**: Classify residues as core (high burial) vs. surface (low burial)
- `with_range/std_reduction()`: Exposure dynamics

**DSSP** (Secondary structure):
- `with_transitions_reduction()`: Structure type changes
  - **Use**: Find regions switching between helix/sheet/coil
- `with_stability_reduction()`: Structure persistence
  - **Use**: Identify stable vs. unstable secondary structure elements

#### 4. Dimensionality Reduction

Four decomposition methods are available:

```python
# Standard PCA
pipeline.decomposition.add.pca(
    n_components=10, selection_name="my_selection"
)

# Kernel PCA with RBF kernel
pipeline.decomposition.add.kernel_pca(
    n_components=10, kernel='rbf', gamma=0.01,
    selection_name="my_selection"
)

# Contact Kernel PCA (optimized for binary contact data)
pipeline.decomposition.add.contact_kernel_pca(
    n_components=10, gamma=0.001, selection_name="contacts_only"
)

# Diffusion Maps
pipeline.decomposition.add.diffusion_maps(
    n_components=10, selection_name="my_selection"
)
```

#### 5. Clustering

Three density-based clustering algorithms:

```python
# Density Peak Advanced (DPA) - automatic cluster number detection
# In our personal experience often good results and easy to use.
pipeline.clustering.add.dpa(
    selection_name="ContactKernelPCA", Z=3.0
)

# DBSCAN
pipeline.clustering.add.dbscan(
    selection_name="my_decomposition", eps=0.5, min_samples=5
)

# HDBSCAN
pipeline.clustering.add.hdbscan(
    selection_name="my_decomposition", min_cluster_size=10
)
```

#### 6. Structural Analysis

Structural metrics provide quantitative measures of protein dynamics and conformational changes. mdxplain offers multiple RMSD/RMSF variants optimized for different analysis scenarios.

**RMSD Metrics - Which Variant to Use:**

**`.rmsd.mean` - Standard RMSD (Root Mean Square Deviation)**
- **What**: Classic RMSD using arithmetic mean
- **When to use**:
  - Standard conformational analysis and literature comparison
  - Systems without highly flexible regions
  - Fastest computation
- **Avoid when**: System has highly flexible regions that would dominate the metric

**`.rmsd.median` - Robust RMSD (Root Median Square Deviation)**
- **What**: Square root of median of squared deviations (instead of mean)
- **When to use**:
  - Systems with occasional outlier atoms (flexible loops + rigid core)
  - Multi-domain proteins with independent movement
  - Combined folded/unstructured regions
- **Why**: Robust against outlier atoms affecting the metric

**`.rmsd.mad` - MAD RMSD (Median Absolute Deviation)**
- **What**: Most robust RMSD variant based on MAD
- **When to use**:
  - Extremely flexible systems
  - Proteins with intrinsically disordered regions
  - Maximum outlier resistance needed
- **Why**: Statistically most robust, least affected by rare large-amplitude moves

**RMSD Modes - What to Measure:**

**`.to_reference(reference_traj, reference_frame, atom_selection="all")`**
- **What**: RMSD to a fixed reference frame
- **Why use it**:
  - Track structural drift from starting structure
  - Measure equilibration (distance from crystal structure)
  - Monitor return to specific conformation
  - Stability relative to known structure (X-ray, cryo-EM)
- **Example**: `rmsd.mean.to_reference(0, 0)` => drift from initial frame

**`.frame_to_frame(lag=1, atom_selection="all")`**
- **What**: RMSD between consecutive or lag-separated frames
- **Why use it**:
  - Quantify local structural fluctuations
  - Identify smooth vs. jerky dynamics
  - Detect conformational transition events
  - lag=1: frame-to-frame noise, lag=10+: larger conformational shifts
- **Example**: `rmsd.mean.frame_to_frame(lag=10)` => local dynamics

**`.window_frame_to_start()`**
- **What**: Sliding window RMSD to window start
- **Why use it**:
  - Assess equilibration within time windows
  - Local stability analysis
  - Detect gradual structural drift

**`.window_frame_to_frame()`**
- **What**: Sliding window frame-to-frame RMSD
- **Why use it**:
  - Track local fluctuations over time
  - Identify periods of high/low dynamics

**RMSF Metrics - Which Variant:**

**`.rmsf.mean` - Standard RMSF (Root Mean Square Fluctuation)**
- **What**: Classic per-residue flexibility metric
- **When to use**:
  - Standard flexibility analysis
  - Comparison with experimental B-factors
  - Identify rigid vs. flexible regions

**`.rmsf.median` - Robust RMSF**
- **What**: Square root of median of squared fluctuations (instead of mean)
- **When to use**: Systems with occasional large-amplitude rare events
- **Why**: Robust against rare outlier movements

**`.rmsf.mad` - MAD RMSF**
- **What**: Most robust fluctuation measure
- **When to use**: Very flexible systems requiring maximum robustness

**RMSF Modes - Resolution Level:**

**`.per_atom`**
- **What**: Atom-level fluctuations
- **Why use it**:
  - Detailed side-chain dynamics
  - Specific functional atoms (active site, binding residues)
  - High-resolution flexibility mapping
- **Example**: Side-chain rotamer dynamics

**`.per_residue`**
- **What**: Residue-aggregated fluctuations
- **Why use it**:
  - Protein-wide flexibility overview
  - Domain mobility comparison
  - Experimental B-factor comparison
  - Flexibility profiles
- **Example**: Identify flexible loops vs. rigid helices

**Per-Residue Aggregation Methods:**

When using `.per_residue`, atom-level RMSF values within each residue must be aggregated to a single per-residue value. Four aggregation strategies are available:

**`.with_mean_aggregation`** (Default, simple average)
- **What**: Arithmetic mean of atom-level RMSF values: `mean(RMSF_atoms)`
- **When to use**:
  - Standard flexibility profiles
  - All atoms in residue have similar flexibility
  - No extreme outlier atoms
- **Sensitivity**: Affected by outlier atoms (very flexible side-chain tips)
- **Example**: Backbone CA atoms or residues with uniform flexibility

**`.with_median_aggregation`** (Robust to outliers)
- **What**: Median of atom-level RMSF values: `median(RMSF_atoms)`
- **When to use**:
  - Long flexible side chains (e.g., LYS, ARG) with rigid backbone
  - Mixed flexibility within residue (some atoms rigid, others flexible)
  - Want typical flexibility, not influenced by extreme atoms
- **Benefit**: Terminal side-chain atoms don't dominate the residue score
- **Example**: Surface residues with flexible tips but stable core

**`.with_rms_aggregation`** (Emphasize larger values)
- **What**: Root-mean-square of RMSF values: `sqrt(mean(RMSF_atoms²))`
- **When to use**:
  - Want to emphasize larger fluctuations more strongly
  - Quadratic weighting is desired (larger RMSF = disproportionately higher weight)
- **Effect**: Residue score dominated by most flexible atoms
- **Example**: Identifying highly dynamic regions where any flexible atom matters

**`.with_rms_median_aggregation`** (Emphasize + robust)
- **What**: Root-median-square of RMSF values: `sqrt(median(RMSF_atoms²))`
- **When to use**:
  - Want quadratic weighting but robust against extreme outliers
  - Very flexible side chains with occasional large-amplitude jumps
- **Effect**: Emphasizes typical flexibility, ignores rare extremes
- **Example**: Disordered regions with occasional extreme conformations

**Aggregation Selection Guide:**

| Your Goal | Recommended Method | Why |
|-----------|-------------------|-----|
| Standard flexibility profile | `mean` | Simple average, standard representation |
| Residues with very flexible tips | `median` | Ignores outlier atoms at side-chain ends |
| Emphasize most flexible atoms | `rms` | Larger fluctuations get more weight (quadratic) |
| Flexibility with outlier protection | `rms_median` | Emphasizes flexibility but ignores extremes |

**Practical Examples:**

```python
# Example 1: Equilibration monitoring
# Question: Has the simulation equilibrated? How far does structure drift from starting point?
# Why: mean.to_reference = standard metric for comparing to crystal structure
# Why: backbone = ignores side-chain noise, focuses on secondary structure stability
# Use case: Quality control, detect slow conformational drift, assess convergence
rmsd_crystal = pipeline.analysis.structure.rmsd.mean.to_reference(
    reference_traj=0, reference_frame=0, atom_selection="backbone"
)

# Example 2: Flexible multi-domain protein
# Question: What is overall conformational change in protein with mobile loops?
# Why: median.to_reference = robust against flexible loop outliers affecting global RMSD
# Why: "protein" selection = all protein atoms, including flexible regions
# Use case: Multi-domain proteins, antibodies, disordered regions that would dominate mean
rmsd_robust = pipeline.analysis.structure.rmsd.median.to_reference(
    reference_traj=0, reference_frame=0, atom_selection="protein"
)

# Example 3: Conformational transition detection
# Question: Are there sudden conformational changes or smooth dynamics?
# Why: frame_to_frame(lag=5) = local structural changes every 5 frames
# Why: mean metric = sufficient for well-behaved dynamics
# Why: CA atoms = coarse-grained, computationally efficient
# Use case: Identify transition events, measure local stability, detect jerky vs smooth motion
rmsd_dynamics = pipeline.analysis.structure.rmsd.mean.frame_to_frame(
    lag=5, atom_selection="name CA"
)

# Example 4: Standard flexibility profile
# Question: Which regions are rigid vs flexible?
# Why: rmsf.mean.per_residue = standard
# Why: with_mean_aggregation (default) = balanced residue flexibility
# Why: CA atoms = one value per residue, standard representation
# Use case: identify flexible loops/hinges
rmsf_profile = pipeline.analysis.structure.rmsf.mean.per_residue.to_mean_reference(
    atom_selection="name CA"
)

# Example 5: Binding site side-chain flexibility
# Question: How do side chains move in the binding pocket?
# Why: per_atom = high-resolution, individual atom fluctuations
# Why: mean metric = standard for well-defined binding site
# Why: resid 120-140 = specific binding site region
# Use case: Ligand binding analysis, induced fit, side-chain conformational sampling
rmsf_detailed = pipeline.analysis.structure.rmsf.mean.per_atom.to_mean_reference(
    atom_selection="resid 120-140"  # binding site
)

# Example 6: Multi-domain protein flexibility
# Question: Which domains are rigid vs mobile, with robustness against outliers?
# Why: rmsf.median = robust against occasional large-amplitude motions
# Why: per_residue.to_median_reference = consistent robust statistics at all levels
# Why: with_median_aggregation (implicit) = prevents outlier atoms from dominating
# Use case: Domain motion analysis, linker flexibility, proteins with mobile tails
rmsf_robust = pipeline.analysis.structure.rmsf.median.per_residue.to_median_reference(
    atom_selection="protein"
)

# Example 7: Side-chain rotamer switching (advanced aggregation)
# Question: Which residues show side-chain flexibility with robust aggregation?
# Why: with_rms_aggregation = combines backbone and side-chain motion correctly
# Why: Emphasizes larger fluctuations, suitable for rotamer analysis
# Use case: Functional side-chain dynamics, allosteric communication pathways
rmsf_rotamers = pipeline.analysis.structure.rmsf.mean.per_residue.with_rms_aggregation(
    atom_selection="protein"
)
```

#### 7. Feature Statistics

Feature statistics provide quantitative insights into molecular interaction patterns, dynamic behavior, and conformational preferences. Use statistics to understand contact formation frequencies, distance distributions, or identify key dynamic residues.

**Available Statistics:** mean, std, min, max, median, variance, cv (coefficient of variation), frequency (contacts), stability (contacts), transitions (contacts/distances)

**Use Cases:**
- Contact frequency: "How often does residue X interact with residue Y?"
- Distance distributions: "What is the typical separation between domains?"
- Dynamic patterns: "Which contacts are stable vs. transient?"
- Trajectory-specific vs. global behavior

```python
# Contact frequency analysis (global)
contact_freq = pipeline.analysis.features.contacts.frequency()
# Returns: contact formation frequency across all frames

# Statistical analysis of distances
mean_distances = pipeline.analysis.features.distances.mean()
std_distances = pipeline.analysis.features.distances.std()
variance_distances = pipeline.analysis.features.distances.variance()

# Coefficient of variation (normalized variability)
cv_distances = pipeline.analysis.features.distances.cv()

# Trajectory-specific statistics
traj0_mean = pipeline.analysis.features.contacts.mean(traj_selection=0)
traj1_mean = pipeline.analysis.features.contacts.mean(traj_selection=1)

# Interpretation example: Contact frequency
# freq = 0.8 => contact formed in 80% of frames (stable interaction)
# freq = 0.2 => contact formed in 20% of frames (transient interaction)
```

#### 8. Data Selection (Frame/Row Selection)

While FeatureSelector defines which features (matrix columns) to analyze, DataSelector chooses which trajectory frames (matrix rows) to include. This enables subset-based analyses focusing on specific conformational states, conditions, or time windows.

**Core Concept:**
- **FeatureSelector**: Defines matrix columns (which features: contacts, distances, etc.)
- **DataSelector**: Defines matrix rows (which frames: states, trajectories, conditions)
- **Combined**: Creates targeted analysis matrices for specific scientific questions

**Why Use Data Selection:**
- **State-specific analysis**: Focus on folded, unfolded, or intermediate conformations
- **Condition comparison**: Wild-type vs. mutant, ligand-bound vs. apo
- **Outlier removal**: Exclude noise clusters or equilibration frames
- **Data reduction**: Sample large datasets for manageable analysis
- **Combined criteria**: Intersection of multiple selection criteria

**Methods:**

**`create(name)`** - Create named frame selection
```python
pipeline.data_selector.create("folded_frames")
pipeline.data_selector.create("active_state")
```

**`select_by_tags(name, tags, match_all=True, mode="add", stride=1)`** - Tag-based selection

**Parameters:**
- `tags`: List of trajectory tags to match
- `match_all`:
  - `True` (default): Frame needs ALL tags (AND logic) - `["wild_type", "production"]` → both required
  - `False`: Frame needs ANY tag (OR logic) - `["system_A", "system_B"]` → either suffices
- `mode`:
  - `"add"` (default): Union - add frames to selection
  - `"subtract"`: Difference - remove frames from selection
  - `"intersect"`: Intersection - keep only overlap
- `stride`: Sample every Nth frame (1=all frames, 10=every 10th)

**Why use it:**
- Condition-based filtering: `tags=["wild_type"]` → only WT trajectory frames
- System organization: `tags=["system_A", "biased"]` → specific simulation type
- Data sampling: `stride=10` → reduce dataset size

**`select_by_cluster(name, clustering_name, cluster_ids, mode="add", stride=1)`** - Cluster-based selection

**Parameters:**
- `clustering_name`: Name of clustering result (e.g., "DPA", "DBSCAN")
- `cluster_ids`: List of cluster numbers `[0, 1, 2]` or names
- `mode`: Same as tags (add/subtract/intersect)
- `stride`: Sample selected clusters

**Why use it:**
- Conformational states: `cluster_ids=[0]` → only folded state
- Multi-state analysis: `cluster_ids=[0, 2]` → active + intermediate
- Comparative studies: Different selectors for each state
- Outlier removal: `cluster_ids=[-1], mode="subtract"` → exclude noise

**`select_by_indices(name, trajectory_indices, mode="add")`** - Direct frame index selection

The most flexible selection method: specify exact frame numbers for each trajectory. Supports various input formats for maximum convenience.

**Parameters:**

**`trajectory_indices`**: Dictionary mapping trajectory selectors to frame specifications

**Trajectory selectors** (keys):
- `int`: Trajectory index → `0`, `1`, `2`
- `str`: Trajectory name → `"system_A"`
- `str`: Tag pattern → `"tag:biased"` (applies to all trajectories with tag)
- `str`: Name pattern → `"system_*"` (glob-style matching)

**Frame specifications** (values):
- `int`: Single frame → `42`
- `List[int]`: Explicit frames → `[10, 20, 30, 50]`
- `str`: Various string formats:
  - Single: `"42"` → frame 42
  - Range: `"10-20"` → frames 10, 11, ..., 20
  - Comma list: `"10,20,30"` → frames 10, 20, 30
  - Combined: `"10-20,30-40,50"` → frames 10-20, 30-40, and 50
  - All: `"all"` → all frames in trajectory
- `dict`: With stride support:
  - `{"frames": frame_spec, "stride": N}` → apply stride to frame selection
  - Example: `{"frames": "0-100", "stride": 10}` → frames 0, 10, 20, ..., 100

**`mode`**: Selection mode (add/subtract/intersect)

**Why use it:**
- **Manual frame selection**: Choose specific frames from analysis (e.g., representative structures)
- **Time window analysis**: Select equilibrated region, exclude equilibration
- **Sparse sampling**: Reduce dataset size with stride for computational efficiency
- **Complex patterns**: Combine multiple ranges and specific frames
- **Cross-trajectory patterns**: Use tags/names to apply same frame selection to multiple trajectories

**Examples:**

```python
# Example 1: Simple frame selection
# Select specific frames from trajectory 0
pipeline.data_selector.create("custom_frames")
pipeline.data_selector.select_by_indices(
    "custom_frames",
    {0: [100, 200, 300, 500]}  # Four specific frames
)

# Example 2: Time window (exclude equilibration)
# Use frames 1000-5000 from all trajectories (first 1000 = equilibration)
pipeline.data_selector.create("equilibrated")
pipeline.data_selector.select_by_indices(
    "equilibrated",
    {"all": "1000-5000"}  # "all" applies to all loaded trajectories
)

# Example 3: Range with stride (sparse sampling)
# Every 50th frame from equilibrated region for computational efficiency
pipeline.data_selector.create("sparse_equilibrated")
pipeline.data_selector.select_by_indices(
    "sparse_equilibrated",
    {0: {"frames": "1000-5000", "stride": 50}}  # 1000, 1050, 1100, ..., 5000
)

# Example 4: Complex combined ranges
# Multiple time windows and specific frames
pipeline.data_selector.create("complex_selection")
pipeline.data_selector.select_by_indices(
    "complex_selection",
    {
        0: "100-200,500-600,1000",  # Two ranges + single frame
        1: "200-400,800-1000"       # Different ranges for trajectory 1
    }
)

# Example 5: Tag-based frame selection
# Apply same frame selection to all trajectories with "biased" tag
pipeline.data_selector.create("biased_frames")
pipeline.data_selector.select_by_indices(
    "biased_frames",
    {"tag:biased": "500-2000"}  # Frames 500-2000 from all biased trajectories
)

# Example 6: Name pattern matching
# Use all frames from trajectories matching pattern
pipeline.data_selector.create("system_A_all")
pipeline.data_selector.select_by_indices(
    "system_A_all",
    {"system_A*": "all"}  # All frames from trajectories starting with "system_A"
)

# Example 7: Multi-trajectory with different frame selections and stride
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
```

**Practical Examples:**

```python
# Example 1: State-specific analysis
# Scientific question: What features characterize the folded state?
# Why: Focus feature importance analysis only on folded conformation
# Use case: Identify stabilizing interactions in folded state, compare folded vs other states
pipeline.data_selector.create("folded")
pipeline.data_selector.select_by_cluster("folded", "DPA", [0])
# Result: Only frames from cluster 0 (folded state) for downstream analysis

# Example 2: Multi-state conformational analysis
# Scientific question: What features distinguish active and intermediate from inactive?
# Why: Combine multiple conformational states for comparison against baseline
# Use case: Activation pathway analysis, identify shared features of non-inactive states
pipeline.data_selector.create("active_states")
pipeline.data_selector.select_by_cluster("active_states", "DPA", [0, 2])
# Result: Frames from clusters 0 (active) + 2 (intermediate), excludes cluster 1 (inactive)

# Example 3: Production run with data reduction
# Scientific question: What is baseline behavior in wild-type production simulations?
# Why: match_all=True ensures only production WT (not equilibration WT)
# Why: stride=5 reduces computational cost while maintaining statistical validity
# Use case: Reference dataset for mutant comparison, representative WT behavior
pipeline.data_selector.create("wt_prod")
pipeline.data_selector.select_by_tags(
    "wt_prod",
    tags=["wild_type", "production"],  # Must have BOTH tags
    match_all=True,                    # AND logic: production AND wild_type
    stride=5                           # Sample every 5th frame for efficiency
)
# Result: Every 5th frame from wild-type production runs only

# Example 4: Multi-step complex selection with set operations
# Scientific question: What features are specific to cluster 1 in biased simulations?
# Why: 3-step refinement isolates specific subset using set operations
# Use case: Enhanced sampling analysis, isolate converged non-noise states
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

# Example 5: Systematic condition comparison (WT vs Mutant in same state)
# Scientific question: What molecular differences exist between WT and mutant in folded state?
# Why: Compare same conformational state across different systems
# Use case: Mutation effect analysis, identify compensatory changes
#
# Strategy: Create two matched selectors for direct comparison
# - Both select folded state (cluster 0)
# - Different genetic backgrounds (WT vs mutant tags)

# Selector A: Wild-type in folded state
pipeline.data_selector.create("wt_folded")
pipeline.data_selector.select_by_tags("wt_folded", ["wild_type"], mode="add")
# Get all WT frames
pipeline.data_selector.select_by_cluster("wt_folded", "conformations", [0], mode="intersect")
# Narrow to folded conformation only
# Result: WT folded state frames

# Selector B: Mutant in folded state
pipeline.data_selector.create("mutant_folded")
pipeline.data_selector.select_by_tags("mutant_folded", ["mutant"], mode="add")
# Get all mutant frames
pipeline.data_selector.select_by_cluster("mutant_folded", "conformations", [0], mode="intersect")
# Narrow to folded conformation only
# Result: Mutant folded state frames

# These two selectors enable apples-to-apples comparison:
# Same conformational state, different genetic backgrounds
# Use in comparison.create_comparison() to find mutation-specific features

# Example 6: Outlier removal (quality control)
# Scientific question: What is protein behavior excluding simulation artifacts?
# Why: DBSCAN noise cluster (-1) often contains artifacts, transitions, rare events
# Use case: Clean analysis dataset, focus on well-sampled conformations
pipeline.data_selector.create("clean_conformations")
# Start with all frames from main clustering
pipeline.data_selector.select_by_cluster("clean_conformations", "DBSCAN", [0, 1, 2], mode="add")
# Explicitly exclude noise (could also start with "all" and subtract)
# Result: Only well-defined clusters, excludes noise/artifacts
```

**Integration with Comparative Analysis:**

Data selectors define the frame sets (rows) for feature importance and comparison analyses:

```python
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
```

**Common Use Cases:**
- **Conformational analysis**: Select frames by cluster assignment
- **Condition comparison**: Select frames by trajectory tags (WT/mutant, apo/holo)
- **Data reduction**: Stride sampling for large datasets
- **Quality control**: Exclude equilibration, outliers, or unstable frames
- **Multi-criteria filtering**: Combine tags and clusters with mode operations

#### 9. Comparative Analysis & Feature Importance

```python
# Create data selectors for clusters
pipeline.data_selector.create("cluster_0")
pipeline.data_selector.select_by_cluster("cluster_0", "DPA", [0])

# One-vs-rest comparison
pipeline.comparison.create_comparison(
    name="cluster_comparison",
    mode="one_vs_rest",
    feature_selector="contacts_only",
    data_selectors=["cluster_0", "cluster_1", "cluster_2"]
)

# Decision tree feature importance
pipeline.feature_importance.add.decision_tree(
    comparison_name="cluster_comparison",
    analysis_name="importance",
    max_depth=3
)

# Get top discriminative features
top_features = pipeline.feature_importance.get_top_features(
    analysis_name="importance",
    comparison_identifier="cluster_0_vs_rest",
    n=5
)
```

#### 10. Saving and Loading

```python
# Save complete pipeline state
pipeline.save("my_analysis.pkl")

# Load saved pipeline
loaded_pipeline = PipelineManager()
loaded_pipeline.load("my_analysis.pkl")

# Print comprehensive pipeline information
loaded_pipeline.print_info()
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

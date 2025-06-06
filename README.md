# MDCatAFlow (MD Catalytic Analysis Flow)

A high-performance Python toolkit that accelerates molecular dynamics trajectory analysis through automated workflows and memory-efficient processing.

**Author:** Maximilian Salomon  
**Version:** 0.1.0

## Name Explanation

**MDCatAFlow** stands for **MD Catalytic Analysis Flow** and describes the core functionality of the tool:

- **MD** - Molecular Dynamics
- **Cat** - Catalyst - the tool accelerates and automates MD analyses
- **A** - Analysis - comprehensive statistical and dynamic analyses
- **Flow** - Workflow/Pipeline - automated analysis process

The tool acts as a **catalyst** for MD analyses by automating and accelerating complex calculations.

## Features (v0.1)

- Memory-mapped processing for large datasets
- Statistical analyses (mean, variance, CV, transitions)
- Dynamic analyses (stability, variability, transitions)
- Automatic unit detection (nm/Angstrom)
- Chunked processing for memory efficiency
- Flexible data formats (Square/Condensed)
- Dimensionality reduction of MD data
- Structure clustering
- Feature selection using Decision Trees
- Results visualization

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/md_analysis_pipeline.git
cd md_analysis_pipeline

# Create and activate virtual environment
python -m venv mdap
source mdap/bin/activate  # Linux/Mac
# or
.\mdap\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

The pipeline can be executed via Jupyter Notebooks or as Python scripts. Examples can be found in the `notebooks` directory.

## Project Structure

```
md_analysis_pipeline/
├── mdcataflow/         # Main Python package
│   ├── data/          # Data loading utilities
│   ├── utils/         # Core calculation modules
│   └── __init__.py    # Package initialization
├── notebooks/          # Jupyter Notebooks
├── data/              # Example data
├── tests/             # Test cases
├── requirements.txt   # Python dependencies
└── LICENSE            # License information
```

## License

This project is licensed under the Apache License 2. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project was developed with assistance from Claude-4-Sonnet and Cursor AI. 

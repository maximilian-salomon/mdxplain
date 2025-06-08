# MDCatAFlow (MD Catalytic Analysis Flow)

A Python toolkit for scalable molecular dynamics trajectory analysis, combining modular workflows, memory-efficient processing and interpretable machine learning via decision trees to identify key conformational features and streamline complex pipelines.

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

## AI-Assisted Development

Development was conducted using the Cursor IDE, an AI-first programming environment with access to multiple large language models (LLMs). The primary models used were Claude Sonnet 3.7 and 4.0 (Anthropic), Gemini 2.5 (Google), and ChatGPT 4 and 4.1 (OpenAI).

The LLMs assisted with code generation, refactoring, optimization and documentation generation, and also served as a sounding board for brainstorming and exploring alternative implementation strategies.

All architectural design, scientific analysis, algorithm development, prompting, and methodological decision-making, along with comprehensive code review and testing, were conducted by the developers. Responsibility for scientific accuracy and methodological responsibility rests entirely with the human contributors.

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL v3.0). See [LICENSE](LICENSE) for details.

# mdxplain

A Python toolkit for scalable molecular dynamics trajectory analysis, combining modular workflows, memory-efficient processing and interpretable machine learning via decision trees to identify key conformational features and streamline complex pipelines.

**Author:** Maximilian Salomon  
**Version:** 0.1.0

## Features (v0.1)

- Memory-mapped processing for large datasets
- Statistical analyses (mean, variance, CV, transitions)
- Dynamic analyses (stability, variability, transitions)
- Automatic unit detection (nm/Angstrom)
- Chunk-based processing for memory efficiency
- Flexible data formats (Square/Condensed)
- Dimensionality reduction of MD data
- Structure clustering
- Feature selection using Decision Trees
- Results visualization

## Installation

```bash
# Clone repository
git clone https://github.com/BioFreak95/mdxplain.git
cd mdxplain

# Create and activate virtual environment
python -m venv mdxplain
source mdxplain/bin/activate  # Linux/Mac
# or
.\mdxplain\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

The pipeline can be executed via Jupyter Notebooks or as Python scripts. Examples can be found in the `notebooks` directory.

## Project Structure

```
md_analysis_pipeline/
├── mdxplain/              # Main Python package
│   ├── data/             # Data loading and feature utilities
│   │   ├── feature_type/ # Feature type implementations
│   │   │   ├── contacts/ # Contact calculations
│   │   │   ├── distances/# Distance calculations
│   │   │   ├── helper/   # Helper functions
│   │   │   └── interfaces/# Base interfaces
│   │   ├── feature_data.py
│   │   ├── trajectory_data.py
│   │   └── trajectory_loader.py
│   ├── clustering/       # Clustering algorithms
│   ├── decomposition/    # Dimensionality reduction
│   ├── feature_analysis/ # Feature analysis tools
│   ├── feature_selection/# Feature selection
│   ├── plots/           # Visualization tools
│   ├── utils/           # General utilities
│   └── __init__.py      # Package initialization
├── notebooks/           # Jupyter Notebooks
├── data/               # Example data
├── tests/              # Test cases
├── requirements.txt    # Python dependencies
└── LICENSE            # License information
```

## Declaration of AI Tool Usage

The software was developed with the support of AI-assisted coding tools, which were used to assist in generating, refining, and debugging the codebase, always under detailed human direction. All AI interactions were based on comprehensive prompting strategies, workflow plans, and technical specifications designed and executed entirely by the author. All conceptual design, scientific reasoning, and methodological decisions were made solely by the human contributors, with AI serving only as an implementation aid.

### Primary Development Environment

The codebase was developed under the full direction and supervision of the author. AI-assisted coding tools were employed to support the implementation process, following detailed architectural and algorithmic specifications provided by the author.

- **Claude Code (Claude Sonnet 4.0)**: CLI-based AI development tool with full codebase context, used to assist in generating, refactoring, and debugging code segments in accordance with predefined designs and workflows.
- **GitHub Copilot (Claude Sonnet 4.0)**: Integrated into Visual Studio Code to provide inline suggestions, supplemental code generation, and context-aware editing assistance.

Claude Sonnet 4.0 served as the primary large language model for coding assistance throughout most of the project.

### Supplementary Tools and Specific Applications

- **Cursor IDE**: AI-first IDE used predominantly during development of the Trajectory and Feature modules. Primarily integrated Claude Sonnet 4.0, with occasional use of Claude Sonnet 3.7 and Gemini 2.5 Pro. Assisted with context-aware inline code suggestions, code and documentation generation, and prototyping under detailed human guidance. All outputs were reviewed, adapted, and integrated by the author.
- **Kiro AI**: AI-first IDE used for the Decomposition module. Integrated Claude Sonnet 4.0 to assist with context-aware implementation planning, code suggestions, and documentation under detailed human guidance. All outputs were reviewed, adapted, and integrated by the author.
- **Additional Large Language Models**: ChatGPT-4o, o1, GPT-4.1 (OpenAI), Claude Sonnet 3.5 and 3.7 (Anthropic), and Gemini 2.5 Pro (Google). Used in early-phase research, fully in chat-mode for prototyping methods, exploring workflows, generating individual code snippets, and brainstorming ideas.

All supplementary AI tools assisted in generating code snippets, prototyping analysis workflows, exploring methodological options, and brainstorming alternative algorithms. But all final method selection, implementation, architectural decisions, and scientific validation were performed solely by the human contributor.

### AI-Assisted Development Tasks

During package development, AI tools were used to assist in the following tasks, always under detailed human guidance:

- **Code generation**: Specified algorithms were translated into code through iterative, dialogue-driven interactions with the AI. The process involved conversational back-and-forth exchanges to progressively refine and adapt code until it met the author’s specifications, with full control and final implementation decisions retained by the author.
- **Refactoring**: Improving code structure, readability, and maintainability according to author-specified patterns and guidelines.
- **Bug fixing**: Providing suggested solutions for issues identified by the author.
- **Code optimization**: Assisting in identifying potential performance improvements based on author-defined benchmarks and criteria.
- **Code-Documentation**: Generating docstrings and inline documentation following author-provided templates to ensure clarity and consistency.
- **Interactive method exploration**: Acting as a discussion partner to propose alternative approaches, explain potential implementations, and provide illustrative code examples. All decisions regarding method selection, workflow design, and implementation were made by the author.
- **Information gathering**: Assisting in identifying relevant existing packages, algorithms, or workflows, analogous to a literature or software survey, with final evaluation and selection performed by the author.

### Early-Phase Research and Prototyping

Before adopting Claude Code and Cursor IDE for the final package implementation, workflows and pipelines for analyzing MD simulations were developed and evaluated through prototyping. During this early phase, various large language models were used in chat-based sessions to generate code fragments, explore potential methods, and discuss implementation strategies. The author also performed molecular dynamics analyses and tested novel workflows for MD data interpretation. Insights from these experiments directly informed the algorithms and functionalities ultimately implemented in the package.

### Author's Independent Contributions

All architectural design, scientific methodology, algorithm selection, and API specification were independently performed by the author. When AI tools reached their technical limitations, such as token restrictions, algorithmic complexity, or domain-specific challenges, the author directly implemented the necessary code and documentation without AI assistance.

The author conducted all prompt engineering, including crafting detailed technical specifications, providing domain-specific molecular dynamics context, and guiding iterative refinement through multi-session interactions. Comprehensive code review, testing, and validation of all generated code were also performed solely by the author.

Responsibility for scientific validity, methodological rigor, code quality, and overall project integrity rests entirely with the human author. The development process involved tens of thousands of prompts and over 2000 hours of hands-on development, code-reviews and methodological refinement, highlighting the central role of human expertise in shaping the final package.

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL v3.0). See [LICENSE](LICENSE) for details.

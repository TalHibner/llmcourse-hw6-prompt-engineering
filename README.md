# Prompt Engineering for Mass Production Optimization

A comprehensive graduate-level research project investigating the impact of various prompt engineering techniques on Large Language Model (LLM) response accuracy and consistency.

## Overview

This project systematically compares different prompt engineering techniques across multiple datasets using vector similarity metrics to quantify response quality. The implementation supports multiple LLM providers with a unified interface, but **uses Ollama (free/local) as the primary execution engine** for all experiments.

### Key Features

- **Multi-LLM Support**: Unified interface for Claude, Gemini, OpenAI, and Ollama
- **Primary Engine**: Ollama for free, local experiment execution
- **Multiple Datasets**: Sentiment analysis and Chain-of-Thought reasoning tasks
- **Prompt Techniques**: Baseline, Standard, Few-Shot, Chain-of-Thought, and ReAct
- **Quantitative Evaluation**: Vector embedding similarity with statistical analysis
- **Publication-Quality Visualizations**: Histograms, bar charts, and box plots

### Research Question

**How do different prompt engineering techniques affect LLM response accuracy and consistency when measured through vector similarity to ground-truth answers?**

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [UV package manager](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai) (for primary experiments)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/TalHibner/llmcourse-hw6-Prompt-Enginnering-for-massprodaction.git
cd llmcourse-hw6-prompt-engineering

# 2. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# 4. Install Ollama
# Visit https://ollama.ai and follow installation instructions for your OS
# Or on Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh

# 5. Pull a model (we recommend llama3.2 or mistral)
ollama pull llama3.2

# 6. Verify Ollama is running
ollama list
```

### Running Experiments

```bash
# Generate datasets (if not already present)
python scripts/generate_datasets.py

# Run all experiments with Ollama (default)
python scripts/run_experiments.py

# Run specific technique
python scripts/run_experiments.py --technique chain_of_thought

# Run on specific dataset
python scripts/run_experiments.py --dataset sentiment_analysis

# Generate analysis and visualizations
python scripts/analyze_results.py
```

---

## Project Structure

```
llmcourse-hw6-prompt-engineering/
├── src/                          # Source code
│   ├── providers/                # LLM provider implementations
│   ├── datasets/                 # Dataset management
│   ├── techniques/               # Prompt technique implementations
│   ├── evaluation/               # Similarity calculation & experiment runner
│   ├── analysis/                 # Statistical analysis & visualization
│   ├── config/                   # Configuration management
│   └── utils/                    # Utilities
├── data/                         # Datasets
│   └── datasets/                 # Generated Q&A pairs (JSON)
├── results/                      # Experiment results
│   ├── experiments/              # Raw experiment data
│   └── visualizations/           # Generated plots
├── tests/                        # Unit and integration tests
├── scripts/                      # Execution scripts
│   ├── generate_datasets.py      # Dataset generation
│   ├── run_experiments.py        # Main experiment runner
│   └── analyze_results.py        # Results analysis
├── config/                       # Configuration files
│   ├── config.yaml               # Main configuration
│   └── example.env               # Environment variable template
├── PRD.md                        # Product Requirements Document
├── DESIGN.md                     # Technical Design Document
├── TASKS.md                      # Implementation checklist
├── README.md                     # This file
└── pyproject.toml                # Project dependencies
```

---

## LLM Provider Support

### Primary: Ollama (FREE & LOCAL) ⭐

**This is the default and recommended provider for all experiments.**

- **Cost**: Free
- **Privacy**: All processing happens locally
- **Models**: llama3.2, mistral, phi3, and many more
- **Setup**: See installation instructions above
- **No API key required**

### Optional: Cloud Providers

The following providers are **implemented but NOT used by default** to avoid costs:

#### Claude (Anthropic)

```bash
# Set API key in .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env

# Run with Claude
python scripts/run_experiments.py --provider claude
```

#### Gemini (Google)

```bash
# Set API key in .env
echo "GOOGLE_API_KEY=your_key_here" >> .env

# Run with Gemini
python scripts/run_experiments.py --provider gemini
```

#### OpenAI

```bash
# Set API key in .env
echo "OPENAI_API_KEY=your_key_here" >> .env

# Run with OpenAI
python scripts/run_experiments.py --provider openai
```

---

## Datasets

### Dataset 1: Sentiment Analysis

Simple Q&A pairs for sentiment classification tasks.

- **Size**: 50-100 examples
- **Categories**: Positive, Negative, Neutral
- **Format**: Short text snippets with expected sentiment labels
- **Use Case**: Testing prompts on straightforward classification tasks

### Dataset 2: Chain-of-Thought

Multi-step reasoning problems requiring logical deduction.

- **Size**: 50-100 examples
- **Categories**: Math problems, logic puzzles, multi-hop reasoning
- **Format**: Questions with step-by-step solutions
- **Use Case**: Testing prompts on complex reasoning tasks

Both datasets are stored in `data/datasets/` as JSON files.

---

## Prompt Engineering Techniques

### 1. Baseline

Simple, direct prompting without enhancements.

```
Answer this question: {question}
```

### 2. Standard Improved

Enhanced prompt with role and structure.

```
You are an expert assistant. Please provide a clear and concise answer.

Question: {question}

Answer:
```

### 3. Few-Shot Learning

Include 2-3 examples before the actual question.

```
Here are some examples:

Q: [example 1 question]
A: [example 1 answer]

Q: [example 2 question]
A: [example 2 answer]

Now answer this question:
Q: {question}
A:
```

### 4. Chain-of-Thought

Request step-by-step reasoning.

```
Answer this question by thinking step by step.

Question: {question}

Let's approach this step by step:
1.
```

### 5. ReAct (Optional)

Reasoning and Acting pattern.

```
Answer using this format:
Thought: [your reasoning]
Action: [what to do]
Observation: [result]
Answer: [final answer]

Question: {question}

Thought:
```

---

## Evaluation Methodology

### Vector Similarity

Responses are evaluated using vector embeddings:

1. Convert LLM response to vector embedding
2. Convert expected answer to vector embedding
3. Calculate cosine similarity (0-1 scale)
4. Higher similarity = better response quality

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

### Statistical Analysis

- **Mean Similarity**: Average performance across all examples
- **Standard Deviation**: Consistency of responses
- **T-Test**: Statistical significance of improvements
- **Cohen's d**: Effect size of improvements
- **Improvement %**: Percentage improvement over baseline

### Visualizations

- **Histograms**: Distribution of similarity scores
- **Bar Charts**: Mean performance comparison across techniques
- **Box Plots**: Distribution and outlier visualization

---

## Configuration

### Main Configuration (config/config.yaml)

```yaml
llm:
  default_provider: "ollama"  # Primary provider

  ollama:
    model: "llama3.2"
    base_url: "http://localhost:11434"
    timeout: 60

experiment:
  batch_size: 10
  max_retries: 3
  retry_delay: 2

evaluation:
  embedding_model: "all-MiniLM-L6-v2"
  similarity_metric: "cosine"

output:
  results_dir: "results"
  save_format: ["json", "csv"]
  visualization_format: ["png", "pdf"]
```

### Environment Variables (.env)

```bash
# Optional: Only needed if using cloud providers
ANTHROPIC_API_KEY=your_claude_key
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

---

## Results

After running experiments, results will be saved to:

- **Raw Data**: `results/experiments/*.json`
- **Visualizations**: `results/visualizations/*.png`
- **Analysis**: `results/analysis/*.csv`

Example results structure:

```json
{
  "experiment_id": "exp_20250115_001",
  "provider": "ollama",
  "model": "llama3.2",
  "technique": "chain_of_thought",
  "dataset": "sentiment_analysis",
  "metrics": {
    "mean_similarity": 0.87,
    "std_similarity": 0.08,
    "improvement_over_baseline": "15.3%"
  },
  "results": [...]
}
```

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_providers.py

# View coverage report
open htmlcov/index.html
```

**Target Coverage**: >70%

---

## Development

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Adding a New Provider

1. Create a new provider class in `src/providers/`
2. Inherit from `LLMProvider` base class
3. Implement required methods
4. Register in provider factory
5. Add configuration in `config.yaml`
6. Add tests

### Adding a New Technique

1. Create a new technique class in `src/techniques/`
2. Inherit from `PromptTechnique` base class
3. Implement `format_prompt()` method
4. Register in technique registry
5. Add tests

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
ollama list

# Start Ollama service (if not running)
ollama serve

# Test connection
curl http://localhost:11434/api/version
```

### Model Not Found

```bash
# Pull the required model
ollama pull llama3.2

# List available models
ollama list

# Remove old models (if needed)
ollama rm <model_name>
```

### GPU/Memory Issues

If you encounter memory issues with local models:

1. Use a smaller model (e.g., `phi3` instead of `llama3.2`)
2. Reduce batch size in configuration
3. Process datasets sequentially instead of in batches

### Dependency Issues

```bash
# Reinstall dependencies
uv pip install -e . --force-reinstall

# Clear UV cache
uv cache clean

# Update UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Performance Expectations

### Typical Execution Times (Ollama on M1 Mac / Modern CPU)

- Dataset Generation: <5 minutes
- Single Experiment (50 examples): 5-10 minutes
- Full Experiment Suite (all techniques, both datasets): 20-30 minutes
- Analysis & Visualization: <1 minute

### Token Usage

- Ollama: Free, unlimited
- Cloud APIs (if used): Varies by model and dataset size

---

## Contributing

This is a research project, but contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

This project is created for academic purposes as part of a graduate-level course in Computer Science.

---

## References

### Academic Papers

- Brown et al. (2020). "Language Models are Few-Shot Learners"
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models"

### Documentation

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Generative AI Documentation](https://ai.google.dev/docs)
- [Sentence Transformers Documentation](https://www.sbert.net/)

---

## Acknowledgments

- Anthropic for Claude
- Google for Gemini
- OpenAI for GPT models
- Ollama team for local LLM execution
- Sentence Transformers team for embedding models

---

## Contact

For questions or issues:
- Create an issue in the GitHub repository
- Refer to documentation in PRD.md and DESIGN.md

---

**Status**: Active Development
**Version**: 1.0.0
**Last Updated**: December 15, 2025

# Data Science Tutorial - HUD Environment

A HUD v5 environment for data science tutorials using Jupyter notebooks. This environment provides scenarios for analyzing the Wine Quality dataset with Python and pandas.

## Features

- **Jupyter Notebook Integration**: Execute Python code in Jupyter kernels
- **Wine Quality Dataset**: Pre-loaded red and white wine quality datasets from UCI ML Repository
- **Multiple Scenarios**: Various data science tasks including:
  - Comparing wine quality between red and white wines
  - Finding correlations between features and quality
  - Calculating statistical measures
  - Identifying high-quality wines

## Project Structure

```
hud-data-science-tutorial/
├── env.py              # Main environment with scenarios and tools
├── Dockerfile.hud      # Docker configuration for HUD
├── pyproject.toml      # Python dependencies
├── test_tasks.json     # Test tasks in v5 format
├── run_eval.py         # Evaluation script with qwen agent
├── local_test.py       # Local testing script
└── .hud_eval.toml      # HUD evaluation configuration
```

## Setup

### 1. Install Dependencies

```bash
pip install -e .
```

Or using uv:

```bash
uv sync
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_actual_api_key_here
```

Get your API key from: https://openrouter.ai/keys

**Note**: The `.env` file is gitignored and will not be committed to the repository.

### 2. Build Docker Image

```bash
docker build -f Dockerfile.hud -t data-science-tutorial:latest .
```

### 3. Test Locally

```bash
# Test scenarios without agent
python local_test.py

# Run full evaluation with qwen agent
python run_eval.py
```

## Usage

### Local Evaluation

Run evaluations locally using the Python script:

```bash
python run_eval.py
```

This uses the qwen/qwen3-coder model via OpenRouter (configured in `.hud_eval.toml`).

### HUD CLI (After Publishing)

Once published to HUD hub:

```bash
# Run all tasks
hud eval test_tasks.json openai --full

# Run specific tasks
hud eval test_tasks.json openai --task-ids compare_wine_quality_001
```

## Scenarios

### 1. compare-wine-quality
Compare average quality scores between red and white wines.

### 2. find-quality-correlation
Find which physicochemical property has the strongest correlation with wine quality.

### 3. calculate-statistics
Calculate mean, median, standard deviation, min, and max for a specified feature.

### 4. identify-high-quality-wines
Identify wines above a quality threshold and calculate the percentage.

## Dataset

The Wine Quality dataset is automatically downloaded during Docker build:
- Red wine: `/app/data/wine_quality/winequality-red.csv`
- White wine: `/app/data/wine_quality/winequality-white.csv`

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## Configuration

- **Agent**: qwen/qwen3-coder via OpenRouter
- **Jupyter Gateway**: Port 8888
- **Python**: 3.11+

## License

[Add your license here]

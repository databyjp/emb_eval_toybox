# Embedding Evaluation Toybox

A Python tool for evaluating vector embedding models, designed for educational purposes. This tool provides simple examples for comparing different embedding models on semantic search tasks.

## Features

- Support for multiple embedding providers:
  - Sentence Transformers
  - Ollama
- Evaluation metrics:
  - NDCG (Normalized Discounted Cumulative Gain) with graded relevance
  - Support for multiple evaluation depths (NDCG@3, NDCG@5, NDCG@10)
  - Precision@k, Recall@k
- Synthetic datasets for testing:

## Installation

```bash
pip install -e .
```

## Usage

### Basic Evaluation

Run the basic evaluation script to compare different embedding models:

```bash
python examples/basic_evaluation.py
```

### NDCG Evaluation

Run the NDCG evaluation script to get detailed ranking quality metrics:

```bash
python examples/ndcg_evaluation.py
```

## Datasets

The tool includes synthetic datasets designed to test semantic understanding:

### Project Structure

```
emb_eval_toybox/
├── data/                    # Dataset files
├── emb_eval_toybox/        # Main package
│   ├── data/              # Dataset loading
│   └── providers/         # Embedding providers
└── examples/              # Example scripts
```

## Dependencies

- sentence-transformers>=2.2.0
- numpy>=1.24.0
- pytest>=7.0.0 (dev)

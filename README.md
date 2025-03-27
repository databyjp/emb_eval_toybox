# Embedding Model Evaluation Project

This project serves as an educational resource for the Weaviate Academy course on embedding models and selection. It demonstrates how to evaluate embedding models specifically for vector search applications.

## Project Overview

This project provides a practical framework for evaluating embedding models in semantic search scenarios:
- Loading and preprocessing search queries and documents
- Generating embeddings using different models
- Evaluating search performance using standard metrics
- Making informed decisions about model selection

## Key Components

### 1. Data Processing
- Query and document preprocessing
- Dataset loading and management
- Support for standard search evaluation datasets

### 2. Embedding Generation
- Integration with popular embedding models
- Batch processing capabilities
- Model configuration management

### 3. Search Evaluation Metrics
- Precision@k (k=1,5,10)
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Query-document similarity scores
- Performance metrics (inference speed, memory usage)

## Project Structure

```
emb_eval/
├── data/             # Sample datasets and processed data
├── src/              # Source code
│   ├── data/         # Data processing modules
│   ├── embeddings/   # Embedding model implementations
│   └── evaluation/   # Search evaluation metrics
├── notebooks/        # Jupyter notebooks for tutorials
├── tests/            # Test cases
└── requirements.txt  # Project dependencies
```

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the example notebooks in the `notebooks/` directory

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Sentence Transformers (for embedding models)

## Contributing

This project is part of the Weaviate Academy curriculum. Contributions are welcome through pull requests.

## License

MIT License

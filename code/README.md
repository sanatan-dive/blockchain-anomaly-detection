# Code Implementation

This directory contains the core implementation of the blockchain anomaly detection system.

## Files

### `Model1.ipynb`
Main Jupyter notebook containing:
- Data preprocessing pipeline
- GNN model architecture implementation
- Training procedures with early stopping
- Comprehensive evaluation and visualization
- Results analysis and interpretation

### `gnn_flagged_transactions.csv`
Generated output file containing:
- Transaction IDs flagged as potentially illicit
- Anomaly scores and confidence levels
- Classification results from the trained model

## Model Architecture
The implementation includes:
- **Graph Convolutional Networks (GCN)** with 3 layers
- **Graph Attention Networks (GAT)** with 8 attention heads
- **LSTM layers** for temporal encoding
- **Attention mechanisms** for improved performance
- **Weighted cross-entropy loss** for handling class imbalance

## Dependencies
Key libraries used:
- PyTorch 2.0.0
- PyTorch Geometric 2.3.0
- NetworkX for graph operations
- Scikit-learn for metrics
- Matplotlib/Seaborn for visualization

## Running the Code
1. Open `Model1.ipynb` in Jupyter Notebook or VS Code
2. Install required dependencies from `../requirements.txt`
3. Run all cells sequentially
4. Results will be saved to `../images/` and `../models/`
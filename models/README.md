# Trained Models

This directory contains the trained model files for the blockchain anomaly detection system.

## Model Files

### `best_gnn_model.pth`
- **Type**: PyTorch model checkpoint
- **Architecture**: Graph Convolutional Network with attention mechanisms
- **Performance**: F1-score of 0.847, ROC-AUC of 0.923
- **Training**: Trained on Elliptic Bitcoin dataset (203,769 transactions)
- **Size**: ~159 KB

## Model Architecture Details
The saved model includes:
- **GCN Layers**: 3 convolutional layers (128, 64, 32 units)
- **GAT Layers**: 2 Graph Attention layers with 8 attention heads each
- **LSTM**: 2-layer LSTM with 64 units for temporal encoding
- **Dropout**: 0.3 dropout rate for regularization
- **Activation**: ReLU and LeakyReLU activations

## Loading the Model
```python
import torch
from torch_geometric.nn import GCNConv, GATConv

# Load the trained model
model = torch.load('models/best_gnn_model.pth')
model.eval()

# Use for inference
with torch.no_grad():
    predictions = model(x, edge_index)
```

## Training Details
- **Optimizer**: AdamW with learning rate 0.001
- **Loss Function**: Weighted cross-entropy
- **Early Stopping**: Patience of 50 epochs
- **Training Time**: ~2 hours on NVIDIA A100 GPU
- **Dataset Split**: 60% train, 20% validation, 20% test
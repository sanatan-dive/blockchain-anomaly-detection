# Novel Graph Neural Network for Real-Time Blockchain Anomaly Detection with Smart Contract Support ⚡

> Enhancing Blockchain Security with Advanced Machine Learning and Automated Smart Contract Integration

## 📝 Overview

This project implements a novel Graph Neural Network (GNN) framework for real-time anomaly detection in blockchain networks, achieving superior performance with an F1-score of 0.847 and ROC-AUC of 0.923. The system leverages Graph Convolutional Networks (GCNs) with attention mechanisms and temporal encoding to identify complex fraudulent activities, including money laundering and ransomware-related payments.

## 🚀 Key Features

- **Real-time Processing**: Handles over 5,000 transactions per second with GPU acceleration
- **Advanced GNN Architecture**: Combines GCN, GAT layers, and LSTM for comprehensive anomaly detection
- **Smart Contract Integration**: Gas-optimized Solidity contract for immutable anomaly logging
- **Cross-blockchain Support**: Tested on Bitcoin, Ethereum, and Monero datasets
- **15% Performance Improvement**: Over state-of-the-art Transformer-based GNNs

## 📁 Repository Structure

```
├── code/                           # Source code and implementations
│   ├── Model1.ipynb               # Main Jupyter notebook with GNN implementation
│   └── gnn_flagged_transactions.csv # Generated flagged transactions dataset
├── images/                         # Performance visualization results
│   ├── confusion_matrix.png       # Model confusion matrix
│   ├── feature_importance.png     # Feature importance analysis
│   ├── performance_metrics.png    # Overall performance metrics
│   ├── precision_recall_curve.png # Precision-recall curve
│   ├── prediction_distribution.png # Prediction probability distribution
│   └── roc_curve.png              # ROC curve visualization
├── models/                         # Trained model files
│   └── best_gnn_model.pth         # Best performing GNN model weights
├── data/                           # Dataset directory (for external datasets)
├── docs/                           # Documentation and research papers
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Node.js (for smart contract deployment)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/sanatan-dive/blockchain-anomaly-detection.git
cd blockchain-anomaly-detection
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install PyTorch with CUDA support**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

## 📊 Dataset Description

### Primary Dataset: Elliptic Bitcoin Dataset
- **Size**: 203,769 transactions across 49 time steps
- **Features**: 166 features per transaction
- **Labels**: 21,564 legitimate and 4,545 illicit transactions
- **Structure**: 234,355 directed edges forming transaction graph

### Additional Datasets
- **Synthetic Bitcoin**: 50,000 transactions for extended testing
- **Ethereum DeFi**: 100,000 DeFi transactions
- **Monero Privacy**: 25,000 privacy-enhanced transactions

## 🎯 Usage

### Running the Analysis

1. **Open the Jupyter Notebook**
```bash
# Navigate to the code directory
cd code

# Launch Jupyter Notebook
jupyter notebook Model1.ipynb
```

2. **Or run in VS Code**
```bash
# Open the notebook in VS Code
code code/Model1.ipynb
```

3. **View Results**
- Check the `images/` directory for performance visualizations
- Review the `models/best_gnn_model.pth` for the trained model weights
- Analyze flagged transactions in `code/gnn_flagged_transactions.csv`

### Reproducing Results
- Run all cells in `Model1.ipynb` sequentially
- Ensure you have the required dependencies installed
- Update dataset paths if using external datasets in the `data/` directory

## 📈 Results Summary

### Performance Metrics
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Proposed GCN (Ours)** | **0.912** | **0.863** | **0.832** | **0.847** | **0.923** |
| Transformer-GNN (GTN) | 0.898 | 0.841 | 0.815 | 0.828 | 0.909 |
| GraphSAGE | 0.876 | 0.812 | 0.798 | 0.805 | 0.874 |
| Random Forest | 0.798 | 0.705 | 0.692 | 0.698 | 0.771 |

### Smart Contract Gas Costs
| Operation | Gas Used | ETH Cost (20 Gwei) | USD Cost ($2,000/ETH) |
|-----------|----------|---------------------|----------------------|
| Contract Deployment | 1,800,000 | 0.036 ETH | $72.00 |
| First Anomaly Log | 195,000 | 0.0039 ETH | $7.80 |
| Subsequent Logs | 35,000 | 0.0007 ETH | $1.40 |

## 🏗️ Architecture

The framework consists of three main components:

1. **GNN Model**: Graph Convolutional Network with attention mechanisms
2. **Temporal Encoding**: LSTM layers for time-series analysis
3. **Smart Contract**: Gas-optimized anomaly logging system

### Model Architecture
- **GCN Layers**: 3 layers (128, 64, 32 units)
- **GAT Layers**: 2 layers with 8 attention heads
- **LSTM**: 2 layers with 64 units each
- **Loss Function**: Weighted cross-entropy for class imbalance

## 👥 Authors

| Name | Affiliation | Email |
|------|-------------|-------|
| **Sanatan Sharma** † | Department of CSE, Chandigarh College of Engineering and Technology | co23355@ccet.ac.in |
| **Sudhakar Kumar** ‡ | Department of CSE, Chandigarh College of Engineering and Technology | sudhakar@ccet.ac.in |
| **Sunil K. Singh** ‡ | Department of CSE, Chandigarh College of Engineering and Technology | sksingh@ccet.ac.in |
| **Ching-Hsien Hsu** | Department of CSIE, Asia University, Taiwan | robertchh@asia.edu.tw |
| **Varsha Arya** | Hong Kong Metropolitan University | varya@hkmu.edu.hk |
| **Kwok Tai Chui** | Hong Kong Metropolitan University | jktchui@hkmu.edu.hk |
| **Brij B. Gupta** * | Department of CSIE, Asia University, Taiwan | bbgupta@asia.edu.tw |

† Corresponding author ‡ Equal contribution * Principal Investigator


## 🙏 Acknowledgments

This research work is supported by National Science and Technology Council (NSTC), Taiwan Grant No. NSTC112-2221-E-468-008-MY3 and NSTC 114-2221-E-468-015.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

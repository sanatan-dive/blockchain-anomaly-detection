# Dataset Directory

This directory is intended for storing blockchain datasets used in the anomaly detection research.

## Recommended Datasets

### Primary Dataset
- **Elliptic Bitcoin Dataset**: Download from [Elliptic](https://www.elliptic.co/blog/elliptic-data-set) or [Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set)
  - Place the CSV files in `data/elliptic/`

### Additional Datasets
- **Bitcoin Transaction Networks**: For extended testing
- **Ethereum DeFi Transactions**: For cross-blockchain validation
- **Synthetic Datasets**: For controlled experiments

## Usage
Place your datasets in appropriate subdirectories within this folder:
```
data/
├── elliptic/
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_classes.csv
│   └── elliptic_txs_edgelist.csv
├── synthetic/
└── ethereum/
```

## Note
Due to file size limitations, datasets are not included in this repository. Please download them from the official sources mentioned above.
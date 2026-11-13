# Real-Time Fraud Detection in Mobile-Money Transactions

ML mini-project: detect fraudulent mobile-money transactions, and score them fast
enough to work on a live transaction stream.

- **`fraud_detection.ipynb`** - the whole project as one executed notebook:
  data, features, logistic regression baseline, small MLP, real metrics, plots,
  and the simulated live-stream demo. Open it and read top to bottom; every
  output cell is from a real run.
- **`explanation.md`** - plain-English walkthrough of the whole thing, written
  for someone new to ML. Start here if you're new.
- **`results/metrics.json`** - the exact numbers from the training run.
- **Trained model weights** - attached to the
  [GitHub release](../../releases) (`lr_model.joblib`, `mlp_model.joblib`,
  `scaler.joblib`, `features.joblib`).

Dataset: [PaySim - Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
(6,362,620 mobile-money transactions, 8,213 fraud). The notebook downloads it
from a public mirror on first run.

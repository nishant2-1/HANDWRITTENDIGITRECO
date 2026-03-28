# Handwritten Digit Recognition

![Accuracy](https://img.shields.io/badge/Accuracy-95%25%2B-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning project that classifies handwritten digits (0–9) from the MNIST dataset using a K-Nearest Neighbours classifier with hyperparameter tuning and cross-validation.

---

## Features

- MNIST data loading, normalization, and preprocessing via TensorFlow/Keras
- Shared constants/configuration module for cleaner maintainability
- KNN classifier with `ball_tree` algorithm for speed-optimised lookups
- `GridSearchCV` over `n_neighbors`, `metric`, and `weights`
- 5-Fold cross-validation for robust performance estimates
- Full evaluation report: precision, recall, F1, confusion matrix
- Single-image CLI prediction with per-class confidence scores
- Results persisted to `results/metrics.json`

---

## Project Structure

```text
handwritten-digit-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── exploration.ipynb
├── examples/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── constants.py
│   └── version.py
├── tests/
│   ├── __init__.py
│   └── test_model.py
└── results/
    ├── metrics.json
    └── training_metrics.json
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/nishant2-1/HANDWRITTENDIGITRECO.git
cd HANDWRITTENDIGITRECO

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
# Fast mode (default, quicker)
python -m src.train

# Full mode (full MNIST + wider grid, slower)
python -m src.train --full
```

### Evaluate on the test set

```bash
python -m src.evaluate
```

### Predict a single image

```bash
python -m src.predict --image path/to/digit.png
```

See example input guidance in `examples/README.md`.

### Run unit tests

```bash
pytest tests/ -v
```

---

## Results

| Metric             | Value   |
|--------------------|---------|
| Test Accuracy      | 95.18%  |
| CV Mean Accuracy   | 95.35%  |
| CV Std Deviation   | 0.24%   |
| Macro F1 Score     | 95.16%  |
| Prediction Latency | 8.28 ms |
| Latency Target     | <100 ms |

Confusion matrix and full classification report are saved to `results/` after running `evaluate.py`.
Training metadata including best hyperparameters and CV scores is stored in `results/training_metrics.json`.

## Release Notes

Project milestones and release notes are tracked in `CHANGELOG.md`.

---

## License

MIT

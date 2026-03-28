"""Model definitions and training utilities for KNN-based digit recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

DEFAULT_CV_SPLITS: int = 5
RANDOM_STATE: int = 42
N_JOBS: int = -1


@dataclass(frozen=True)
class TrainingResult:
    """Container for model training outputs."""

    best_model: KNeighborsClassifier
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: np.ndarray


def create_knn_model(n_neighbors: int = 3) -> KNeighborsClassifier:
    """Create a baseline KNN classifier.

    Args:
        n_neighbors: Number of neighbors for KNN.

    Returns:
        KNeighborsClassifier: Initialized KNN model.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="minkowski",
        algorithm="ball_tree",
    )


def tune_and_train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = DEFAULT_CV_SPLITS,
    fast_mode: bool = True,
) -> TrainingResult:
    """Run grid search and K-fold cross-validation to train the best KNN model.

    Args:
        x_train: Training features of shape (n_samples, 784).
        y_train: Training labels of shape (n_samples,).

        cv_splits: Number of K-Fold splits.
        fast_mode: Whether to use a lighter parameter grid for quicker iteration.

    Returns:
        TrainingResult: Best model, best params, and CV scores.
    """
    base_model: KNeighborsClassifier = create_knn_model()

    if fast_mode:
        param_grid: Dict[str, List[Any]] = {
            "n_neighbors": [3, 5],
            "metric": ["minkowski"],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree"],
        }
    else:
        param_grid = {
            "n_neighbors": [3, 5, 7],
            "metric": ["minkowski", "euclidean"],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree"],
        }

    cv: KFold = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    grid_search: GridSearchCV = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=N_JOBS,
        verbose=1,
    )
    grid_search.fit(x_train, y_train)

    best_model: KNeighborsClassifier = grid_search.best_estimator_
    cv_scores: np.ndarray = cross_val_score(
        best_model,
        x_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=N_JOBS,
    )

    return TrainingResult(
        best_model=best_model,
        best_params=grid_search.best_params_,
        best_score=float(grid_search.best_score_),
        cv_scores=cv_scores,
    )

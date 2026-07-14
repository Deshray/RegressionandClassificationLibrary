"""
islpstat
--------
Complete statistical learning library from ISLP Chapters 3 and 4.
Written in C with Python/ctypes wrappers. sklearn-style API + ISLP summaries.

Chapter 3 — Linear Regression
  LinearRegression        core OLS, CIs, prediction intervals
  forward_selection       AIC/BIC/adjR2 based variable selection
  backward_elimination    p-value or criterion based
  stepwise_selection      mixed forward+backward
  make_interaction        X_i * X_j interaction term
  add_interactions        batch interaction terms
  make_polynomial         polynomial basis [x, x^2, ..., x^d]
  dummy_encode            one-hot encoding for qualitative predictors
  correlation_matrix      predictor correlation matrix

Chapter 4 — Classification
  LogisticRegression             binary, IRLS, odds ratios, LRT, Wald CIs
  MultinomialLogisticRegression  K > 2 classes via softmax IRLS
  LinearDiscriminantAnalysis     pooled covariance
  QuadraticDiscriminantAnalysis  class-specific covariance
  GaussianNaiveBayes             conditional independence
  KNeighborsClassifier           KNN classifier
  KNeighborsRegressor            KNN regressor
  PoissonRegression              GLM with log link

Evaluation
  confusion_matrix_summary  full Table 4.6/4.7 metrics
  roc_curve                 FPR/TPR at all thresholds
  auc_score                 scalar AUC
  threshold_analysis        Figure 4.7 style error rate breakdown
"""

from .linear_model import (
    LinearRegression,
    forward_selection,
    backward_elimination,
    stepwise_selection,
    make_interaction,
    add_interactions,
    make_polynomial,
    dummy_encode,
    correlation_matrix,
)

from .classification import (
    LogisticRegression,
    MultinomialLogisticRegression,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    GaussianNaiveBayes,
    confusion_matrix_summary,
    roc_curve,
    auc_score,
    threshold_analysis,
)

from .neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)

from .glm import (
    PoissonRegression,
)

__all__ = [
    # Chapter 3
    'LinearRegression',
    'forward_selection', 'backward_elimination', 'stepwise_selection',
    'make_interaction', 'add_interactions', 'make_polynomial',
    'dummy_encode', 'correlation_matrix',
    # Chapter 4
    'LogisticRegression', 'MultinomialLogisticRegression',
    'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis',
    'GaussianNaiveBayes',
    'KNeighborsClassifier', 'KNeighborsRegressor',
    'PoissonRegression',
    # Evaluation
    'confusion_matrix_summary', 'roc_curve', 'auc_score', 'threshold_analysis',
]

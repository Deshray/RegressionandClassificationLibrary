"""
islpstat
--------
A statistical learning library implementing the exact methodology from
An Introduction to Statistical Learning (ISLP), Chapters 3 and 4.

Written in C (for numerical correctness and speed) with a Python/ctypes
wrapper that mirrors the sklearn API and produces ISLP-style summaries.

Chapter 3 — Linear Regression
    from islpstat import LinearRegression

Chapter 4 — Classification
    from islpstat import (LogisticRegression,
                          LinearDiscriminantAnalysis,
                          QuadraticDiscriminantAnalysis,
                          GaussianNaiveBayes,
                          confusion_matrix_summary)
"""

from .linear_model import LinearRegression
from .classification import (
    LogisticRegression,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    GaussianNaiveBayes,
    confusion_matrix_summary,
)

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis',
    'GaussianNaiveBayes',
    'confusion_matrix_summary',
]

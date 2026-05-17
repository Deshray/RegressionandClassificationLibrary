"""
python/classification.py
------------------------
Classification classes from ISLP Chapter 4:
  LogisticRegression  — Section 4.3  (binary, IRLS)
  LinearDiscriminantAnalysis  — Section 4.4.1/4.4.2
  QuadraticDiscriminantAnalysis — Section 4.4.3
  NaiveBayes          — Section 4.4.4  (Gaussian)

Each class follows the sklearn API:  .fit(X, y)  .predict(X)  .predict_proba(X)
and adds  .summary()  for ISLP-style statistical output.
"""

import numpy as np
import ctypes
from ._core import (_lib,
                    _BinaryConfusion,
                    _dbl_ptr, _int_ptr,
                    _out_dbl, _out_int)
from .linear_model import _stars


# ==================================================================
# Logistic Regression  (ISLP 4.3)
# ==================================================================

class LogisticRegression:
    """
    Binary logistic regression fitted by IRLS (Newton-Raphson), ISLP §4.3.

    Model:  log( p(X)/(1-p(X)) ) = beta_0 + beta_1*X_1 + ... + beta_p*X_p
    Estimation: maximum likelihood via IRLS.

    Attributes (after fit)
    ----------------------
    coef_          : ndarray (p,)    — slope coefficients
    intercept_     : float           — intercept
    se_            : ndarray (p+1,)  — standard errors
    z_stat_        : ndarray (p+1,)  — z-statistics  (ISLP 4.3.2)
    p_value_       : ndarray (p+1,)  — two-sided p-values
    null_deviance_ : float           — deviance of intercept-only model
    deviance_      : float           — deviance of fitted model
    aic_           : float           — AIC
    converged_     : bool
    """

    def __init__(self, max_iter=100, tol=1e-8, feature_names=None):
        self.max_iter      = max_iter
        self.tol           = tol
        self.feature_names = feature_names
        self._result_ptr   = None

    def fit(self, X, y, feature_names=None):
        """
        Fit binary logistic regression.

        X : (n, p) predictors — no intercept column.
        y : (n,)  binary labels, must be 0 or 1.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape

        if feature_names is not None: self.feature_names = feature_names
        if self.feature_names is None: self.feature_names = [f'X{j+1}' for j in range(p)]

        if self._result_ptr is not None:
            _lib.logreg_free(self._result_ptr)

        X_ptr, _X = _dbl_ptr(X.ravel())
        y_ptr, _y = _dbl_ptr(y)

        self._result_ptr = _lib.logreg_fit(
            X_ptr, y_ptr,
            ctypes.c_int(n), ctypes.c_int(p),
            ctypes.c_int(self.max_iter),
            ctypes.c_double(self.tol))

        if not self._result_ptr:
            raise RuntimeError("logreg_fit failed.")

        r = self._result_ptr.contents
        q = p + 1
        self._n = n; self._p = p

        self.intercept_     = r.beta[0]
        self.coef_          = np.array([r.beta[j] for j in range(1, q)])
        self.se_            = np.array([r.se[j]      for j in range(q)])
        self.z_stat_        = np.array([r.z_stat[j]  for j in range(q)])
        self.p_value_       = np.array([r.p_value[j] for j in range(q)])
        self.null_deviance_ = r.null_deviance
        self.deviance_      = r.deviance
        self.aic_           = r.aic
        self.converged_     = bool(r.converged)
        self.n_iter_        = r.n_iter
        return self

    def predict_proba(self, X):
        """Predicted probabilities P(Y=1|X), shape (m,)."""
        X = self._validate_X(X)
        m = X.shape[0]
        out_ptr, out = _out_dbl(m)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.logreg_predict_prob(self._result_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out

    def predict(self, X, threshold=0.5):
        """Predicted class labels (0 or 1) using the given threshold."""
        X = self._validate_X(X)
        m = X.shape[0]
        out_ptr, out = _out_int(m)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.logreg_predict(self._result_ptr, X_ptr, out_ptr,
                             ctypes.c_int(m), ctypes.c_double(threshold))
        return out.astype(int)

    def summary(self):
        """
        ISLP-style logistic regression summary.
        Matches Table 4.1/4.3 from the book.
        """
        SEP  = "=" * 70
        LINE = "-" * 70
        print(SEP)
        print("  Logistic Regression (Binary) — ISLP Chapter 4")
        print(SEP)
        print(f"  n = {self._n}   p = {self._p}")
        print(f"  Converged: {self.converged_}  (iterations: {self.n_iter_})")
        print()

        names = ['(Intercept)'] + list(self.feature_names)
        betas = [self.intercept_] + list(self.coef_)

        hdr = f"{'':20s} {'Estimate':>12} {'Std. Error':>12} {'z value':>10} {'Pr(>|z|)':>12}"
        print(hdr)
        print(LINE)
        for j, name in enumerate(names):
            pv    = self.p_value_[j]
            stars = _stars(pv)
            print(f"{name:20s} {betas[j]:>12.4f} {self.se_[j]:>12.4f} "
                  f"{self.z_stat_[j]:>10.3f} {pv:>12.4e}  {stars}")
        print()
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print()
        print(LINE)
        print(f"  Null deviance     : {self.null_deviance_:.4f}  "
              f"(on {self._n - 1} df)")
        print(f"  Residual deviance : {self.deviance_:.4f}  "
              f"(on {self._n - self._p - 1} df)")
        print(f"  AIC               : {self.aic_:.4f}")
        print(SEP)

    def _validate_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        assert X.shape[1] == self._p
        return X

    def __del__(self):
        if self._result_ptr is not None:
            try: _lib.logreg_free(self._result_ptr)
            except Exception: pass


# ==================================================================
# Linear Discriminant Analysis  (ISLP 4.4.1 / 4.4.2)
# ==================================================================

class LinearDiscriminantAnalysis:
    """
    LDA — ISLP Section 4.4.2.

    Assumes X|Y=k ~ N(mu_k, Sigma) with a common covariance matrix Sigma.
    Assigns x to the class k that maximises:

       delta_k(x) = x' Sigma^{-1} mu_k  -  0.5 mu_k' Sigma^{-1} mu_k  +  log(pi_k)

    Parameters
    ----------
    n_classes : int   — number of distinct classes (must be set if auto-detection fails).
    """

    def __init__(self, n_classes=None):
        self.n_classes   = n_classes
        self._model_ptr  = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape

        K = self.n_classes or int(np.unique(y).shape[0])
        self._n = n; self._p = p; self._K = K

        if self._model_ptr is not None:
            _lib.lda_free(self._model_ptr)

        X_ptr, _X = _dbl_ptr(X.ravel())
        y_ptr, _y = _int_ptr(y)

        self._model_ptr = _lib.lda_fit(X_ptr, y_ptr,
                                        ctypes.c_int(n), ctypes.c_int(p),
                                        ctypes.c_int(K))
        if not self._model_ptr:
            raise RuntimeError("lda_fit failed — singular covariance matrix?")

        m = self._model_ptr.contents
        self.classes_   = np.array([m.class_labels[k] for k in range(K)], dtype=int)
        self.priors_    = np.array([m.pi[k]            for k in range(K)])
        self.means_     = np.array([[m.mu[k*p+j] for j in range(p)] for k in range(K)])
        return self

    def predict(self, X):
        X = self._validate_X(X)
        m = X.shape[0]
        out_ptr, out = _out_int(m)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.lda_predict(self._model_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out.astype(int)

    def predict_proba(self, X):
        """Posterior probabilities P(Y=k|X), shape (m, K)."""
        X = self._validate_X(X)
        m, K = X.shape[0], self._K
        out_ptr, out = _out_dbl(m * K)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.lda_posterior(self._model_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out.reshape(m, K)

    def summary(self):
        SEP = "=" * 60
        print(SEP)
        print("  Linear Discriminant Analysis — ISLP §4.4.2")
        print(SEP)
        print(f"  n={self._n}  p={self._p}  K={self._K}")
        print()
        print("  Prior probabilities pi_k:")
        for k in range(self._K):
            print(f"    Class {self.classes_[k]:4d} : {self.priors_[k]:.4f}")
        print()
        print("  Group means mu_k:")
        for k in range(self._K):
            print(f"    Class {self.classes_[k]:4d} : {np.round(self.means_[k], 4)}")
        print(SEP)

    def _validate_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return np.ascontiguousarray(X)

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.lda_free(self._model_ptr)
            except Exception: pass


# ==================================================================
# Quadratic Discriminant Analysis  (ISLP 4.4.3)
# ==================================================================

class QuadraticDiscriminantAnalysis:
    """
    QDA — ISLP Section 4.4.3.

    Assumes X|Y=k ~ N(mu_k, Sigma_k) with CLASS-SPECIFIC covariance.
    Assigns x to the class that maximises:

       delta_k(x) = -0.5*(x-mu_k)' Sigma_k^{-1} (x-mu_k)
                    - 0.5*log|Sigma_k|  +  log(pi_k)
    """

    def __init__(self, n_classes=None):
        self.n_classes  = n_classes
        self._model_ptr = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape
        K = self.n_classes or int(np.unique(y).shape[0])
        self._n = n; self._p = p; self._K = K

        if self._model_ptr is not None:
            _lib.qda_free(self._model_ptr)

        X_ptr, _X = _dbl_ptr(X.ravel())
        y_ptr, _y = _int_ptr(y)
        self._model_ptr = _lib.qda_fit(X_ptr, y_ptr,
                                        ctypes.c_int(n), ctypes.c_int(p),
                                        ctypes.c_int(K))
        if not self._model_ptr:
            raise RuntimeError("qda_fit failed.")

        m = self._model_ptr.contents
        self.classes_ = np.array([m.class_labels[k] for k in range(K)], dtype=int)
        self.priors_  = np.array([m.pi[k]            for k in range(K)])
        self.means_   = np.array([[m.mu[k*p+j] for j in range(p)] for k in range(K)])
        return self

    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m = X.shape[0]
        out_ptr, out = _out_int(m)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.qda_predict(self._model_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out.astype(int)

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m, K = X.shape[0], self._K
        out_ptr, out = _out_dbl(m * K)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.qda_posterior(self._model_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out.reshape(m, K)

    def summary(self):
        SEP = "=" * 60
        print(SEP)
        print("  Quadratic Discriminant Analysis — ISLP §4.4.3")
        print(SEP)
        print(f"  n={self._n}  p={self._p}  K={self._K}")
        print("  (Class-specific covariance matrices)")
        print()
        print("  Prior probabilities pi_k:")
        for k in range(self._K):
            print(f"    Class {self.classes_[k]:4d} : {self.priors_[k]:.4f}")
        print()
        print("  Group means mu_k:")
        for k in range(self._K):
            print(f"    Class {self.classes_[k]:4d} : {np.round(self.means_[k], 4)}")
        print(SEP)

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.qda_free(self._model_ptr)
            except Exception: pass


# ==================================================================
# Naive Bayes  (ISLP 4.4.4, Gaussian)
# ==================================================================

class GaussianNaiveBayes:
    """
    Naive Bayes with Gaussian class-conditional densities — ISLP §4.4.4.

    Assumes conditional independence of features given class:
       f_k(x) = prod_j N(x_j; mu_kj, sigma_kj^2)

    Log posterior:
       log P(Y=k|x) ∝ log(pi_k) + sum_j log N(x_j; mu_kj, sigma_kj^2)
    """

    def __init__(self, n_classes=None):
        self.n_classes  = n_classes
        self._model_ptr = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape
        K = self.n_classes or int(np.unique(y).shape[0])
        self._n = n; self._p = p; self._K = K

        if self._model_ptr is not None:
            _lib.nb_free(self._model_ptr)

        X_ptr, _X = _dbl_ptr(X.ravel())
        y_ptr, _y = _int_ptr(y)
        self._model_ptr = _lib.nb_fit(X_ptr, y_ptr,
                                       ctypes.c_int(n), ctypes.c_int(p),
                                       ctypes.c_int(K))
        if not self._model_ptr:
            raise RuntimeError("nb_fit failed.")

        m = self._model_ptr.contents
        self.classes_ = np.array([m.class_labels[k] for k in range(K)], dtype=int)
        self.priors_  = np.array([m.pi[k]            for k in range(K)])
        self.means_   = np.array([[m.mu[k*p+j]  for j in range(p)] for k in range(K)])
        self.var_     = np.array([[m.var[k*p+j] for j in range(p)] for k in range(K)])
        return self

    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m = X.shape[0]
        out_ptr, out = _out_int(m)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.nb_predict(self._model_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out.astype(int)

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m, K = X.shape[0], self._K
        out_ptr, out = _out_dbl(m * K)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.nb_posterior(self._model_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out.reshape(m, K)

    def summary(self):
        SEP = "=" * 70
        LINE = "-" * 70
        print(SEP)
        print("  Gaussian Naive Bayes — ISLP §4.4.4")
        print(SEP)
        print(f"  n={self._n}  p={self._p}  K={self._K}")
        print()
        print("  Prior probabilities pi_k:")
        for k in range(self._K):
            print(f"    Class {self.classes_[k]} : {self.priors_[k]:.4f}")
        print()
        print("  Feature means mu_kj:")
        for k in range(self._K):
            print(f"    Class {self.classes_[k]} : {np.round(self.means_[k], 4)}")
        print()
        print("  Feature variances sigma_kj^2:")
        for k in range(self._K):
            print(f"    Class {self.classes_[k]} : {np.round(self.var_[k], 4)}")
        print(SEP)

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.nb_free(self._model_ptr)
            except Exception: pass


# ==================================================================
# Confusion Matrix  (ISLP Table 4.4)
# ==================================================================

def confusion_matrix_summary(y_true, y_pred, positive_class=1):
    """
    Compute and print binary confusion matrix following ISLP Table 4.4.

    Parameters
    ----------
    y_true          : array-like of int
    y_pred          : array-like of int
    positive_class  : int — which label is the 'positive' class (default 1)

    Returns
    -------
    dict with keys: TP, TN, FP, FN, sensitivity, specificity, error_rate, precision
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    assert y_true.shape == y_pred.shape
    n = len(y_true)

    yt_ptr, _yt = _int_ptr(y_true)
    yp_ptr, _yp = _int_ptr(y_pred)

    cm = _lib.confusion_matrix_binary(yt_ptr, yp_ptr,
                                       ctypes.c_int(n),
                                       ctypes.c_int(positive_class))

    neg_class = 1 - positive_class
    SEP = "=" * 50
    print(SEP)
    print("  Confusion Matrix — ISLP Table 4.4 style")
    print(SEP)
    print(f"  Positive class: {positive_class}   Negative class: {neg_class}")
    print()
    print(f"                    True {positive_class}    True {neg_class}")
    print(f"  Predicted {positive_class}   : {cm.TP:>8d}   {cm.FP:>8d}")
    print(f"  Predicted {neg_class}   : {cm.FN:>8d}   {cm.TN:>8d}")
    print()
    print(f"  Sensitivity (recall) = {cm.sensitivity:.4f}   "
          f"[TP/(TP+FN), fraction of true {positive_class}s caught]")
    print(f"  Specificity          = {cm.specificity:.4f}   "
          f"[TN/(TN+FP), fraction of true {neg_class}s caught]")
    print(f"  Precision            = {cm.precision:.4f}   "
          f"[TP/(TP+FP)]")
    print(f"  Overall error rate   = {cm.error_rate:.4f}")
    print(SEP)

    return dict(TP=cm.TP, TN=cm.TN, FP=cm.FP, FN=cm.FN,
                sensitivity=cm.sensitivity, specificity=cm.specificity,
                error_rate=cm.error_rate, precision=cm.precision)

"""
python/neighbors.py  —  KNN classifier and regressor, ISLP §3.5 and §4.5.2.

KNNClassifier  (ISLP §4.5.2):
  Given K and test point x0, find K closest training points.
  P(Y=k|X=x0) = (1/K) * #{i in N0 : y_i = k}
  Classify to the class with highest posterior probability.

KNNRegressor   (ISLP §3.5):
  f_hat(x0) = (1/K) * sum_{i in N0} y_i
  The average response over the K nearest training points.

ISLP key observations (§4.5.2):
  - KNN dominates parametric methods when decision boundary is non-linear.
  - KNN requires n >> p (curse of dimensionality).
  - Unlike logistic regression, KNN gives no coefficient interpretability.
"""
import numpy as np
import ctypes
from ._core import _lib, _dbl_ptr, _int_ptr, _out_dbl, _out_int


class KNeighborsClassifier:
    """
    KNN Classifier — ISLP §4.5.2.

    Brute-force Euclidean (L2) distance.  No model assumptions.
    """
    def __init__(self, K=5):
        self.K = K; self._model_ptr = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape
        self._n = n; self._p = p
        if self._model_ptr is not None: _lib.knn_classifier_free(self._model_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _int_ptr(y)
        self._model_ptr = _lib.knn_classifier_fit(Xp, yp, ctypes.c_int(n),
                                                    ctypes.c_int(p), ctypes.c_int(self.K))
        if not self._model_ptr: raise RuntimeError("knn_classifier_fit failed")
        m = self._model_ptr.contents
        self._K_actual = m.K
        self.classes_ = np.array([m.class_labels[k] for k in range(m.n_classes)], dtype=int)
        self._n_classes = m.n_classes
        return self

    def predict(self, X):
        X = self._vX(X); m = X.shape[0]
        op, out = _out_int(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.knn_classifier_predict(self._model_ptr, Xp, op, ctypes.c_int(m))
        return out.astype(int)

    def predict_proba(self, X):
        """Empirical class probabilities, shape (m, K_classes)."""
        X = self._vX(X); m = X.shape[0]; C = self._n_classes
        op, out = _out_dbl(m * C); Xp, _X = _dbl_ptr(X.ravel())
        _lib.knn_classifier_proba(self._model_ptr, Xp, op, ctypes.c_int(m))
        return out.reshape(m, C)

    def _vX(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        assert X.shape[1] == self._p
        return np.ascontiguousarray(X)

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.knn_classifier_free(self._model_ptr)
            except: pass


class KNeighborsRegressor:
    """
    KNN Regressor — ISLP §3.5.

    f_hat(x0) = (1/K) * sum_{i in N(x0)} y_i

    ISLP notes: KNN regression often suffers in high dimensions.
    When the true relationship is roughly linear, OLS often outperforms.
    """
    def __init__(self, K=5):
        self.K = K; self._model_ptr = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape; self._n = n; self._p = p
        if self._model_ptr is not None: _lib.knn_regressor_free(self._model_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _dbl_ptr(y)
        self._model_ptr = _lib.knn_regressor_fit(Xp, yp, ctypes.c_int(n),
                                                   ctypes.c_int(p), ctypes.c_int(self.K))
        if not self._model_ptr: raise RuntimeError("knn_regressor_fit failed")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        assert X.shape[1] == self._p
        X = np.ascontiguousarray(X); m = X.shape[0]
        op, out = _out_dbl(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.knn_regressor_predict(self._model_ptr, Xp, op, ctypes.c_int(m))
        return out

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.knn_regressor_free(self._model_ptr)
            except: pass

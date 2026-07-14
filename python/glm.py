"""
python/glm.py  —  Generalized Linear Models, ISLP §4.6.

PoissonRegression  (ISLP §4.6.2):
  Model: log(E[Y|X]) = beta_0 + beta_1*X_1 + ... + beta_p*X_p
  Y|X ~ Poisson(exp(X'beta))
  Estimation: IRLS (same framework as logistic regression).

ISLP §4.6.2 context:
  "Poisson regression is used to model count data — data where the
   response Y takes non-negative integer values: 0, 1, 2, ..."
  Examples: number of bikes rented, number of hospital admissions.

Key output:
  exp(beta_j) = multiplicative factor on E[Y] for a one-unit increase in X_j.
  Deviance = 2*sum[y*log(y/mu) - (y-mu)] — analogous to RSS in OLS.
"""
import numpy as np
import ctypes
from ._core import _lib, _dbl_ptr, _int_ptr, _out_dbl
from .linear_model import _stars


class PoissonRegression:
    """
    Poisson regression (GLM, log link) — ISLP §4.6.2.
    """
    def __init__(self, max_iter=200, tol=1e-8, feature_names=None):
        self.max_iter      = max_iter
        self.tol           = tol
        self.feature_names = feature_names
        self._result_ptr   = None

    def fit(self, X, y, feature_names=None):
        """
        X : (n, p) predictor matrix.
        y : (n,)   non-negative count response.
        """
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape
        if feature_names: self.feature_names = feature_names
        if not self.feature_names: self.feature_names = [f'X{j+1}' for j in range(p)]
        if self._result_ptr is not None: _lib.poisson_free(self._result_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _dbl_ptr(y)
        self._result_ptr = _lib.poisson_fit(Xp, yp, ctypes.c_int(n), ctypes.c_int(p),
                                             ctypes.c_int(self.max_iter), ctypes.c_double(self.tol))
        if not self._result_ptr: raise RuntimeError("poisson_fit failed")
        r = self._result_ptr.contents; q = p + 1
        self._n = n; self._p = p
        self.intercept_     = r.beta[0]
        self.coef_          = np.array([r.beta[j]    for j in range(1, q)])
        self.se_            = np.array([r.se[j]       for j in range(q)])
        self.z_stat_        = np.array([r.z_stat[j]   for j in range(q)])
        self.p_value_       = np.array([r.p_value[j]  for j in range(q)])
        self.null_deviance_ = r.null_deviance
        self.deviance_      = r.deviance
        self.aic_           = r.aic
        self.converged_     = bool(r.converged)
        self.n_iter_        = r.n_iter
        return self

    def predict(self, X):
        """Predicted count mu_hat = exp(X'beta)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        assert X.shape[1] == self._p
        X = np.ascontiguousarray(X); m = X.shape[0]
        op, out = _out_dbl(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.poisson_predict(self._result_ptr, Xp, op, ctypes.c_int(m))
        return out

    def rate_ratios(self, alpha=0.05):
        """
        exp(beta_j) — rate ratios with confidence intervals.

        ISLP: exp(beta_j) is the multiplicative factor by which the
        expected count E[Y] changes for a one-unit increase in X_j,
        holding all other predictors constant.

        Returns dict: {feature_name: {'RR': val, 'CI': (lo, hi)}}
        """
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        result = {}
        for j, nm in enumerate(self.feature_names):
            b = self.coef_[j]; se = self.se_[j+1]
            result[nm] = {'RR': round(float(np.exp(b)), 6),
                          'CI': (round(float(np.exp(b - z*se)), 6),
                                 round(float(np.exp(b + z*se)), 6))}
        return result

    def summary(self):
        SEP = "=" * 68; LINE = "-" * 68
        print(SEP); print("  Poisson Regression (GLM, log link)  —  ISLP §4.6.2"); print(SEP)
        print(f"  n={self._n}  p={self._p}  converged={self.converged_}  iter={self.n_iter_}")
        print()
        names = ['(Intercept)'] + list(self.feature_names)
        betas = [self.intercept_] + list(self.coef_)
        print(f"  {'':20} {'Estimate':>12} {'Std.Error':>12} {'z value':>10} {'Pr(>|z|)':>12}   {'exp(coef)':>10}")
        print("  " + LINE)
        for j, nm in enumerate(names):
            pv = self.p_value_[j]
            rr = f"{np.exp(betas[j]):.4f}" if j > 0 else "—"
            print(f"  {nm:20} {betas[j]:>12.4f} {self.se_[j]:>12.4f} "
                  f"{self.z_stat_[j]:>10.3f} {pv:>12.4e}  {_stars(pv):3}  {rr:>10}")
        print()
        print("  Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")
        print(); print("  " + LINE)
        print(f"  Null deviance:     {self.null_deviance_:.4f}  (df={self._n-1})")
        print(f"  Residual deviance: {self.deviance_:.4f}  (df={self._n-self._p-1})")
        print(f"  AIC: {self.aic_:.4f}")
        print(SEP)

    def __del__(self):
        if self._result_ptr is not None:
            try: _lib.poisson_free(self._result_ptr)
            except: pass

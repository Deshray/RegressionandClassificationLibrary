"""
python/linear_model.py
----------------------
LinearRegression class: sklearn-like API, ISLP-style statistical output.

Usage
-----
    from islpstat import LinearRegression
    import numpy as np

    X = np.column_stack([TV, radio, newspaper])
    model = LinearRegression()
    model.fit(X, sales)
    model.summary()

    # predict with confidence and prediction intervals
    X_new = np.array([[100.0, 20.0, 5.0]])
    y_hat  = model.predict(X_new)
    ci_lo, ci_hi = model.predict_ci(X_new)
    pi_lo, pi_hi = model.predict_pi(X_new)
"""

import numpy as np
from ._core import (_lib,
                    _LinRegResult,
                    _dbl_ptr, _int_ptr,
                    _out_dbl, _out_int)
import ctypes


class LinearRegression:
    """
    Ordinary Least Squares linear regression, ISLP Chapter 3.

    Attributes (after fit)
    ----------------------
    coef_       : ndarray (p,)    — slope coefficients beta_1, ..., beta_p
    intercept_  : float           — intercept beta_0
    se_         : ndarray (p+1,)  — standard errors  [intercept, slopes...]
    t_stat_     : ndarray (p+1,)  — t-statistics
    p_value_    : ndarray (p+1,)  — two-sided p-values
    ci_lower_   : ndarray (p+1,)  — 95% CI lower bounds
    ci_upper_   : ndarray (p+1,)  — 95% CI upper bounds
    rse_        : float           — Residual Standard Error
    r_squared_  : float           — R^2
    adj_r2_     : float           — Adjusted R^2
    f_stat_     : float           — F-statistic
    f_pvalue_   : float           — p-value for F-test
    feature_names_ : list[str]    — predictor names (set via feature_names param)
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self._result_ptr   = None
        self.coef_         = None
        self.intercept_    = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X, y, feature_names=None):
        """
        Fit OLS regression.

        Parameters
        ----------
        X : array-like (n, p)   — predictor matrix (no intercept column).
        y : array-like (n,)     — continuous response.
        feature_names : list of str, optional — names for the p predictors.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape
        assert y.shape == (n,), "y must be 1-D with length n"

        if feature_names is not None:
            self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = [f'X{j+1}' for j in range(p)]

        X_ptr, _X = _dbl_ptr(X.ravel())
        y_ptr, _y = _dbl_ptr(y)

        # Free previous result if refitting
        if self._result_ptr is not None:
            _lib.linreg_free(self._result_ptr)

        self._result_ptr = _lib.linreg_fit(X_ptr, y_ptr,
                                            ctypes.c_int(n),
                                            ctypes.c_int(p))
        if not self._result_ptr:
            raise RuntimeError("linreg_fit failed — X'X may be singular.")

        r = self._result_ptr.contents
        q = p + 1

        self.intercept_ = r.beta[0]
        self.coef_      = np.array([r.beta[j] for j in range(1, q)])
        self.se_        = np.array([r.se[j]      for j in range(q)])
        self.t_stat_    = np.array([r.t_stat[j]  for j in range(q)])
        self.p_value_   = np.array([r.p_value[j] for j in range(q)])
        self.ci_lower_  = np.array([r.ci_lower[j]for j in range(q)])
        self.ci_upper_  = np.array([r.ci_upper[j]for j in range(q)])
        self.rse_       = r.rse
        self.r_squared_ = r.r_squared
        self.adj_r2_    = r.adj_r2
        self.f_stat_    = r.f_stat
        self.f_pvalue_  = r.f_pvalue
        self._n         = n
        self._p         = p
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X):
        """Point predictions  y_hat = X @ coef_ + intercept_."""
        X = self._validate_X(X)
        m = X.shape[0]
        out_ptr, out = _out_dbl(m)
        X_ptr, _X   = _dbl_ptr(X.ravel())
        _lib.linreg_predict(self._result_ptr, X_ptr, out_ptr, ctypes.c_int(m))
        return out

    # ------------------------------------------------------------------
    # predict_ci  — confidence interval for E[Y|x_0]
    # ------------------------------------------------------------------
    def predict_ci(self, X, alpha=0.05):
        """
        95% confidence interval for the mean response (default alpha=0.05).

        CI = y_hat +/- t_{n-p-1, alpha/2} * RSE * sqrt(x_0'(X'X)^{-1}x_0)

        Returns (lower, upper) each of shape (m,).
        """
        X = self._validate_X(X)
        m = X.shape[0]
        lo_ptr, lo = _out_dbl(m)
        hi_ptr, hi = _out_dbl(m)
        X_ptr, _X  = _dbl_ptr(X.ravel())
        _lib.linreg_predict_ci(self._result_ptr, X_ptr,
                                ctypes.c_int(m), ctypes.c_double(alpha),
                                lo_ptr, hi_ptr)
        return lo, hi

    # ------------------------------------------------------------------
    # predict_pi  — prediction interval for individual Y
    # ------------------------------------------------------------------
    def predict_pi(self, X, alpha=0.05):
        """
        95% prediction interval for an individual response (default alpha=0.05).

        PI = y_hat +/- t_{n-p-1, alpha/2} * RSE * sqrt(1 + x_0'(X'X)^{-1}x_0)

        Wider than the CI because it includes the irreducible error (ISLP 3.2.2).

        Returns (lower, upper) each of shape (m,).
        """
        X = self._validate_X(X)
        m = X.shape[0]
        lo_ptr, lo = _out_dbl(m)
        hi_ptr, hi = _out_dbl(m)
        X_ptr, _X  = _dbl_ptr(X.ravel())
        _lib.linreg_predict_pi(self._result_ptr, X_ptr,
                                ctypes.c_int(m), ctypes.c_double(alpha),
                                lo_ptr, hi_ptr)
        return lo, hi

    # ------------------------------------------------------------------
    # summary  — ISLP-style table (mirrors R's summary(lm(...)))
    # ------------------------------------------------------------------
    def summary(self):
        """
        Print an ISLP/R-style regression summary.

        Matches the format of Tables 3.1, 3.4, and 3.6 in the book.
        """
        if self._result_ptr is None:
            print("Model not fitted yet. Call .fit(X, y) first.")
            return

        SEP  = "=" * 70
        LINE = "-" * 70

        print(SEP)
        print(f"  OLS Linear Regression — ISLP Chapter 3")
        print(SEP)
        print(f"  n = {self._n}   p = {self._p}   df_residual = {self._n - self._p - 1}")
        print()

        # ---- Coefficient table (ISLP Table 3.1 / 3.4 style) ----
        names = ['(Intercept)'] + list(self.feature_names)
        betas = [self.intercept_] + list(self.coef_)

        hdr = f"{'':20s} {'Estimate':>12} {'Std. Error':>12} {'t value':>10} {'Pr(>|t|)':>12}   {'95% CI':>22}"
        print(hdr)
        print(LINE)
        for j, name in enumerate(names):
            pv    = self.p_value_[j]
            stars = _stars(pv)
            ci_str= f"[{self.ci_lower_[j]:.4f}, {self.ci_upper_[j]:.4f}]"
            print(f"{name:20s} {betas[j]:>12.4f} {self.se_[j]:>12.4f} "
                  f"{self.t_stat_[j]:>10.3f} {pv:>12.4e}  {stars:3s}  {ci_str:>22s}")
        print()
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print()
        print(LINE)

        # ---- Model statistics (ISLP Table 3.2 / 3.6 style) ----
        print(f"  Residual Standard Error : {self.rse_:.4f}  "
              f"(on {self._n - self._p - 1} degrees of freedom)")
        print(f"  R²                      : {self.r_squared_:.4f}")
        print(f"  Adjusted R²             : {self.adj_r2_:.4f}")
        print(f"  F-statistic             : {self.f_stat_:.4f}  "
              f"on {self._p} and {self._n - self._p - 1} DF,  "
              f"p-value: {self.f_pvalue_:.4e}")
        print(SEP)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert X.shape[1] == self._p, \
            f"Expected {self._p} predictors, got {X.shape[1]}"
        return X

    def __del__(self):
        if self._result_ptr is not None:
            try:
                _lib.linreg_free(self._result_ptr)
            except Exception:
                pass


# ------------------------------------------------------------------
# Utility: significance stars (standard R convention)
# ------------------------------------------------------------------
def _stars(p):
    if   p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    elif p < 0.1:   return '.'
    else:           return ' '

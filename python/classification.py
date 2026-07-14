"""
python/classification.py  —  Full ISLP Chapter 4 classification toolkit.

  §4.3   LogisticRegression (binary IRLS)
           .ci()           Wald confidence intervals for beta
           .odds_ratio()   exp(beta) with CIs and interpretation
           .lrt()          likelihood ratio test vs reduced model

  §4.3.5 MultinomialLogisticRegression  (K > 2 via softmax IRLS)

  §4.4.1/4.4.2  LinearDiscriminantAnalysis
  §4.4.3  QuadraticDiscriminantAnalysis
  §4.4.4  GaussianNaiveBayes

  §4.4.2 Table 4.6/4.7  — full confusion matrix metrics:
           sensitivity, specificity, precision, NPV, FDR, FPR,
           F1, balanced accuracy, MCC

  §4.4.2  roc_curve()   — (fpr, tpr, thresholds) arrays
           auc_score()   — scalar AUC via trapezoidal rule

  §4.5.2  threshold_analysis() — error rates at every threshold
           (ISLP Figure 4.7)
"""

import numpy as np
import ctypes
from ._core import (_lib,
                    _BinaryConfusion, _BinaryMetrics,
                    _dbl_ptr, _int_ptr, _out_dbl, _out_int)
from .linear_model import _stars, _f_pvalue_py


# =====================================================================
# Logistic Regression  (ISLP §4.3)
# =====================================================================

class LogisticRegression:
    """
    Binary logistic regression via IRLS — ISLP §4.3.

    Model (ISLP Eq 4.7):
      log[ p(X)/(1-p(X)) ] = beta_0 + beta_1*X_1 + ... + beta_p*X_p

    Attributes after fit
    --------------------
    intercept_     : float
    coef_          : ndarray (p,)
    se_            : ndarray (p+1,) — standard errors
    z_stat_        : ndarray (p+1,) — z-statistics
    p_value_       : ndarray (p+1,) — two-sided p-values
    null_deviance_ : float
    deviance_      : float
    aic_           : float
    converged_     : bool
    """

    def __init__(self, max_iter=100, tol=1e-8, feature_names=None):
        self.max_iter      = max_iter
        self.tol           = tol
        self.feature_names = feature_names
        self._result_ptr   = None

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape
        if feature_names: self.feature_names = feature_names
        if not self.feature_names: self.feature_names = [f'X{j+1}' for j in range(p)]
        if self._result_ptr is not None: _lib.logreg_free(self._result_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _dbl_ptr(y)
        self._result_ptr = _lib.logreg_fit(Xp, yp, ctypes.c_int(n), ctypes.c_int(p),
                                            ctypes.c_int(self.max_iter), ctypes.c_double(self.tol))
        if not self._result_ptr: raise RuntimeError("logreg_fit failed.")
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

    def predict_proba(self, X):
        """P(Y=1|X), shape (m,)."""
        X = self._vX(X); m = X.shape[0]
        op, out = _out_dbl(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.logreg_predict_prob(self._result_ptr, Xp, op, ctypes.c_int(m))
        return out

    def predict(self, X, threshold=0.5):
        """Predicted class labels using threshold (ISLP §4.3.3)."""
        X = self._vX(X); m = X.shape[0]
        op, out = _out_int(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.logreg_predict(self._result_ptr, Xp, op, ctypes.c_int(m), ctypes.c_double(threshold))
        return out.astype(int)

    # ------------------------------------------------------------------
    # Wald confidence intervals for beta  (ISLP §4.3.2)
    # ------------------------------------------------------------------
    def ci(self, alpha=0.05):
        """
        Wald confidence intervals:  beta_j ± z_{alpha/2} * SE_j

        Returns dict: {feature_name: (lower, upper), ...}
        """
        from scipy.stats import norm
        z_crit = norm.ppf(1 - alpha / 2)
        names  = ['(Intercept)'] + list(self.feature_names)
        betas  = np.concatenate([[self.intercept_], self.coef_])
        return {nm: (round(betas[j] - z_crit * self.se_[j], 6),
                     round(betas[j] + z_crit * self.se_[j], 6))
                for j, nm in enumerate(names)}

    # ------------------------------------------------------------------
    # Odds ratios  exp(beta)  (ISLP §4.3.2)
    # ------------------------------------------------------------------
    def odds_ratio(self, alpha=0.05):
        """
        Odds ratios exp(beta_j) with confidence intervals.

        exp(beta_j) = odds multiplier for one-unit increase in X_j.
        ISLP interprets: exp(beta_1) is the factor by which the odds
        of Y=1 multiply for a one-unit increase in X_1.

        Returns dict: {feature_name: {'OR': or_val, 'CI': (lo, hi)}}
        """
        from scipy.stats import norm
        z_crit = norm.ppf(1 - alpha / 2)
        betas  = self.coef_   # only slopes, not intercept
        ses    = self.se_[1:] # se for slopes

        result = {}
        for j, nm in enumerate(self.feature_names):
            lo  = np.exp(betas[j] - z_crit * ses[j])
            hi  = np.exp(betas[j] + z_crit * ses[j])
            result[nm] = {'OR': round(float(np.exp(betas[j])), 6),
                          'CI': (round(float(lo), 6), round(float(hi), 6))}
        return result

    # ------------------------------------------------------------------
    # Likelihood Ratio Test  (ISLP §4.3.2)
    # ------------------------------------------------------------------
    def lrt(self, reduced_model):
        """
        Likelihood ratio test: full model (self) vs reduced model.

        H0: the extra predictors in the full model are jointly zero.
        LRT statistic = deviance_reduced - deviance_full ~ chi^2(q)
        where q = p_full - p_reduced.

        Returns dict: {chi2, df, p_value}
        """
        from scipy.stats import chi2 as chi2_dist
        lrt_stat = reduced_model.deviance_ - self.deviance_
        df       = self._p - reduced_model._p
        p_val    = chi2_dist.sf(lrt_stat, df) if df > 0 else 1.0
        return dict(chi2=round(float(lrt_stat), 6), df=df,
                    p_value=round(float(p_val), 8))

    def summary(self):
        SEP = "=" * 68; LINE = "-" * 68
        print(SEP); print("  Logistic Regression (Binary)  —  ISLP §4.3"); print(SEP)
        print(f"  n={self._n}  p={self._p}  converged={self.converged_}  iter={self.n_iter_}")
        print()
        names = ['(Intercept)'] + list(self.feature_names)
        betas = [self.intercept_] + list(self.coef_)
        print(f"  {'':20} {'Estimate':>12} {'Std.Error':>12} {'z value':>10} {'Pr(>|z|)':>12}   {'Odds Ratio':>12}")
        print("  " + LINE)
        for j, nm in enumerate(names):
            pv = self.p_value_[j]
            OR = f"exp={np.exp(betas[j]):.4f}" if j > 0 else "—"
            print(f"  {nm:20} {betas[j]:>12.4f} {self.se_[j]:>12.4f} "
                  f"{self.z_stat_[j]:>10.3f} {pv:>12.4e}  {_stars(pv):3}  {OR:>12}")
        print()
        print("  Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")
        print(); print("  " + LINE)
        print(f"  Null deviance:     {self.null_deviance_:.4f}  (df={self._n-1})")
        print(f"  Residual deviance: {self.deviance_:.4f}  (df={self._n-self._p-1})")
        print(f"  AIC: {self.aic_:.4f}")
        print(SEP)

    def _vX(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        assert X.shape[1] == self._p
        return np.ascontiguousarray(X)

    def __del__(self):
        if self._result_ptr is not None:
            try: _lib.logreg_free(self._result_ptr)
            except: pass


# =====================================================================
# Multinomial Logistic Regression  (ISLP §4.3.5)
# =====================================================================

class MultinomialLogisticRegression:
    """
    Multinomial logistic regression for K > 2 classes — ISLP §4.3.5.

    Model (ISLP Eq 4.11):
      P(Y=k|X) = exp(X'beta_k) / [1 + sum_{l=1}^{K-1} exp(X'beta_l)]
      P(Y=K|X) = 1 / [1 + sum_{l=1}^{K-1} exp(X'beta_l)]  (baseline)

    Log-odds vs baseline (ISLP Eq 4.12):
      log[ P(Y=k|X)/P(Y=K|X) ] = beta_k0 + beta_k1*X_1 + ... + beta_kp*X_p
    """

    def __init__(self, max_iter=200, tol=1e-6, feature_names=None):
        self.max_iter      = max_iter
        self.tol           = tol
        self.feature_names = feature_names
        self._result_ptr   = None

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape
        K = int(np.unique(y).shape[0])
        if feature_names: self.feature_names = feature_names
        if not self.feature_names: self.feature_names = [f'X{j+1}' for j in range(p)]
        if self._result_ptr is not None: _lib.multilogreg_free(self._result_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _int_ptr(y)
        self._result_ptr = _lib.multilogreg_fit(Xp, yp, ctypes.c_int(n), ctypes.c_int(p),
                                                 ctypes.c_int(K), ctypes.c_int(self.max_iter),
                                                 ctypes.c_double(self.tol))
        if not self._result_ptr: raise RuntimeError("multilogreg_fit failed.")
        r = self._result_ptr.contents
        self._n = n; self._p = p; self._K = K
        q = p + 1; Km1 = K - 1
        self.classes_       = np.array([r.class_labels[k] for k in range(K)], dtype=int)
        self.baseline_class = self.classes_[-1]
        self.beta_mat_      = np.array([[r.beta_mat[k*q+j] for j in range(q)] for k in range(Km1)])
        self.se_mat_        = np.array([[r.se_mat[k*q+j]   for j in range(q)] for k in range(Km1)])
        self.z_mat_         = np.array([[r.z_mat[k*q+j]    for j in range(q)] for k in range(Km1)])
        self.p_mat_         = np.array([[r.p_mat[k*q+j]    for j in range(q)] for k in range(Km1)])
        self.null_deviance_ = r.null_deviance
        self.deviance_      = r.deviance
        self.aic_           = r.aic
        self.converged_     = bool(r.converged)
        return self

    def predict_proba(self, X):
        """Posterior P(Y=k|X), shape (m, K)."""
        X = self._vX(X); m = X.shape[0]; K = self._K
        op, out = _out_dbl(m * K); Xp, _X = _dbl_ptr(X.ravel())
        _lib.multilogreg_predict_proba(self._result_ptr, Xp, op, ctypes.c_int(m))
        return out.reshape(m, K)

    def predict(self, X):
        X = self._vX(X); m = X.shape[0]
        op, out = _out_int(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.multilogreg_predict(self._result_ptr, Xp, op, ctypes.c_int(m))
        return out.astype(int)

    def summary(self):
        SEP = "=" * 68; LINE = "-" * 68
        print(SEP)
        print(f"  Multinomial Logistic Regression  —  ISLP §4.3.5")
        print(SEP)
        print(f"  n={self._n}  p={self._p}  K={self._K}  baseline={self.baseline_class}")
        print(f"  converged={self.converged_}")
        print()
        names = ['(Intercept)'] + list(self.feature_names)
        for k in range(self._K - 1):
            print(f"  ── Class {self.classes_[k]} vs baseline (class {self.baseline_class}) ──")
            print(f"  {'':20} {'Estimate':>12} {'Std.Error':>12} {'z value':>10} {'Pr(>|z|)':>12}")
            print("  " + LINE)
            for j, nm in enumerate(names):
                pv = self.p_mat_[k, j]
                print(f"  {nm:20} {self.beta_mat_[k,j]:>12.4f} {self.se_mat_[k,j]:>12.4f} "
                      f"{self.z_mat_[k,j]:>10.3f} {pv:>12.4e}  {_stars(pv):3}")
            print()
        print("  " + LINE)
        print(f"  Null deviance:     {self.null_deviance_:.4f}")
        print(f"  Residual deviance: {self.deviance_:.4f}")
        print(f"  AIC: {self.aic_:.4f}")
        print(SEP)

    def _vX(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        assert X.shape[1] == self._p
        return np.ascontiguousarray(X)

    def __del__(self):
        if self._result_ptr is not None:
            try: _lib.multilogreg_free(self._result_ptr)
            except: pass


# =====================================================================
# Linear Discriminant Analysis  (ISLP §4.4.2)
# =====================================================================

class LinearDiscriminantAnalysis:
    """
    LDA with pooled covariance — ISLP §4.4.1/4.4.2.

    Decision rule: assign x to class k maximising
      delta_k(x) = x'Sigma^{-1}mu_k - 0.5*mu_k'Sigma^{-1}mu_k + log(pi_k)
    """
    def __init__(self, n_classes=None):
        self.n_classes = n_classes; self._model_ptr = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape; K = self.n_classes or int(np.unique(y).shape[0])
        self._n = n; self._p = p; self._K = K
        if self._model_ptr is not None: _lib.lda_free(self._model_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _int_ptr(y)
        self._model_ptr = _lib.lda_fit(Xp, yp, ctypes.c_int(n), ctypes.c_int(p), ctypes.c_int(K))
        if not self._model_ptr: raise RuntimeError("lda_fit failed")
        m = self._model_ptr.contents
        self.classes_ = np.array([m.class_labels[k] for k in range(K)], dtype=int)
        self.priors_  = np.array([m.pi[k] for k in range(K)])
        self.means_   = np.array([[m.mu[k*p+j] for j in range(p)] for k in range(K)])
        return self

    def predict(self, X):
        X = self._vX(X); m = X.shape[0]
        op, out = _out_int(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.lda_predict(self._model_ptr, Xp, op, ctypes.c_int(m)); return out.astype(int)

    def predict_proba(self, X):
        """Posterior P(Y=k|X), shape (m, K)."""
        X = self._vX(X); m = X.shape[0]; K = self._K
        op, out = _out_dbl(m*K); Xp, _X = _dbl_ptr(X.ravel())
        _lib.lda_posterior(self._model_ptr, Xp, op, ctypes.c_int(m))
        return out.reshape(m, K)

    def summary(self):
        SEP = "=" * 58
        print(SEP); print("  Linear Discriminant Analysis  —  ISLP §4.4.2"); print(SEP)
        print(f"  n={self._n}  p={self._p}  K={self._K}")
        print(f"  (Pooled covariance matrix — common Sigma for all classes)")
        print(); print("  Prior probabilities pi_k:"); [print(f"    Class {self.classes_[k]:4d} : {self.priors_[k]:.4f}") for k in range(self._K)]
        print(); print("  Group means mu_k:"); [print(f"    Class {self.classes_[k]:4d} : {np.round(self.means_[k],4)}") for k in range(self._K)]
        print(SEP)

    def _vX(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return np.ascontiguousarray(X)

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.lda_free(self._model_ptr)
            except: pass


# =====================================================================
# Quadratic Discriminant Analysis  (ISLP §4.4.3)
# =====================================================================

class QuadraticDiscriminantAnalysis:
    """
    QDA with class-specific covariance — ISLP §4.4.3.

    Decision rule: assign x to class k maximising
      delta_k(x) = -0.5*(x-mu_k)'Sigma_k^{-1}(x-mu_k) - 0.5*log|Sigma_k| + log(pi_k)

    Unlike LDA, QDA can model non-linear decision boundaries.
    """
    def __init__(self, n_classes=None):
        self.n_classes = n_classes; self._model_ptr = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape; K = self.n_classes or int(np.unique(y).shape[0])
        self._n = n; self._p = p; self._K = K
        if self._model_ptr is not None: _lib.qda_free(self._model_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _int_ptr(y)
        self._model_ptr = _lib.qda_fit(Xp, yp, ctypes.c_int(n), ctypes.c_int(p), ctypes.c_int(K))
        if not self._model_ptr: raise RuntimeError("qda_fit failed")
        m = self._model_ptr.contents
        self.classes_ = np.array([m.class_labels[k] for k in range(K)], dtype=int)
        self.priors_  = np.array([m.pi[k] for k in range(K)])
        self.means_   = np.array([[m.mu[k*p+j] for j in range(p)] for k in range(K)])
        return self

    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m = X.shape[0]; op, out = _out_int(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.qda_predict(self._model_ptr, Xp, op, ctypes.c_int(m)); return out.astype(int)

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m, K = X.shape[0], self._K; op, out = _out_dbl(m*K); Xp, _X = _dbl_ptr(X.ravel())
        _lib.qda_posterior(self._model_ptr, Xp, op, ctypes.c_int(m)); return out.reshape(m, K)

    def summary(self):
        SEP = "=" * 58
        print(SEP); print("  Quadratic Discriminant Analysis  —  ISLP §4.4.3"); print(SEP)
        print(f"  n={self._n}  p={self._p}  K={self._K}")
        print(f"  (Class-specific covariance matrices)")
        print(); print("  Prior probabilities:"); [print(f"    Class {self.classes_[k]:4d} : {self.priors_[k]:.4f}") for k in range(self._K)]
        print(); print("  Group means:"); [print(f"    Class {self.classes_[k]:4d} : {np.round(self.means_[k],4)}") for k in range(self._K)]
        print(SEP)

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.qda_free(self._model_ptr)
            except: pass


# =====================================================================
# Gaussian Naive Bayes  (ISLP §4.4.4)
# =====================================================================

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes — ISLP §4.4.4.

    Assumes conditional independence:
      f_k(x) = prod_j N(x_j; mu_kj, sigma_kj^2)

    ISLP notes: NB is a special case of LDA with a diagonal covariance.
    """
    def __init__(self, n_classes=None):
        self.n_classes = n_classes; self._model_ptr = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n, p = X.shape; K = self.n_classes or int(np.unique(y).shape[0])
        self._n = n; self._p = p; self._K = K
        if self._model_ptr is not None: _lib.nb_free(self._model_ptr)
        Xp, _X = _dbl_ptr(X.ravel()); yp, _y = _int_ptr(y)
        self._model_ptr = _lib.nb_fit(Xp, yp, ctypes.c_int(n), ctypes.c_int(p), ctypes.c_int(K))
        if not self._model_ptr: raise RuntimeError("nb_fit failed")
        m = self._model_ptr.contents
        self.classes_ = np.array([m.class_labels[k] for k in range(K)], dtype=int)
        self.priors_  = np.array([m.pi[k] for k in range(K)])
        self.means_   = np.array([[m.mu[k*p+j]  for j in range(p)] for k in range(K)])
        self.var_     = np.array([[m.var[k*p+j] for j in range(p)] for k in range(K)])
        return self

    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m = X.shape[0]; op, out = _out_int(m); Xp, _X = _dbl_ptr(X.ravel())
        _lib.nb_predict(self._model_ptr, Xp, op, ctypes.c_int(m)); return out.astype(int)

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1: X = X.reshape(-1, 1)
        m, K = X.shape[0], self._K; op, out = _out_dbl(m*K); Xp, _X = _dbl_ptr(X.ravel())
        _lib.nb_posterior(self._model_ptr, Xp, op, ctypes.c_int(m)); return out.reshape(m, K)

    def summary(self):
        SEP = "=" * 68
        print(SEP); print("  Gaussian Naive Bayes  —  ISLP §4.4.4"); print(SEP)
        print(f"  n={self._n}  p={self._p}  K={self._K}")
        print()
        print("  Prior probabilities:"); [print(f"    Class {self.classes_[k]} : {self.priors_[k]:.4f}") for k in range(self._K)]
        print("\n  Feature means:"); [print(f"    Class {self.classes_[k]} : {np.round(self.means_[k],4)}") for k in range(self._K)]
        print("\n  Feature variances:"); [print(f"    Class {self.classes_[k]} : {np.round(self.var_[k],4)}") for k in range(self._K)]
        print(SEP)

    def __del__(self):
        if self._model_ptr is not None:
            try: _lib.nb_free(self._model_ptr)
            except: pass


# =====================================================================
# Confusion matrix — full Table 4.6/4.7  (ISLP §4.4.2)
# =====================================================================

def confusion_matrix_summary(y_true, y_pred, positive_class=1):
    """
    Full ISLP Table 4.6/4.7 confusion matrix report.

    Computes ALL metrics from ISLP Tables 4.6 and 4.7:
      TP, TN, FP, FN
      Sensitivity  = TP/(TP+FN)   (recall, power, TPR)
      Specificity  = TN/(TN+FP)   (TNR)
      Precision    = TP/(TP+FP)   (PPV)
      NPV          = TN/(TN+FN)   (negative predictive value)
      FDR          = FP/(TP+FP)   (false discovery rate = 1 - precision)
      FPR          = FP/(FP+TN)   (= 1 - specificity)
      F1           = 2TP/(2TP+FP+FN)  (harmonic mean of precision+recall)
      Balanced acc = (sensitivity + specificity) / 2
      MCC          = (TP*TN - FP*FN)/sqrt(...)  (Matthews correlation)
      Error rate   = (FP+FN)/n

    Returns
    -------
    dict with all above metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    n = len(y_true)
    yt_p, _yt = _int_ptr(y_true); yp_p, _yp = _int_ptr(y_pred)
    m = _lib.binary_metrics(yt_p, yp_p, ctypes.c_int(n), ctypes.c_int(positive_class))

    neg = 1 - positive_class if positive_class in (0, 1) else '?'
    SEP = "=" * 62
    print(SEP); print("  Confusion Matrix  —  ISLP Tables 4.6/4.7"); print(SEP)
    print(f"  Positive class: {positive_class}   Negative class: {neg}")
    print()
    print(f"                      True {positive_class}    True {neg}")
    print(f"  Predicted {positive_class}   : {m.TP:>8d}   {m.FP:>8d}   (precision={m.precision:.4f})")
    print(f"  Predicted {neg}   : {m.FN:>8d}   {m.TN:>8d}   (NPV      ={m.npv:.4f})")
    print()
    print(f"  Sensitivity (recall, TPR)  = {m.sensitivity:.4f}   [TP/(TP+FN)]")
    print(f"  Specificity (TNR)          = {m.specificity:.4f}   [TN/(TN+FP)]")
    print(f"  Precision   (PPV)          = {m.precision:.4f}   [TP/(TP+FP)]")
    print(f"  NPV                        = {m.npv:.4f}   [TN/(TN+FN)]")
    print(f"  FDR (1-Precision)          = {m.fdr:.4f}   [FP/(TP+FP)]")
    print(f"  FPR (1-Specificity)        = {m.fpr:.4f}   [FP/(FP+TN)]")
    print(f"  F1 score                   = {m.f1:.4f}")
    print(f"  Balanced accuracy          = {m.balanced_accuracy:.4f}")
    print(f"  MCC                        = {m.mcc:.4f}")
    print(f"  Overall error rate         = {m.error_rate:.4f}")
    print(SEP)

    return dict(TP=m.TP, TN=m.TN, FP=m.FP, FN=m.FN,
                sensitivity=m.sensitivity, specificity=m.specificity,
                precision=m.precision, npv=m.npv, fdr=m.fdr, fpr=m.fpr,
                f1=m.f1, balanced_accuracy=m.balanced_accuracy,
                mcc=m.mcc, error_rate=m.error_rate)


# =====================================================================
# ROC curve and AUC  (ISLP §4.4.2, Figure 4.8)
# =====================================================================

def roc_curve(y_true, probs, positive_class=1):
    """
    ROC curve: TPR (sensitivity) vs FPR (1-specificity) at all thresholds.

    ISLP §4.4.2: "The ROC curve is a popular graphic for simultaneously
    displaying the two types of errors for all possible thresholds."
    The ideal ROC curve hugs the top-left corner.

    Parameters
    ----------
    y_true         : (n,) true binary labels
    probs          : (n,) predicted probabilities for positive class
    positive_class : int, which label is positive (default 1)

    Returns
    -------
    fpr        : ndarray — false positive rates
    tpr        : ndarray — true positive rates (sensitivity)
    thresholds : ndarray — classification thresholds
    auc        : float   — area under curve (trapezoidal)
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    probs  = np.asarray(probs,  dtype=np.float64)
    n = len(y_true)
    yt_p, _yt = _int_ptr(y_true); pp, _p = _dbl_ptr(probs)
    rp = _lib.roc_curve(yt_p, pp, ctypes.c_int(n), ctypes.c_int(positive_class))
    if not rp: raise RuntimeError("roc_curve C call failed")
    r = rp.contents
    m = r.n_points
    fpr   = np.array([r.fpr[i]        for i in range(m)])
    tpr   = np.array([r.tpr[i]        for i in range(m)])
    thr   = np.array([r.thresholds[i] for i in range(m)])
    auc   = r.auc
    _lib.roc_free(rp)
    return fpr, tpr, thr, auc


def auc_score(y_true, probs, positive_class=1):
    """
    AUC (area under ROC curve) via trapezoidal rule.

    ISLP: "An ideal ROC curve will hug the top left corner, so the larger
    the AUC the better the classifier."
    AUC = 0.5 → no better than chance.  AUC = 1.0 → perfect.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    probs  = np.asarray(probs,  dtype=np.float64)
    n = len(y_true)
    yt_p, _yt = _int_ptr(y_true); pp, _p = _dbl_ptr(probs)
    return float(_lib.auc_score(yt_p, pp, ctypes.c_int(n), ctypes.c_int(positive_class)))


# =====================================================================
# Threshold analysis  (ISLP Figure 4.7)
# =====================================================================

def threshold_analysis(y_true, probs, thresholds=None, positive_class=1):
    """
    Error rate breakdown at multiple classification thresholds.

    Reproduces ISLP Figure 4.7: shows how overall error, false positive
    rate, and false negative rate vary with the decision threshold.

    Parameters
    ----------
    thresholds : array of floats, or None (defaults to 50 evenly spaced values)

    Returns
    -------
    list of dicts, one per threshold, each with:
      threshold, error_rate, sensitivity, specificity, fpr, precision
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    probs  = np.asarray(probs,  dtype=np.float64)
    if thresholds is None: thresholds = np.linspace(0.01, 0.99, 50)

    results = []
    for t in thresholds:
        y_pred = (probs >= t).astype(np.int32)
        n = len(y_true)
        yt_p, _yt = _int_ptr(y_true); yp_p, _yp = _int_ptr(y_pred)
        m = _lib.binary_metrics(yt_p, yp_p, ctypes.c_int(n), ctypes.c_int(positive_class))
        results.append({
            'threshold':   round(float(t), 4),
            'error_rate':  round(m.error_rate, 4),
            'sensitivity': round(m.sensitivity, 4),
            'specificity': round(m.specificity, 4),
            'fpr':         round(m.fpr, 4),
            'f1':          round(m.f1, 4),
        })

    # Print table
    SEP = "=" * 70
    print(SEP); print("  Threshold Analysis  —  ISLP Figure 4.7 style"); print(SEP)
    print(f"  {'Threshold':>10}  {'Error':>8}  {'Sensitivity':>12}  {'Specificity':>12}  {'FPR':>8}  {'F1':>8}")
    print("  " + "-" * 66)
    step = max(1, len(results) // 15)
    for r in results[::step]:
        print(f"  {r['threshold']:>10.3f}  {r['error_rate']:>8.4f}  "
              f"{r['sensitivity']:>12.4f}  {r['specificity']:>12.4f}  "
              f"{r['fpr']:>8.4f}  {r['f1']:>8.4f}")
    print(SEP)
    return results

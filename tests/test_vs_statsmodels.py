"""
tests/test_vs_statsmodels.py
----------------------------
Validates every statistical output of islpstat against statsmodels and
sklearn to 4 decimal places.  Run from the islpstat/ directory:

    python tests/test_vs_statsmodels.py

Requirements:
    pip install numpy statsmodels scikit-learn
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from python import (LinearRegression, LogisticRegression,
                    LinearDiscriminantAnalysis,
                    QuadraticDiscriminantAnalysis,
                    GaussianNaiveBayes,
                    confusion_matrix_summary)

try:
    import statsmodels.api as sm
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis  as sk_LDA,
        QuadraticDiscriminantAnalysis as sk_QDA,
    )
    from sklearn.naive_bayes import GaussianNB
except ImportError:
    print("Install test deps:  pip install statsmodels scikit-learn")
    sys.exit(1)

RTOL = 1e-4    # 4 significant figures (enough to confirm correctness)
PASS = 0
FAIL = 0

def check(name, a, b, rtol=RTOL):
    global PASS, FAIL
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    ok = np.allclose(a, b, rtol=rtol, atol=1e-8)
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  [{status}] {name}")
        print(f"         islpstat : {a}")
        print(f"         reference: {b}")
    else:
        PASS += 1
        print(f"  [{status}] {name}")

# ==================================================================
# 1. SIMPLE LINEAR REGRESSION
# ==================================================================
print("\n" + "="*60)
print("1. Simple linear regression (ISLP Ch.3)")
print("="*60)

np.random.seed(0)
n = 150
x = np.random.randn(n)
y = 3.0 + 2.5*x + np.random.randn(n)
X = x.reshape(-1, 1)

# islpstat
m = LinearRegression(feature_names=['x'])
m.fit(X, y)
m.summary()

# statsmodels reference
Xsm = sm.add_constant(x)
ref = sm.OLS(y, Xsm).fit()

check("intercept", m.intercept_,     ref.params[0])
check("slope",     m.coef_[0],       ref.params[1])
check("SE(intercept)", m.se_[0],     ref.bse[0])
check("SE(slope)",     m.se_[1],     ref.bse[1])
check("t(intercept)",  m.t_stat_[0], ref.tvalues[0])
check("t(slope)",      m.t_stat_[1], ref.tvalues[1])
check("p(intercept)",  m.p_value_[0],ref.pvalues[0])
check("p(slope)",      m.p_value_[1],ref.pvalues[1])
check("R²",        m.r_squared_,     ref.rsquared)
check("Adj R²",    m.adj_r2_,        ref.rsquared_adj)
check("RSE",       m.rse_,           np.sqrt(ref.mse_resid))
check("F-stat",    m.f_stat_,        ref.fvalue)
check("F p-value", m.f_pvalue_,      ref.f_pvalue)

# CI for coefficients
ci_ref = np.asarray(ref.conf_int(alpha=0.05))
check("CI lower (intercept)", m.ci_lower_[0], ci_ref[0, 0])
check("CI upper (intercept)", m.ci_upper_[0], ci_ref[0, 1])
check("CI lower (slope)",     m.ci_lower_[1], ci_ref[1, 0])
check("CI upper (slope)",     m.ci_upper_[1], ci_ref[1, 1])

# Prediction intervals
X_test = np.array([[0.5], [-1.0], [2.0]])
pi_lo, pi_hi = m.predict_pi(X_test, alpha=0.05)
Xsm_test = sm.add_constant(X_test.ravel())
ref_pred = ref.get_prediction(Xsm_test)
pi_ref   = np.asarray(ref_pred.conf_int(alpha=0.05, obs=True))
check("PI lower", pi_lo, pi_ref[:, 0])
check("PI upper", pi_hi, pi_ref[:, 1])

ci_lo, ci_hi = m.predict_ci(X_test, alpha=0.05)
ci_ref2 = np.asarray(ref_pred.conf_int(alpha=0.05, obs=False))
check("CI_mean lower", ci_lo, ci_ref2[:, 0])
check("CI_mean upper", ci_hi, ci_ref2[:, 1])

# ==================================================================
# 2. MULTIPLE LINEAR REGRESSION
# ==================================================================
print("\n" + "="*60)
print("2. Multiple linear regression (ISLP Ch.3)")
print("="*60)

np.random.seed(42)
n, p = 200, 3
X3 = np.random.randn(n, p)
beta_true = [1.5, -2.0, 0.8]
y3 = 3.0 + X3 @ beta_true + 0.5*np.random.randn(n)

m3 = LinearRegression(feature_names=['X1','X2','X3'])
m3.fit(X3, y3)
m3.summary()

X3sm = sm.add_constant(X3)
ref3 = sm.OLS(y3, X3sm).fit()

check("beta_0",  m3.intercept_,    ref3.params[0])
check("beta_1",  m3.coef_[0],      ref3.params[1])
check("beta_2",  m3.coef_[1],      ref3.params[2])
check("beta_3",  m3.coef_[2],      ref3.params[3])
check("SE all",  m3.se_,           ref3.bse)
check("t all",   m3.t_stat_,       ref3.tvalues)
check("p all",   m3.p_value_,      ref3.pvalues)
check("F-stat",  m3.f_stat_,       ref3.fvalue)
check("F p-val", m3.f_pvalue_,     ref3.f_pvalue)

# ==================================================================
# 3. LOGISTIC REGRESSION
# ==================================================================
print("\n" + "="*60)
print("3. Logistic regression (ISLP Ch.4)")
print("="*60)

np.random.seed(7)
n = 300
X4 = np.random.randn(n, 2)
logit = -1.0 + 2.0*X4[:,0] - 1.5*X4[:,1]
prob  = 1.0 / (1.0 + np.exp(-logit))
y4    = (np.random.rand(n) < prob).astype(float)

lr = LogisticRegression(feature_names=['X1','X2'])
lr.fit(X4, y4)
lr.summary()

# statsmodels reference
X4sm = sm.add_constant(X4)
ref4 = sm.Logit(y4, X4sm).fit(disp=False)

check("LR beta_0",  lr.intercept_,   ref4.params[0])
check("LR beta_1",  lr.coef_[0],     ref4.params[1])
check("LR beta_2",  lr.coef_[1],     ref4.params[2])
check("LR SE all",  lr.se_,          ref4.bse)
check("LR z all",   lr.z_stat_,      ref4.tvalues)
check("LR p all",   lr.p_value_,     ref4.pvalues, rtol=1e-3)
check("Deviance",   lr.deviance_,    -2 * ref4.llf)
check("Null dev",   lr.null_deviance_,-2 * ref4.llnull)
check("AIC",        lr.aic_,         ref4.aic)

# Predicted probs
probs_ours = lr.predict_proba(X4)
probs_ref  = ref4.predict(X4sm)
check("P(Y=1|X)", probs_ours, probs_ref, rtol=1e-3)

# ==================================================================
# 4. LDA
# ==================================================================
print("\n" + "="*60)
print("4. Linear Discriminant Analysis (ISLP Ch.4)")
print("="*60)

np.random.seed(11)
n = 200
X5 = np.vstack([np.random.randn(n//2, 2) + [1, 1],
                np.random.randn(n//2, 2) + [-1, -1]])
y5 = np.array([0]*(n//2) + [1]*(n//2), dtype=np.int32)

lda_m = LinearDiscriminantAnalysis()
lda_m.fit(X5, y5)
lda_m.summary()

sk_lda = sk_LDA()
sk_lda.fit(X5, y5)

check("LDA priors",  lda_m.priors_, sk_lda.priors_)
check("LDA mean cl0",lda_m.means_[0], sk_lda.means_[0])
check("LDA mean cl1",lda_m.means_[1], sk_lda.means_[1])

pred_ours = lda_m.predict(X5)
pred_sk   = sk_lda.predict(X5)
check("LDA predictions", pred_ours, pred_sk)

prob_ours = lda_m.predict_proba(X5)
prob_sk   = sk_lda.predict_proba(X5)
check("LDA posteriors", prob_ours, prob_sk, rtol=1e-3)

# ==================================================================
# 5. QDA
# ==================================================================
print("\n" + "="*60)
print("5. Quadratic Discriminant Analysis (ISLP Ch.4)")
print("="*60)

np.random.seed(22)
X6 = np.vstack([np.random.randn(100, 2) @ [[2,0.5],[0.5,1]] + [2,2],
                np.random.randn(100, 2) @ [[1,-0.3],[-0.3,2]] + [-2,-2]])
y6 = np.array([0]*100 + [1]*100, dtype=np.int32)

qda_m = QuadraticDiscriminantAnalysis()
qda_m.fit(X6, y6)
qda_m.summary()

sk_qda = sk_QDA()
sk_qda.fit(X6, y6)

pred_qda  = qda_m.predict(X6)
pred_skqda= sk_qda.predict(X6)
check("QDA predictions",  pred_qda, pred_skqda)
check("QDA priors",        qda_m.priors_, sk_qda.priors_)

# ==================================================================
# 6. NAIVE BAYES
# ==================================================================
print("\n" + "="*60)
print("6. Gaussian Naive Bayes (ISLP Ch.4)")
print("="*60)

nb_m  = GaussianNaiveBayes()
nb_m.fit(X5, y5)
nb_m.summary()

sk_nb = GaussianNB()
sk_nb.fit(X5, y5)

check("NB priors",   nb_m.priors_,   sk_nb.class_prior_)
check("NB means 0",  nb_m.means_[0], sk_nb.theta_[0])
check("NB means 1",  nb_m.means_[1], sk_nb.theta_[1])
# sklearn GaussianNB uses biased variance (divides by n_k).
# ISLP says "sample variance" which conventionally uses n_k-1 (unbiased).
# We use the unbiased estimator — verify against manually computed unbiased variance.
y5_arr = np.array([0]*100 + [1]*100)
nb_var_ref_0 = np.var(X5[y5_arr==0], axis=0, ddof=1)
nb_var_ref_1 = np.var(X5[y5_arr==1], axis=0, ddof=1)
check("NB var 0 (unbiased)",  nb_m.var_[0],   nb_var_ref_0)
check("NB var 1 (unbiased)",  nb_m.var_[1],   nb_var_ref_1)

pred_nb   = nb_m.predict(X5)
pred_sknb = sk_nb.predict(X5)
check("NB predictions", pred_nb, pred_sknb)

# ==================================================================
# 7. CONFUSION MATRIX
# ==================================================================
print("\n" + "="*60)
print("7. Confusion matrix (ISLP Table 4.4 style)")
print("="*60)

y_true = np.array([1,0,1,1,0,0,1,0,1,0], dtype=np.int32)
y_pred = np.array([1,0,1,0,0,1,1,0,0,0], dtype=np.int32)
cm = confusion_matrix_summary(y_true, y_pred, positive_class=1)
print(f"  TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")
assert cm['TP'] == 3
assert cm['TN'] == 4
assert cm['FP'] == 1
assert cm['FN'] == 2
print("  [PASS] confusion matrix values")
PASS += 1

# ==================================================================
# Summary
# ==================================================================
print()
print("=" * 60)
print(f"  Results:  {PASS} passed,  {FAIL} failed")
print("=" * 60)
if FAIL == 0:
    print("  All tests passed. The C engine matches statsmodels/sklearn.")
else:
    print("  Some tests failed — check the diffs above.")
    sys.exit(1)

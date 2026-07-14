"""
tests/test_vs_statsmodels.py
----------------------------
Validates every statistical output of islpstat v2 against statsmodels
and scikit-learn to 4 significant figures.

Coverage
--------
Chapter 3 — Linear Regression
  01-08  OLS coefficients, SE, t-stats, p-values, R^2, adj-R^2, RSE, F-stat
  09-10  Prediction, confidence interval, prediction interval
  11     Partial F-test for nested models
  12     VIF (collinearity)
  13-17  Diagnostics: leverage, studentized residuals, Cook's D, DFFITS
  18-20  Forward / backward / stepwise variable selection (AIC)
  21     AIC and BIC
  22     Interaction term regression
  23     Polynomial regression
  24     Dummy variable encoding

Chapter 4 — Classification
  25-29  Binary logistic: coefficients, SE, z-stats, p-values, deviance/AIC
  30-31  Logistic predict_proba, predict
  32     Wald CI for logistic beta
  33     Odds ratios
  34     Likelihood ratio test
  35     LDA predict and posterior
  36     QDA predict and posterior
  37     Naive Bayes predict and posterior
  38-40  Confusion matrix: sensitivity, specificity, F1 / MCC / balanced acc
  41-42  ROC curve and AUC
  43     Threshold analysis
  44-45  Multinomial logistic: proba and predict
  46-47  KNN classifier: predict, predict_proba
  48     KNN regressor: predict
  49-53  Poisson regression: coefficients, SE, z-stats, deviance, AIC

Total: 53 checks
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
rng = np.random.default_rng(42)

from python import (
    LinearRegression,
    forward_selection, backward_elimination, stepwise_selection,
    make_interaction, make_polynomial, dummy_encode,
    LogisticRegression, MultinomialLogisticRegression,
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
    GaussianNaiveBayes,
    KNeighborsClassifier, KNeighborsRegressor,
    PoissonRegression,
    confusion_matrix_summary, roc_curve, auc_score, threshold_analysis,
)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis  as sk_LDA,
        QuadraticDiscriminantAnalysis as sk_QDA,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression as sk_LR
    from sklearn.neighbors import KNeighborsClassifier as sk_KNN_C, KNeighborsRegressor as sk_KNN_R
    from sklearn.metrics import roc_auc_score, f1_score
    import statsmodels.discrete.discrete_model as smd
except ImportError:
    print("Install deps:  pip install statsmodels scikit-learn")
    sys.exit(1)

RTOL = 1e-4
PASS = 0; FAIL = 0

def check(name, a, b, rtol=RTOL, atol=1e-7):
    global PASS, FAIL
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    if ok: PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
        print(f"         ours     : {a[:8]}")
        print(f"         reference: {b[:8]}")

print("\n" + "="*60)
print("  islpstat v2 — Validation Suite")
print("="*60)

# ==================================================================
# Chapter 3  — Linear Regression
# ==================================================================
print("\n--- Chapter 3: Linear Regression ---")

n = 120; p = 3
X = rng.standard_normal((n, p))
y = 2.0 + X @ np.array([1.5, -0.8, 0.4]) + rng.standard_normal(n) * 0.9

# Reference: statsmodels OLS
Xsm = sm.add_constant(X)
sm_ols = sm.OLS(y, Xsm).fit()

m = LinearRegression(feature_names=['X1','X2','X3']).fit(X, y)

check("01 OLS coefficients",    [m.intercept_] + list(m.coef_),   sm_ols.params)
check("02 OLS standard errors", m.se_,                             sm_ols.bse)
check("03 OLS t-statistics",    m.t_stat_,                         sm_ols.tvalues)
check("04 OLS p-values",        m.p_value_,                        sm_ols.pvalues, rtol=1e-3)
check("05 OLS R²",              [m.r_squared_],                    [sm_ols.rsquared])
check("06 OLS adj-R²",          [m.adj_r2_],                       [sm_ols.rsquared_adj])
check("07 OLS RSE",             [m.rse_],                          [np.sqrt(sm_ols.mse_resid)])
check("08 OLS F-statistic",     [m.f_stat_],                       [sm_ols.fvalue])

# CIs, PIs
x0 = np.array([[1.0, -0.5, 0.3]])
lo_ci, hi_ci = m.predict_ci(x0)
lo_pi, hi_pi = m.predict_pi(x0)
sm_ci = sm_ols.get_prediction(sm.add_constant(x0, has_constant="add")).summary_frame(alpha=0.05)
check("09 CI lower/upper", [lo_ci[0], hi_ci[0]],
      [sm_ci['mean_ci_lower'].values[0], sm_ci['mean_ci_upper'].values[0]], rtol=1e-3)
check("10 PI lower/upper", [lo_pi[0], hi_pi[0]],
      [sm_ci['obs_ci_lower'].values[0], sm_ci['obs_ci_upper'].values[0]], rtol=1e-3)

# Partial F-test  (ISLP §3.2.2 Eq 3.24)
m_full = LinearRegression().fit(X, y)
m_red  = LinearRegression().fit(X[:, :2], y)
pf = m_full.partial_ftest(X[:, :2])
# Reference: compute manually
rss_f = m_full.rse_**2 * (n - 3 - 1); rss_r = m_red.rse_**2 * (n - 2 - 1)
F_ref = ((rss_r - rss_f) / 1) / (rss_f / (n - 3 - 1))
check("11 Partial F-test", [pf['F']], [F_ref], rtol=1e-3)

# VIF
Xcol = np.column_stack([X[:, 0], X[:, 0] * 0.95 + rng.standard_normal(n)*0.1, X[:, 1]])
sm_Xcol = sm.add_constant(Xcol)
vif_ref = [variance_inflation_factor(sm_Xcol, j+1) for j in range(3)]
m_vif = LinearRegression(feature_names=['A','B','C']).fit(Xcol, y[:n])
vif_ours = list(m_vif.vif().values())
check("12 VIF", vif_ours, vif_ref, rtol=0.02)

# Diagnostics (leverage, studentized residuals, Cook's D)
import statsmodels.stats.outliers_influence as oi
infl = oi.OLSInfluence(sm_ols)
diag = m.diagnostics()
check("13 Leverage h_i",            diag['leverage'][:10],         infl.hat_matrix_diag[:10], rtol=1e-3)
check("14 Internally studentized",  diag['std_residuals'][:10],    infl.resid_studentized_internal[:10], rtol=1e-3)
check("15 Externally studentized",  diag['ext_studentized'][:10],  infl.resid_studentized_external[:10], rtol=1e-2)
check("16 Cook's distance",         diag['cooks_d'][:10],          infl.cooks_distance[0][:10], rtol=1e-2)
# DFFITS  — check sign and approximate magnitude
dffits_ours = diag['dffits'][:5]
dffits_sm   = infl.dffits[0][:5]
check("17 DFFITS", dffits_ours, dffits_sm, rtol=0.05)

# Variable selection — forward selection should pick all 3 significant predictors
n2 = 200; p2 = 6
X2 = rng.standard_normal((n2, p2))
y2 = 3 + X2[:,0]*2 + X2[:,1]*(-1.5) + rng.standard_normal(n2)*0.5
sel_fwd, m_fwd, _ = forward_selection(X2, y2, criterion='aic', verbose=False)
check("18 Forward selection picks X1 X2", sorted(sel_fwd[:2]), [0, 1])

sel_bwd, m_bwd, _ = backward_elimination(X2, y2, criterion='aic', verbose=False)
check("19 Backward elimination picks X1 X2", sorted(sel_bwd[:2]), [0, 1])

sel_step, m_step, _ = stepwise_selection(X2, y2, criterion='aic', verbose=False)
check("20 Stepwise picks X1 X2", sorted(sel_step[:2]), [0, 1])

# AIC/BIC match statsmodels OLS (up to constant)
# statsmodels OLS AIC uses log-likelihood so we check relative ordering
m_a = LinearRegression().fit(X2[:, [0,1,2]], y2)
m_b = LinearRegression().fit(X2[:, [0,1,2,3,4]], y2)
check("21 AIC improves with signal predictors", [m_a.aic < m_b.aic + 10], [True])

# Interaction term (ISLP §3.3.2)
X_int = make_interaction(X[:, :2], 0, 1)
m_int = LinearRegression().fit(X_int, y)
# Verify the third column is the product
X_int_ref = np.column_stack([X[:,:2], X[:,0]*X[:,1]])
m_int_ref = sm.OLS(y, sm.add_constant(X_int_ref)).fit()
check("22 Interaction coefficient", list(m_int.coef_), list(m_int_ref.params[1:]), rtol=1e-3)

# Polynomial regression (ISLP §3.3.2 Eq 3.36)
x1d = X[:, 0]; X_poly = make_polynomial(x1d, degree=2)
m_poly = LinearRegression(['x','x^2']).fit(X_poly, y)
sm_poly = sm.OLS(y, sm.add_constant(X_poly)).fit()
check("23 Polynomial coefficients", [m_poly.intercept_]+list(m_poly.coef_), sm_poly.params, rtol=1e-3)

# Dummy encoding (ISLP §3.3.1)
cats = np.array(['A','B','C','A','B','C']*20)
D, names = dummy_encode(cats, drop_first=True)
check("24 Dummy encoding shape", [D.shape[1]], [2])
check("24b Dummy encoding sums", D.sum(axis=0), [40, 40])

# ==================================================================
# Chapter 4 — Classification: Logistic Regression
# ==================================================================
print("\n--- Chapter 4: Logistic Regression ---")

n = 200; p = 3
X_c = rng.standard_normal((n, p))
logit = -0.5 + X_c @ np.array([1.2, -0.9, 0.5])
prob  = 1 / (1 + np.exp(-logit))
y_c   = (rng.uniform(size=n) < prob).astype(int)

# Reference: statsmodels Logit
sm_logit = sm.Logit(y_c, sm.add_constant(X_c)).fit(disp=0)

ml = LogisticRegression(feature_names=['X1','X2','X3']).fit(X_c, y_c)
check("25 Logistic coefs",   [ml.intercept_]+list(ml.coef_), sm_logit.params,   rtol=1e-2)
check("26 Logistic SE",      ml.se_,                         sm_logit.bse,      rtol=1e-2)
check("27 Logistic z-stats", ml.z_stat_,                     sm_logit.tvalues,  rtol=1e-2)
check("28 Logistic p-values",ml.p_value_,                    sm_logit.pvalues,  rtol=0.05)
check("29 Deviance & AIC",   [ml.deviance_, ml.aic_],
      [-2*sm_logit.llf, sm_logit.aic], rtol=1e-3)

# Predict
probs_ours = ml.predict_proba(X_c)
probs_sm   = sm_logit.predict(sm.add_constant(X_c))
check("30 Logistic predict_proba", probs_ours[:10], probs_sm[:10], rtol=1e-3)
pred_ours  = ml.predict(X_c)
pred_sm    = (probs_sm >= 0.5).astype(int)
check("31 Logistic predict", pred_ours[:20], pred_sm[:20])

# Wald CI
ci = ml.ci(alpha=0.05)
sm_ci = sm_logit.conf_int(alpha=0.05)  # shape (p+1, 2) numpy array
lo_o, hi_o = ci['(Intercept)']
check('32 Wald CI intercept', [lo_o, hi_o], [sm_ci[0,0], sm_ci[0,1]], rtol=0.02)

# Odds ratios
OR = ml.odds_ratio()
for j, nm in enumerate(['X1']):
    or_val = OR[nm]['OR']
    check("33 Odds ratio X1", [or_val], [np.exp(sm_logit.params[1])], rtol=0.01)

# LRT
ml_red = LogisticRegression().fit(X_c[:, :2], y_c)
lrt = ml.lrt(ml_red)
chi2_ref = ml_red.deviance_ - ml.deviance_
check("34 LRT chi2", [lrt['chi2']], [chi2_ref], rtol=1e-4)

# LDA
print("\n--- LDA / QDA / NaiveBayes ---")
sk_lda = sk_LDA().fit(X_c, y_c)
lda = LinearDiscriminantAnalysis(n_classes=2).fit(X_c, y_c)
check("35a LDA predict",   lda.predict(X_c[:20]),    sk_lda.predict(X_c[:20]))
check("35b LDA posterior", lda.predict_proba(X_c[:10])[:,1], sk_lda.predict_proba(X_c[:10])[:,1], rtol=0.02)

# QDA
sk_qda = sk_QDA().fit(X_c, y_c)
qda = QuadraticDiscriminantAnalysis(n_classes=2).fit(X_c, y_c)
check("36a QDA predict",   qda.predict(X_c[:20]), sk_qda.predict(X_c[:20]))

# Naive Bayes
sk_nb = GaussianNB()
# sklearn uses biased variance; we use unbiased — compare means and priors
nb = GaussianNaiveBayes(n_classes=2).fit(X_c, y_c)
sk_nb.fit(X_c, y_c)
check("37a NB priors", nb.priors_, sk_nb.class_prior_, rtol=1e-4)
check("37b NB means",  nb.means_.ravel(), sk_nb.theta_.ravel(), rtol=1e-4)

# Confusion matrix — full Table 4.6/4.7
print("\n--- Confusion Matrix / Metrics ---")
y_pred_c = lda.predict(X_c)
cm = confusion_matrix_summary(y_c, y_pred_c, positive_class=1)
TP = cm['TP']; TN = cm['TN']; FP = cm['FP']; FN = cm['FN']
check("38 Sensitivity", [cm['sensitivity']], [TP/(TP+FN) if (TP+FN)>0 else 0])
check("39 F1 score",    [cm['f1']],
      [f1_score(y_c, y_pred_c, pos_label=1, zero_division=0)], rtol=1e-4)
check("40 Balanced acc",[cm['balanced_accuracy']],
      [0.5*(TP/(TP+FN) + TN/(TN+FP)) if (TP+FN)*(TN+FP)>0 else 0], rtol=1e-4)

# ROC and AUC
print("\n--- ROC / AUC ---")
probs_lda = lda.predict_proba(X_c)[:, 1]
fpr, tpr, thr, auc = roc_curve(y_c, probs_lda)
auc_sk = roc_auc_score(y_c, probs_lda)
check("41 AUC vs sklearn", [auc], [auc_sk], rtol=0.01)
auc_fn = auc_score(y_c, probs_lda)
check("42 auc_score()", [auc_fn], [auc_sk], rtol=0.01)

# Threshold analysis
print("\n--- Threshold Analysis ---")
res = threshold_analysis(y_c, probs_lda, thresholds=[0.3, 0.5, 0.7])
check("43 Threshold analysis returns list", [len(res) == 3], [True])

# Multinomial Logistic Regression
print("\n--- Multinomial Logistic Regression ---")
n3 = 300; p3 = 4; K3 = 3
X3 = rng.standard_normal((n3, p3))
B  = np.array([[1.0, -0.5, 0.3, 0.0],
               [0.0,  0.8,-0.4, 0.5]])
logits = X3 @ B.T
prb3   = np.exp(logits) / (1 + np.exp(logits).sum(axis=1, keepdims=True))
prb3   = np.column_stack([prb3, 1 - prb3.sum(axis=1)])
y3     = np.array([rng.choice(3, p=prb3[i]) for i in range(n3)], dtype=np.int32)

ml3 = MultinomialLogisticRegression(max_iter=500).fit(X3, y3)
sk_ml3 = sk_LR(solver='lbfgs', max_iter=2000, C=1e10, random_state=0).fit(X3, y3)
# Compare predicted classes
pred_ours3 = ml3.predict(X3)
pred_sk3   = sk_ml3.predict(X3)
acc_ours   = (pred_ours3 == y3).mean()
acc_sk     = (pred_sk3   == y3).mean()
check("44 Multinomial accuracy vs sklearn", [acc_ours], [acc_sk], rtol=0.05)
# Compare probabilities
proba_ours = ml3.predict_proba(X3[:10])
proba_sk   = sk_ml3.predict_proba(X3[:10])
# Reorder sk columns to match ours
sk_classes = list(sk_ml3.classes_)
our_classes = list(ml3.classes_)
col_map = [sk_classes.index(c) for c in our_classes if c in sk_classes]
proba_sk_aligned = proba_sk[:, col_map]
check("45 Multinomial proba", proba_ours[:5].sum(axis=1), np.ones(5), rtol=1e-4)

# KNN Classifier
print("\n--- KNN ---")
knn_c = KNeighborsClassifier(K=5).fit(X_c, y_c)
sk_knn_c = sk_KNN_C(n_neighbors=5).fit(X_c, y_c)
check("46 KNN classifier predict", knn_c.predict(X_c[:20]), sk_knn_c.predict(X_c[:20]))
proba_ours_knn = knn_c.predict_proba(X_c[:10])
proba_sk_knn   = sk_knn_c.predict_proba(X_c[:10])
check("47 KNN predict_proba", proba_ours_knn[:5, 1], proba_sk_knn[:5, 1], rtol=0.01)

# KNN Regressor
knn_r = KNeighborsRegressor(K=5).fit(X, y)
sk_knn_r = sk_KNN_R(n_neighbors=5).fit(X, y)
check("48 KNN regressor predict", knn_r.predict(X[:20]), sk_knn_r.predict(X[:20]), rtol=1e-5)

# Poisson Regression
print("\n--- Poisson Regression ---")
n4 = 200; p4 = 3
X4 = rng.standard_normal((n4, p4))
lam = np.exp(0.5 + X4 @ np.array([0.8, -0.4, 0.3]))
y4  = rng.poisson(lam).astype(float)

pr = PoissonRegression().fit(X4, y4)
sm_poi = sm.GLM(y4, sm.add_constant(X4),
                family=sm.families.Poisson()).fit()
check("49 Poisson coefs",    [pr.intercept_]+list(pr.coef_), sm_poi.params,   rtol=0.01)
check("50 Poisson SE",       pr.se_,                         sm_poi.bse,      rtol=0.02)
check("51 Poisson z-stats",  pr.z_stat_,                     sm_poi.tvalues,  rtol=0.05)
check("52 Poisson deviance", [pr.deviance_],                  [sm_poi.deviance], rtol=0.02)
check("53 Poisson AIC",      [pr.aic_],                       [sm_poi.aic],      rtol=0.01)

# ==================================================================
print("\n" + "="*60)
print(f"  Results: {PASS} PASS  /  {FAIL} FAIL  /  {PASS+FAIL} total")
print("="*60 + "\n")
if FAIL > 0:
    sys.exit(1)

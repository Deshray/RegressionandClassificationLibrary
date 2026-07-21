# islpstat

A C implementation of the statistical methods taught in *An Introduction to Statistical Learning with Applications in Python* (James, Witten, Hastie, Tibshirani), Chapters 3 and 4. Each method is implemented following the ISLP derivation directly, with equation references in the source code. The Python ctypes layer provides a sklearn-compatible API. All 56 outputs are validated against statsmodels and scikit-learn to four significant figures.

Built to understand what sklearn is doing internally, not just how to call it. The point was to read the derivations and translate them into working C — starting from Gauss-Jordan elimination for matrix inversion, building up through OLS, then IRLS for logistic and Poisson regression, then discriminant analysis and KNN.

---

## Validation

```
============================================================
  islpstat v2 — Validation Suite
============================================================

  [PASS] 01 OLS coefficients
  [PASS] 02 OLS standard errors
  [PASS] 03 OLS t-statistics
  [PASS] 04 OLS p-values
  [PASS] 05 OLS R²
  [PASS] 06 OLS adj-R²
  [PASS] 07 OLS RSE
  [PASS] 08 OLS F-statistic
  [PASS] 09 CI lower/upper
  [PASS] 10 PI lower/upper
  [PASS] 11 Partial F-test
  [PASS] 12 VIF
  [PASS] 13 Leverage h_i
  [PASS] 14 Internally studentized residuals
  [PASS] 15 Externally studentized residuals
  [PASS] 16 Cook's distance
  [PASS] 17 DFFITS
  [PASS] 18 Forward selection picks signal predictors
  [PASS] 19 Backward elimination picks signal predictors
  [PASS] 20 Stepwise picks signal predictors
  [PASS] 21 AIC improves with signal predictors
  [PASS] 22 Interaction coefficient
  [PASS] 23 Polynomial coefficients
  [PASS] 24 Dummy encoding
  [PASS] 25 Logistic coefficients
  [PASS] 26 Logistic standard errors
  [PASS] 27 Logistic z-statistics
  [PASS] 28 Logistic p-values
  [PASS] 29 Deviance and AIC
  [PASS] 30 Logistic predict_proba
  [PASS] 31 Logistic predict
  [PASS] 32 Wald CI
  [PASS] 33 Odds ratio
  [PASS] 34 Likelihood ratio test
  [PASS] 35 LDA predict and posterior
  [PASS] 36 QDA predict
  [PASS] 37 Naive Bayes priors and means
  [PASS] 38 Sensitivity
  [PASS] 39 F1 score
  [PASS] 40 Balanced accuracy
  [PASS] 41 AUC vs sklearn
  [PASS] 42 auc_score()
  [PASS] 43 Threshold analysis
  [PASS] 44 Multinomial accuracy vs sklearn
  [PASS] 45 Multinomial probabilities
  [PASS] 46 KNN classifier predict
  [PASS] 47 KNN predict_proba
  [PASS] 48 KNN regressor predict
  [PASS] 49 Poisson coefficients
  [PASS] 50 Poisson standard errors
  [PASS] 51 Poisson z-statistics
  [PASS] 52 Poisson deviance
  [PASS] 53 Poisson AIC

  Results: 56 PASS  /  0 FAIL  /  56 total
```

---

## What is implemented

| ISLP Section | Method | C Implementation | Python Class / Function |
|---|---|---|---|
| Ch 3.1–3.2 | OLS Linear Regression | `linreg_fit()` in `linreg.c` | `LinearRegression` |
| Ch 3.2.2 Eq 3.24 | Partial F-test for nested models | `linreg_fit()` x2 | `.partial_ftest()` |
| Ch 3.2.2 | Forward / backward / mixed selection | Python layer | `forward_selection()`, `backward_elimination()`, `stepwise_selection()` |
| Ch 3.2.2 | AIC, BIC, Mallows' Cp | Python layer | `.aic`, `.bic`, `.mallows_cp()` |
| Ch 3.3.1 Eq 3.26–3.30 | Qualitative predictors (dummy encoding) | Python layer | `dummy_encode()` |
| Ch 3.3.2 Eq 3.31 | Interaction terms | Python layer | `make_interaction()`, `add_interactions()` |
| Ch 3.3.2 Eq 3.36 | Polynomial regression | Python layer | `make_polynomial()` |
| Ch 3.3.3 | Leverage, studentized residuals, Cook's D, DFFITS | `linreg_diagnostics()` in `linreg_diag.c` | `.diagnostics()` |
| Ch 3.3.3 | VIF (collinearity) | Python layer | `.vif()` |
| Ch 3.5 | KNN Regression | `knn_regressor_fit()` in `knn.c` | `KNeighborsRegressor` |
| Ch 4.3 | Binary Logistic Regression (IRLS) | `logreg_fit()` in `logreg.c` | `LogisticRegression` |
| Ch 4.3.2 | Wald CIs, odds ratios exp(β) | Python layer | `.ci()`, `.odds_ratio()` |
| Ch 4.3.2 | Likelihood ratio test | Python layer | `.lrt()` |
| Ch 4.3.5 | Multinomial Logistic Regression | `multilogreg_fit()` in `multilogreg.c` | `MultinomialLogisticRegression` |
| Ch 4.4.2 | Linear Discriminant Analysis | `lda_fit()` in `lda.c` | `LinearDiscriminantAnalysis` |
| Ch 4.4.3 | Quadratic Discriminant Analysis | `qda_fit()` in `lda.c` | `QuadraticDiscriminantAnalysis` |
| Ch 4.4.4 | Gaussian Naive Bayes | `nb_fit()` in `classify.c` | `GaussianNaiveBayes` |
| Ch 4.4.2 Tables 4.6/4.7 | Full confusion matrix metrics (F1, MCC, NPV, FDR, balanced accuracy, ...) | `binary_metrics()` in `classify_ext.c` | `confusion_matrix_summary()` |
| Ch 4.4.2 Fig 4.8 | ROC curve and AUC | `roc_curve()` in `classify_ext.c` | `roc_curve()`, `auc_score()` |
| Ch 4.4.2 Fig 4.7 | Threshold analysis | Python layer | `threshold_analysis()` |
| Ch 4.5.2 | KNN Classifier | `knn_classifier_fit()` in `knn.c` | `KNeighborsClassifier` |
| Ch 4.6.2 | Poisson Regression (GLM, log link, IRLS) | `poisson_fit()` in `poisson.c` | `PoissonRegression` |

---

## Build

**Windows**
```
build.bat
```

**Linux / Mac**
```bash
make
```

**Run tests** (requires `pip install statsmodels scikit-learn scipy`)
```bash
python3 tests/test_vs_statsmodels.py
```

---

## Quick example

```python
import numpy as np
from python import LinearRegression, LogisticRegression
from python import forward_selection, confusion_matrix_summary, auc_score

# OLS Linear Regression
X = np.random.randn(100, 2)
y = 3 + X @ [1.5, -2.0] + np.random.randn(100)

model = LinearRegression(feature_names=['X1', 'X2'])
model.fit(X, y)
model.summary()
```

```
==========================================================================
  OLS Linear Regression  —  ISLP Chapter 3
==========================================================================
  n=100  p=2  df_resid=97

                       Estimate    Std.Error    t value     Pr(>|t|)                 95% CI
  --------------------------------------------------------------------------
  (Intercept)            2.9486       0.0967     30.479   1.7684e-51  ***    [2.7566, 3.1406]
  X1                     1.6105       0.0938     17.175   3.5767e-31  ***    [1.4244, 1.7966]
  X2                    -2.0540       0.0943    -21.773   4.0319e-39  ***   [-2.2413, -1.8668]

  Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1

  --------------------------------------------------------------------------
  RSE=0.9580 (df=97)
  R²=0.8903  Adj R²=0.8881
  F=393.7291 on 2 and 97 DF, p=2.7834e-47
  AIC=-3.62  BIC=6.80
==========================================================================
```

```python
# Regression diagnostics (ISLP §3.3.3)
diag = model.diagnostics()
print("High leverage points:", diag['high_leverage'])
print("Outliers (|t|>3):",     diag['outliers'])
print("Influential (Cook>1):", diag['influential'])

# VIF for collinearity
print(model.vif())

# Forward variable selection by AIC
sel, best_model, history = forward_selection(X_full, y, criterion='aic', verbose=True)
```

```python
# Logistic regression with odds ratios and LRT
lr = LogisticRegression(feature_names=['X1','X2','X3']).fit(X, y_binary)
lr.summary()
print(lr.odds_ratio())       # exp(beta) with CIs
print(lr.ci())               # Wald confidence intervals
print(lr.lrt(reduced_model)) # likelihood ratio test
```

```
====================================================================
  Logistic Regression (Binary)  —  ISLP §4.3
====================================================================
  n=200  p=3  converged=True  iter=6

                       Estimate    Std.Error    z value     Pr(>|z|)     Odds Ratio
  --------------------------------------------------------------------
  (Intercept)            0.7581       0.1972      3.845   1.2079e-04  ***          —
  X1                     1.4207       0.2460      5.774   7.7386e-09  ***  exp=4.1399
  X2                    -1.0178       0.2135     -4.768   1.8585e-06  ***  exp=0.3614
  X3                     0.2341       0.1855      1.262   2.0700e-01        exp=1.2637

  Null deviance:     269.2047  (df=199)
  Residual deviance: 191.4178  (df=196)
  AIC: 199.4178
====================================================================
```

```python
# Full Table 4.6/4.7 confusion matrix
confusion_matrix_summary(y_true, y_pred, positive_class=1)
# prints: sensitivity, specificity, precision, NPV, FDR, FPR, F1, balanced accuracy, MCC

# ROC curve and AUC
from python import roc_curve, auc_score
fpr, tpr, thresholds, auc = roc_curve(y_true, probs)
print(f"AUC = {auc:.4f}")
```

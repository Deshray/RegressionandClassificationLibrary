/*
 * logreg.h
 * --------
 * Binary logistic regression via IRLS (Newton-Raphson) following ISLP Ch. 4.
 *
 * Model (ISLP Eq 4.2):
 *   p(X) = e^{beta_0 + beta_1*X_1 + ... + beta_p*X_p}
 *          / (1 + e^{beta_0 + ... + beta_p*X_p})
 *
 * which can also be written as the logit (log-odds) being linear in X (Eq 4.4):
 *   log( p(X)/(1-p(X)) ) = beta_0 + beta_1*X_1 + ... + beta_p*X_p
 *
 * Estimation: Maximum likelihood via IRLS (Iteratively Reweighted Least Squares).
 * This is the standard Newton-Raphson algorithm applied to the log-likelihood.
 *
 *   Iteration:
 *     eta   = X @ beta                         (linear predictor)
 *     mu    = sigmoid(eta)                     (fitted probabilities)
 *     W     = diag(mu * (1 - mu))              (IRLS weight matrix)
 *     z     = eta + W^{-1} @ (y - mu)          (working response)
 *     beta  = (X'WX)^{-1} X'Wz                (WLS normal equations)
 *
 * Standard errors and inference (ISLP 4.3.2):
 *   Cov(beta_hat) = (X'WX)^{-1}               (information matrix inverse)
 *   SE(beta_hat_j)= sqrt( Cov[j,j] )
 *   z_j           = beta_hat_j / SE_j          (z-statistic, analogous to t in linreg)
 *   p_j           = 2 * P(Z > |z_j|)           (two-sided, Z ~ N(0,1))
 *
 * Deviance (ISLP 4.6 / generalised linear model framework):
 *   log-likelihood: ell(beta) = sum_i [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
 *   Null deviance:     D_null = -2 * ell(null model, intercept only)
 *   Residual deviance: D_resid= -2 * ell(fitted model)
 */

#ifndef LOGREG_H
#define LOGREG_H

/* ------------------------------------------------------------------ */
/* Result structure                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;            /* number of observations                       */
    int     p;            /* number of predictors (excluding intercept)   */

    /* Coefficients — length (p+1), index 0 is intercept */
    double *beta;         /* beta_hat[j]                                  */
    double *se;           /* SE(beta_hat[j])  via (X'WX)^{-1}            */
    double *z_stat;       /* z_j = beta_hat[j] / SE[j]  (ISLP 4.3.2)    */
    double *p_value;      /* 2*P(Z > |z_j|),  Z ~ N(0,1)                */

    /* Deviance (ISLP 4.6 / GLM framework) */
    double  null_deviance;   /* -2 * log-lik of intercept-only model     */
    double  deviance;        /* -2 * log-lik of fitted model             */
    double  aic;             /* AIC = deviance + 2*(p+1)                 */

    int     converged;       /* 1 if IRLS converged, 0 otherwise         */
    int     n_iter;          /* number of IRLS iterations taken           */

} LogRegResult;

/* ------------------------------------------------------------------ */
/* API                                                                  */
/* ------------------------------------------------------------------ */

/*
 * Fit binary logistic regression via IRLS.
 *
 *   X     : (n x p) predictor matrix WITHOUT intercept column.
 *   y     : (n) binary response: values must be 0 or 1.
 *   n     : number of observations.
 *   p     : number of predictors.
 *   max_iter : maximum IRLS iterations (100 is usually sufficient).
 *   tol   : convergence tolerance on max |delta_beta|  (e.g. 1e-8).
 *
 * Returns a heap-allocated LogRegResult.  Caller must call logreg_free().
 * Returns NULL on failure.
 */
LogRegResult *logreg_fit(const double *X, const double *y, int n, int p,
                         int max_iter, double tol);

/*
 * Free a LogRegResult.
 */
void logreg_free(LogRegResult *r);

/*
 * Predicted probabilities  p_hat = sigmoid(X_new @ beta_hat).
 *
 *   X_new  : (m x p) new predictor matrix (no intercept column).
 *   prob   : (m) output — caller allocates.
 */
void logreg_predict_prob(const LogRegResult *r,
                         const double *X_new, double *prob, int m);

/*
 * Predicted class labels using a threshold.
 *
 *   X_new    : (m x p) new predictor matrix.
 *   labels   : (m) integer output (0 or 1) — caller allocates.
 *   threshold: classify as 1 if p_hat > threshold (typically 0.5).
 */
void logreg_predict(const LogRegResult *r,
                    const double *X_new, int *labels,
                    int m, double threshold);

#endif /* LOGREG_H */

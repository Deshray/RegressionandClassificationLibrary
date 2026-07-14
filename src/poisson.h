/*
 * poisson.h
 * ---------
 * Poisson regression (GLM with log link), ISLP Section 4.6.2.
 *
 * Model (ISLP Eq 4.34):
 *   log(E[Y|X]) = beta_0 + beta_1*X_1 + ... + beta_p*X_p
 *   E[Y|X] = exp(X'beta)
 *   Y|X ~ Poisson(exp(X'beta))
 *
 * The log link is the natural link for the Poisson family and ensures
 * the fitted values are always positive.
 *
 * Estimation via IRLS (same framework as logistic regression, ISLP §4.6.3):
 *   IRLS weights:   w_i = mu_i = exp(x_i'beta)
 *   Working response: z_i = x_i'beta + (y_i - mu_i)/mu_i
 *   Update: beta = (X'WX)^{-1} X'Wz
 *
 * Inference:
 *   Cov(beta_hat) = (X'WX)^{-1}    (observed Fisher information)
 *   SE(beta_hat_j) = sqrt(Cov[j,j])
 *   z_j = beta_hat_j / SE_j         (z-statistic, Normal approximation)
 *   p_j = 2*P(Z > |z_j|)
 *
 * Deviance (ISLP §4.6.2):
 *   Residual deviance D = 2 * sum_i [y_i*log(y_i/mu_i) - (y_i - mu_i)]
 *   (saturated - fitted model, two times negative log-likelihood ratio)
 *   Null deviance: deviance of intercept-only model.
 *   AIC = -2*log-lik + 2*(p+1) = D_resid - 2*sum_i(-log(y_i!)) + 2*(p+1)
 *         (simplified to: deviance + 2*(p+1) when comparing same data)
 */

#ifndef POISSON_H
#define POISSON_H

typedef struct {
    int     n;
    int     p;

    double *beta;       /* (p+1) coefficient estimates                  */
    double *se;         /* (p+1) standard errors                        */
    double *z_stat;     /* (p+1) z-statistics                           */
    double *p_value;    /* (p+1) two-sided p-values                     */

    double  null_deviance;    /* deviance of intercept-only model       */
    double  deviance;         /* residual deviance                      */
    double  aic;              /* AIC = deviance + 2*(p+1)               */

    int     converged;
    int     n_iter;
} PoissonResult;

/*
 * Fit Poisson regression via IRLS.
 *   X       : (n x p) predictor matrix (no intercept).
 *   y       : (n) non-negative integer response (count data).
 *   n, p    : dimensions.
 *   max_iter, tol : convergence parameters.
 */
PoissonResult *poisson_fit(const double *X, const double *y, int n, int p,
                            int max_iter, double tol);

void poisson_free(PoissonResult *r);

/*
 * Predicted mean response mu_hat = exp(X_new @ beta).
 *   X_new  : (m x p) new predictors (no intercept column).
 *   mu_hat : (m) output.
 */
void poisson_predict(const PoissonResult *r, const double *X_new,
                     double *mu_hat, int m);

#endif /* POISSON_H */
